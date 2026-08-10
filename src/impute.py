from sklearn.preprocessing import LabelEncoder

# Column groupings by NaN cause, established via full audit of
# build_feature_table()'s output (2022-2024, 1359 rows x 59 cols).
# Cold-start columns: NaN because the driver/team/constructor has no
# prior history yet within the dataset window (rookie debut, new team,
# or simply early rows in the 2022-2024 table before enough races have
# accumulated). NaN rate naturally decays over the dataset's timeline.
COLD_START_COLS = [
    "DriverCircuitAvgFinish",
    "DriverDNF_Rate",
    "MechanicalDNF_Rate",
    "PU_DNF_Rate",
    "TeamStrategyScore",
    "RollingFinish",
    "DiscountedCareerPosition",
    "CareerMechanicalDNFRate",
    "CareerDriverErrorDNFRate",
    "ConstructorArchetypePerformance",
]

# Structural columns: NaN because Imola and Le Castellet raced 2022-2024
# but are off the 2026 calendar, so they're deliberately excluded from
# tire_deg_ratings.csv. This is permanent, not a cold-start pattern, 
# these specific rows will never have this data.
STRUCTURAL_NAN_COLS = [
    "deg_rating",
    "track_data_available",
    "archetype_balanced",
    "archetype_cold_track",
    "archetype_high_altitude",
    "archetype_high_deg",
    "archetype_high_downforce",
    "archetype_high_speed",
    "archetype_high_speed_sweeping",
    "archetype_power",
    "archetype_qualifying",
    "archetype_street",
    "archetype_technical",
]

# Edge case: exactly 2 rows (Stroll 2023 R15, Schumacher 2022 R2), both
# Status == "Withdrew" as the driver never started, so no grid/finish position exists.
WITHDREW_EDGE_COLS = [
    "Position",
    "GridPosition",
]

# Columns describing the race's outcome: post-race information, would
# leak the answer directly into the model if used as inputs.
LEAKAGE_COLS = [
    "Position",
    "ClassifiedPosition",
    "Status",
    "Points",
    "Laps",
    "mechanical_dnf",
    "driver_dnf",
]

# Pure identifier/display columns, has no predictive value.
IDENTIFIER_COLS = [
    "BroadcastName",
    "FirstName",
    "LastName",
    "FullName",
    "HeadshotUrl",
    "TeamColor",
    "CountryCode",
    "Abbreviation",
    "DriverNumber",
    "TeamName",
]

# Rebrand consolidation: AlphaTauri->RB and Alfa Romeo->Kick Sauber are the
# same underlying team across the rename. Standardized on the *current* (2024+) identity, 
# since that's what the model will be predicting on going into 2026.
TEAM_REBRAND_MAP = {
    "alphatauri": "rb",
    "alfa": "sauber",
}

def verify_nan_coverage(df):
    """
    Confirms every column with NaN values is accounted for in one of the
    three categorized lists (COLD_START_COLS, STRUCTURAL_NAN_COLS,
    WITHDREW_EDGE_COLS). Raises if an uncategorized NaN column appears —
    catches silent gaps before they reach imputation, e.g. if a future
    feature change introduces a new NaN pattern that hasn't been audited.
    """
    nan_cols = set(df.columns[df.isna().any()])
    categorized = set(COLD_START_COLS + STRUCTURAL_NAN_COLS + WITHDREW_EDGE_COLS)
    uncategorized = nan_cols - categorized

    if uncategorized:
        raise ValueError(f"Uncategorized NaN columns found: {uncategorized}")

    print("All NaN columns accounted for.")

def drop_unusable_rows(df):
    """
    Drops rows that can't be meaningfully used for training or imputed
    sensibly:
    - Imola/Le Castellet rows (STRUCTURAL_NAN_COLS gap), these are permanently
      missing tire/archetype data since these circuits are off the 2026
      calendar; imputing a fake value for a track the model will never
      predict on adds noise, not signal.
    - The 2 Withdrew rows (WITHDREW_EDGE_COLS gap), because the driver never started,
      no real grid/finish position exists to impute.
    """
    before = len(df)

    df = df[~df["Location"].isin(["Imola", "Le Castellet"])]
    df = df[df["GridPosition"].notna() & df["Position"].notna()]

    after = len(df)
    print(f"Dropped {before - after} rows ({before} -> {after})")

    return df

def split_time_based(df, val_start_round=1, val_end_round=12, test_start_round=13):
    """
    Time-based train/val/test split, respecting chronological order to
    avoid leakage (per project specas random k-fold would let the model
    train on future seasons and validate on the past).

    Train: 2022 + 2023 (all rounds)
    Val:   2024, rounds val_start_round through val_end_round
    Test:  2024, rounds test_start_round onward

    Mirrors the real live-prediction scenario: we're using all prior seasons
    to predict an unseen season, split by round to preserve chronology
    within 2024 as well.
    """
    train = df[df["year"].isin([2022, 2023])]

    val = df[
        (df["year"] == 2024)
        & (df["round_number"] >= val_start_round)
        & (df["round_number"] <= val_end_round)
    ]

    test = df[
        (df["year"] == 2024)
        & (df["round_number"] >= test_start_round)
    ]

    print(f"Train: {len(train)} rows ({train['year'].min()}-{train['year'].max()})")
    print(f"Val:   {len(val)} rows (2024 R{val_start_round}-R{val_end_round})")
    print(f"Test:  {len(test)} rows (2024 R{test_start_round}+)")
    print(f"Total: {len(train)+len(val)+len(test)} (original: {len(df)})")

    return train, val, test

def impute_cold_start(train, val, test, cold_start_cols=COLD_START_COLS):
    """
    Imputes COLD_START_COLS using means computed from the training set
    only, applied identically across train/val/test to avoid leakage
    (val/test never contribute to their own imputation values).

    Adds a single shared is_cold_start flag: True if any COLD_START_COLS
    value was NaN before imputation so the model can learn to weight
    imputed rows differently from rows with real historical data.
    """
    train = train.copy()
    val = val.copy()
    test = test.copy()

    # flag BEFORE imputing — must capture NaN state prior to fill
    train["is_cold_start"] = train[cold_start_cols].isna().any(axis=1)
    val["is_cold_start"] = val[cold_start_cols].isna().any(axis=1)
    test["is_cold_start"] = test[cold_start_cols].isna().any(axis=1)

    # means from train only
    train_means = train[cold_start_cols].mean()

    train[cold_start_cols] = train[cold_start_cols].fillna(train_means)
    val[cold_start_cols] = val[cold_start_cols].fillna(train_means)
    test[cold_start_cols] = test[cold_start_cols].fillna(train_means)

    return train, val, test

def prepare_model_features(df, label_encoders=None):
    """
    Builds the binary Podium target (Position <= 3), drops leakage columns
    (post-race outcomes: Position, Status, Points, Laps, etc.) and pure
    identifier/display columns (names, headshot URLs, etc.), and label-
    encodes categorical columns (DriverId, TeamId, PU_Group, Location).

    label_encoders: dict of {column: LabelEncoder}, fit on train and
    passed in when encoding val/test and ensures consistent encoding across
    splits and avoids fitting encoders on val/test data (leakage).
    Pass None when encoding the training set for the first time; the
    function will fit new encoders and return them for reuse.
    """
    from sklearn.preprocessing import LabelEncoder

    df = df.copy()
    df["Podium"] = (df["Position"] <= 3).astype(int)

    df = df.drop(columns=LEAKAGE_COLS + IDENTIFIER_COLS + ["TeamName"])

    df["TeamId"] = df["TeamId"].replace(TEAM_REBRAND_MAP)

    categorical_cols = ["DriverId", "TeamId", "PU_Group", "Location"]

    fit_new = label_encoders is None
    if fit_new:
        label_encoders = {}

    for col in categorical_cols:
        if fit_new:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        else:
            le = label_encoders[col]
            df[col] = df[col].astype(str).map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    # bool columns to int, for model compatibility
    df["is_cold_start"] = df["is_cold_start"].astype(int)
    df["rain_flag"] = df["rain_flag"].astype(int)

    return df, label_encoders