import fastf1
import numpy as np
import pandas as pd
from starlette import requests
FINISHED = {"Finished", "+1 Lap", "+2 Laps", "+6 Laps", "+3 Laps", "+4 Laps", "+5 Laps", "Lapped"}
DRIVER_ERROR = {"Collision damage", "Collision", "Accident", "Spun off"}
MECHANICAL = {"Fuel pressure", "Power Unit", "Cooling system", "Water pump", "Gearbox", "Fuel leak",
                   "Power loss", "Turbo", "Mechanical", "Water pressure", "Vibrations", "Hydraulics", "Engine",
                   "Fuel pump", "Undertray", "Differential", "Oil leak", "Front wing", "Rear wing", "Clutch",
                   "Suspension", "Brakes", "Electrical", "Did not start", "Retired", "DNS"}
TEAM_NAME_TO_ID = {
    "Alpine": "alpine",
    "Aston Martin": "aston_martin",
    "Ferrari": "ferrari",
    "Haas F1 Team": "haas",
    "McLaren": "mclaren",
    "Mercedes": "mercedes",
    "Red Bull Racing": "red_bull",
    "Williams": "williams",
    "AlphaTauri": "alphatauri",
    "RB": "rb",
    "Alfa Romeo": "alfa",
    "Kick Sauber": "sauber",
    "Cadillac": "cadillac",
}

LOCATION_NORMALIZE_MAP = {
    "Miami Gardens": "Miami",
    "Monte Carlo": "Monaco",
}

def load_results(path: str = "data/processed/race_results.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["Location"] = df["Location"].replace(LOCATION_NORMALIZE_MAP)
    df = df.sort_values(by=["year", "round_number", "Position"])
    return df

def add_rolling_finish(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    df["RollingFinish"] = df.groupby("DriverId")["Position"].transform(lambda x: x.shift(1).rolling(window=n, min_periods=1).mean())
    return df

# Rolling mean of last n finishing positions = recent form signal.
# shift(1) excludes the current race (prevents leakage: which means we 
# can't use the result we're predicting). 
# min_periods=1 lets early-career/early-season
# rows compute from fewer races instead of returning NaN.

def add_dnf_rate(df: pd.DataFrame) -> pd.DataFrame:
    # Split DNFs by cause — two different signals:
    # mechanical = car/PU reliability (out of driver's control)
    # driver = driver error (crashes, spins)
    # Retired -> mechanical (catch-all, almost always a failure).
    # Withdrew / Disqualified -> neither (precautionary/technical, not a fault signal).
    # DNS -> neither.
    # Season-to-date rate for each, going INTO the race.
    # group driver+year (resets each season), shift(1) excludes current race (leakage),
    # expanding().mean() = cumulative fraction over all prior races this season.

    df["mechanical_dnf"] = df["Status"].isin(MECHANICAL).astype(int)
    df["driver_dnf"] = df["Status"].isin(DRIVER_ERROR).astype(int)
    df["MechanicalDNF_Rate"] = df.groupby(["DriverId", "year"])["mechanical_dnf"].transform(lambda x: x.shift(1).expanding().mean())
    df["DriverDNF_Rate"] = df.groupby(["DriverId", "year"])["driver_dnf"].transform(lambda x: x.shift(1).expanding().mean())
    return df

def map_team_to_id(team_name, year):
    return TEAM_NAME_TO_ID.get(team_name, team_name)

def add_standings(df: pd.DataFrame) -> pd.DataFrame:
    df["Points"] = df["Points"].fillna(0)

    # Driver championship points going INTO the race.
    # shift(1) excludes current race; cumsum = season-to-date total; fillna(0) = race 1 starts at 0.

    df["SeasonPoints"] = df.groupby(["DriverId", "year"])["Points"].transform(lambda x: x.shift(1).cumsum()).fillna(0)
    
    # Constructor points: two driver rows per team per round, so a direct transform double-counts.
    # Fix: collapse to one row per team per round (sum both drivers), cumsum that, merge back.

    team_round = df.groupby(["TeamId", "year", "round_number"], as_index=False)["Points"].sum()
    team_round["ConstructorPoints"] = team_round.groupby(["TeamId", "year"])["Points"].transform(
    lambda x: x.shift(1).cumsum()).fillna(0)
    df = df.merge(
    team_round[["TeamId", "year", "round_number", "ConstructorPoints"]],
    on=["TeamId", "year", "round_number"],
    how="left")
    return df

def add_reg_era_flags(df: pd.DataFrame) -> pd.DataFrame:
    # Era flags: simple binary indicators for major regulation eras (PU, aero, tires).
    # These capture broad shifts in competitive dynamics and team performance.
    df["GroundEffectEra"] = ((df["year"] >= 2022) & (df["year"] < 2026)).astype(int)
    df["ActiveAeroEra"] = ((df["year"] >= 2026) & (df["year"] < 2030)).astype(int)
    era_start_map = {2022: 2022, 2023: 2022, 2024: 2022, 2025: 2022, 2026: 2026}
    df["EraStartYear"] = df["year"].map(era_start_map)
    era_races = (
    df[["EraStartYear", "year", "round_number"]]
    .drop_duplicates()
    .sort_values(["year", "round_number"])
    )
    era_races["EraRaceNumber"] = era_races.groupby("EraStartYear").cumcount() + 1
    df = df.merge(
    era_races[["year", "round_number", "EraRaceNumber"]],
    on=["year", "round_number"],
    how="left"
    )
    return df

def add_pu_dnf_rate(df: pd.DataFrame) -> pd.DataFrame:
    # Power Unit DNF rate: season-to-date number of PU DNFs for each PU supplier for the season
    # going INTO the race. Captures reliability signal at the team level (since teams share PU suppliers).
    # group by PU supplier + year (resets each season), shift(1) excludes current race (leakage)
    pu_df = pd.read_csv('data/processed/pu_supplier_history.csv')
    merged = df.merge(pu_df, on='TeamId', how='left')
    filtered = merged[(merged["year"] >= merged["start_year"]) & (merged["year"] <= merged["end_year"])]
    filtered = filtered.drop(columns=["start_year", "end_year", "note"])

    pu_round_summary = (
        filtered
        .groupby(["year", "round_number", "PU_Group"])
        .agg(
            dnf_count=("mechanical_dnf", "sum"),
            car_count=("mechanical_dnf", "count")
        )
        .reset_index()
    )
    pu_round_summary["PU_DNF_Rate"] = pu_round_summary["dnf_count"] / pu_round_summary["car_count"]
    pu_round_summary["PU_DNF_Rate"] = (
        pu_round_summary.sort_values(["year", "round_number"])
        .groupby(["year", "PU_Group"])["PU_DNF_Rate"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    df = df.merge(pu_df[["TeamId", "PU_Group", "start_year", "end_year"]], on="TeamId", how="left")
    df = df[(df["year"] >= df["start_year"]) & (df["year"] <= df["end_year"])]
    df = df.drop(columns=["start_year", "end_year"])
    df = df.merge(pu_round_summary[["year", "round_number", "PU_Group", "PU_DNF_Rate"]], on=["year", "round_number", "PU_Group"], how="left")

    return df

def add_standings_position(df: pd.DataFrame) -> pd.DataFrame:
    # Add current championship position for driver and constructor.
    df["Drivers_Standings"] = df.groupby(["year", "round_number"])["SeasonPoints"].rank(method='min', ascending=False)
    df["Constructors_Standings"] = df.groupby(["year", "round_number"])["ConstructorPoints"].rank(method='min', ascending=False)
    return df

def add_tire_deg_ratings(df: pd.DataFrame) -> pd.DataFrame:
    # Tire degradation rating per circuit, ordinal-encoded (low=1, medium=2, high=3) since
    # the categories have genuine order that preserves order.
    # Joined on 'Location' (FastF1's circuit-locale string, e.g. "Sakhir"), not the CSV's 'circuit'
    # column (official name, e.g. "Bahrain International Circuit") because FastF1 doesn't expose official
    # circuit names, so the CSV's 'location' column was manually built to match FastF1's exact strings.
    # 'unknown' (Madring/new tracks with no history) maps to NaN via .map(), this is intentional, not a bug.
    # Pairs with track_data_available flag so the model/imputation step knows that it's missing,
    # consistent with the a "honest NaN over a blanket imputation" principle used elsewhere.
    # Imola and Le Castellet (raced 2022-2024, off the 2026 calendar) are intentionally excluded from
    # the reference CSV as those historical rows will show NaN here.
    
    tire_deg_csv = pd.read_csv('data/processed/tire_deg_ratings.csv')
    ordinal_map = {"low": 1, "medium": 2, "high": 3}
    tire_deg_csv["deg_rating"] = tire_deg_csv["deg_rating"].map(ordinal_map)
    df = df.merge(tire_deg_csv, left_on="Location", right_on="location", how="left")
    df = df.drop(columns=['circuit', 'deg_notes', 'country', 'track_archetypes', 'location'])
    return df

def add_circuit_history(df: pd.DataFrame) -> pd.DataFrame:
    # A driver's historical average finish on a given circuit. 
    # Grouped by DriverID and Location, as we're looking at their finishes across all
    # races they've driven at this circuit, not on a specific year(s)
    # This gives us a good idea of what circuits drivers tend to score well at, and which ones 
    # They struggle on.
    df["DriverCircuitAvgFinish"] = df.groupby(["DriverId", "Location"])["Position"].transform(
    lambda x: x.shift(1).expanding().mean()
    )
    return df

def add_track_archetype(df: pd.DataFrame) -> pd.DataFrame:
    # Multi-hot encodes the semicolon-separated track_archetypes tags from tire_deg_ratings.csv
    # (e.g. "street; balanced" -> archetype_street=1, archetype_balanced=1).
    # Tags are stripped of whitespace before get_dummies to avoid " balanced" vs "balanced" duplicates.
    # Imola/Le Castellet (off 2026 calendar, absent from CSV) deliberately preserved as honest NaN
    # across all archetype columns rather than 0 "no data" != "confirmed not this archetype."
    # Required casting dummy columns to nullable Int64, since plain int64 can't hold pd.NA.
    
    track_archetypes = pd.read_csv('data/processed/tire_deg_ratings.csv')[['location', 'track_archetypes']]
    df = df.merge(track_archetypes, left_on="Location", right_on='location', how="left")
    df = df.drop(columns=['location'])
    missing_mask = df['track_archetypes'].isna()
    cleaned = df['track_archetypes'].apply(
    lambda val: ';'.join(t.strip() for t in val.split(';')) if isinstance(val, str) else val
    )
    dummies = cleaned.str.get_dummies(sep=';').add_prefix('archetype_')
    df = pd.concat([df, dummies], axis=1)
    archetype_cols = dummies.columns
    df[archetype_cols] = df[archetype_cols].astype('Int64')
    df.loc[missing_mask, archetype_cols] = pd.NA
    df = df.drop(columns=['track_archetypes'])
    return df

def build_circuit_tag_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by Location, columns = archetype_* dummies,
    one row per unique circuit, values 0/1 (Int64, NaN-circuits dropped or kept—decide).
    """
    archetype_cols = [col for col in df.columns if col.startswith("archetype_")]
    # one row per unique circuit, every race at a given circuit has identical tags
    tag_matrix = df[["Location"] + archetype_cols].drop_duplicates(subset="Location")
    # drop circuits with missing archetype data (Imola, Le Castellet — off 2026 calendar)
    # honest-NaN principle: exclude rather than impute
    tag_matrix = tag_matrix.dropna(subset=archetype_cols)
    tag_matrix = tag_matrix.set_index("Location")
    # cast to plain int — safe now that NaN rows are gone, needed for matrix mult below
    tag_matrix[archetype_cols] = tag_matrix[archetype_cols].astype(int)
    M = tag_matrix.values
    # dot product of binary vectors = count of shared tags (intersection size)
    intersection = M @ M.T
    # tags per circuit
    row_sums = M.sum(axis=1)
    # |A union B| = |A| + |B| - |A intersect B|
    union = row_sums[:, None] + row_sums[None, :] - intersection
    # Jaccard similarity = |intersection| / |union|, elementwise
    jaccard = intersection / union
    jaccard_df = pd.DataFrame(jaccard, index=tag_matrix.index, columns=tag_matrix.index)
    return jaccard_df


def add_constructor_archetype_performance(df: pd.DataFrame, jaccard_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (TeamId, race), computes Jaccard-weighted average of the
    constructor's finishing position at prior circuits, weighted by archetype
    similarity to the current circuit.
    """
    df = df.sort_values(["TeamId", "year", "round_number"])
    results = {}
    for team_id, group in df.groupby("TeamId"):
        history = []
        for idx, row in group.iterrows():
            current_location = row["Location"]

            if len(history) == 0:
                weighted_avg = np.nan
            else:
                if current_location not in jaccard_df.index:
                     weighted_avg = np.nan
                else:
                    weights = [jaccard_df.loc[current_location, loc] for loc, pos in history if loc in jaccard_df.index]
                    values = [pos for loc, pos in history if loc in jaccard_df.index]
                    total_weight = sum(weights)
                    if total_weight == 0:
                        weighted_avg = np.nan
                    else:
                        weighted_avg = sum(w * v for w, v in zip(weights, values)) / total_weight
            results[idx] = weighted_avg

            is_classified = row["Status"] in FINISHED
            if is_classified:
                history.append((current_location, row["Position"]))

    df["ConstructorArchetypePerformance"] = df.index.map(results)
    return df

def add_team_strategy_score(df: pd.DataFrame, pit_stops_path="data/processed/pit_stops.parquet") -> pd.DataFrame:
    # Team strategy score: a composite metric capturing a team's strategic performance
    # across races, based on pit stop efficiency
    max_duration = 60 # exclude non-representative stops (red flags, stoppages) as real stops are ~15-35s
    pits = pd.read_parquet(pit_stops_path)
    pits["TeamId"] = pits["Team"].map(TEAM_NAME_TO_ID)

    # exclude non-representative stops (red flags, stoppages) — real stops are ~15-35s
    pits = pits[pits["pit_duration"] <= max_duration]

    # team's avg pit duration per race
    team_race_avg = pits.groupby(["year", "round_number", "TeamId"])["pit_duration"].mean()
    team_race_avg = team_race_avg.reset_index().rename(columns={"pit_duration": "team_avg_pit"})

    # field avg pit duration per race (all teams combined)
    field_avg = pits.groupby(["year", "round_number"])["pit_duration"].mean()
    field_avg = field_avg.reset_index().rename(columns={"pit_duration": "field_avg_pit"})

    merged = team_race_avg.merge(field_avg, on=["year", "round_number"])
    merged["strategy_delta"] = merged["team_avg_pit"] - merged["field_avg_pit"]

    # merge onto main df
    df = df.merge(
        merged[["year", "round_number", "TeamId", "strategy_delta"]],
        on=["year", "round_number", "TeamId"],
        how="left"
    )

    df["TeamStrategyScore"] = df.groupby(["TeamId", "year"])["strategy_delta"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    df = df.drop(columns=["strategy_delta"])
    return df

def add_weather_features(df, weather_path="data/processed/weather.parquet"):
    """
    Merges weather data (actual or forecast) onto the main feature
    table, joined on (year, round_number). No leakage-safe shift
    needed, weather is per-race exogenous data, not derived from
    past race outcomes.
    """
    weather = pd.read_csv(weather_path) if weather_path.endswith(".csv") else pd.read_parquet(weather_path)

    weather_cols = ["year", "round_number", "rain_flag", "avg_track_temp", "avg_air_temp", "humidity"]
    weather = weather[weather_cols]

    merged = df.merge(weather, on=["year", "round_number"], how="left")

    return merged

def add_career_dnf_rate(df, extended_results_path="data/processed/race_results_extended.parquet"):
    """
    Adds CareerMechanicalDNFRate and CareerDriverErrorDNFRate, a driver's
    mechanical and driver-error DNF rate across their full career history
    (2018+), not just the current season.

    Uses a separate, wider historical table (race_results_extended.parquet,
    2018-2024) rather than the main df, since veterans need career depth
    that the main 2022+ table doesn't have. Leakage-safe: shift(1) before
    expanding() ensures each race only sees DNFs from strictly prior races.
    No year reset in the groupby as this is career-spanning, unlike the
    season-level DNF rate.

    Merged onto df via left join on (DriverId, year, round_number), so row
    count is preserved. Rows for drivers/races outside the extended table's
    coverage will have NaN, consistent with the project's honest-NaN
    convention (no fillna at this stage).

    Known limitation: 2024 Round 3 (Australian GP) is missing Ollie
    Bearman's row in the extended table — a one-off substitute driver not
    captured in FastF1/Jolpica's session data for that race.
    """
    hist = pd.read_parquet(extended_results_path)
    hist = hist.sort_values(["DriverId", "year", "round_number"])

    hist["is_mechanical_dnf"] = hist["Status"].isin(MECHANICAL)
    hist["is_driver_error_dnf"] = hist["Status"].isin(DRIVER_ERROR)

    hist["CareerMechanicalDNFRate"] = (
        hist.groupby("DriverId")["is_mechanical_dnf"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    hist["CareerDriverErrorDNFRate"] = (
        hist.groupby("DriverId")["is_driver_error_dnf"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    merge_cols = ["DriverId", "year", "round_number", "CareerMechanicalDNFRate", "CareerDriverErrorDNFRate"]
    df = df.merge(hist[merge_cols], on=["DriverId", "year", "round_number"], how="left")

    return df

DECAY_RATE = 0.7

def add_multiseason_history(df, extended_results_path="data/processed/race_results_extended.parquet", decay_rate=DECAY_RATE):
    """
    Adds DiscountedCareerPosition: a driver's finishing position history
    weighted so recent seasons count more than distant ones, using
    exponential decay: weight = decay_rate ** years_ago.

    Uses the same extended historical table (race_results_extended.parquet,
    2018-2024) as add_career_dnf_rate(), this is because veterans need
    career depth the main 2022+ table doesn't have.

    decay_rate=0.7 is an initial estimate, not yet tuned against model
    performance.
    """
    hist = pd.read_parquet(extended_results_path)
    hist = hist.sort_values(["DriverId", "year", "round_number"])

    def weighted_history_for_driver(group):
        positions = group["Position"].values
        years = group["year"].values
        n = len(group)
        result = [None] * n

        for i in range(n):
            if i == 0:
                continue  # no prior races at all

            prior_positions = positions[:i]
            prior_years = years[:i]

            valid = ~pd.isna(prior_positions)  # drop DNF rows from the average
            if not valid.any():
                continue  # all prior races were DNFs, nothing to average

            prior_positions = prior_positions[valid]
            prior_years = prior_years[valid]

            years_ago = years[i] - prior_years
            weights = decay_rate ** years_ago
            result[i] = (prior_positions * weights).sum() / weights.sum()

        return pd.Series(result, index=group.index)

    hist["DiscountedCareerPosition"] = (
        hist.groupby("DriverId", group_keys=False)[["year", "round_number", "Position"]]
        .apply(weighted_history_for_driver, include_groups=False)
    )

    merge_cols = ["DriverId", "year", "round_number", "DiscountedCareerPosition"]
    df = df.merge(hist[merge_cols], on=["DriverId", "year", "round_number"], how="left")

    return df

def estimate_tire_degradation(year, round_number):
    """
    Estimates per-stint tire performance from lap times: degradation_slope
    (seconds lost per lap of tire wear) and baseline_pace (estimated
    fresh-tire lap time), fit via linear regression on clean in-stint laps
    (excludes out/in-laps and non-green-flag laps).

    Limitation: slope reflects net pace evolution, not isolated tire
    degradation as fuel burn (opposite direction) is not corrected for.
    """
    session = fastf1.get_session(year, round_number, 'R')
    session.load(laps=True, telemetry=False, weather=False, messages=False)
    laps = session.laps

    results = []
    for (driver, stint), stint_laps in laps.groupby(["Driver", "Stint"]):
        stint_laps = stint_laps.sort_values("LapNumber")

        if len(stint_laps) <= 2:
            continue
        clean_laps = stint_laps.iloc[1:-1]
        clean_laps = clean_laps[clean_laps["TrackStatus"] == "1"]
        clean_laps = clean_laps[clean_laps["LapTime"].notna()]

        if len(clean_laps) < 6:  
            continue

        tyre_life = clean_laps["TyreLife"].values
        lap_times = clean_laps["LapTime"].dt.total_seconds().values

        if len(set(tyre_life)) < 2:  # zero variance, can't fit a line
            continue

        try:
            slope, intercept = np.polyfit(tyre_life, lap_times, 1)
        except np.linalg.LinAlgError:
            continue

        results.append({
            "year": year,
            "round_number": round_number,
            "Location": session.event["Location"],
            "DriverId": driver,
            "Stint": stint,
            "Compound": clean_laps["Compound"].iloc[0],
            "degradation_slope": slope,       # seconds lost per lap of tire wear
            "baseline_pace": intercept,        # estimated fresh-tire lap time (sec)
            "n_laps": len(clean_laps),
        })

    return pd.DataFrame(results)

def aggregate_circuit_tire_degradation(stint_data_path="data/processed/tire_degradation.parquet"):
    """
    Aggregates per-stint tire degradation estimates to circuit x compound
    level: average degradation_slope, average baseline_pace, and stint
    count (n_stints) as a sample-size/confidence signal for downstream use.

    Drops rows with unknown compound ('None' string, not NaN — a FastF1
    data gap for irregular stints) before aggregating.
    """
    stints = pd.read_parquet(stint_data_path)
    stints = stints[stints["Compound"] != "None"]

    circuit_agg = (
        stints.groupby(["Location", "Compound"])
        .agg(
            avg_degradation_slope=("degradation_slope", "mean"),
            avg_baseline_pace=("baseline_pace", "mean"),
            n_stints=("degradation_slope", "count"),
        )
        .reset_index()
    )

    return circuit_agg

def add_telemetry_tire_deg(df, stint_data_path="data/processed/tire_degradation.parquet"):
    """
    Adds TelemetryDegradationSlope and TelemetryBaselinePace, a circuit-level
    tire degradation that estimates derived objectively from lap-time telemetry,
    as an alternative to the manually-curated add_tire_deg_ratings().

    Collapses the compound-level aggregation (aggregate_circuit_tire_degradation)
    further to Location only, averaging across all compounds, so this merges
    onto df at the same grain (one row per circuit) as the manual ratings 
    enable a clean model comparison between the two tire-deg approaches.
    """
    circuit_compound_agg = aggregate_circuit_tire_degradation(stint_data_path)

    circuit_agg = (
        circuit_compound_agg.groupby("Location")
        .agg(
            TelemetryDegradationSlope=("avg_degradation_slope", "mean"),
            TelemetryBaselinePace=("avg_baseline_pace", "mean"),
            TelemetryTotalStints=("n_stints", "sum"),
        )
        .reset_index()
    )

    df = df.merge(circuit_agg, on="Location", how="left")

    return df

def build_feature_table(use_telemetry_tire_deg=False):
    """
    Assembles the full feature table by chaining all Tier 1-3 add_* functions
    in dependency order, starting from load_results().

    use_telemetry_tire_deg toggles between the manual CSV-based tire deg
    rating (add_tire_deg_ratings) and the objective telemetry-derived one
    (add_telemetry_tire_deg) in line with the dual planned model which compares 
    the two models while all else held identical.
    """
    df = load_results()

    # Tier 1 features
    df = add_rolling_finish(df)
    df = add_dnf_rate(df)
    df = add_standings(df)
    df = add_reg_era_flags(df)
    df = add_pu_dnf_rate(df)
    df = add_standings_position(df)

    # Tier 2 features
    if use_telemetry_tire_deg:
        df = add_telemetry_tire_deg(df)
    else:
        df = add_tire_deg_ratings(df)
    df = add_circuit_history(df)
    df = add_track_archetype(df)

    jaccard_df = build_circuit_tag_matrix(df)
    df = add_constructor_archetype_performance(df, jaccard_df)

    df = add_team_strategy_score(df)
    df = add_weather_features(df)

    # Tier 3 features
    df = add_career_dnf_rate(df)
    df = add_multiseason_history(df)
    
    # GridPosition == 0 represents pit-lane starts, not an actual grid slot.
    # Remapping to max_grid_position + 1 so it's treated as worse than last,
    # not better than P1, otherwise the model reads it as an advantage
    max_grid = df["GridPosition"].max()
    df["GridPosition"] = df["GridPosition"].replace(0, max_grid + 1)
    
    # As per the project spec, qualifying times are not relevant, as the
    # model should only use starting position (GridPosition) as a feature, 
    # not the raw Q1/Q2/Q3 times.
    df = df.drop(columns=['Q1', 'Q2', 'Q3'])

    # Time is the race finish time (which is post-race) not something known
    # before lights out. Keeping it as a model input would be direct leakage
    # (the model would essentially see the answer). Dropping it here: it's
    # metadata about the race result, not a valid predictive feature.
    df = df.drop(columns=["Time"])

    return df