import pandas as pd

def load_results(path: str = "data/processed/race_results.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
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
    FINISHED = {"Finished", "+1 Lap", "+2 Laps", "+6 Laps", "+3 Laps", "+4 Laps", "+5 Laps", "Lapped"}
    MECHANICAL = {"Fuel pressure", "Power Unit", "Cooling system", "Water pump", "Gearbox", "Fuel leak",
                   "Power loss", "Turbo", "Mechanical", "Water pressure", "Vibrations", "Hydraulics", "Engine",
                   "Fuel pump", "Undertray", "Differential", "Oil leak", "Front wing", "Rear wing", "Clutch",
                   "Suspension", "Brakes", "Electrical", "Did not start", "Retired", "DNS"}
    DRIVER_ERROR = {"Collision damage", "Collision", "Accident", "Spun off"}

    # Season-to-date rate for each, going INTO the race.
    # group driver+year (resets each season), shift(1) excludes current race (leakage),
    # expanding().mean() = cumulative fraction over all prior races this season.

    df["mechanical_dnf"] = df["Status"].isin(MECHANICAL).astype(int)
    df["driver_dnf"] = df["Status"].isin(DRIVER_ERROR).astype(int)
    df["MechanicalDNF_Rate"] = df.groupby(["DriverId", "year"])["mechanical_dnf"].transform(lambda x: x.shift(1).expanding().mean())
    df["DriverDNF_Rate"] = df.groupby(["DriverId", "year"])["driver_dnf"].transform(lambda x: x.shift(1).expanding().mean())
    return df

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
    pu_round_summary["PU_DNF_Rate"] = pu_round_summary.sort_values(["year", "round_number"]).groupby(["year", "PU_Group"])["PU_DNF_Rate"].transform(lambda x: x.shift(1).expanding().mean())
    df = df.merge(pu_round_summary, on=["year", "round_number", "PU_Group"], how="left")
    return df

def add_standings_position(df: pd.DataFrame) -> pd.DataFrame:
    # Add current championship position for driver and constructor.
    df["Drivers_Standings"] = df.groupby(["year", "round_number"])["SeasonPoints"].rank(method='min', ascending=False)
    df["Constructors_Standings"] = df.groupby(["year", "round_number"])["ConstructorPoints"].rank(method='min', ascending=False)
    return df

def add_tire_deg_ratings(df: pd.DataFrame) -> pd.DataFrame:
    tire_deg_csv = pd.read_csv('data/processed/tire_deg_ratings.csv')
    ordinal_map = {"low": 1, "medium": 2, "high": 3}
    tire_deg_csv["deg_rating"] = tire_deg_csv["deg_rating"].map(ordinal_map)
    df = df.merge(tire_deg_csv, left_on="Location", right_on="location", how="left")
    df = df.drop(columns=['circuit', 'deg_notes', 'country', 'track_archetypes'])
    return df