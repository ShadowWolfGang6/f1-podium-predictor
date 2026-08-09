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
}

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
    pu_round_summary["PU_DNF_Rate"] = pu_round_summary.sort_values(["year", "round_number"]).groupby(["year", "PU_Group"])["PU_DNF_Rate"].transform(lambda x: x.shift(1).expanding().mean())
    df = df.merge(pu_round_summary, on=["year", "round_number", "PU_Group"], how="left")
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
    df = df.drop(columns=['circuit', 'deg_notes', 'country', 'track_archetypes'])
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

    # merge the two, and then compute the delta
    merged = team_race_avg.merge(field_avg, on=["year", "round_number"])
    merged["strategy_delta"] = merged["team_avg_pit"] - merged["field_avg_pit"]

    # merge onto main df
    df = df.merge(
        merged[["year", "round_number", "TeamId", "strategy_delta"]],
        on=["year", "round_number", "TeamId"],
        how="left"
    )

    # leakage-safe season-to-date score 
    df["TeamStrategyScore"] = df.groupby(["TeamId", "year"])["strategy_delta"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    df = df.drop(columns=["strategy_delta"])
    return df

def add_weather_features(df, weather_path):
    """
    Merges weather data (actual or forecast) onto the main feature
    table, joined on (year, round_number). No leakage-safe shift
    needed — weather is per-race exogenous data, not derived from
    past race outcomes.
    """
    weather = pd.read_csv(weather_path) if weather_path.endswith(".csv") else pd.read_parquet(weather_path)

    # keep only the columns you actually want as model features
    weather_cols = ["year", "round_number", "rain_flag", "avg_track_temp", "avg_air_temp", "humidity"]
    weather = weather[weather_cols]

    merged = df.merge(weather, on=["year", "round_number"], how="left")

    return merged