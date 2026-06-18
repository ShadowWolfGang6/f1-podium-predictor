import os
import fastf1
import pandas as pd

def setup_cache():
    cache_dir = "data/raw"
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

def get_race_results(year, round_number):
    session = fastf1.get_session(year, round_number, 'R')
    session.load(telemetry=False, weather=False, messages=False)
    df = session.results
    df["year"] = year
    df["round_number"] = round_number
    return df

def get_all_results(start_year, end_year):
    all_results = []
    for year in range(start_year, end_year + 1):
        for round_number in range(1, 25):
            try:
                race_results = get_race_results(year, round_number)
                all_results.append(race_results)
            except Exception as e:
                    print(f"Skipped {year} R{round_number}: {e}")
                    continue
    os.makedirs("data/processed", exist_ok=True)
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_parquet("data/processed/race_results.parquet")
    return combined


if __name__ == "__main__":
    setup_cache()
    get_all_results(2022, 2024)