import os
import fastf1
import pandas as pd
import time
import requests
from src.features import estimate_tire_degradation

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
    df["Location"] = session.event["Location"]
    return df

def get_all_results(start_year, end_year, output_path="data/processed/race_results.parquet"):
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
    combined.to_parquet(output_path)
    return combined

def get_all_results_safe(start_year, end_year, output_path="data/processed/race_results.parquet"):
    """
    Same as get_all_results, but merges with existing data at output_path
    instead of overwriting it. Prevents accidentally wiping out previously
    ingested seasons when pulling a new one, the exact mistake that
    happened pulling 2025 while 2022 to 2024 was already saved.
    """
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
    new_data = pd.concat(all_results, ignore_index=True)

    if os.path.exists(output_path):
        existing = pd.read_parquet(output_path)
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=["year", "round_number", "DriverId"], keep="last")
    else:
        combined = new_data

    combined.to_parquet(output_path)
    return combined

def get_pit_stops(year, round_number):
     session = fastf1.get_session(year, round_number, 'R')
     session.load(laps=True, telemetry=False, weather=False, messages=False)
     laps = session.laps
     pit_ins = laps[laps["PitInTime"].notna()]
     records = []
     for _, row in pit_ins.iterrows():
        driver = row["Driver"]
        # find this driver's next lap (has PitOutTime)
        next_lap = laps[(laps["Driver"] == driver) & (laps["LapNumber"] == row["LapNumber"] + 1)]
        if next_lap.empty or next_lap["PitOutTime"].isna().all():
            continue  # skip if no next lap or no PitOutTime
        duration = (next_lap["PitOutTime"].iloc[0] - row["PitInTime"]).total_seconds()
        records.append({
            "year": year, "round_number": round_number,
            "DriverId": driver, "LapNumber": row["LapNumber"],
            "pit_duration": duration, "Team": row["Team"]
        })
     return pd.DataFrame(records)
     

def get_all_pit_stops(start_year, end_year):
    os.makedirs("data/processed", exist_ok=True)
    all_stops = []

    for year in range(start_year, end_year + 1):
        year_stops = []
        for round_number in range(1, 25):
            try:
                stops = get_pit_stops(year, round_number)
                year_stops.append(stops)
                time.sleep(5)  # throttle to stay under 500 calls/hour
            except Exception as e:
                print(f"Skipped {year} R{round_number}: {e}")
                continue

        # checkpoint: save this year's results immediately
        year_df = pd.concat(year_stops, ignore_index=True)
        year_df.to_parquet(f"data/processed/pit_stops_{year}.parquet")
        print(f"Checkpointed {year}: {year_df.shape}")

        all_stops.append(year_df)

    combined = pd.concat(all_stops, ignore_index=True)
    combined.to_parquet("data/processed/pit_stops.parquet")
    return combined

def get_weather_session(year, round_number):
    """
    Loads one race session's weather data and collapses it to a
    single summary row: rain_flag, avg_track_temp, avg_air_temp, humidity.
    """
    session = fastf1.get_session(year, round_number, 'R')
    session.load(telemetry=False, weather=True, messages=False)
    weather = session.weather_data
    if weather.empty:
        return pd.DataFrame([{
            "year": year,
            "round_number": round_number,
            "rain_flag": None,
            "avg_track_temp": None,
            "avg_air_temp": None,
            "humidity": None,
            "Location": session.event["Location"],
            "source": "actual"
        }])
    rain_flag = (weather["Rainfall"] > 0).any()
    avg_track_temp = weather["TrackTemp"].mean()
    avg_air_temp = weather["AirTemp"].mean()
    humidity = weather["Humidity"].mean()
    return pd.DataFrame([{
        "year": year,
        "round_number": round_number,
        "rain_flag": rain_flag,
        "avg_track_temp": avg_track_temp,
        "avg_air_temp": avg_air_temp,
        "humidity": humidity,
        "Location": session.event["Location"],
        "source": "actual"
    }])

def get_all_weather_sessions(start_year, end_year):
    os.makedirs("data/processed", exist_ok=True)
    all_weather = []
    for year in range(start_year, end_year + 1):
        year_weather = []
        for round_number in range(1, 25):
            try:
                weather = get_weather_session(year, round_number)
                year_weather.append(weather)
                time.sleep(5)  # throttle to stay under 500 calls/hour
            except Exception as e:
                print(f"Skipped {year} R{round_number}: {e}")
                continue

        if not year_weather:
            print(f"No data for {year}, skipping checkpoint")
            continue

        # checkpoint: save this year's results immediately
        year_df = pd.concat(year_weather, ignore_index=True)
        year_df.to_parquet(f"data/processed/weather_{year}.parquet")
        print(f"Checkpointed {year}: {year_df.shape}")

        all_weather.append(year_df)

    combined = pd.concat(all_weather, ignore_index=True)
    combined.to_parquet("data/processed/weather.parquet")
    return combined

def get_forecast_weather(year, round_number, location):
    """
    Pulls a pre-race forecast for a specific upcoming round from
    Open-Meteo, using the circuit lat/lon lookup. Returns a single-row
    DataFrame matching the weather.parquet schema, source='forecast'.
    """
    # 1. look up lat/lon for `location`
    coords = pd.read_csv("data/processed/circuits_lat_lon.csv")
    match = coords[coords["Location"] == location]
    if match.empty:
        raise ValueError(f"No lat/lon found for Location='{location}' — check for encoding/naming mismatch")
    lat, lon = match["latitude"].values[0], match["longitude"].values[0]

    # 2. get race start date/time
    event = fastf1.get_event_schedule(year).query(f"RoundNumber == {round_number}")
    assert event["Session5"].values[0] == "Race", "Session5 isn't the Race for this round — check EventFormat"
    race_time = pd.to_datetime(event["Session5Date"].values[0])
    race_hour = race_time.floor("h")
    race_hour_str = race_hour.strftime("%Y-%m-%dT%H:%M")

    # 3. call Open-Meteo
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": "precipitation_probability,temperature_2m,relative_humidity_2m",
            "timezone": "auto",
            "forecast_days": 16,
        }
    )
    data = response.json()

    # 4. extract the matching hour
    hour_index = data['hourly']['time'].index(race_hour_str)
    rain_flag = data['hourly']['precipitation_probability'][hour_index] > 50
    avg_air_temp = data['hourly']['temperature_2m'][hour_index]
    humidity = data['hourly']['relative_humidity_2m'][hour_index]
    avg_track_temp = None

    return pd.DataFrame([{
        "year": year,
        "round_number": round_number,
        "Location": location,
        "rain_flag": rain_flag,
        "avg_track_temp": avg_track_temp,
        "avg_air_temp": avg_air_temp,
        "humidity": humidity,
        "source": "forecast",
    }])

def get_all_results_extended(start_year, end_year, output_path="data/processed/race_results_extended.parquet"):
    os.makedirs("data/processed", exist_ok=True)
    all_results = []

    for year in range(start_year, end_year + 1):
        year_results = []
        for round_number in range(1, 25):
            try:
                race_results = get_race_results(year, round_number)
                year_results.append(race_results)
                time.sleep(5)  # throttle to stay under 500 calls/hour
            except Exception as e:
                print(f"Skipped {year} R{round_number}: {e}")
                continue

        if not year_results:
            print(f"No data for {year}, skipping checkpoint")
            continue

        year_df = pd.concat(year_results, ignore_index=True)
        year_df.to_parquet(f"data/processed/race_results_extended_{year}.parquet")
        print(f"Checkpointed {year}: {year_df.shape}")

        all_results.append(year_df)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_parquet(output_path)
    return combined

def get_all_tire_degradation(start_year, end_year):
    os.makedirs("data/processed", exist_ok=True)
    all_stints = []

    for year in range(start_year, end_year + 1):
        year_stints = []
        for round_number in range(1, 25):
            try:
                stints = estimate_tire_degradation(year, round_number)
                year_stints.append(stints)
                time.sleep(5)
            except Exception as e:
                print(f"Skipped {year} R{round_number}: {e}")
                continue

        if not year_stints:
            print(f"No data for {year}, skipping checkpoint")
            continue

        year_df = pd.concat(year_stints, ignore_index=True)
        year_df.to_parquet(f"data/processed/tire_degradation_{year}.parquet")
        print(f"Checkpointed {year}: {year_df.shape}")

        all_stints.append(year_df)

    combined = pd.concat(all_stints, ignore_index=True)
    combined.to_parquet("data/processed/tire_degradation.parquet")
    return combined

if __name__ == "__main__":
    setup_cache()
    get_all_results(2022, 2024)
    get_all_pit_stops(2022, 2024)
    get_all_weather_sessions(2022, 2024)