import time
import requests
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
import os
from data.preprocessing import standardize_columns
from utils.config import forecast_weather_api_url, historical_weather_api_url, velib_api_url, csv_url, table_name
import tempfile
from utils.utils import get_max_date
from sqlalchemy import Engine
import json


# ==============================
# Configuration
# ==============================

# VELIB_API_URL = os.getenv("VELIB_API_URL")
# HISTORICAL_WEATHER_API_URL = os.getenv("HISTORICAL_WEATHER_API_URL")
# FORECAST_WEATHER_API_URL = os.getenv("FORECAST_WEATHER_API_URL")


# col_map = {
#     "id_compteur": "identifiant_du_compteur",
#     "nom_compteur": "nom_du_compteur",
#     "id": "identifiant_du_site_de_comptage",
#     "name": "nom_du_site_de_comptage",
#     "sum_counts": "comptage_horaire",
#     "date": "date_et_heure_de_comptage",
#     "installation_date": "date_d'installation_du_site_de_comptage",
#     "url_photos_n1": "lien_vers_photo_du_site_de_comptage",
#     "coordinates": "coordonnées_géographiques",
#     "counter": "identifiant_technique_compteur",
#     "photos": "id_photos",
#     "test_lien_vers_photos_du_site_de_comptage_": "test_lien_vers_photos_du_site_de_comptage_",
#     "id_photo_1": "id_photo_1",
#     "url_sites": "url_sites",
#     "type_dimage": "type_dimage",
#     "mois_annee_comptage": "mois_annee_comptage"
# }
col_map = {
    "id_compteur": "identifiant_du_compteur",
    "nom_compteur": "nom_du_compteur",
    "id": "identifiant_du_site_de_comptage",
    "name": "nom_du_site_de_comptage",
    "sum_counts": "comptage_horaire",
    "date": "date_et_heure_de_comptage",
    "coordinates": "coordonnées_géographiques",
}


def _as_date(value) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, str):
        return datetime.fromisoformat(value).date()
    raise TypeError(f"Unsupported date value type: {type(value).__name__}")


# Velib API expects: YYYY/MM/DD, not not ISO: YYYY-MM-DD
def fetch_velib_data(
    start_date: str,
    end_date: Optional[str] = None,
    limit: int = 100,
    sleep: float = 0.2,
) -> pd.DataFrame:
    """
    Fetch Velib data for a given date range using pagination.

    Parameters
    ----------
    start_date : str
        Start date (YYYY/MM/DD)
    end_date : str, optional
        End date (YYYY/MM/DD). If None, fetch only start_date.
    limit : int
        Records per API call.
    sleep : float
        Delay between API calls.

    Returns
    -------
    pd.DataFrame
    """

    if end_date is None:
        end_date = start_date

    # API filter
    where_clause = (
        f"date >= date'{start_date}' "
        f"AND date <= date'{end_date}'"
    )

    offset = 0
    all_records: List[Dict] = []

    while True:
        # params = {
        #     "where": where_clause,
        #     "limit": limit,
        #     "offset": offset,
        # }
        params = {
            "where": where_clause,
            "limit": limit,
            "offset": offset,
            "select": "id_compteur, nom_compteur, id, name, sum_counts, date, coordinates" # To avoid fetching unnecessary columns and reduce memory usage
        }

        response = requests.get(velib_api_url(), params=params, timeout=10)

        if response.status_code != 200:
            raise RuntimeError(f"Velib API error {response.status_code} - {response.text}")

        data = response.json()
        results = data.get("results", [])
        total_count = data.get("total_count", 0)

        if not results:
            break

        all_records.extend(results)
        offset += len(results)
        print(f"Fetched {offset}/{total_count} records")

        if offset >= total_count:
            break

        time.sleep(sleep)
    
    df = pd.DataFrame(all_records).rename(columns=col_map)

    return df


# fetch_velib_range_by_day is a helper that calls fetch_velib_data day by day to avoid potential API issues with large date ranges (e.g. timeouts, memory errors, etc.)
# Indeed, there seems to be a 10000 record limit on the API side. It is safe to expect less than 10000 records per day, so we can fetch day by day to ensure we get all data without hitting API limits.
def fetch_velib_range_by_day(start_date: str, end_date: str, **kwargs) -> pd.DataFrame:
    start = datetime.strptime(start_date, "%Y/%m/%d").date()
    end = datetime.strptime(end_date, "%Y/%m/%d").date()

    dfs = []
    d = start
    while d <= end:
        print(f"Fetching Velib data for {d}...")
        day = d.strftime("%Y/%m/%d")
        dfs.append(fetch_velib_data(start_date=day, end_date=day, **kwargs))
        d += timedelta(days=1)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def fetch_weather_data(
    start_date: str,
    end_date: Optional[str] = None,
    latitude: float = 48.8575,
    longitude: float = 2.3514,
) -> pd.DataFrame:
    """
    Fetch historical weather data from Open-Meteo.

    Parameters
    ----------
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    latitude : float
    longitude : float

    Returns
    -------
    pd.DataFrame
        Weather dataframe indexed by timestamp
    """

    if end_date is None:
        end_date = start_date

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "rain,snowfall,apparent_temperature,wind_speed_10m",
    }

    today = datetime.now(ZoneInfo("Europe/Paris")).date() # Respects daylight saving automatically (matches the semantics of raw dataset which is in Paris local time)
    
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d").date() # Convert velib_max string to date object to compare with yesterday date object
    
    if end_date_dt >= today:
        response = requests.get(forecast_weather_api_url(), params=params, timeout=10)
    else: 
        response = requests.get(historical_weather_api_url(), params=params, timeout=10)
        
    if response.status_code != 200:
        # include response.text to see API error message/details
        raise RuntimeError(f"Weather API error {response.status_code}: {response.text}")

    print("Weather data retrieved successfully.")

    data = response.json()
    records = data.get("hourly", {})

    df_weather = pd.DataFrame(records)

    # Clean timestamp
    df_weather["time"] = pd.to_datetime(df_weather["time"], utc=True)
    df_weather["time"] = df_weather["time"].dt.tz_convert(None)

    return df_weather


# -------------------------------
# Velib Update
# -------------------------------
def update_velib(engine: Engine):
    print("🚲 Checking Velib data...")

    min_date, max_date = get_max_date("velib_raw", "date_et_heure_de_comptage")

    today = datetime.now(ZoneInfo("Europe/Paris")).date() # Respects daylight saving automatically (matches the semantics of raw dataset which is in Paris local time)
    yesterday = today - timedelta(days=1)

    if max_date is None:
        print("Velib table empty → full load required.")
        # Define initial start date manually
        start_date = "2020/01/01"  # replace with your real start
    else:
        # velib_max_dt = datetime.strptime(velib_max, "%Y/%m/%d").date() # Convert velib_max string to date object to compare with yesterday date object
        # velib_start = (velib_max_dt + timedelta(days=1)).strftime("%Y/%m/%d")
        max_date = _as_date(max_date)
        if max_date >= yesterday:
            print("✅ Velib already up to date.")
            return

        start_date = (max_date + timedelta(days=1)).strftime("%Y/%m/%d")

    end_date = yesterday.strftime("%Y/%m/%d")

    print(f"Fetching Velib from {start_date} to {end_date}")

    # df = fetch_velib_data(start_date=start_date, end_date=end_date)
    df = fetch_velib_range_by_day(start_date=start_date, end_date=end_date)
    if "coordonnées_géographiques" in df.columns:
        df["coordonnées_géographiques"] = df["coordonnées_géographiques"].apply(
            lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v
        ) # Convert dict to JSON string for storage in SQL database, while ensuring non-ASCII characters (e.g. accents) are preserved correctly. This is necessary because the raw API returns coordinates as a dict like {"latitude": 48.8575, "longitude": 2.3514}, but we want to store it as a string in the database.
    # We will parse it back to dict and split it into latitude and longitude in preprocessing step when needed.
    print(df.head())

    if df.empty:
        print("No new Velib data.")
        return

    df.to_sql("velib_raw", engine, if_exists="append", index=False, method="multi", chunksize=2000)
    print(f"✅ Inserted {len(df)} Velib rows.")


# -------------------------------
# Weather Update
# -------------------------------
def update_weather(engine: Engine):
    print("🌦 Checking weather data...")

    velib_min, velib_max = get_max_date("velib_raw", "date_et_heure_de_comptage")
    weather_min, weather_max = get_max_date("weather_raw", "time")

    if velib_max is None:
        print("Velib not populated yet — skipping weather.")
        return

    velib_min = _as_date(velib_min)
    velib_max = _as_date(velib_max)

    if weather_max is None:
        print("Weather table empty → bootstrapping.")
        start_date = velib_min
    else:
        weather_max = _as_date(weather_max)
        if weather_max >= velib_max:
            print("✅ Weather already up to date.")
            return
        start_date = weather_max + timedelta(days=1)

    end_date = velib_max

    print(f"Fetching weather from {start_date} to {end_date}")

    df = fetch_weather_data(
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
    )

    if df.empty:
        print("No new weather data.")
        return

    df.to_sql("weather_raw", engine, if_exists="append", index=False, method="multi", chunksize=2000)
    print(f"✅ Inserted {len(df)} weather rows.")

import pandas as pd
from sqlalchemy.engine import Engine
from utils.utils import get_max_date

def update_weather_forecast(engine: Engine, horizon_hours: int = 168) -> None:
    """
    Fetch and overwrite forecast weather aligned to the last available Velib timestamp.

    - start_ts = MAX(velib_raw.date_et_heure_de_comptage)
    - end_ts   = start_ts + horizon_hours
    - Fetch weather by date window that covers [start_ts, end_ts]
    - Filter safely with consistent timezone handling
    - Overwrite weather_forecast_raw each run
    """
    print("🌦 Updating weather forecast (aligned to last Velib timestamp)...")

    _, max_ts = get_max_date("velib_raw", "date_et_heure_de_comptage")
    if max_ts is None:
        print("⚠️ velib_raw missing or empty. Skipping weather forecast update.")
        return

    # Force velib max timestamp to UTC tz-aware
    start_ts = pd.to_datetime(max_ts, utc=True, errors="coerce")
    if pd.isna(start_ts):
        print(f"⚠️ Could not parse max velib timestamp: {max_ts!r}. Skipping.")
        return

    end_ts = start_ts + pd.Timedelta(hours=horizon_hours)

    # Date window (no extra day buffer): just cover start/end dates
    start_date = start_ts.date().isoformat()
    end_date = end_ts.date().isoformat()

    print(
        f"Last velib ts: {start_ts} | "
        f"fetching weather {start_date} → {end_date} | "
        f"keeping [{start_ts} → {end_ts}]"
    )

    df = fetch_weather_data(start_date=start_date, end_date=end_date)
    if df is None or df.empty:
        print("⚠️ No weather data returned.")
        return

    # Force weather timestamps to UTC tz-aware as well
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"])

    # Filter to exact horizon
    df = df[(df["time"] >= start_ts) & (df["time"] <= end_ts)].copy()
    df = df.sort_values("time").drop_duplicates(subset=["time"], keep="last")

    if df.empty:
        print("⚠️ Weather data returned but none overlaps desired forecast window.")
        return

    # Optional: store as naive UTC timestamps (consistent with many DB schemas)
    df["time"] = df["time"].dt.tz_convert("UTC").dt.tz_localize(None)

    df.to_sql(
        "weather_forecast_raw",
        engine,
        if_exists="replace",
        index=False,
        method="multi",
        chunksize=2000,
    )

    print(
        f"✅ Wrote {len(df)} rows to weather_forecast_raw "
        f"covering {df['time'].min()} → {df['time'].max()}"
    )

# def download_and_insert_in_chunks(engine, chunksize=50000):
#     url = csv_url()
#     print(f"Streaming CSV from {url} ...")

#     chunk_iter = pd.read_csv(url, sep=";", chunksize=chunksize)

#     first_chunk = True

#     for chunk in chunk_iter:
#         print(f"Inserting chunk with {len(chunk)} rows")

#         # 🔹 Standardize column names
#         chunk = standardize_columns(chunk)

#         chunk.to_sql(
#             table_name(),
#             engine,
#             if_exists="replace" if first_chunk else "append",
#             index=False
#         )

#         first_chunk = False

#     print("All chunks inserted successfully.")

# print(fetch_velib_data("2026/01/30", "2026/01/30")) # Works
# print(fetch_velib_data("2026/01/30")) # Works
# print(fetch_velib_data("2026/01/30").columns) # Works
# print(fetch_weather_data("2026-01-29", "2026-01-30"))  # Works
# print(fetch_weather_data("2026-01-29")["time"])  # Works


def _download_to_tempfile(url: str, *, retries: int = 3, timeout=(5, 120)) -> str:
    """
    Download URL to a temporary file and return its filename.
    Uses retries + timeouts to avoid hanging forever.
    """
    last_err = None

    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()

                tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
                try:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            tmp.write(chunk)
                    tmp.flush()
                    return tmp.name
                finally:
                    tmp.close()

        except Exception as e:
            last_err = e
            sleep_s = min(2 ** attempt, 10)
            print(f"Download attempt {attempt}/{retries} failed: {e}. Retrying in {sleep_s}s...", flush=True)
            time.sleep(sleep_s)

    raise RuntimeError(f"Failed to download CSV after {retries} attempts: {last_err}")


def download_and_insert_in_chunks(engine, chunksize=50_000):
    url = csv_url()
    print(f"Downloading CSV from {url} ...", flush=True)

    csv_path = _download_to_tempfile(url, retries=4, timeout=(5, 180))

    try:
        print(f"Reading chunks from {csv_path} ...", flush=True)
        chunk_iter = pd.read_csv(csv_path, sep=";", chunksize=chunksize, usecols=["Identifiant du compteur","Nom du compteur","Identifiant du site de comptage",
                                                                                  "Nom du site de comptage","Comptage horaire","Date et heure de comptage", 
                                                                                  "Coordonnées géographiques"])

        first_chunk = True
        total_rows = 0

        for i, chunk in enumerate(chunk_iter, start=1):
            rows = len(chunk)
            total_rows += rows
            print(f"Inserting chunk {i} with {rows} rows (total so far: {total_rows})", flush=True)

            chunk = standardize_columns(chunk)

            chunk.to_sql(
                table_name(),
                engine,
                if_exists="replace" if first_chunk else "append",
                index=False,
                method="multi",        # usually faster
                chunksize=2000         # controls INSERT batching, independent of read chunksize
            )

            first_chunk = False

        print(f"All chunks inserted successfully. Total rows inserted: {total_rows}", flush=True)

    finally:
        # Clean up temp file
        try:
            os.remove(csv_path)
        except OSError:
            pass
