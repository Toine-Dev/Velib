import time
import requests
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
import os
from data.preprocessing import coerce_velib_types, standardize_columns
# from db.db import ensure_velib_raw_schema
from models.features import _drop_tz_if_any
from utils.config import forecast_weather_api_url, historical_weather_api_url, velib_api_url, csv_url, table_name
import tempfile
from utils.utils import get_max_date, insert_on_conflict_do_nothing
from sqlalchemy import DateTime, Engine, Float, text
import json


def upsert_velib_sites(engine: Engine) -> None:
    stmt = text("""
        INSERT INTO velib_sites (
            identifiant_du_site_de_comptage,
            nom_du_site_de_comptage,
            latitude,
            longitude
        )
        SELECT DISTINCT
            identifiant_du_site_de_comptage,
            nom_du_site_de_comptage,
            NULLIF(trim(split_part(coordonnees_geographiques, ',', 1)), '')::double precision AS latitude,
            NULLIF(trim(split_part(coordonnees_geographiques, ',', 2)), '')::double precision AS longitude
        FROM velib_raw
        WHERE coordonnees_geographiques IS NOT NULL
        ON CONFLICT (identifiant_du_site_de_comptage) DO UPDATE
        SET
            nom_du_site_de_comptage = EXCLUDED.nom_du_site_de_comptage,
            latitude = EXCLUDED.latitude,
            longitude = EXCLUDED.longitude;
    """)
    with engine.begin() as conn:
        conn.execute(stmt)


def ensure_weather_raw_schema(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS weather_raw (
                time TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                rain DOUBLE PRECISION,
                snowfall DOUBLE PRECISION,
                apparent_temperature DOUBLE PRECISION,
                wind_speed_10m DOUBLE PRECISION
            );
        """))

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS weather_forecast_raw (
                time TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                rain DOUBLE PRECISION,
                snowfall DOUBLE PRECISION,
                apparent_temperature DOUBLE PRECISION,
                wind_speed_10m DOUBLE PRECISION
            );
        """))

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
    # "id_compteur": "identifiant_du_compteur",
    # "nom_compteur": "nom_du_compteur",
    "id": "identifiant_du_site_de_comptage",
    "name": "nom_du_site_de_comptage",
    "sum_counts": "comptage_horaire",
    "date": "date_et_heure_de_comptage",
    "coordinates": "coordonnees_geographiques",
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
    timezone = "Europe/Paris"  # Velib data is in Paris local time, so we use that timezone for any date parsing/handling to ensure consistency with the raw dataset semantics. We will convert to UTC or naive timestamps as needed later in the pipeline, but using the correct local timezone here is crucial to avoid off-by-one-day errors around midnight and daylight saving time changes.

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
            "timezone": timezone, # Ensure API returns timestamps in correct timezone (if supported by API) to avoid timezone-related bugs. If API does not support this parameter, we will handle timezone conversion ourselves in preprocessing step, but it's worth trying to get it right from the source if possible.
            "select": "id,name,sum_counts,date,coordinates" # To avoid fetching unnecessary columns and reduce memory usage
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
    print(df["date_et_heure_de_comptage"].dtype)
    s = pd.to_datetime(df["date_et_heure_de_comptage"], errors="coerce")

    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_localize(None)

    df["date_et_heure_de_comptage"] = s.dt.floor("h")
    # if df["date_et_heure_de_comptage"].dt.tz is not None:
    #     df["date_et_heure_de_comptage"] = (
    #         df["date_et_heure_de_comptage"].dt.tz_localize(None)
    # ) # Convert to naive timestamps in local timezone (Paris time) to match the semantics of the raw dataset. This is important to avoid timezone-related bugs later in the pipeline. We will handle timezone conversion to UTC or other formats as needed in preprocessing, but using consistent naive local timestamps here is a simpler approach that matches the source data semantics and avoids common pitfalls with timezone-aware timestamps in pandas.
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
    timezone = "Europe/Paris"  # Weather data should also be in Paris local time to align with Velib data semantics. We will handle timezone conversion in preprocessing step as needed, but using correct local timezone here is important to avoid timezone-related bugs.
    
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "rain,snowfall,apparent_temperature,wind_speed_10m",
        "timezone": timezone, # Ensure API returns timestamps in correct timezone (if supported by API) to avoid timezone-related bugs. If API does not support this parameter, we will handle timezone conversion ourselves in preprocessing step, but it's worth trying to get it right from the source if possible.
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
    print(df_weather["time"].dtype)
    s = pd.to_datetime(df_weather["time"], errors="coerce")

    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_localize(None)

    df_weather["time"] = s.dt.floor("h")
    # # Clean timestamp
    # if df_weather["time"].dt.tz is not None:
    #     df_weather["time"] = (
    #         df_weather["time"].dt.tz_localize(None)
    # ) # Convert to naive timestamps in local timezone (Paris time) to match the semantics of the raw dataset. This is important to avoid timezone-related bugs later in the pipeline. We will handle timezone conversion to UTC or other formats as needed in preprocessing, but using consistent naive local timestamps here is a simpler approach that matches the source data semantics and avoids common pitfalls with timezone-aware timestamps in pandas.
    
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
    if "coordonnees_geographiques" in df.columns:
    #     df["coordonnees_geographiques"] = df["coordonnees_geographiques"].apply(
    #         lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v
    #     ) # Convert dict to JSON string for storage in SQL database, while ensuring non-ASCII characters (e.g. accents) are preserved correctly. This is necessary because the raw API returns coordinates as a dict like {"latitude": 48.8575, "longitude": 2.3514}, but we want to store it as a string in the database.
    # # We will parse it back to dict and split it into latitude and longitude in preprocessing step when needed.
        df["coordonnees_geographiques"] = df["coordonnees_geographiques"].apply(
            lambda v: f"{v['lat']},{v['lon']}" if isinstance(v, dict) else v
        ) # Convert dict to "lat,lon" string format for storage in SQL database. This is a simpler alternative to JSON string and is sufficient for our use case since we only have latitude and longitude. We can parse it back to dict or separate lat/lon in preprocessing step when needed.
    print(df.head())

    if df.empty:
        print("No new Velib data.")
        return

    # df.to_sql("velib_raw", engine, if_exists="append", index=False, method="multi", chunksize=2000)
    # print(f"✅ Inserted {len(df)} Velib rows.")

    df = coerce_velib_types(df)
    # Drop rows with missing key columns (cannot be used for training/features)
    before = len(df)
    df = df.dropna(subset=["identifiant_du_site_de_comptage", "date_et_heure_de_comptage"])
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with missing station_id or datetime.", flush=True)
    inserted = insert_on_conflict_do_nothing(engine, df)
    print(f"✅ Inserted {inserted} new Velib rows (after deduplication).")
    


# -------------------------------
# Weather Update
# -------------------------------
def update_weather(engine: Engine):
    print("🌦 Checking weather data...")

    ensure_weather_raw_schema(engine)

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

    df.to_sql(
        "weather_raw", engine, 
        if_exists="append", index=False, 
        method="multi", chunksize=2000, 
        dtype={
        "time": DateTime(),
        "rain": Float(),
        "snowfall": Float(),
        "apparent_temperature": Float(),
        "wind_speed_10m": Float(),
        }
        )

    print(f"✅ Inserted {len(df)} weather rows.")



def update_weather_forecast(engine: Engine, horizon_hours: int = 168) -> None:
    """
    Fetch and overwrite forecast weather aligned to the last available Velib timestamp.

    Canonical time convention:
    - All timestamps in this pipeline are tz-naive and represent Europe/Paris local clock time.
    - Never tz_convert.
    - Never attach tz.
    - Only drop tz if present.
    """
    print("🌦 Updating weather forecast (aligned to last Velib timestamp)...")

    _, max_ts = get_max_date("velib_raw", "date_et_heure_de_comptage")
    if max_ts is None:
        print("⚠️ velib_raw missing or empty. Skipping weather forecast update.")
        return

    # Parse velib max timestamp as tz-naive Paris local time (DO NOT use utc=True)
    start_ts = pd.to_datetime(max_ts, errors="coerce")
    if pd.isna(start_ts):
        print(f"⚠️ Could not parse max velib timestamp: {max_ts!r}. Skipping.")
        return

    # If tz-aware sneaks in, drop tz WITHOUT converting (keep clock time)
    if start_ts.tzinfo is not None:
        start_ts = start_ts.tz_localize(None)

    start_ts = start_ts.floor("h")
    end_ts = (start_ts + pd.Timedelta(hours=horizon_hours)).floor("h")

    start_date = start_ts.date().isoformat()
    end_date = end_ts.date().isoformat()

    print(
        f"Last velib ts (Paris-naive): {start_ts} | "
        f"fetching weather {start_date} → {end_date} | "
        f"keeping [{start_ts} → {end_ts}]"
    )

    # Your fetch_weather_data should already query timezone=Europe/Paris
    df = fetch_weather_data(start_date=start_date, end_date=end_date)
    if df is None or df.empty:
        print("⚠️ No weather data returned.")
        return

    # Parse weather timestamps WITHOUT utc=True; then drop tz if any
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).copy()

    # Drop tz if present (your helper)
    df["time"] = _drop_tz_if_any(df["time"]).dt.floor("h")

    # Filter to exact horizon (naive Paris clock time)
    df = df[(df["time"] >= start_ts) & (df["time"] <= end_ts)].copy()
    df = df.sort_values("time").drop_duplicates(subset=["time"], keep="last")

    if df.empty:
        print("⚠️ Weather data returned but none overlaps desired forecast window.")
        return

    # At this point df["time"] is tz-naive Paris time and matches weather_forecast_raw schema
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
        f"covering {df['time'].min()} → {df['time'].max()} (Paris-naive)"
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
            print(f"Attempting to download CSV (attempt {attempt}/{retries})...")
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()

                tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
                print(f"Downloading CSV to temporary file {tmp.name} ...")
                try:
                    print("Streaming download...")
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            tmp.write(chunk)
                    tmp.flush()
                    print("Download completed successfully.")
                    return tmp.name
                finally:
                    tmp.close()

        except Exception as e:
            last_err = e
            sleep_s = min(2 ** attempt, 10)
            print(f"Download attempt {attempt}/{retries} failed: {e}. Retrying in {sleep_s}s...", flush=True)
            time.sleep(sleep_s)

    raise RuntimeError(f"Failed to download CSV after {retries} attempts: {last_err}")


# def download_and_insert_in_chunks(engine, chunksize=50_000):
#     url = csv_url()
#     print(f"Downloading CSV from {url} ...", flush=True)

#     csv_path = _download_to_tempfile(url, retries=4, timeout=(5, 180))

#     try:
#         print(f"Reading chunks from {csv_path} ...", flush=True)
#         chunk_iter = pd.read_csv(csv_path, sep=";", chunksize=chunksize, usecols=["Identifiant du site de comptage",
#                                                                                   "Nom du site de comptage","Comptage horaire","Date et heure de comptage", 
#                                                                                   "Coordonnées géographiques"])

#         first_chunk = True
#         total_rows = 0

#         for i, chunk in enumerate(chunk_iter, start=1):
#             rows = len(chunk)
#             total_rows += rows
#             print(f"Inserting chunk {i} with {rows} rows (total so far: {total_rows})", flush=True)

#             chunk = standardize_columns(chunk)

#             chunk.to_sql(
#                 table_name(),
#                 engine,
#                 if_exists="replace" if first_chunk else "append",
#                 index=False,
#                 method="multi",        # usually faster
#                 chunksize=2000         # controls INSERT batching, independent of read chunksize
#             )

#             first_chunk = False

#         print(f"All chunks inserted successfully. Total rows inserted: {total_rows}", flush=True)

#     finally:
#         # Clean up temp file
#         try:
#             os.remove(csv_path)
#         except OSError:
#             pass

# import os
# import pandas as pd

# def download_and_insert_in_chunks(engine, chunksize=50_000):
#     url = csv_url()
#     print(f"Downloading CSV from {url} ...", flush=True)

#     # 1) Ensure schema exists with correct types
#     ensure_velib_raw_schema(engine)

#     csv_path = _download_to_tempfile(url, retries=4, timeout=(5, 180))

#     try:
#         print(f"Reading chunks from {csv_path} ...", flush=True)

#         chunk_iter = pd.read_csv(
#             csv_path,
#             sep=";",
#             chunksize=chunksize,
#             usecols=[
#                 "Identifiant du site de comptage",
#                 "Nom du site de comptage",
#                 "Comptage horaire",
#                 "Date et heure de comptage",
#                 "Coordonnées géographiques",
#             ],
#         )

#         total_rows = 0

#         for i, chunk in enumerate(chunk_iter, start=1):
#             rows = len(chunk)
#             total_rows += rows
#             print(f"Inserting chunk {i} with {rows} rows (total so far: {total_rows})", flush=True)

#             # 2) Standardize columns (your existing function)
#             chunk = standardize_columns(chunk)

#             # 3) Rename the coordinates column to match DB schema
#             # After standardize_columns, it will likely be "coordonnées_géographiques" (still accented)
#             # Normalize it once here:
#             if "coordonnées_géographiques" in chunk.columns:
#                 chunk = chunk.rename(columns={"coordonnées_géographiques": "coordonnees_geographiques"})

#             # 4) Coerce types so they match the table schema
#             chunk = coerce_velib_types(chunk)

#             # 5) Append only (do not replace)
#             chunk.to_sql(
#                 "velib_raw",
#                 engine,
#                 if_exists="append",
#                 index=False,
#                 method="multi",
#                 chunksize=2000,
#             )

#         print(f"All chunks inserted successfully. Total rows inserted: {total_rows}", flush=True)

#     finally:
#         try:
#             os.remove(csv_path)
#         except OSError:
#             pass