import pandas as pd
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import sys

# Add project root to path otherwise imports below fail when running this script directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.metadata import init_metadata_if_missing, save_metadata
from data.ingestion import fetch_velib_data, fetch_weather_data

RAW_VELIB_PATH = Path("./comptage_velo_donnees_compteurs.csv")
WEATHER_PATH = Path("./weather_data.csv")

def append_to_csv(df_new: pd.DataFrame, path: Path) -> None:
    """Append new data to CSV, creating file if needed."""
    header = not path.exists() # Write header if file does not exist
    df_new.to_csv(path, mode="a", header=header, index=False) # DataFrame index will not be written to file (index=False)


def update_data():
    # --------------------------------------------------
    # 1. Load or initialize metadata (cheap)
    # --------------------------------------------------
    state = init_metadata_if_missing()
    velib_min = state["velib"]["min_date"]
    velib_max = state["velib"]["max_date"]

    if velib_min is None or velib_max is None:
        print("Error: velib min_date or max_date in dataset_state.json is None. Please make sure raw Velib CSV dataset exists and is functional and try again.")
        return

    today = datetime.now(ZoneInfo("Europe/Paris")).date() # Respects daylight saving automatically (matches the semantics of raw dataset which is in Paris local time)
    print(datetime.now(ZoneInfo("Europe/Paris"))) # Debugging line
    print(f"Today's date (Paris time): {today}")  # Debugging line
    yesterday = today - timedelta(days=1)


    # --------------------------------------------------
    # 2. Velib ingestion (primary timeline)
    # --------------------------------------------------

    velib_max_dt = datetime.strptime(velib_max, "%Y/%m/%d").date() # Convert velib_max string to date object to compare with yesterday date object
    velib_start = (velib_max_dt + timedelta(days=1)).strftime("%Y/%m/%d")

    if velib_max_dt < yesterday:
        print("📥 Fetching new Velib data...")
        new_velib_df = fetch_velib_data(
            start_date=velib_start,
            end_date=yesterday.strftime("%Y/%m/%d"),
        )

        if not new_velib_df.empty:
            print(f"Yesterday's date (Paris time): {yesterday}")
            print("Updating up to yesterday's date inclusive.")  # Debugging line
############################# DO NOT FORGET TO UNCOMMENT BELOW #############################
            # append_to_csv(new_velib_df, RAW_VELIB_PATH)
############################# DO NOT FORGET TO UNCOMMENT BELOW #############################
            new_velib_df['date_et_heure_de_comptage'] = pd.to_datetime(new_velib_df['date_et_heure_de_comptage'])
            state["velib"]["max_date"] = new_velib_df["date_et_heure_de_comptage"].max().strftime("%Y/%m/%d")
            save_metadata(state)
            print("✅ Velib data ingestion complete.")
            print(f"✅ Added {len(new_velib_df)} new rows.")
            print(f"✅ new_velib_df['date_et_heure_de_comptage']:\n{new_velib_df['date_et_heure_de_comptage']}")
            print(f"Dates ranging from {new_velib_df['date_et_heure_de_comptage'].min()} to {new_velib_df['date_et_heure_de_comptage'].max()}")
        else:
            print("✅ Velib data ingestion complete.")
            print(f"✅ No new data to add (data already up-to-date).")


    # --------------------------------------------------
    # 3. Weather ingestion (depends on Velib)
    # --------------------------------------------------
    # Reassign state to ensure we have the latest velib_min and velib_max after Velib ingestion above as this may have changed Velib metadata
    state = init_metadata_if_missing() # Not useful I think, I have to check the logic here again
    velib_min = state["velib"]["min_date"]
    velib_max = state["velib"]["max_date"]

    # weather_min = state["weather"]["min_date"]
    weather_max = state["weather"]["max_date"]

    # Case A: weather file does not exist yet
    if not WEATHER_PATH.exists():
        print("🌦️ Weather file missing — bootstrapping from Velib range")
        weather_start = velib_min
        weather_end = velib_max
        weather_max = velib_min

    # Case B: weather exists but is behind Velib
    weather_max_dt = datetime.strptime(weather_max, "%Y/%m/%d").date()
    velib_max_dt = datetime.strptime(velib_max, "%Y/%m/%d").date()

    if weather_max_dt < velib_max_dt:
        print("🌦️ Fetching missing weather data...")
        weather_start = weather_max
        # weather_start = (weather_max_dt + timedelta(days=1)).strftime("%Y/%m/%d")
        weather_end = velib_max

    else:
        print("✅ Weather data already up to date")
        save_metadata(state)
        return

    if weather_start != velib_min:
        weather_start = (weather_max_dt + timedelta(days=1)).strftime("%Y/%m/%d")

    # Weather API expect "YYYY-MM-DD" format and not "YYYY/MM/DD"
    weather_start_formatted = datetime.strptime(weather_start, "%Y/%m/%d").strftime("%Y-%m-%d")
    weather_end_formatted = datetime.strptime(weather_end, "%Y/%m/%d").strftime("%Y-%m-%d")

    print(f"Fetching weather data from {weather_start} to {weather_end}")
    weather_df = fetch_weather_data(
        start_date=weather_start_formatted,
        end_date=weather_end_formatted,
    )

    if not weather_df.empty:
        append_to_csv(weather_df, WEATHER_PATH)
        state["weather"]["min_date"] = velib_min
        state["weather"]["max_date"] = weather_end

    # --------------------------------------------------
    # 4. Persist metadata once
    # --------------------------------------------------
    save_metadata(state)
    print("✅ Dataset successfully updated")

if __name__ == "__main__":
    update_data()

# yesterday (31/01/2026) API returned no data for that same date (and onwards) but did for previous dates (30/01/2026 and before)