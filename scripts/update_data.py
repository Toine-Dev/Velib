import pandas as pd
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import sys

# Add project root to path otherwise imports below fail when running this script directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.metadata import get_date_range
from data.ingestion import fetch_velib_data

RAW_DATA_PATH = Path("./comptage_velo_donnees_compteurs.csv")

def append_to_csv(df_new: pd.DataFrame, path: Path) -> None:
    """Append new data to CSV, creating file if needed."""
    header = not path.exists() # Write header if file does not exist
    df_new.to_csv(path, mode="a", header=header, index=False) # DataFrame index will not be written to file (index=False)


def main():
    _, last_date = get_date_range()

    # today = datetime.utcnow().date()
    # today = datetime.now(timezone.utc).date()

    # yesterday = (today - timedelta(days=1)).strftime("%Y/%m/%d")

    # if last_date is not None:
    #     if last_date >= yesterday:
    #         print("✅ Data already up to date. No ingestion needed.")
    #         return

    #     start_date = (
    #         datetime.strptime(last_date, "%Y/%m/%d").date()
    #         + timedelta(days=1)
    #     ).strftime("%Y/%m/%d")
    # else:
    #     # First run: choose a sensible historical start
    #     start_date = "2024/01/01"

    today = datetime.now(ZoneInfo("Europe/Paris")).date() # Respects daylight saving automatically (matches the semantics of raw dataset which is in Paris local time)
    print(datetime.now(ZoneInfo("Europe/Paris"))) # Debugging line
    print(f"Today's date (Paris time): {today}")  # Debugging line
    yesterday = today - timedelta(days=1)
    print(f"Yesterday's date (Paris time): {yesterday}")  # Debugging line

    if last_date is not None:
        last_date_dt = datetime.strptime(last_date, "%Y/%m/%d").date() # Convert last_date string to date object to compare with yesterday date object
        print(f"Last date in dataset: {last_date_dt}, Yesterday: {yesterday}")  # Debugging line
        if last_date_dt >= yesterday:
            print("✅ Data already up to date. No ingestion needed.")
            return

        start_date = (last_date_dt + timedelta(days=1)).strftime("%Y/%m/%d")

    else:
        # First run or missing/corrupted dataset → fetch last 12 months
        start_date = (today - timedelta(days=365)).strftime("%Y/%m/%d")

    end_date = yesterday.strftime("%Y/%m/%d")

    print(f"📥 Fetching Velib data from {start_date} to {end_date}")

    df_new = fetch_velib_data(start_date=start_date, end_date=end_date)

    if df_new.empty:
        print("⚠️ API returned no new records.")
        return

    append_to_csv(df_new, RAW_DATA_PATH)

    print(f"✅ Added {len(df_new)} new rows.")


if __name__ == "__main__":
    main()

# yesterday (31/01/2026) API returned no data for that same date (and onwards) but did for previous dates (30/01/2026 and before)