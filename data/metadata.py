import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict
import json
import shutil


# Paths
RAW_VELIB_PATH = Path("./comptage_velo_donnees_compteurs.csv")
WEATHER_PATH = Path("./weather_data.csv")
METADATA_PATH = Path("metadata/dataset_state.json")

# ---------------------------------------------------------------------
# Low-level helpers (JSON only, fast)
# ---------------------------------------------------------------------

def load_metadata() -> Optional[Dict]:
    if not RAW_VELIB_PATH.exists():
        # Delete metadata folder if it exists
        metadata_folder = METADATA_PATH.parent
        if metadata_folder.exists():
            shutil.rmtree(metadata_folder)
        # Delete weather_data.csv if it exists
        if WEATHER_PATH.exists():
            WEATHER_PATH.unlink()
        raise FileNotFoundError(f"Raw Velib data not found at {RAW_VELIB_PATH}")
    
    if not METADATA_PATH.exists():
        return None

    with open(METADATA_PATH, "r") as f:
        return json.load(f)


def save_metadata(state: Dict) -> None:
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_PATH, "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------
# One-time bootstrap helpers (expensive, used once)
# ---------------------------------------------------------------------

def _get_velib_date_range_from_csv(path: Path) -> Tuple[Optional[str], Optional[str]]:
    if not path.exists():
        return None, None

    df = pd.read_csv(
        path,
        sep=";",
        usecols=["Date et heure de comptage"]
    )

    # Force UTC to avoid DST issues without altering instants
    dates = pd.to_datetime(df["Date et heure de comptage"], utc=True)

    return (
        dates.min().strftime("%Y/%m/%d"),
        dates.max().strftime("%Y/%m/%d"),
    )


def _get_weather_date_range(path: Path) -> Tuple[Optional[str], Optional[str]]:
    if not path.exists():
        return None, None

    df = pd.read_csv(path, usecols=["time"])
    dates = pd.to_datetime(df["time"])

    return (
        dates.min().strftime("%Y/%m/%d"),
        dates.max().strftime("%Y/%m/%d"),
    )


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def init_metadata_if_missing() -> Dict:
    """
    Initializes metadata.json if it does not exist.
    This function may scan raw files ONCE.
    """
    state = load_metadata()
    if state is not None:
        return state
    
    velib_min, velib_max = _get_velib_date_range_from_csv(RAW_VELIB_PATH)
    weather_min, weather_max = _get_weather_date_range(WEATHER_PATH)

    state = {
        "velib": {
            "min_date": velib_min,
            "max_date": velib_max,
        },
        "weather": {
            "min_date": weather_min,
            "max_date": weather_max,
        },
    }

    save_metadata(state)
    return state


# def update_velib_dates(state: Dict, new_max_date: str) -> None:
#     state["velib"]["max_date"] = new_max_date
#     save_metadata(state)


# def update_weather_dates(state: Dict, new_max_date: str) -> None:
#     state["weather"]["max_date"] = new_max_date
#     save_metadata(state)