import time
import requests
import pandas as pd
from typing import List, Dict, Optional


# VELIB_URL = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/comptage-velo-donnees-compteurs/records"
# "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/comptage-velo-donnees-compteurs/records?where=date%3E%3Ddate%272026%2F01%2F29%27"

# def fetch_velib_data(limit: int = 100, sleep: float = 0.2) -> pd.DataFrame:
#     """
#     Fetch all Velib records using pagination.

#     Parameters
#     ----------
#     limit : int
#         Number of records per API call (max allowed by API).
#     sleep : float
#         Delay between calls to avoid hammering the API.

#     Returns
#     -------
#     pd.DataFrame
#         DataFrame containing all fetched records.
#     """
#     offset = 0
#     all_records: List[Dict] = []

#     while True:
#         params = {
#             "limit": limit,
#             "offset": offset,
#         }

#         response = requests.get(VELIB_URL, params=params, timeout=10)

#         if response.status_code != 200:
#             raise RuntimeError(f"Velib API error {response.status_code}")

#         data = response.json()
#         results = data.get("results", [])
#         total_count = data.get("total_count", 0)

#         print(f"Fetched {offset}/{total_count} records")

#         if not results:
#             break  # safety exit

#         all_records.extend(results)
#         offset += len(results)

#         if offset >= total_count:
#             break

#         time.sleep(sleep)

#     df = pd.DataFrame(all_records)
#     return df


VELIB_URL = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/comptage-velo-donnees-compteurs/records"

col_map = {
    "id_compteur": "identifiant_du_compteur",
    "nom_compteur": "nom_du_compteur",
    "id": "identifiant_du_site_de_comptage",
    "name": "nom_du_site_de_comptage",
    "sum_counts": "comptage_horaire",
    "date": "date_et_heure_de_comptage",
    "installation_date": "date_d'installation_du_site_de_comptage",
    "url_photos_n1": "lien_vers_photo_du_site_de_comptage",
    "coordinates": "coordonnées_géographiques",
    "counter": "identifiant_technique_compteur",
    "photos": "id_photos",
    "test_lien_vers_photos_du_site_de_comptage_": "test_lien_vers_photos_du_site_de_comptage_",
    "id_photo_1": "id_photo_1",
    "url_sites": "url_sites",
    "type_dimage": "type_dimage",
    "mois_annee_comptage": "mois_annee_comptage"
}

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
        params = {
            "where": where_clause,
            "limit": limit,
            "offset": offset,
        }

        response = requests.get(VELIB_URL, params=params, timeout=10)

        if response.status_code != 200:
            raise RuntimeError(f"Velib API error {response.status_code}")

        data = response.json()
        results = data.get("results", [])
        total_count = data.get("total_count", 0)

        print(f"Fetched {offset}/{total_count} records")

        if not results:
            break

        all_records.extend(results)
        offset += len(results)

        if offset >= total_count:
            break

        time.sleep(sleep)
    
    df = pd.DataFrame(all_records).rename(columns=col_map)

    return df


# def fetch_weather_data(
#     start_date: str,
#     end_date: str,
#     latitude: float = 48.8575,
#     longitude: float = 2.3514,
# ) -> pd.DataFrame:
#     """
#     Fetch historical weather data from Open-Meteo.

#     Parameters
#     ----------
#     start_date : str
#         Start date (YYYY-MM-DD)
#     end_date : str
#         End date (YYYY-MM-DD)
#     latitude : float
#     longitude : float

#     Returns
#     -------
#     pd.DataFrame
#         Weather dataframe indexed by timestamp
#     """
#     "https://archive-api.open-meteo.com/v1/archive?latitude=48.8575&longitude=2.3514&start_date={start_date}&end_date={end_date}&hourly=rain,snowfall,apparent_temperature,wind_speed_10m"
#     url = (
#         "https://archive-api.open-meteo.com/v1/archive"
#         f"?latitude={latitude}"
#         f"&longitude={longitude}"
#         f"&start_date={start_date}"
#         f"&end_date={end_date}"
#         "&hourly=rain,snowfall,apparent_temperature,wind_speed_10m"
#     )

#     response = requests.get(url, timeout=10)

#     if response.status_code != 200:
#         raise RuntimeError(f"Weather API error {response.status_code}")

#     print("Weather data retrieved successfully.")

#     data = response.json()
#     records = data.get("hourly", {})

#     df_weather = pd.DataFrame(records)

#     # Clean timestamp
#     df_weather["time"] = pd.to_datetime(df_weather["time"], utc=True)
#     df_weather["time"] = df_weather["time"].dt.tz_convert(None)

#     return df_weather


WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"

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

    response = requests.get(WEATHER_URL, params=params, timeout=10)

    if response.status_code != 200:
        raise RuntimeError(f"Weather API error {response.status_code}")

    print("Weather data retrieved successfully.")

    data = response.json()
    records = data.get("hourly", {})

    df_weather = pd.DataFrame(records)

    # Clean timestamp
    df_weather["time"] = pd.to_datetime(df_weather["time"], utc=True)
    df_weather["time"] = df_weather["time"].dt.tz_convert(None)

    return df_weather


# print(fetch_velib_data("2026/01/30", "2026/01/30")) # Works
print(fetch_velib_data("2026/01/30")) # Works
# print(fetch_velib_data("2026/01/30").columns) # Works
# print(fetch_weather_data("2026-01-29", "2026-01-30"))  # Works
# print(fetch_weather_data("2026-01-29")["time"])  # Works











