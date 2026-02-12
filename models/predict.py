import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
from models.features import add_time_features, add_site_statistics, build_history, cross_join_sites, merge_weather, recursive_forecast 
from data.preprocessing import *
from data.ingestion import *
from data.loader import *
import os
from models.model_utils import load_model
from data.metadata import last_cached_datetime
from models.train import train_model

def predict_model(datetime_pred):

    # Load cached data
    raw_df_velib = load_raw_velib_data()
    raw_df_weather = load_raw_weather_data()
    processed_df, feature_names = load_processed_data(raw_df_velib, raw_df_weather)
    df_forecast_weather = load_forecast_weather_data()

    locations = None

    # start_datetime = pd.to_datetime(raw_df_velib['date_et_heure_de_comptage'].max())
    start_datetime, start_date = last_cached_datetime()

    pipeline = None

    # Vérifier si le modèle existe
    if os.path.exists("model.pkl"):
        pipeline = load_model()
    else:
        train_model()
        pipeline = load_model()

    # -------------------------
    # 1) Préparer les heures futures
    # -------------------------

    # The "periods=x number of hours" parameter includes the start_datetime if inclusive="left".
    # A restriction must be set on the UI to force the user to select a future time which does not go over that limit (ex: max 48h in the future after start_datetime).
    # Keep in mind that start_datetime should always have 23:00 as its time component (based on the observed downloadable CSV data), so the future hours will be from 00:00 of the next day until 23:00 of the day after tomorrow (if we set periods=49 for instance).
    future_hours = pd.date_range(start=start_datetime, periods=49, freq='h', inclusive="right") # 49 to have 48 future hours after start_datetime, inclusive="right" to exclude start_datetime and include the last hour
    hours_df = add_time_features(future_hours, datetime_pred)

    # -------------------------
    # 2) Récupérer les données météo
    # -------------------------

    df_weather_merge = df_forecast_weather.copy() # Create a copy to avoid modifying cached dataframe
    df_merged = merge_weather(hours_df, df_weather_merge)  # This will modify df_weather_merge to have the merged result with hours_df

    # -------------------------
    # 3) Encodage cyclique (si nécessaire)
    # -------------------------

    df_merged = add_cyclic_features(df_merged)

    # -------------------------
    # 4) Cross join avec les sites
    # -------------------------

    unique_sites, cartesian_df = cross_join_sites(processed_df, df_merged)

    # -------------------------
    # 5) Ajouter statistiques par site
    # -------------------------

    site_stats, cartesian_df = add_site_statistics(cartesian_df, processed_df)

    # -------------------------
    # 6) Prépare ajout des valeurs historiques récursives (lags et rolling) / Préparer historique par site (tri chronologique du processed_df)
    # -------------------------

    history_dict = build_history(processed_df, unique_sites)

    # -------------------------
    # 7) Forecast récursif : par site, heures triées
    # -------------------------

    cartesian_df = recursive_forecast(cartesian_df, history_dict, pipeline, feature_names, site_stats)

    # -------------------------
    # 8) Préparer les coordonnées et la carte de chaleur
    # -------------------------

    coords_df = raw_df_velib[['identifiant_du_site_de_comptage', 'coordonnées_géographiques', 'nom_du_site_de_comptage']].dropna().drop_duplicates()
    coords_split = coords_df['coordonnées_géographiques'].str.split(",", expand=True)
    coords_df['latitude'] = pd.to_numeric(coords_split[0], errors="coerce")
    coords_df['longitude'] = pd.to_numeric(coords_split[1], errors="coerce")
    coords_df = coords_df.dropna(subset=['latitude', 'longitude'])

    locations = pd.merge(coords_df, cartesian_df, on='identifiant_du_site_de_comptage', how='inner')

    if locations is None:
        raise RuntimeError("Prediction failed: locations never created")

    return locations