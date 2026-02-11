import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st
from models.features import add_time_features, add_site_statistics, build_history, cross_join_sites, merge_weather, recursive_forecast 
from data.preprocessing import *
from data.ingestion import *
from datetime import timedelta 
from data.loader import *
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
import os
from model_utils import train_final_model, save_model, load_model
import json
from data.metadata import last_cached_datetime

def predict_model():

    # Load cached data
    raw_df_velib = load_raw_velib_data()
    raw_df_weather = load_raw_weather_data()
    processed_df, feature_names = load_processed_data(raw_df_velib, raw_df_weather)
    df_forecast_weather = load_forecast_weather_data()

    # start_datetime = pd.to_datetime(raw_df_velib['date_et_heure_de_comptage'].max())
    start_datetime, start_date = last_cached_datetime()

    # à mettre dans prediction.py
    st.write(f"Données disponibles jusqu'à : {start_datetime} (heure de Paris).")
    
    st.header("Prédiction pour une date future")

    # start_date = start_datetime.date()
    st.write(f"Veuillez choisir une date/heure STRICTEMENT après {start_datetime}.")
    col1, col2 = st.columns(2)

    pred_date = col1.date_input("Sélectionnez la date", min_value=start_date)
    pred_time = col2.time_input("Sélectionnez l'heure")
    # The tz_localize method does not convert the time, it just assigns the timezone.
    pred_datetime = pd.Timestamp.combine(pred_date, pred_time).tz_localize(start_datetime.tz)

    if pred_datetime <= start_datetime or pred_datetime > (start_datetime + timedelta(hours=48)):
        st.error(f"Veuillez choisir une combinaison date et heure STRICTEMENT après {start_datetime} mais pas plus de 48 heures après cette dernière.")
        st.stop()

    radius = st.slider("Rayon de la heatmap", min_value=10, max_value=50, value=30)

    pipeline = None
    metrics = None

    # Vérifier si le modèle existe
    if os.path.exists("model.pkl"):
        pipeline = load_model()
        # model = pipeline.named_steps['model']
        if os.path.exists("metrics.json"):
            with open("metrics.json", "r") as f:
                metrics = json.load(f)
    else:
        st.info("Aucun modèle trouvé. Entraînement en cours...")
        with st.spinner("Training model..."):
            target_cols = ['identifiant_du_site_de_comptage']
            numeric_cols = [col for col in feature_names if col not in target_cols]
            # Préprocesseur
            # preprocessor = make_preprocessor(target_cols, numeric_cols)

            model_params = {
                "n_estimators" : 500, 
                "learning_rate" : 0.05,
                "max_depth" : -1,
                "random_state" : 42
            }

            pipeline, metrics = train_final_model(processed_df[feature_names], processed_df[['comptage_horaire']], model_params, target_cols, numeric_cols, test_size_ratio=0.1)
            save_model(pipeline)

            with open("metrics.json", "w") as f:
                json.dump(metrics, f)
            st.success("Modèle entraîné et sauvegardé dans model.pkl")

    if st.button("Afficher la carte de chaleur des prédictions"):
        pred_time = pred_time.replace(minute=0, second=0, microsecond=0)
        datetime_pred = pd.Timestamp.combine(pred_date, pred_time)
        st.write(datetime_pred, datetime_pred.tzinfo) # pas de fuseau horaire, bonne date + heure

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

        # Heatmap
        m = folium.Map(location=[48.8566, 2.3522], zoom_start=12, tiles="CartoDB positron")
        HeatMap(data=locations[['latitude', 'longitude', 'comptage_horaire']].values.tolist(), radius=radius, max_zoom=13).add_to(m)
        st.subheader("Carte de chaleur des prédictions")
        folium_static(m)

        # Stocker pour graphique
        st.session_state['locations'] = locations

    # Graphique après heatmap
    if 'locations' in st.session_state:
        st.subheader("Choisissez un site pour voir son évolution")
        site_choice = st.selectbox("Site :", st.session_state['locations']['nom_du_site_de_comptage'].unique())
        site_data = st.session_state['locations'][st.session_state['locations']['nom_du_site_de_comptage'] == site_choice]
        site_data_sorted = site_data.sort_values('date_et_heure_de_comptage')
        st.line_chart(site_data_sorted[['date_et_heure_de_comptage', 'comptage_horaire']].set_index('date_et_heure_de_comptage'))


if __name__ == "__main__":
    predict_model()