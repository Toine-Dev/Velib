import sys
from pathlib import Path 
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
import streamlit as st
from data.preprocessing import *
from data.ingestion import *
from datetime import timedelta 
from data.loader import *
from utils.config import DATA, MODELS

def predict_model():

    raw_df_velib = load_raw_velib_data()
    raw_df_weather = load_raw_weather_data()
    processed_df, feature_names = load_processed_data(raw_df_velib, raw_df_weather)

    start_time = pd.to_datetime(raw_df_velib['date_et_heure_de_comptage'].max())
    
    st.header("Prédiction pour une date future")

    min_date = start_time.date()
    st.write(f"Veuillez choisir une date/heure STRICTEMENT après {start_time}")
    col1, col2 = st.columns(2)

    pred_date = col1.date_input("Sélectionnez la date", min_value=min_date)
    pred_time = col2.time_input("Sélectionnez l'heure")
    pred_datetime = pd.Timestamp.combine(pred_date, pred_time)
    pred_datetime = pred_datetime.tz_localize(start_time.tz)

    if pred_datetime <= start_time:
        st.error(f"Veuillez choisir une date/heure STRICTEMENT après {start_time}")
        st.stop()

    radius = st.slider("Rayon de la heatmap", min_value=10, max_value=50, value=30)

    if st.button("Prédire la heatmap"):
        pred_time = pred_time.replace(minute=0, second=0, microsecond=0)
        datetime_pred = pd.Timestamp.combine(pred_date, pred_time)
        #st.write(datetime_pred, datetime_pred.tzinfo) -> pas de fuseau horaire, bonne date + heure

        # -------------------------
        # 1) Préparer les heures futures
        # -------------------------
        # future_hours = pd.date_range(start=datetime_pred, periods=24, freq='H')
        future_hours = pd.date_range(start=start_time, periods=24, freq='H')
        hours_df = pd.DataFrame({'date_et_heure_de_comptage': future_hours})
        hours_df['jour'] = hours_df['date_et_heure_de_comptage'].dt.weekday
        hours_df['saison'] = hours_df['date_et_heure_de_comptage'].apply(get_season_from_date)
        hours_df['nom_jour'] = hours_df['date_et_heure_de_comptage'].dt.day_name(locale='fr_FR.UTF-8')
        hours_df['mois'] = hours_df['date_et_heure_de_comptage'].dt.month
        hours_df['heure'] = hours_df['date_et_heure_de_comptage'].dt.hour
        hours_df['nuit'] = hours_df.apply(is_night, axis=1)
        hours_df['vacances'] = hours_df['date_et_heure_de_comptage'].apply(is_vacances)
        hours_df['heure_de_pointe'] = hours_df['date_et_heure_de_comptage'].apply(is_rush_hour)

        st.write(hours_df['date_et_heure_de_comptage'].iloc[0], hours_df['date_et_heure_de_comptage'].iloc[0].tz)
        # -------------------------
        # 2) Récupérer les données météo
        # -------------------------
        df_weather = fetch_weather_data(
            datetime_pred.strftime("%Y-%m-%d"),
            (future_hours.max() + timedelta(days=1)).strftime("%Y-%m-%d")
        )

        st.write(df_weather['time'].iloc[0], df_weather['time'].iloc[0].tz)

        # Convertir en datetime et ajuster fuseau horaire Paris
        df_weather['time'] = pd.to_datetime(df_weather['time']).dt.tz_localize('UTC').dt.tz_convert('Europe/Paris').dt.floor('H')

        st.write(df_weather['time'].iloc[-1], df_weather['time'].iloc[-1].tz)

        #VERIFIER SUBTILITE FUSEAU HORAIRE
        hours_df['date_et_heure_de_comptage'] = pd.to_datetime(hours_df['date_et_heure_de_comptage']).dt.tz_localize('UTC').dt.tz_convert('Europe/Paris').dt.floor('H')
        st.write(hours_df['date_et_heure_de_comptage'].iloc[0], hours_df['date_et_heure_de_comptage'].iloc[0].tz)
        # Merge sécurisé
        df_merged = pd.merge(hours_df, df_weather, how="left",
                            left_on="date_et_heure_de_comptage",
                            right_on="time").drop(columns=["time"])

        # Créer les colonnes météo booléennes
        df_merged['pluie'] = (df_merged['rain'].fillna(0) > 0).astype(int)
        df_merged['neige'] = (df_merged['snowfall'].fillna(0) > 0).astype(int)
        df_merged['vent'] = (df_merged['wind_speed_10m'].fillna(0) > 15).astype(int)
        df_merged['apparent_temperature'] = df_merged['apparent_temperature'].fillna(df_merged['apparent_temperature'].mean())

        st.dataframe(df_merged)

        # -------------------------
        # 3) Encodage cyclique (si nécessaire)
        # -------------------------
        df_merged = add_cyclic_features(df_merged)

        # -------------------------
        # 4) Cross join avec les sites
        # -------------------------
        unique_sites = processed_df[['identifiant_du_site_de_comptage']].drop_duplicates()
        cartesian_df = unique_sites.merge(df_merged, how='cross')

        # -------------------------
        # 5) Ajouter statistiques par site
        # -------------------------
        site_stats = processed_df.groupby('identifiant_du_site_de_comptage')['comptage_horaire'].agg(['mean','std','max','min']).rename(columns={
            'mean':'site_mean_usage',
            'std':'site_usage_variability',
            'max':'site_max_usage',
            'min':'site_min_usage'
        }).reset_index()
        cartesian_df = cartesian_df.merge(site_stats, on='identifiant_du_site_de_comptage', how='left')

        # -------------------------
        # 6) Ajouter valeurs historiques récursives (lags et rolling)/ Préparer historique par site (tri chronologique du processed_df)
        # -------------------------
        processed_df_sorted = processed_df.sort_values(['identifiant_du_site_de_comptage', 'date_et_heure_de_comptage'])
        # Construire dictionnaire history_dict[site] = list des dernières 24 valeurs (chronologiques)
        history_dict = {}
        for site in unique_sites['identifiant_du_site_de_comptage'].unique():
            site_hist = processed_df_sorted[processed_df_sorted['identifiant_du_site_de_comptage'] == site]
            vals = site_hist['comptage_horaire'].tolist()
            # On prend les 24 dernières valeurs (si moins, on pad avec la moyenne du site ou 0)
            if len(vals) >= 24:
                last24 = vals[-24:]
            else:
                pad_val = int(np.round(np.mean(vals))) if len(vals) > 0 else 0
                last24 = ([pad_val] * (24 - len(vals))) + vals
            history_dict[site] = list(last24)  # ordre chronologique, oldest...newest

        # 7) Forecast récursif : par site, heures triées
        preds_list = []  # (idx, pred) pour recoller ensuite
        # trier cartesian_df pour avoir un ordre stable
        cartesian_df = cartesian_df.sort_values(['identifiant_du_site_de_comptage', 'date_et_heure_de_comptage']).reset_index(drop=False)
        # 'index' contient l'ancien index qui nous permet de recoller
        for site, site_df in cartesian_df.groupby('identifiant_du_site_de_comptage', sort=True):
            history = history_dict.get(site, [0]*24)  # oldest...newest
            # s'assurer que site_df est trié par date
            site_df = site_df.sort_values('date_et_heure_de_comptage')
            for idx, row in site_df.iterrows():
                # calcul des lags à partir de l'historique courant
                lag_1 = history[-1] if len(history) >= 1 else 0
                lag_24 = history[0] if len(history) >= 24 else history[0] if len(history) > 0 else 0
                rolling_mean_24 = float(np.mean(history)) if len(history) >= 1 else 0.0

                # préparer une copie de la ligne et injecter les lags dans les noms attendus
                row_for_pred = row.copy()
                row_for_pred['lag_1'] = lag_1
                row_for_pred['lag_24'] = lag_24
                row_for_pred['rolling_mean_24'] = rolling_mean_24

                # Construire X_row avec les mêmes feature_names utilisés au training
                X_row = pd.DataFrame([row_for_pred[feature_names]])

    #             # prédiction via pipeline (préprocesseur + modèle)
    #             try:
    #                 pred = float(pipeline.predict(X_row)[0])
    #             except Exception as e:
    #                 # fallback sécuritaire si problème : 0 ou moyenne site
    #                 pred = float(max(0, site_stats.loc[site_stats['identifiant_du_site_de_comptage']==site, 'site_mean_usage'].values[0]
    #                                 if site in site_stats['identifiant_du_site_de_comptage'].values else 0))

    #             # clip à 0 positif
    #             pred = max(pred, 0.0)

    #             # stocker prediction pour recoller
    #             preds_list.append((idx, pred))

    #             # mise à jour FIFO de l'historique : on ajoute la pred comme nouvelle valeur la plus récente
    #             history.append(pred)
    #             # garder seulement les 24 dernières
    #             if len(history) > 24:
    #                 history = history[-24:]
    #             history_dict[site] = history

    #     # Recréer une colonne predictions alignée avec cartesian_df index
    #     preds_df = pd.DataFrame(preds_list, columns=['orig_index', 'pred'])
    #     preds_df.set_index('orig_index', inplace=True)
    #     # cartesian_df.index correspond aux 'idx' utilisés plus haut
    #     cartesian_df['comptage_horaire'] = cartesian_df.index.map(lambda i: preds_df.loc[i, 'pred'] if i in preds_df.index else 0.0)

    #     # 8) Préparer les coordonnées et la heatmap
    #     coords_df = raw_df[['identifiant_du_site_de_comptage', 'coordonnées_géographiques', 'nom_du_site_de_comptage']].dropna().drop_duplicates()
    #     coords_split = coords_df['coordonnées_géographiques'].str.split(",", expand=True)
    #     coords_df['latitude'] = pd.to_numeric(coords_split[0], errors="coerce")
    #     coords_df['longitude'] = pd.to_numeric(coords_split[1], errors="coerce")
    #     coords_df = coords_df.dropna(subset=['latitude', 'longitude'])

    #     locations = pd.merge(coords_df, cartesian_df, on='identifiant_du_site_de_comptage', how='inner')

    #     # Heatmap
    #     m = folium.Map(location=[48.8566, 2.3522], zoom_start=12, tiles="CartoDB positron")
    #     HeatMap(data=locations[['latitude', 'longitude', 'comptage_horaire']].values.tolist(), radius=radius, max_zoom=13).add_to(m)
    #     st.subheader("Heatmap des prédictions")
    #     folium_static(m)

    #     # Stocker pour graphique
    #     st.session_state['locations'] = locations

    # # Graphique après heatmap
    # if 'locations' in st.session_state:
    #     st.subheader("Choisissez un site pour voir son évolution")
    #     site_choice = st.selectbox("Site :", st.session_state['locations']['nom_du_site_de_comptage'].unique())
    #     site_data = st.session_state['locations'][st.session_state['locations']['nom_du_site_de_comptage'] == site_choice]
    #     site_data_sorted = site_data.sort_values('date_et_heure_de_comptage')
    #     st.line_chart(site_data_sorted[['date_et_heure_de_comptage', 'comptage_horaire']].set_index('date_et_heure_de_comptage'))


if __name__ == "__main__":
    predict_model()