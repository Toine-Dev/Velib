import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
from datetime import timedelta
from data.metadata import last_cached_datetime
from models.train import train_model
from models.predict import predict_model
from utils.config import load_model

#-------------------------------------------------------PAGE MODEL PREDICTIONS-------------------------------------------------------------------
def show_prediction():

    st.title("Model Training & Predictions")

    pipeline = None
    metrics = None

    # Vérifier si le modèle existe
    if os.path.exists("model.pkl"):
        pipeline = load_model()
        model = pipeline.named_steps['model']
        if os.path.exists("metrics.json"):
            with open("metrics.json", "r") as f:
                metrics = json.load(f)
    else:
        st.info("Aucun modèle trouvé. Entraînement en cours...")
        with st.spinner("Training model..."):
            train_model()
            pipeline = load_model()

    # Afficher les métriques
    if metrics:
        st.write("Validation Metrics:")
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{metrics['MAE']:.4f}")
        col2.metric("RMSE", f"{metrics['RMSE']:.4f}")
        col3.metric("R²", f"{metrics['R2']:.4f}")

# # Feature Importance Plot
    st.subheader("Top Features Importantes")
    transformed_feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    feature_importances = pipeline.named_steps['model'].feature_importances_
    importances_df = pd.DataFrame({'Feature': transformed_feature_names , 'Importance': feature_importances}).sort_values(by='Importance', ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.barh(importances_df['Feature'], importances_df['Importance'], color='skyblue')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title('Top 10 features importantes')
    ax.invert_yaxis()
    st.pyplot(fig)

# --- Section prédiction interactive ---
    start_datetime, start_date = last_cached_datetime()
    st.header("Prédiction pour une date future")
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

    if st.button("Prédire la heatmap"):
        pred_time = pred_time.replace(minute=0, second=0, microsecond=0)
        datetime_pred = pd.Timestamp.combine(pred_date, pred_time)
        locations = predict_model(datetime_pred)
        
        # Heatmap
        m = folium.Map(location=[48.8566, 2.3522], zoom_start=12, tiles="CartoDB positron")
        HeatMap(data=locations[['latitude', 'longitude', 'comptage_horaire']].values.tolist(), radius=radius, max_zoom=13).add_to(m)
        st.subheader("Heatmap des prédictions")
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