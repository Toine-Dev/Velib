from data.preprocessing import preprocess_velib_data, preprocess_weather_data, preprocess_merged_data
import streamlit as st
import pandas as pd
from utils.config import DATA, MODELS

VELIB_PATH = "comptage_velo_donnees_compteurs.csv"
WEATHER_PATH = "weather_data.csv"


@st.cache_data
def load_raw_velib_data(velib_path=VELIB_PATH):
    df_velib = pd.read_csv(velib_path, sep=";")
    df_velib.columns = [col.strip().replace(" ", "_").lower() for col in df_velib.columns]
    return df_velib

@st.cache_data
def load_raw_weather_data(weather_path=WEATHER_PATH):
    df_weather = pd.read_csv(weather_path)
    return df_weather

@st.cache_data
def load_processed_data(raw_velib_df, raw_weather_df):
    processed_velib_data = preprocess_velib_data(raw_velib_df)
    processed_weather_data = preprocess_weather_data(raw_weather_df)
    df_merged = pd.merge(processed_velib_data, processed_weather_data, how="left", left_on="date_et_heure_de_comptage", right_on="time").drop(columns=["time"])
    processed_df, feature_names = preprocess_merged_data(df_merged)
    return processed_df, feature_names


# fast API
# côté admin : gestion d'utilisateurs (suppression ou ajout), redéclencher pipeline de récupération (modelling ou data), avoir accès à un lien qui pointe vers registre MLflow pour voir les différents modèles et entraînements
# côté client : lancer une prédiction en fonction de certaines données d'entrée (date, météo, etc.) et afficher les résultats sous forme graphique (graphiques de tendances, corrélations, etc.)