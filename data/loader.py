from data.preprocessing import preprocess_data
import streamlit as st
import pandas as pd

VELIB_PATH = "comptage_velo_donnees_compteurs.csv"
WEATHER_PATH = "weather_data.csv"

@st.cache_data
def load_raw_data(csv_path=VELIB_PATH):
    # if not csv_path.exists():
    #     raise FileNotFoundError(
    #         "Raw dataset not found. Run scripts/update_data.py first."
    #     )
    df_velib = pd.read_csv(csv_path, sep=";")
    df_velib.columns = [col.strip().replace(" ", "_").lower() for col in df_velib.columns]
    return df_velib

@st.cache_data
def load_processed_data(raw_df):
    processed_df, feature_names = preprocess_data(raw_df)
    return processed_df, feature_names


# fast API
# côté admin : gestion d'utilisateurs (suppression ou ajout), redéclencher pipeline de récupération (modelling ou data), avoir accès à un lien qui pointe vers registre MLflow pour voir les différents modèles et entraînements
# côté client : lancer une prédiction en fonction de certaines données d'entrée (date, météo, etc.) et afficher les résultats sous forme graphique (graphiques de tendances, corrélations, etc.)