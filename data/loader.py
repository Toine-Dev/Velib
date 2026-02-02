from data.preprocessing import preprocess_data
import streamlit as st
import pandas as pd

CSV_PATH = "comptage_velo_donnees_compteurs.csv"

@st.cache_data
def load_raw_data(csv_path=CSV_PATH):
    # if not csv_path.exists():
    #     raise FileNotFoundError(
    #         "Raw dataset not found. Run scripts/update_data.py first."
    #     )
    df = pd.read_csv(csv_path, sep=";")
    df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]
    return df

@st.cache_data
def load_processed_data(raw_df):
    processed_df, feature_names = preprocess_data(raw_df)
    return processed_df, feature_names