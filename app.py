import streamlit as st
import pandas as pd
# from my_utils import *
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
from datetime import timedelta
import matplotlib.pyplot as plt
import statsmodels.api as sm
# import pmdarima as pm
import os
# from my_utils import save_model, load_model
import streamlit.components.v1 as components
from scipy.stats import pearsonr, ttest_ind, f_oneway, chi2_contingency
# import plotly.express as px
# import plotly.figure_factory as ff
import seaborn as sns
import json

from data.loader import load_raw_data, load_processed_data

# ------------------------------ PATHS ------------------------------
# CSV_PATH = "comptage_velo_donnees_compteurs.csv"
PLOT_DIR = "assets/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ------------------------------ CACHING ------------------------------
@st.cache_data
def ensure_png_hourly(df, path=os.path.join(PLOT_DIR, "hourly.png")):
    # si le fichier existe déjà, retourner son chemin (pas de recalcul)
    if os.path.exists(path):
        return path
    # sinon dessiner et sauvegarder
    df_heure = df.groupby('heure', as_index=False)['comptage_horaire'].mean()
    fig, ax = plt.subplots(figsize=(12,6))
    sns.barplot(data=df_heure, x='heure', y='comptage_horaire', color='steelblue', ax=ax)
    ax.set_title("Comptage horaire moyen selon l'heure")
    ax.set_ylabel("Comptage horaire")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path

@st.cache_data
def ensure_png_weather(df, path=os.path.join(PLOT_DIR, "weather_effects.png")):
    if os.path.exists(path):
        return path
    df['pluie'] = df['rain'] > 0
    df['neige'] = df['snowfall'] > 0
    df['vent'] = df['wind_speed_10m'] > 15
    fig, axes = plt.subplots(2,2, figsize=(14,10))
    axes = axes.flatten()
    sns.barplot(x='pluie', y='comptage_horaire', data=df, ax=axes[0])
    axes[0].set_title("Pluie")
    sns.barplot(x='neige', y='comptage_horaire', data=df, ax=axes[1])
    axes[1].set_title("Neige")
    sns.barplot(x='vent', y='comptage_horaire', data=df, ax=axes[2])
    axes[2].set_title("Vent")
    sns.scatterplot(x='apparent_temperature', y='comptage_horaire', data=df, alpha=0.3, ax=axes[3])
    axes[3].set_title("Température")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path

@st.cache_data
def ensure_png_corr(df, path=os.path.join(PLOT_DIR, "corr_matrix.png")):
    if os.path.exists(path):
        return path
    corr_matrix = df[['comptage_horaire','nuit','vacances','heure_de_pointe','pluie','neige','apparent_temperature','vent']].corr()
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path

@st.cache_data
def ensure_png_seasons(df, path=os.path.join(PLOT_DIR, "seasons.png")):
    if os.path.exists(path):
        return path
    fig, axes = plt.subplots(2,2, figsize=(14,10))
    axes = axes.flatten()
    sns.barplot(x='saison', y='comptage_horaire', order=['winter','spring','summer','autumn'], data=df, ax=axes[0])
    axes[0].set_title("Saisons")
    sns.barplot(x=df['date_et_heure_de_comptage'].dt.month, y='comptage_horaire', data=df, ax=axes[1])
    axes[1].set_title("Mois")
    sns.barplot(x='vacances', y='comptage_horaire', data=df, ax=axes[2])
    axes[2].set_title("Vacances")
    sns.barplot(x='heure_de_pointe', y='comptage_horaire', data=df, ax=axes[3])
    axes[3].set_title("Heures de pointe")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path

# @st.cache_data
# def load_raw_data(csv_path=CSV_PATH):
#     df = pd.read_csv(csv_path, sep=";")
#     df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]
#     return df

# @st.cache_data
# def load_processed_data(raw_df):
#     processed_df, feature_names = preprocess_data(raw_df)
#     return processed_df, feature_names

# -----------------------------------------------------LOAD DATA--------------------------------------------------------------------
raw_df = load_raw_data()
processed_df, feature_names = load_processed_data(raw_df)

#------------------------------------------------------- SIDE BAR-------------------------------------------------------------------
with st.sidebar:
    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Data Analysis",
            "Model & Predictions",
            "Modèle Sarimax"
        ],
        index=0
    )

    st.markdown("""
    <div style="
        background-color: #f0f4fa;
        padding: 10px 15px;
        border-radius: 8px;
        font-size: 0.9em;
        ">
        <b>Auteurs :</b><br>
Antoine Scarcella<br>
Nathan Vitse<br>
Nikhil Teja Bellamkonda<br><br>

<b>Données :
<a href="https://opendata.paris.fr/explore/dataset/comptage-velo-donnees-compteurs/download/?format=csv&timezone=Europe/Paris&lang=fr&use_labels_for_header=true&csv_separator=%3B" target="_self" style="color:#0066cc; text-decoration:none;">
data.gouv.fr
</a>
</div>
""", unsafe_allow_html=True)

#-------------------------------------------------------PAGE CONFIG-------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard afflluence vélos à Paris",
    layout="wide"
)

st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 3rem;
            padding-right: 3rem;
            max-width: 1400px;
            margin: auto;
        }
    </style>
""", unsafe_allow_html=True)

st.dataframe(raw_df.head())
st.write(raw_df.columns)