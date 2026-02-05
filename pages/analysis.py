import streamlit as st
from data.loader import load_raw_velib_data, load_raw_weather_data, load_processed_data
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ttest_ind, chi2_contingency
import pandas as pd
from data.preprocessing import *
from utils.config import DATA, MODELS

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
    df['pluie'] = df['pluie'] > 0
    df['neige'] = df['neige'] > 0
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
    sns.barplot(x=df['date_et_heure_de_comptage'].apply(get_season_from_date), y='comptage_horaire', order=['winter','spring','summer','autumn'], data=df, ax=axes[0])
    axes[0].set_title("Saisons")
    sns.barplot(x=df['date_et_heure_de_comptage'].dt.month, y='comptage_horaire', data=df, ax=axes[1])
    axes[1].set_title("Mois")
    sns.barplot(x=df['vacances'], y='comptage_horaire', data=df, ax=axes[2])
    axes[2].set_title("Vacances")
    sns.barplot(x=df['heure_de_pointe'], y='comptage_horaire', data=df, ax=axes[3])
    axes[3].set_title("Heures de pointe")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


#-------------------------------------------------------PAGE DATA ANALYSIS-------------------------------------------------------------------
def show_analysis():
    raw_df_velib = load_raw_velib_data()
    raw_df_weather = load_raw_weather_data()
    processed_df, feature_names = load_processed_data(raw_df_velib, raw_df_weather)

    st.title("Data Analysis : Étude statistique et visualisation du trafic cycliste")
    
    processed_df["heure"] = processed_df["date_et_heure_de_comptage"].dt.hour

    # Ensure PNGs exist (first call will generate and cache them)
    hourly_path = ensure_png_hourly(processed_df)
    weather_path = ensure_png_weather(processed_df)
    corr_path = ensure_png_corr(processed_df)
    seasons_path = ensure_png_seasons(processed_df)

    st.markdown("""
    Cette page présente une **analyse statistique complète** du trafic cycliste à Paris à partir des données des compteurs Vélib’.
    Nous examinons l’influence des **conditions météorologiques**, des **variables temporelles**
    (heure, saison, vacances) et d'autres facteurs sur le **comptage horaire des vélos**.
    """)

    #Corrélations météo
    st.header("Influence des conditions météorologiques")
    st.markdown("""
    Nous testons la corrélation entre le trafic cycliste (`comptage_horaire`) et les variables météorologiques : pluie, neige, vent et température apparente.
    """)

    #Corrélations météo
    st.markdown("On teste ici la corrélation linéaire entre la météo et le trafic cycliste (Pearson).")

    corr_results = []

    with st.expander("Test de corrélation de Pearson"):
        st.markdown("""
    **Pourquoi ?**  
    Ce test mesure **la relation linéaire** entre deux variables numériques continues.  
    Par exemple, on cherche à savoir si le **trafic cycliste** augmente quand la **température** monte, ou diminue quand la **pluie** augmente.

    **Comment interpréter ?**  
    - **Corrélation (r)** : varie entre -1 et 1  
      - *r > 0* → les deux variables augmentent ensemble  
      - *r < 0* → quand l’une augmente, l’autre diminue  
      - *r ≈ 0* → pas de relation linéaire claire  
    - **p-value < 0.05** → la corrélation est **statistiquement significative**

    **Exemple de résultat :**  
    - Température : r = 0.35, p < 0.001 → plus il fait chaud, plus il y a de cyclistes  
    - Pluie : r = -0.08, p > 0.05 → effet faible et non significatif  
    """)

    variables = ['pluie', 'neige', 'apparent_temperature', 'vent']
    for var in variables:
        temp_df = processed_df[['comptage_horaire', var]].dropna()
        if len(temp_df) > 1:  # Vérifier qu'on a au moins 2 points
            corr, p_value = pearsonr(temp_df['comptage_horaire'], temp_df[var])
            st.write(f"**{var}** → Corrélation : {corr:.3f} | p-value : {p_value:.3f}")
            corr_results.append((var, corr, p_value))
        else:
            st.write(f"**{var}** → Pas assez de données pour calculer la corrélation")

    st.markdown("""
        **Interprétation :**
        - La **température** est légèrement corrélée positivement à l’affluence.
        - La **pluie**, le vent et la **neige** ont des effets faibles sur la corrélation linéaire, 
          mais ils peuvent avoir des effets de seuil (non linéaires).
        """)

    st.markdown("""
     **Interprétation :**
    - Une corrélation négative indique que la variable réduit le trafic vélo (ex. pluie, neige).
    - Une corrélation positive indique une augmentation (ex. température).
    - Une *p-value < 0.05* signifie une corrélation statistiquement significative.
    """)

    #Visualisations météo
    st.header("Visualisations météo")
    st.image(weather_path, width=1000)
    st.markdown("""
     **Analyse :**
    - Le trafic vélo diminue en cas de pluie ou de neige.
    - L’effet du vent est limité sauf en cas de rafales fortes.
    - La température a un effet positif modéré : plus il fait doux (15-25°C), plus le trafic est fort.
    """)

    st.header("Corrélations globales entre variables")
    st.header("Matrice de corrélation")
    st.image(corr_path, width=800)

    st.markdown("""
     **Interprétation :**
    - Le trafic vélo est **plus élevé de jour** (r ≈ -0.36 avec `nuit`).
    - Les **heures de pointe** et la **température** influencent positivement le trafic.
    - Les effets météo (pluie, vent, neige) sont faibles, mais parfois non linéaires.
    """)

    #Études temporelles
    st.header("Influence du temps et du calendrier")
    st.header("Répartition horaire")
    st.image(hourly_path, width=1000)
    st.markdown("""
    On observe un pic net **le matin (8h–9h)** et **en fin de journée (17h–19h)**,
    correspondant aux déplacements domicile-travail (heures de pointe).
    """)

    #Tests statistiques temporels
    st.subheader("Tests statistiques temporels")

    tests = [
    ('Jour vs Nuit', 'nuit'),
    ('Vacances vs Non-vacances', 'vacances'),
    ('Heures de pointe vs Heures creuses', 'heure_de_pointe')
]

    with st.expander("Test t de Student"):
        st.markdown("""
    **Pourquoi ?**  
    Le test *t* permet de **comparer la moyenne** d’une variable entre **deux groupes** distincts.

    **Exemples :**
    - Jour vs Nuit : p < 0.001 → trafic plus élevé le jour  
    - Vacances vs Non-vacances : p < 0.001 → trafic plus faible pendant les vacances  

    **Interprétation :**
    - p < 0.05 → différences significatives  
    - p > 0.05 → différences non significatives
    """)

    for label, col in tests:
        # Convertir la colonne en int (0 ou 1)
        temp_df = processed_df[['comptage_horaire', col]].dropna().copy()
        temp_df[col] = temp_df[col].astype(int)

        # Grouper
        groups = temp_df.groupby(col)['comptage_horaire'].apply(list)

        if len(groups) == 2:
            # Récupérer les deux listes correctement
            group1, group2 = groups.values[0], groups.values[1]
            stat, p = ttest_ind(group1, group2)
            st.markdown(f"**{label} :** p-value = {p:.3e}")
        else:
            st.markdown(f"**{label} :** pas assez de données pour effectuer le test")

    st.markdown("""
     **Interprétation :**
    - Le trafic vélo est **nettement plus élevé le jour que la nuit**.
    - **Les saisons** influencent fortement le trafic (plus en été, moins en hiver).
    - Pendant les **vacances scolaires**, le trafic est nettement plus faible.
    - Les **heures de pointe** constituent une variable très explicative du comptage.
    """)

    #Variables catégorielles
    st.header("Dépendance entre variables")

    cat_tests = [
        ('Nuit / Vacances', 'nuit', 'vacances'),
        ('Nuit / Heure de pointe', 'nuit', 'heure_de_pointe')
    ]

    with st.expander("Test du Chi-Deux"):
        st.markdown("""
    **Pourquoi ?**  
    Ce test vérifie si **deux variables catégorielles** sont **indépendantes ou liées**.

    **Exemples :**
    - Nuit / Heure de pointe : p < 0.001 → dépendance forte  
    - Nuit / Vacances : p = 0.12 → pas de lien significatif

    **Interprétation :**
    - p < 0.05 → dépendance entre variables  
    - p > 0.05 → indépendance
    """)
    for label, col1, col2 in cat_tests:
            temp_df = processed_df[[col1, col2]].dropna()
            contingency = pd.crosstab(temp_df[col1], temp_df[col2])
            if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                chi2, p, _, _ = chi2_contingency(contingency)
                st.markdown(f"**{label} :** chi2 = {chi2:.3e} | p-value = {p:.3e}")
            else:
                st.markdown(f"**{label} :** pas assez de données pour effectuer le test")

    st.markdown("""
     **Conclusion :**
    Certaines variables catégorielles présentent une dépendance :
    - La **nuit** et les **heures de pointe** ne sont pas indépendantes (trafic concentré le jour).
    - Ces relations doivent être prises en compte lors du modèle de prédiction.
    """)

    #Saisons et périodes
    st.header("Effet saisonnier et périodique")
    st.image(seasons_path, width=1000)
    st.markdown("""
     **Analyse finale :**
    Le trafic cycliste suit une **forte saisonnalité** :
    - Plus élevé en été et au printemps,
    - Réduit en hiver,
    - Très marqué par les heures de pointe.
    """)

    st.info("Ces analyses serviront de base à la modélisation de la prédiction du trafic sur la page suivante.")