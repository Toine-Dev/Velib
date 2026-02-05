import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
from data.loader import load_raw_velib_data, load_raw_weather_data, load_processed_data
from utils.config import DATA, MODELS


#-------------------------------------------------------PAGE OVERVIEW-------------------------------------------------------------------
def show_overview():
    raw_df_velib = load_raw_velib_data()
    raw_df_weather = load_raw_weather_data()
    processed_df, feature_names = load_processed_data(raw_df_velib, raw_df_weather)
    
    raw_df_velib["date_et_heure_de_comptage"] = pd.to_datetime(raw_df_velib["date_et_heure_de_comptage"])
    date_min = raw_df_velib["date_et_heure_de_comptage"].min()
    date_max = raw_df_velib["date_et_heure_de_comptage"].max()
    nb_mois = ((date_max.year - date_min.year) * 12 + (date_max.month - date_min.month) + 1)

    st.title("Analyse de l'affluence des vélos à Paris")

    st.markdown("""
Bienvenue dans ce projet d’analyse des données de comptage vélos à Paris.  
Ce tableau de bord a pour objectif d’explorer, visualiser et comprendre les tendances de fréquentation des pistes cyclables grâce aux bornes de comptages dans la capitale.
La ville de Paris déploie depuis plusieurs années des compteurs à vélo permanents pour évaluer le développement de la pratique cycliste. Ce projet a pour objectif d’effectuer une analyse des données récoltées par ces compteurs vélo afin de visualiser les horaires et les zones d’affluences. Ceci aura pour but de fournir des outils à la mairie de Paris afin qu’elle juge des améliorations à apporter sur les différents endroits cyclables de la ville. 
""")

    st.subheader("Localisation des bornes de comptage à Paris")

    try:
        with open("compteur_paris.html", "r", encoding="utf-8") as f:
            map_html = f.read()
        components.html(map_html, height=600, scrolling=False)
    except FileNotFoundError:
        st.error("Le fichier 'compteur_paris.html' est introuvable. Assurez-vous qu'il se trouve dans le dossier du projet.")

    st.markdown(f"""
Cette carte interactive présente les emplacements des bornes de comptage vélos à Paris. Chaque point correspond à un site de comptage, permettant de suivre les flux de cyclistes à différents moments de la journée. En passant la souris dessus nous pouvons voir le nom de la rue où la borne est implantée. Dans notre jeu de données nous avons **{raw_df_velib["identifiant_du_site_de_comptage"].nunique()} sites** de comptage.
""")

    st.subheader("Structure et description des données")

    st.markdown("### Période des données")
    st.markdown(f"La période analysée contient **{nb_mois} mois** de données allant du **{date_min.strftime('%d %B %Y')} au {date_max.strftime('%d %B %Y')}**.")

    st.markdown("### Dictionnaire des données d'origine")

    data_description = pd.DataFrame({
        "Nom de la donnée": raw_df_velib.columns,
    "Description": [
        "Identifiant du compteur, de type string.",
        "Rue du compteur + points cardinaux (N-S-E-O), de type string.",
        "Identifiant du site de comptage, de type string.",
        "Rue du site de comptage, de type string.",
        "Nombre de vélo utilisant la piste cyclable à chaque heure pour une borne donnée, de type integer.",
        "Date et heure du comptage de vélo de la station, de type datetime yyyy-mm-dd-hh-mm-ss.",
        "Date d'installation du site de comptage, de type date date dd/mm/yyyy.",
        "Lien https de la photo du site de comptage, de type string.",
        "Géolocalisation du site de comptage (latitude longitude), de type string.",
        "Identifiant technique du compteur, de type string.",
        "Photo de la borne, de type string.",
        "Photo de la borne, de type string.",
        "Photo de la borne, de type string.",
        "Photo de la borne, de type string.",
        "Type d'image jpg, png, etc, de type string.",
        "Date année et mois du comptage, de type string."
    ]
})
    st.dataframe(data_description, use_container_width=True, height= 597)

    st.markdown("""
Notre jeu de données initial provient du site de la **mairie de Paris** (*disponible dans la sidebar*) qui propose une data open source issue des différentes sites de comptage de la capitale. On compte **16** colonnes mais elles contiennent des redondances, **5** colonnes pour l'identifiant ou numéro de site de comptage et **5** colonnes pour la photo du site. De ce dataset nous garderons seulement l'identifiant du site de comptage, la date et heure du comptage ainsi que le comptage horaire. Avant de commencer l'étape de modélisation et il important de bien nettoyer nos données et de bien comprendre quelles données pourront nous être utiles pour prédire l'affluence des vélos à Paris.
""")

    st.markdown(f"""
### Étapes principales du Feature Engineering

- Suppression des colonnes doublons ou inutiles
- Nettoyage des valeurs manquantes et standardisation des noms de colonnes  
- Extraction de variables temporelles (heure, jour, mois, saison) depuis la colonne date et heure de comptage    
- Création de variables dérivées (heures de pointe, vacances, nuit)  
- Calcul de statistiques par site (moyenne, écart-type, maximum, minimum)  
- Création de variables décalées (lag) et moyennes glissantes
- Fusion avec les données d'une API météo (température, vent, pluie, neige)
- Encodage des valeurs temporels (heure, jour, mois, saison) en valeurs cycliques (sin/cos).
                
Grâce à cette étape de feature engineering nous sommes passés d'un dataset de **{len(raw_df_velib.columns)}** colonnes avec **{len(raw_df_velib)}** lignes à un dataset de **{len(processed_df.columns)}** colonnes avec **{len(processed_df)}** lignes de données prêtes à être utilisées.
""")
    
    st.markdown("### Dictionnaire des données utilisées")

    data_description = pd.DataFrame({
        "Nom de la donnée": processed_df.columns,
    "Description": [
        "Identifiant du site de comptage, de type string.",
        "Nombre de vélo utilisant la piste cyclable à chaque heure pour une borne donnée, de type integer.",
        "Date et heure du comptage de vélo de la station, de type datetime yyyy-mm-dd-hh-mm-ss.",
        "Vacances (1) ou non (0), de type booléen.",
        "Heure de pointe (1) ou non (0), de type booléen.",
        "Nuit (1), jour (0), de type booléen.",
        "Température dans l'air en degré Celsius, de type float.",
        "Quantité de pluie tombée en millimètre, transformé en booléen pour la modélisation.",
        "Rafales de vent de plus de 30km/h, transformé en booléen pour la modélisation.",
        "Quantité de neige tombée en millimètre, transformé en booléen pour la modélisation.",
        "Moyenne du nombre de vélos comptés pour ce site, de type float.",
        "Variance du nombre de vélos comptés pour ce site, de type float.",
        "Nombre maximum de vélos comptés pour ce site, de type float.",
        "Nombre minimum de vélos comptés pour ce site, de type float.",
        "Nombre de comptage vélos 1 heure avant, de type float.",
        "Nombre de comptage vélos 24 heures avant, de type float.",
        "Moyenne roulante sur les dernières 24 heures, de type float.",
        "Représentation cyclique du jour du mois (1-7) pour montrer sa position dans la semaine.",
        "Représentation cyclique du jour du mois (1-7) pour montrer sa position dans la semaine.",
        "Représentation cyclique du mois de l’année (1–12) pour capturer la saisonnalité annuelle.",
        "Représentation cyclique du mois de l’année (1–12) pour capturer la saisonnalité annuelle.",
        "Représentation cyclique de l’heure de la journée (0–23) pour tenir compte du cycle journalier.",
        "Représentation cyclique de l’heure de la journée (0–23) pour tenir compte du cycle journalier.",
        "Représentation cyclique de la saison (hiver → printemps → été → automne).",
        "Représentation cyclique de la saison (hiver → printemps → été → automne)."
    ]
})
    st.dataframe(data_description, use_container_width=True, height= 913)

    st.subheader("Comparaison des données brutes et finales")
    raw_nonzero = raw_df_velib[raw_df_velib['comptage_horaire'] > 0]
    merged_nonzero = processed_df[processed_df['comptage_horaire'] > 0]
    try:
        tab1, tab2 = st.tabs(["Données brutes", "Données finales"])

        with tab1:
            st.subheader("Données brutes (originales)")
            st.dataframe(raw_nonzero.head(20), use_container_width=True)
            st.markdown(f"Nombre de lignes : **{len(raw_df_velib):,}** | Nombre de colonnes : **{len(raw_df_velib.columns)}**")

        with tab2:
            st.subheader("Données après feature engineering")
            st.dataframe(merged_nonzero.head(20), use_container_width=True)
            st.markdown(f"Nombre de lignes : **{len(processed_df):,}** | Nombre de colonnes : **{len(processed_df.columns)}**")
    except NameError:
        st.warning("Les DataFrames `df` et `df_merged` ne sont pas encore chargés. Importez-les avant d'afficher les tableaux.")