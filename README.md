# 🚴 Prediction de l'affluence des vélos dans Paris

## 📑 Sommaire
- [Description du projet](#1--description-du-projet)
- [Architecture du projet](#2--architecture-du-projet)
- [Installation](#3-️-installation)
- [Création de l’environnement virtuel](#31-création-de-lenvironnement-virtuel)
- [Installation des dépendances](#32-installation-des-dépendances)
- [Téléchargement des données](#33-téléchargement-des-données)
- [Usage](#4--usage)
- [Lancer l’API FastAPI](#41-lancer-lapi-fastapi)
- [Lancer Streamlit](#42-lancer-streamlit)
- [Scripts utiles](#5--scripts-utiles)
- [Tests unitaires](#6--tests-unitaires)
- [Modèle et performances](#7--modèle-et-performances)

---

## 1. 📌 Description du projet

Ce projet a pour objectif de prédire l’affluence des vélos en libre-service à Paris en utilisant :
- Les données historiques de comptage vélo fournies par la Mairie de Paris
- Les données météo
- Le modèle de prédiction utilise LightGBM Regressor et un pipeline de features incluant :
    - 📊 Statistiques par site : moyenne, max, min, écart-type
    - 🔄 Valeurs historiques récursives : lags et rolling
    - 🕒 Features temporelles et cycliques : jour, heure, saison
    - 🌦 Conditions météorologiques : pluie, neige, vent, température apparente
    - 🏖 Indicateurs vacances et heures de pointe
- Une API FastAPI permet d’effectuer des prédictions en temps réel, et une interface Streamlit fournit des visualisations interactives (heatmap et tendances horaires).

---

## 2. 🏗 Architecture du projet
```bash
VELIB/
├─ api/                  # API FastAPI pour les prédictions
│  └─ main.py
├─ data/                 # Gestion des données
│  ├─ ingestion.py       # Scripts d’ingestion
│  ├─ loader.py          # Chargement des CSV et données traitées
│  ├─ preprocessing.py   # Nettoyage et feature engineering
│  └─ metadata.py        # Gestion de l’état des données
├─ models/               # Modèles et fonctions associées
│  ├─ features.py        # Feature engineering
│  ├─ inference.py       # Fonctions de prédiction
│  ├─ model_utils.py     # Chargement du modèle
│  ├─ predict.py         # Pipeline de prédiction
│  ├─ train.py           # Entraînement du modèle
│  └─ model.pkl          # Modèle LightGBM entraîné
├─ pages/                # Streamlit : interface utilisateur
│  ├─ analysis.py
│  ├─ overview.py
│  └─ prediction.py
├─ scripts/              # Scripts utilitaires
│  ├─ run_eval.sh
│  ├─ run_predict.sh
│  ├─ run_train.sh
│  ├─ run_update.sh
│  └─ update_data.py     # Mise à jour des données vélo et météo
├─ tests/                # Tests unitaires
│  ├─ test_api.py
│  ├─ test_model.py
│  ├─ test_predict.py
│  └─ test_recursive.py
├─ utils/
├─ requirements.txt
├─ Makefile
└─ README.md
```

### 🔄 Flux global
- update_data.py : Mise à jour des CSV depuis la Mairie de Paris et données météo (weather API)
- train.py : Entraînement du modèle LightGBM
- predict.py / FastAPI : Prédictions pour une date/heure donnée
- Streamlit : Visualisation des prédictions sur carte et graphiques horaires

---

## 3. ⚙️ Installation
### 3.1 Création de l’environnement virtuel
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3.2 Installation des dépendances
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3.3 Téléchargement des données
Téléchargez le CSV de la Mairie de Paris pour les comptages vélos et placez-le dans le dossier VELIB/  
Nommez le : comptage_velo_donnees_compteurs.csv  
Pour mettre à jour automatiquement les données lancer le script: 
```bash
python scripts/update_data.py
```
Le fichier metadata/data_state.json permet de définir les dates de début et fin des données à charger.

---

## 4. 🚀 Usage
### 4.1 Lancer l’API FastAPI
```bash
uvicorn api.main:app --reload
http://127.0.0.1:8000/docs

Endpoint santé :
GET http://127.0.0.1:8000/health

Endpoint prédiction :
POST http://127.0.0.1:8000/predict

Exemple JSON pour la prédiction :
{
    "datetime": "2026-02-12 12:00:00"
}
```

### 4.2 Lancer Streamlit
```bash
streamlit run app.py
```
- Page Overview : présentation du projet et des données brutes.
- Page Analysis : tests statistiques et visualisations des données.
- Page Model & Prédictions : prédire l'affluence à une heure donnée, avec heatmap de Paris et graphique de comptage horaire pour chaque site.

---

## 5. 🛠 Scripts utiles
```bash
run_update.sh	Mise à jour des CSV vélo et météo
run_train.sh	Entraînement du modèle LightGBM
run_predict.sh	Lancer une prédiction en ligne de commande
run_eval.sh	Évaluation des métriques du modèle
update_data.py	Script Python pour télécharger et nettoyer les données
```

---

## 6. ✅ Tests unitaires
Pour exécuter tous les tests, dans le terminal :
```bash
pytest
```
Les tests couvrent :
- Chargement du modèle
- Existence et cohérence des features
- Fonctionnement du forecast récursif
- Prédictions positives et robustes
- API FastAPI (/predict et /health)

## 7. 📈 Modèle et performances
- Modèle : LightGBM Regressor
- Features : Historique récursif, statistiques site, météo, temps, vacances, cycles horaire/jour
- Metrics : MAE, RMSE, R² (stockées dans metrics.json)