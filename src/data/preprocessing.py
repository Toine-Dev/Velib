import json
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from utils.config import database_url

def standardize_columns(df):
    df.columns = [
        col.strip().replace(" ", "_").lower()
        for col in df.columns
    ]
    return df

def coerce_velib_types(df: pd.DataFrame) -> pd.DataFrame:
    # Station id: may come as float/string; coerce safely
    df["identifiant_du_site_de_comptage"] = pd.to_numeric(
        df["identifiant_du_site_de_comptage"], errors="coerce"
    ).astype("Int64")  # pandas nullable integer

    # Counts: integers
    df["comptage_horaire"] = pd.to_numeric(
        df["comptage_horaire"], errors="coerce"
    ).astype("Int64")

    # Datetime: parse
    df["date_et_heure_de_comptage"] = pd.to_datetime(
        df["date_et_heure_de_comptage"], errors="coerce"
    )

    # Coordinates: keep as text
    if "coordonnees_geographiques" in df.columns:
        df["coordonnees_geographiques"] = df["coordonnees_geographiques"].astype(str)

    return df

# Function to determine the season from a date
def get_season_from_date(date) -> str:
    """
    Determine season using local calendar date (month/day).
    Works with tz-naive or tz-aware inputs, but always compares using tz-naive
    clock time (no conversions, no tz attachment).
    """
    d = pd.Timestamp(date)

    # If tz-aware, drop tz WITHOUT converting (keeps clock time)
    if d.tzinfo is not None:
        d = d.tz_localize(None)

    year = d.year
    spring = pd.Timestamp(f"{year}-03-20")
    summer = pd.Timestamp(f"{year}-06-21")
    autumn = pd.Timestamp(f"{year}-09-22")
    winter = pd.Timestamp(f"{year}-12-21")

    if spring <= d < summer:
        return "spring"
    elif summer <= d < autumn:
        return "summer"
    elif autumn <= d < winter:
        return "autumn"
    else:
        return "winter"
    

def is_night(row):
    # Example rough night hours per season (24h format)
    night_hours = {
        'winter':    {'start': 17, 'end': 8},
        'spring':{'start': 20.5, 'end': 6},   # 20:30
        'summer':      {'start': 22, 'end': 5},
        'autumn':  {'start': 19, 'end': 7},
    }

    season = row['saison'].lower()
    dt = row['date_et_heure_de_comptage']
    hour = dt.hour + dt.minute/60  # fractional hour
    
    nh = night_hours.get(season)
    if nh is None:
        # if season is unknown, consider not night
        return False
    
    start, end = nh['start'], nh['end']
    
    # Since all seasons cross midnight, we only need this check
    return hour >= start or hour < end


# Define function to test if date falls in a holiday
def is_vacances(date):
    # Define vacation periods inside the function
    vacances_periods = [
        ('2024-10-19', '2024-11-05'),  # Toussaint
        ('2024-12-21', '2025-01-07'),  # Noël
        ('2025-02-15', '2025-03-04'),  # Hiver
        ('2025-04-12', '2025-04-29'),  # Printemps
        ('2025-05-29', '2025-06-01'),  # Ascension + pont (29, 30, 31)
        ('2025-07-05', '2025-09-02'),  # Summer begins 5 July to 1 Sept
    ]
    
    # Convert to datetime timestamps
    vacances_intervals = [
        (pd.Timestamp(start), pd.Timestamp(end))
        for start, end in vacances_periods
    ]
    
    # Check if date falls in any vacation period
    for start, end in vacances_intervals:
        if start <= date < end:
            return True
    return False


# Function to classify rush hour
def is_rush_hour(dt):
    hour = dt.hour
    return (7 <= hour < 10) or (17 <= hour < 20)


def static_features(df):
    site_stats = (
        df.groupby('identifiant_du_site_de_comptage')['comptage_horaire']
        .agg(['mean', 'std', 'max', 'min'])
        .rename(columns={
            'mean': 'site_mean_usage',
            'std': 'site_usage_variability',
            'max': 'site_max_usage',
            'min': 'site_min_usage'
        })
    )
    df = df.merge(site_stats, on='identifiant_du_site_de_comptage', how='left')
    return df
 

def time_varying_features(df):
    # Time-varying features per site
    df = df.sort_values(['identifiant_du_site_de_comptage', 'date_et_heure_de_comptage'])

    df['lag_1'] = df.groupby('identifiant_du_site_de_comptage')['comptage_horaire'].shift(1)
    df['lag_24'] = df.groupby('identifiant_du_site_de_comptage')['comptage_horaire'].shift(24)
    df['rolling_mean_24'] = (
        df.groupby('identifiant_du_site_de_comptage')['comptage_horaire']
        .shift(1).rolling(24).mean()
    )
    return df


#Fonction pour encodage cyclique
def add_cyclic_features(df):

    # Encodage cyclique pour mois (1-12)
    df['jour_sin'] = np.sin(2 * np.pi * df['jour'] / 7)
    df['jour_cos'] = np.cos(2 * np.pi * df['jour'] / 7)

    # Encodage cyclique pour mois (1-12)
    df['mois_sin'] = np.sin(2 * np.pi * df['mois'] / 12)
    df['mois_cos'] = np.cos(2 * np.pi * df['mois'] / 12)

    # Encodage cyclique pour heure (0-23)
    df['heure_sin'] = np.sin(2 * np.pi * df['heure'] / 24)
    df['heure_cos'] = np.cos(2 * np.pi * df['heure'] / 24)

    # Encodage cyclique pour saison 
    df["saison_sin"]  = np.sin(2 * np.pi * df["saison"].map({'winter':0, 'spring':1, 'summer':2, 'autumn':3}) / 4)
    df["saison_cos"]  = np.cos(2 * np.pi * df["saison"].map({'winter':0, 'spring':1, 'summer':2, 'autumn':3}) / 4)

    df = df.drop(columns=["jour", "saison", "heure", "mois"])

    return df


# Load and preprocess data
def preprocess_velib_data(df):
    print('Preprocessing has started.')
    df = df.dropna().copy()
    df['date_et_heure_de_comptage'] = pd.to_datetime(df['date_et_heure_de_comptage'].astype(str), errors='coerce')
    df = df.dropna(subset=['date_et_heure_de_comptage']).copy()
    if df["date_et_heure_de_comptage"].dt.tz is not None:
        df['date_et_heure_de_comptage'] = df['date_et_heure_de_comptage'].dt.tz_localize(None)

    # Features temporelles
    df['heure'] = df['date_et_heure_de_comptage'].dt.hour
    df['mois'] = df['date_et_heure_de_comptage'].dt.month
    df['jour'] = df['date_et_heure_de_comptage'].dt.day
    df['saison'] = df['date_et_heure_de_comptage'].apply(get_season_from_date)
    df['vacances'] = df['date_et_heure_de_comptage'].apply(is_vacances)
    df['heure_de_pointe'] = df['date_et_heure_de_comptage'].apply(is_rush_hour)
    df['nuit'] = df.apply(is_night, axis=1)

    # # Coordonnées
    # coords = df["coordonnées_géographiques"].str.split(",", expand=True)
    # df["latitude"] = coords[0].astype(float)
    # df["longitude"] = coords[1].astype(float)

    # Alternative parsing of coordinates if they are stored as JSON strings (e.g. '{"lat": 48.8575, "lon": 2.3514}')
    coords = df["coordonnees_geographiques"].apply(
        lambda s: json.loads(s) if isinstance(s, str) and s.strip().startswith("{") else {}
    )

    df["latitude"] = pd.to_numeric(coords.apply(lambda d: d.get("lat")), errors="coerce")
    df["longitude"] = pd.to_numeric(coords.apply(lambda d: d.get("lon") or d.get("lng")), errors="coerce")

    return df


def preprocess_weather_data(df):
    # Ajout météo
    df['pluie'] = (df['rain'] > 0)
    df['vent'] = (df['wind_speed_10m'] > 30)
    df['neige'] = (df['snowfall'] > 0)

    # Convert 'time' column to datetime
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    # Check for any conversion errors
    if df['time'].isnull().any():
        print("Warning: Some time values could not be converted to datetime.")

    return df


def preprocess_merged_data(df):

    # Nettoyage colonnes inutiles
    # df = df.drop(columns=["latitude", "longitude", "date_d'installation_du_site_de_comptage",
    #                                     "identifiant_technique_compteur", "mois_annee_comptage", "identifiant_du_compteur",
    #                                     "nom_du_site_de_comptage", "nom_du_compteur", "snowfall", "rain",
    #                                     "wind_speed_10m", 'lien_vers_photo_du_site_de_comptage', 'id_photos',
    #                                     'test_lien_vers_photos_du_site_de_comptage_', 'id_photo_1', 'url_sites', 'type_dimage',
    #                                     "coordonnees_geographiques"])
    df = df.drop(columns=["latitude", "longitude", "nom_du_site_de_comptage", "snowfall", 
                          "rain","wind_speed_10m", "coordonnees_geographiques"])
                                        

    # Ajout des features statiques et dynamiques
    # df = static_features(df)
    df = time_varying_features(df)
    df = df.dropna()

    # Ajout des features cycliques
    df_encoded = add_cyclic_features(df)
    print("df_encoded:",df_encoded.columns)

    # Sélection des features
    features = [col for col in df_encoded.columns if col not in ['comptage_horaire', 'date_et_heure_de_comptage']]
    df_encoded = df_encoded.sort_values(by='date_et_heure_de_comptage', ascending=True).reset_index(drop = True)
    return df_encoded, features






DB_URL = database_url()

RAW_TABLE = "velib_raw"
FEATURES_TABLE = "site_features"

engine = create_engine(DB_URL, future=True)

def main():
    with engine.begin() as conn:  # auto-commit/rollback
        conn.execute(text(f"""
        CREATE TABLE IF NOT EXISTS {FEATURES_TABLE} (
          identifiant_du_site_de_comptage text PRIMARY KEY,
          n bigint NOT NULL,
          mean double precision,
          std double precision,
          min double precision,
          max double precision
        );
        """))

        conn.execute(text(f"""
        CREATE INDEX IF NOT EXISTS idx_raw_site
        ON {RAW_TABLE} (identifiant_du_site_de_comptage);
        """))

        conn.execute(text(f"""
        INSERT INTO {FEATURES_TABLE} (
          identifiant_du_site_de_comptage,
          n,
          mean,
          std,
          min,
          max
        )
        SELECT
          identifiant_du_site_de_comptage,
          COUNT(comptage_horaire) AS n,
          AVG(comptage_horaire)::double precision AS mean,
          STDDEV_SAMP(comptage_horaire)::double precision AS std,
          MIN(comptage_horaire)::double precision AS min,
          MAX(comptage_horaire)::double precision AS max
        FROM {RAW_TABLE}
        WHERE comptage_horaire IS NOT NULL
        GROUP BY identifiant_du_site_de_comptage
        ON CONFLICT (identifiant_du_site_de_comptage) DO UPDATE SET
          n = EXCLUDED.n,
          mean = EXCLUDED.mean,
          std = EXCLUDED.std,
          min = EXCLUDED.min,
          max = EXCLUDED.max;
        """))

    print("✅ site_features refreshed")