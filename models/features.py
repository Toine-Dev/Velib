import pandas as pd
import numpy as np


def add_time_features(dt_index: pd.DatetimeIndex, datetime_pred: pd.Timestamp, datetime_col: str="date_et_heure_de_comptage") -> pd.DataFrame:
    df = pd.DataFrame({datetime_col: dt_index})
    df[datetime_col] = pd.to_datetime(df[datetime_col]).dt.tz_localize(None).dt.floor('H')

    df["jour"] = df[datetime_col].dt.weekday
    df['saison'] = df[datetime_col].apply(get_season_from_date)
    df["nom_jour"] = df[datetime_col].dt.day_name(locale="fr_FR.UTF-8")
    df["mois"] = df[datetime_col].dt.month
    df["heure"] = df[datetime_col].dt.hour
    df['nuit'] = df.apply(is_night, axis=1)
    df["vacances"] = df[datetime_col].apply(is_vacances)
    df["heure_de_pointe"] = df[datetime_col].apply(is_rush_hour)
    df = df[df[datetime_col] <= datetime_pred].copy()

    return df

def merge_weather(hours_df, df_weather):
    # Ensure time column is also timezone-naive at hourly precision for proper merge
    df_weather['time'] = pd.to_datetime(df_weather['time'], errors='coerce').dt.tz_localize(None).dt.floor('H')

    # hours_df['date_et_heure_de_comptage'] = pd.to_datetime(hours_df['date_et_heure_de_comptage']).dt.tz_localize('UTC').dt.tz_convert('Europe/Paris').dt.floor('H')

    df = pd.merge(hours_df, df_weather, how="left",
                            left_on="date_et_heure_de_comptage",
                            right_on="time").drop(columns=["time"])

    df['pluie'] = (df['rain'].fillna(0) > 0).astype(int)
    df['neige'] = (df['snowfall'].fillna(0) > 0).astype(int)
    df['vent'] = (df['wind_speed_10m'].fillna(0) > 15).astype(int)
    df['apparent_temperature'] = df['apparent_temperature'].fillna(df['apparent_temperature'].mean())

    return df

def cross_join_sites(processed_df, future_df):
    unique_sites = processed_df[['identifiant_du_site_de_comptage']].drop_duplicates()
    return unique_sites, unique_sites.merge(future_df, how='cross')

def add_site_statistics(cartesian_df, processed_df):

    site_stats = processed_df.groupby('identifiant_du_site_de_comptage')['comptage_horaire'].agg(['mean','std','max','min']).rename(columns={
        'mean':'site_mean_usage',
        'std':'site_usage_variability',
        'max':'site_max_usage',
        'min':'site_min_usage'
    }).reset_index()

    return site_stats, cartesian_df.merge(site_stats, on='identifiant_du_site_de_comptage', how='left')


def build_history(processed_df, unique_sites):

    processed_df_sorted = processed_df.sort_values(['identifiant_du_site_de_comptage', 'date_et_heure_de_comptage'])
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
        history_dict[site] = list(last24) # Chronological order, oldest...newest

    return history_dict

def recursive_forecast(cartesian_df, history_dict, pipeline, feature_names, site_stats):
    cartesian_df = cartesian_df.sort_values(['identifiant_du_site_de_comptage','date_et_heure_de_comptage']).reset_index(drop=False) # trier cartesian_df pour avoir un ordre stable
    preds = [] # (idx, pred) pour recoller ensuite

    for site, site_df in cartesian_df.groupby('identifiant_du_site_de_comptage', sort=True):
        history = history_dict.get(site, [0]*24) # oldest...newest
        site_df = site_df.sort_values('date_et_heure_de_comptage') # s'assurer que site_df est trié par date
        for idx, row in site_df.iterrows():
             # calcul des lags à partir de l'historique courant
            lag_1 = history[-1] if len(history) >= 1 else 0
            lag_24 = history[0] if len(history) >= 24 else history[0] if len(history) > 0 else 0
            rolling_mean_24 = float(np.mean(history)) if len(history) >= 1 else 0.0

            # préparer une copie de la ligne et injecter les lags dans les noms attendus
            row = row.copy()
            row['lag_1'] = lag_1
            row['lag_24'] = lag_24
            row['rolling_mean_24'] = rolling_mean_24

            # Construire X_row avec les mêmes feature_names utilisés au training
            X = pd.DataFrame([row[feature_names]])

            # prédiction via pipeline (préprocesseur + modèle)
            try:
                pred = float(pipeline.predict(X)[0])
            except:
                # fallback sécuritaire si problème : 0 ou moyenne site
                pred = float(max(0, site_stats.loc[site_stats['identifiant_du_site_de_comptage']==site, 'site_mean_usage'].values[0] 
                                 if site in site_stats['identifiant_du_site_de_comptage'].values else 0))

            pred = max(pred, 0.0) # clip à 0 positif

            preds.append((idx, pred)) # stocker prediction pour recoller

            history.append(pred) # mise à jour FIFO de l'historique : on ajoute la pred comme nouvelle valeur la plus récente
            history = history[-24:]
            # garder seulement les 24 dernières
            if len(history) > 24:
                history = history[-24:]

    # Recréer une colonne predictions alignée avec cartesian_df index
    preds_df = pd.DataFrame(preds, columns=['orig_index', 'pred'])
    preds_df.set_index('orig_index', inplace=True)

    # cartesian_df.index correspond aux 'idx' utilisés plus haut
    cartesian_df['comptage_horaire'] = cartesian_df.index.map(
        lambda i: preds_df.loc[i,'pred'] if i in preds_df.index else 0.0
    )

    return cartesian_df


# Function to determine the season from a date
def get_season_from_date(date):
    # Ensure date is timezone-aware, if not, assume UTC
    if date.tz is None:
        date = pd.Timestamp(date, tz='UTC')
    
    year = date.year
    # Create timezone-aware seasonal boundary dates
    spring = pd.Timestamp(f'{year}-03-20', tz='UTC')
    summer = pd.Timestamp(f'{year}-06-21', tz='UTC')
    autumn = pd.Timestamp(f'{year}-09-22', tz='UTC')
    winter = pd.Timestamp(f'{year}-12-21', tz='UTC')

    # Convert input date to UTC for comparison
    date_utc = date.tz_convert('UTC')

    if spring <= date_utc < summer:
        return 'spring'
    elif summer <= date_utc < autumn:
        return 'summer'
    elif autumn <= date_utc < winter:
        return 'autumn'
    else:
        return 'winter'
    

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


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["pluie"] = (df["rain"].fillna(0) > 0).astype(int)
    df["neige"] = (df["snowfall"].fillna(0) > 0).astype(int)
    df["vent"] = (df["wind_speed_10m"].fillna(0) > 15).astype(int)

    df["apparent_temperature"] = df["apparent_temperature"].fillna(
        df["apparent_temperature"].mean()
    )

    return df

def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["heure_sin"] = np.sin(2 * np.pi * df["heure"] / 24)
    df["heure_cos"] = np.cos(2 * np.pi * df["heure"] / 24)

    df["jour_sin"] = np.sin(2 * np.pi * df["jour"] / 7)
    df["jour_cos"] = np.cos(2 * np.pi * df["jour"] / 7)

    return df

def compute_site_stats(processed_df: pd.DataFrame) -> pd.DataFrame:
    return (
        processed_df
        .groupby("identifiant_du_site_de_comptage")["comptage_horaire"]
        .agg(["mean", "std", "max", "min"])
        .rename(columns={
            "mean": "site_mean_usage",
            "std": "site_usage_variability",
            "max": "site_max_usage",
            "min": "site_min_usage",
        })
        .reset_index()
    )

def build_future_timeframe(
    start_datetime: pd.Timestamp,
    end_datetime: pd.Timestamp,
) -> pd.DataFrame:
    future_hours = pd.date_range(
        start=start_datetime,
        end=end_datetime,
        freq="h",
        inclusive="right"
    )

    df = pd.DataFrame({"date_et_heure_de_comptage": future_hours})
    df["date_et_heure_de_comptage"] = (
        pd.to_datetime(df["date_et_heure_de_comptage"])
        .dt.tz_localize(None)
        .dt.floor("H")
    )

    return df

def build_future_features(
    processed_df: pd.DataFrame,
    weather_forecast_df: pd.DataFrame,
    start_datetime: pd.Timestamp,
    end_datetime: pd.Timestamp,
) -> pd.DataFrame:
    df = build_future_timeframe(start_datetime, end_datetime)

    df = add_time_features(df, "date_et_heure_de_comptage")
    df["saison"] = df["date_et_heure_de_comptage"].apply(get_season_from_date)
    df["nuit"] = df.apply(is_night, axis=1)
    df["vacances"] = df["date_et_heure_de_comptage"].apply(is_vacances)
    df["heure_de_pointe"] = df["date_et_heure_de_comptage"].apply(is_rush_hour)

    # merge weather
    weather = weather_forecast_df.copy()
    weather["time"] = (
        pd.to_datetime(weather["time"])
        .dt.tz_localize(None)
        .dt.floor("H")
    )

    df = df.merge(
        weather,
        left_on="date_et_heure_de_comptage",
        right_on="time",
        how="left"
    ).drop(columns=["time"])

    df = add_weather_features(df)
    df = add_cyclic_features(df)

    # cross join with sites
    sites = processed_df[["identifiant_du_site_de_comptage"]].drop_duplicates()
    df = sites.merge(df, how="cross")

    # add site stats
    site_stats = compute_site_stats(processed_df)
    df = df.merge(site_stats, on="identifiant_du_site_de_comptage", how="left")

    return df