import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
import mlflow

# Reuse your existing logic
# - compute_site_stats(processed_df)
# - build_history(processed_df, unique_sites)
# - recursive_forecast(cartesian_df, history_dict, pipeline, feature_names, site_stats)
from models.features import compute_site_stats, build_history, recursive_forecast

DB_URL = os.environ["DATABASE_URL"]
MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "velib_forecast")
MODEL_URI = os.getenv("MLFLOW_MODEL_URI")  # optional
FORECAST_HOURS = int(os.getenv("FORECAST_HOURS", "48"))
HISTORY_WINDOW_DAYS = int(os.getenv("HISTORY_WINDOW_DAYS", "30"))


def _drop_tz_if_any(s: pd.Series) -> pd.Series:
    if hasattr(s.dt, "tz") and s.dt.tz is not None:
        return s.dt.tz_localize(None)
    return s


def ensure_forecast_table(engine):
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS velib_forecast (
                identifiant_du_site_de_comptage BIGINT NOT NULL,
                date_et_heure_de_comptage TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                comptage_horaire DOUBLE PRECISION NOT NULL,
                generated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW(),
                model_uri TEXT NULL,
                PRIMARY KEY (identifiant_du_site_de_comptage, date_et_heure_de_comptage)
            );
        """))


def load_latest_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"MLflow experiment not found: {EXPERIMENT_NAME}")

    if MODEL_URI:
        uri = MODEL_URI
    else:
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )
        if runs.empty:
            raise RuntimeError(f"No MLflow runs found in experiment: {EXPERIMENT_NAME}")
        run_id = runs.loc[0, "run_id"]
        uri = f"runs:/{run_id}/model"

    model = mlflow.pyfunc.load_model(uri)
    return model, uri


# ---------- feature building (matching velib_weather_processed schema) ----------

def season_name(ts: pd.Timestamp) -> str:
    y = ts.year
    spring = pd.Timestamp(f"{y}-03-20")
    summer = pd.Timestamp(f"{y}-06-21")
    autumn = pd.Timestamp(f"{y}-09-22")
    winter = pd.Timestamp(f"{y}-12-21")
    if spring <= ts < summer:
        return "spring"
    if summer <= ts < autumn:
        return "summer"
    if autumn <= ts < winter:
        return "autumn"
    return "winter"


def is_rush_hour(ts: pd.Timestamp) -> bool:
    h = ts.hour
    return (7 <= h < 10) or (17 <= h < 20)


def is_night(ts: pd.Timestamp) -> bool:
    # simple seasonal night bounds (clock-time based)
    s = season_name(ts)
    h = ts.hour + ts.minute / 60.0
    bounds = {
        "winter": (17, 8),
        "spring": (20.5, 6),
        "summer": (22, 5),
        "autumn": (19, 7),
    }
    start, end = bounds[s]
    return (h >= start) or (h < end)


def is_vacances(ts: pd.Timestamp) -> bool:
    # replace with your holiday-calendar logic if you have one;
    # keep consistent with what you trained on.
    d = ts
    periods = [
        ("2024-10-19", "2024-11-05"),
        ("2024-12-21", "2025-01-07"),
        ("2025-02-15", "2025-03-04"),
        ("2025-04-12", "2025-04-29"),
        ("2025-05-29", "2025-06-01"),
        ("2025-07-05", "2025-09-02"),
    ]
    for a, b in periods:
        if pd.Timestamp(a) <= d < pd.Timestamp(b):
            return True
    return False


def build_future_feature_frame(start_dt: pd.Timestamp, end_dt: pd.Timestamp, weather_fc: pd.DataFrame) -> pd.DataFrame:
    # hourly timestamps strictly after start_dt, up to end_dt inclusive
    hours = pd.date_range(start=start_dt, end=end_dt, freq="h", inclusive="right")
    df = pd.DataFrame({"date_et_heure_de_comptage": hours})
    df["date_et_heure_de_comptage"] = pd.to_datetime(df["date_et_heure_de_comptage"], errors="coerce").dt.floor("h")

    # time booleans
    dt = df["date_et_heure_de_comptage"]
    df["vacances"] = dt.apply(is_vacances).astype(bool)
    df["heure_de_pointe"] = dt.apply(is_rush_hour).astype(bool)
    df["nuit"] = dt.apply(is_night).astype(bool)

    # cyclic features expected by your model
    weekday = dt.dt.weekday.astype(float)  # 0..6
    month = dt.dt.month.astype(float)      # 1..12
    hour = dt.dt.hour.astype(float)        # 0..23

    df["jour_sin"] = np.sin(2 * np.pi * weekday / 7.0)
    df["jour_cos"] = np.cos(2 * np.pi * weekday / 7.0)

    df["mois_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["mois_cos"] = np.cos(2 * np.pi * month / 12.0)

    df["heure_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["heure_cos"] = np.cos(2 * np.pi * hour / 24.0)

    season_map = {"winter": 0, "spring": 1, "summer": 2, "autumn": 3}
    s_idx = dt.apply(lambda x: season_map[season_name(x)]).astype(float)
    ang = 2 * np.pi * s_idx / 4.0
    df["saison_sin"] = np.sin(ang)
    df["saison_cos"] = np.cos(ang)

    # merge weather_forecast_raw (already tz-naive Paris clock time)
    w = weather_fc.copy()
    w["time"] = pd.to_datetime(w["time"], errors="coerce").dt.floor("h")
    df = df.merge(w, left_on="date_et_heure_de_comptage", right_on="time", how="left").drop(columns=["time"])

    if df["apparent_temperature"].isna().all():
        raise RuntimeError("weather_forecast_raw has no apparent_temperature for the horizon.")

    # convert raw weather to processed booleans
    df["pluie"] = (df["rain"].fillna(0) > 0).astype(bool)
    df["neige"] = (df["snowfall"].fillna(0) > 0).astype(bool)
    df["vent"] = (df["wind_speed_10m"].fillna(0) > 15).astype(bool)

    # keep only expected numeric weather
    df["apparent_temperature"] = df["apparent_temperature"].astype(float)

    # drop raw numeric cols not used by model (your processed schema doesn’t include them)
    df = df.drop(columns=["rain", "snowfall", "wind_speed_10m"], errors="ignore")

    return df


def main():
    engine = create_engine(DB_URL)
    ensure_forecast_table(engine)

    # 1) start_dt from velib_raw (tz-naive Paris clock time)
    start_dt = pd.read_sql("SELECT MAX(date_et_heure_de_comptage) AS max_dt FROM velib_raw;", engine).loc[0, "max_dt"]
    if pd.isna(start_dt):
        raise RuntimeError("velib_raw is empty; cannot forecast.")
    start_dt = pd.to_datetime(start_dt).floor("h")
    end_dt = start_dt + pd.Timedelta(hours=FORECAST_HOURS)

    # last hour in processed
    last_proc = pd.read_sql(
        "SELECT MAX(date_et_heure_de_comptage) AS max_dt FROM velib_weather_processed;",
        engine
    ).loc[0, "max_dt"]

    if pd.isna(last_proc):
        raise RuntimeError("velib_weather_processed is empty; cannot forecast.")

    last_proc = pd.to_datetime(last_proc, errors="coerce")
    if getattr(last_proc, "tzinfo", None) is not None:
        last_proc = last_proc.tz_localize(None)
    last_proc = last_proc.floor("h")

    if last_proc != start_dt:
        raise RuntimeError(
            f"Processed table is not aligned with raw. "
            f"velib_raw max={start_dt} but velib_weather_processed max={last_proc}. "
            f"Run processing pipeline before forecasting."
        )

###### USE THIS IF YOU WANT TO AUTO-ALIGN FORECAST START TO PROCESSED MAX (RISKY, BE CAREFUL) INSTEAD OF RAISING RUNTIME ERROR ABOVE ######
    # if last_proc != start_dt:
    #     print(
    #         f"⚠️ processed lags raw: raw max={start_dt}, processed max={last_proc}. "
    #         f"Forecast will start from processed max."
    #     )
    #     start_dt = last_proc
    #     end_dt = start_dt + pd.Timedelta(hours=FORECAST_HOURS)
###### USE THIS IF YOU WANT TO AUTO-ALIGN FORECAST START TO PROCESSED MAX (RISKY, BE CAREFUL) INSTEAD OF RAISING RUNTIME ERROR ABOVE ######

    # 2) history from velib_weather_processed (need comptage_horaire for lags)
    min_hist = start_dt - pd.Timedelta(days=HISTORY_WINDOW_DAYS)
    processed = pd.read_sql(
        text("""
            SELECT *
            FROM velib_weather_processed
            WHERE date_et_heure_de_comptage >= :min_dt
              AND date_et_heure_de_comptage <= :start_dt
        """),
        engine,
        params={"min_dt": min_hist, "start_dt": start_dt},
    )
    if processed.empty:
        raise RuntimeError("velib_weather_processed has no history rows in the selected window.")

    processed["date_et_heure_de_comptage"] = pd.to_datetime(processed["date_et_heure_de_comptage"], errors="coerce")
    processed["date_et_heure_de_comptage"] = _drop_tz_if_any(processed["date_et_heure_de_comptage"]).dt.floor("h")

    # 3) weather forecast for horizon
    weather_fc = pd.read_sql(
        text("""
            SELECT *
            FROM weather_forecast_raw
            WHERE time > :start_dt AND time <= :end_dt
        """),
        engine,
        params={"start_dt": start_dt, "end_dt": end_dt},
    )
    if weather_fc.empty:
        raise RuntimeError("weather_forecast_raw has no rows for the forecast horizon.")

    # 4) build future feature frame
    future_feats = build_future_feature_frame(start_dt, end_dt, weather_fc)
    print("Future feature frame for forecast horizon has columns:", future_feats.columns.tolist())

    
    # sites = processed[["identifiant_du_site_de_comptage"]].drop_duplicates()
    # print("sites DataFrame has columns:", sites.columns.tolist())
    # cartesian = sites.merge(future_feats, how="cross")
    # print("cartesian DataFrame after cross join has columns:", cartesian.columns.tolist())

    
    # 5) cross join with sites
    site_stats = pd.read_sql(
    "SELECT identifiant_du_site_de_comptage, mean, std, min, max FROM site_features",
    engine
)
    sites = processed[["identifiant_du_site_de_comptage"]].drop_duplicates()
    sites = sites.merge(site_stats, on="identifiant_du_site_de_comptage", how="left")
    cartesian = sites.merge(future_feats, how="cross")

    # 6) history dict + site stats
    history_dict = build_history(processed, sites)


    # placeholders for lags (will be filled per-row in recursion)
    for c in ["lag_1", "lag_24", "rolling_mean_24"]:
        if c not in cartesian.columns:
            cartesian[c] = np.nan

    # 7) model + feature list (exactly: processed columns minus target)
    model, model_uri = load_latest_model()
    feature_names = [c for c in processed.columns if c != "comptage_horaire"]
    print(f"Using model from {model_uri} with features: {feature_names}")

    missing = [c for c in feature_names if c not in cartesian.columns]
    if missing:
        raise RuntimeError(f"cartesian_df missing model features: {missing}")

    # 8) recursive prediction
    forecast_full = recursive_forecast(
        cartesian_df=cartesian,
        history_dict=history_dict,
        pipeline=model,
        feature_names=feature_names,
        site_stats=site_stats,
    )

    out = forecast_full[["identifiant_du_site_de_comptage", "date_et_heure_de_comptage", "comptage_horaire"]].copy()
    out["model_uri"] = model_uri

    # 9) upsert into velib_forecast
    upsert = text("""
        INSERT INTO velib_forecast (
            identifiant_du_site_de_comptage, date_et_heure_de_comptage, comptage_horaire, model_uri
        ) VALUES (
            :site, :dt, :y, :model_uri
        )
        ON CONFLICT (identifiant_du_site_de_comptage, date_et_heure_de_comptage)
        DO UPDATE SET
            comptage_horaire = EXCLUDED.comptage_horaire,
            generated_at = NOW(),
            model_uri = EXCLUDED.model_uri;
    """)

    with engine.begin() as conn:
        conn.execute(
            upsert,
            [
                {
                    "site": int(r.identifiant_du_site_de_comptage),
                    "dt": pd.Timestamp(r.date_et_heure_de_comptage).to_pydatetime(),
                    "y": float(r.comptage_horaire),
                    "model_uri": r.model_uri,
                }
                for r in out.itertuples(index=False)
            ],
        )

    print(f"✅ Wrote {len(out)} forecast rows for {FORECAST_HOURS}h beyond {start_dt}.")
    print(f"   Model: {model_uri}")


if __name__ == "__main__":
    main()