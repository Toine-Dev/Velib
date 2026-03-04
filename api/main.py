import os
from datetime import datetime, timedelta
from typing import List, Optional, Literal
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from passlib.context import CryptContext
from jose import jwt, JWTError
from fastapi.responses import RedirectResponse
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi.responses import StreamingResponse
from io import BytesIO



# ---------------- Config ----------------
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://ml_user:ml_password@db:5432/ml_db")
MLFLOW_UI_URL = os.getenv("MLFLOW_UI_URL", "http://mlflow:5000")

JWT_SECRET = os.getenv("JWT_SECRET", "change-me")
JWT_ALG = "HS256"
ACCESS_TOKEN_MINUTES = int(os.getenv("ACCESS_TOKEN_MINUTES", "120"))

# engine = create_engine(DATABASE_URL)
engine = create_engine(
    DATABASE_URL,
    connect_args={"options": "-csearch_path=api,public"}
)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

app = FastAPI(title="Velib Prediction API")

# ---------------- Schemas ----------------
class PredictRequest(BaseModel):
    datetime: str  # "YYYY-MM-DD HH:00:00"

class PredictionRow(BaseModel):
    identifiant_du_site_de_comptage: int
    nom_du_site_de_comptage: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    comptage_horaire: float

class UserCreate(BaseModel):
    username: str
    password: str
    role: Literal["admin", "client"] = "client"

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class PipelineTriggerRequest(BaseModel):
    job_type: Literal["data", "model", "forecast"]

class TrendPoint(BaseModel):
    date_et_heure_de_comptage: datetime
    comptage_horaire: float

# ---------------- DB bootstrap (tables) ----------------
def ensure_api_tables():
    with engine.begin() as conn:
#         conn.execute(text("""
#             CREATE TABLE IF NOT EXISTS users (
#                 username TEXT PRIMARY KEY,
#                 password_hash TEXT NOT NULL,
#                 role TEXT NOT NULL CHECK (role IN ('admin', 'client')),
#                 created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW()
#             );
#         """))
#         conn.execute(text("""
#             CREATE TABLE IF NOT EXISTS pipeline_jobs (
#                 id BIGSERIAL PRIMARY KEY,
#                 job_type TEXT NOT NULL CHECK (job_type IN ('data','model','forecast')),
#                 status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued','running','success','failed')),
#                 created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW(),
#                 started_at TIMESTAMP WITHOUT TIME ZONE NULL,
#                 finished_at TIMESTAMP WITHOUT TIME ZONE NULL,
#                 message TEXT NULL
#             );
#         """))

        # Create a default admin if none exists (optional)
        res = conn.execute(text("SELECT COUNT(*) FROM users WHERE role='admin';")).scalar()
        if res == 0:
            conn.execute(
                text("INSERT INTO users(username, password_hash, role) VALUES (:u,:p,'admin')"),
                {"u": "admin", "p": pwd_context.hash("admin")}
            )

ensure_api_tables()

# ---------------- Auth helpers ----------------
def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_access_token(sub: str, role: str) -> str:
    exp = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_MINUTES)
    payload = {"sub": sub, "role": role, "exp": exp}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

def get_user(username: str):
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT username, password_hash, role FROM users WHERE username=:u"),
            {"u": username},
        ).mappings().first()
    return row

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        username = payload.get("sub")
        role = payload.get("role")
        if not username or not role:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"username": username, "role": role}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def require_admin(user=Depends(get_current_user)):
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    return user

# ---------------- Routes ----------------
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {"status": "ok"}

# ---- Auth ----
@app.post("/auth/token", response_model=Token)
def login(form: OAuth2PasswordRequestForm = Depends()):
    user = get_user(form.username)
    if not user or not verify_password(form.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = create_access_token(user["username"], user["role"])
    return Token(access_token=token)

# ---- Client: Predict (DB forecast) ----
@app.get("/client/forecast-heatmap")
def forecast_heatmap(datetime: str, user=Depends(get_current_user)):
    """
    Return a PNG heatmap of predicted Velib traffic for a given hour.
    Uses velib_forecast_geo (view: velib_forecast + velib_sites).
    """
    try:
        dt = pd.to_datetime(datetime, errors="raise").floor("H").to_pydatetime()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid datetime. Use 'YYYY-MM-DD HH:00:00'.")

    q = text("""
        SELECT latitude, longitude, comptage_horaire
        FROM velib_forecast_geo
        WHERE date_et_heure_de_comptage = :dt
          AND latitude IS NOT NULL
          AND longitude IS NOT NULL
    """)
    df = pd.read_sql(q, engine, params={"dt": dt})

    if df.empty:
        raise HTTPException(status_code=404, detail="No forecast rows for that datetime.")

    # ---- Create a simple 2D heatmap over lon/lat ----
    # (This is not a map tile; it’s a geographic density heatmap on axes.)
    lats = df["latitude"].to_numpy()
    lons = df["longitude"].to_numpy()
    weights = df["comptage_horaire"].to_numpy()

    # Define grid bounds (tight around Paris sites)
    lat_min, lat_max = lats.min() - 0.01, lats.max() + 0.01
    lon_min, lon_max = lons.min() - 0.01, lons.max() + 0.01

    # Grid resolution (increase for smoother but heavier)
    bins = 150

    # Weighted 2D histogram
    heat, xedges, yedges = np.histogram2d(
        lats, lons,
        bins=bins,
        range=[[lat_min, lat_max], [lon_min, lon_max]],
        weights=weights
    )

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(
        heat.T,
        origin="lower",
        extent=[lat_min, lat_max, lon_min, lon_max],
        aspect="auto"
    )
    ax.set_title(f"Velib Forecast Heatmap — {dt.strftime('%Y-%m-%d %H:00')}")
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Longitude")

    buf = BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")



@app.post("/client/predict", response_model=List[PredictionRow])
def predict(req: PredictRequest, user=Depends(get_current_user)):
    try:
        dt = pd.to_datetime(req.datetime, errors="raise").floor("h").to_pydatetime()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid datetime format. Use 'YYYY-MM-DD HH:00:00'.")

    q = text("""
        SELECT
            identifiant_du_site_de_comptage,
            nom_du_site_de_comptage,
            latitude,
            longitude,
            comptage_horaire
        FROM velib_forecast_geo
        WHERE date_et_heure_de_comptage = :dt
        ORDER BY comptage_horaire DESC
    """)
    df = pd.read_sql(q, engine, params={"dt": dt})
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail="No forecast found for this datetime (is recursive_forecast up to date?)"
        )
    return df.to_dict(orient="records")

# ---- Client: Trends for one site (historical + forecast) ----
@app.get("/client/trends")
def trends(site_id: int, start: str, end: str, user=Depends(get_current_user)):

    try:
        start_dt = pd.to_datetime(start).floor("H").to_pydatetime()
        end_dt = pd.to_datetime(end).floor("H").to_pydatetime()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid start/end datetime.")

    # Historical data
    q_hist = text("""
        SELECT date_et_heure_de_comptage, comptage_horaire
        FROM velib_weather_processed
        WHERE identifiant_du_site_de_comptage = :site
          AND date_et_heure_de_comptage >= :start
          AND date_et_heure_de_comptage <= :end
        ORDER BY date_et_heure_de_comptage
    """)

    hist = pd.read_sql(q_hist, engine, params={
        "site": site_id,
        "start": start_dt,
        "end": end_dt
    })

    # Forecast data
    q_fc = text("""
        SELECT date_et_heure_de_comptage, comptage_horaire
        FROM velib_forecast
        WHERE identifiant_du_site_de_comptage = :site
          AND date_et_heure_de_comptage >= :start
          AND date_et_heure_de_comptage <= :end
        ORDER BY date_et_heure_de_comptage
    """)

    fc = pd.read_sql(q_fc, engine, params={
        "site": site_id,
        "start": start_dt,
        "end": end_dt
    })

    if hist.empty and fc.empty:
        raise HTTPException(status_code=404, detail="No data found for selected window.")

    # Convert datetime
    hist["date_et_heure_de_comptage"] = pd.to_datetime(hist["date_et_heure_de_comptage"])
    fc["date_et_heure_de_comptage"] = pd.to_datetime(fc["date_et_heure_de_comptage"])

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 5))

    if not hist.empty:
        ax.plot(
            hist["date_et_heure_de_comptage"],
            hist["comptage_horaire"],
            label="Historical",
            color="blue"
        )

    if not fc.empty:
        ax.plot(
            fc["date_et_heure_de_comptage"],
            fc["comptage_horaire"],
            label="Forecast",
            color="red",
            linestyle="--"
        )

    ax.set_title(f"Velib Traffic Trend — Site {site_id}")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Hourly Count")
    ax.legend()
    ax.grid(True)

    buf = BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)

    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")

# @app.get("/client/trends", response_model=List[TrendPoint])
# def trends(
#     site_id: int,
#     start: str,
#     end: str,
#     user=Depends(get_current_user),
# ):
#     try:
#         start_dt = pd.to_datetime(start).floor("h").to_pydatetime()
#         end_dt = pd.to_datetime(end).floor("h").to_pydatetime()
#     except Exception:
#         raise HTTPException(status_code=400, detail="Invalid start/end datetime.")

#     q = text("""
#         SELECT date_et_heure_de_comptage, comptage_horaire
#         FROM velib_weather_processed
#         WHERE identifiant_du_site_de_comptage = :site
#           AND date_et_heure_de_comptage >= :start
#           AND date_et_heure_de_comptage <= :end
#         ORDER BY date_et_heure_de_comptage ASC
#     """)
#     hist = pd.read_sql(q, engine, params={"site": site_id, "start": start_dt, "end": end_dt})

#     q2 = text("""
#         SELECT date_et_heure_de_comptage, comptage_horaire
#         FROM velib_forecast
#         WHERE identifiant_du_site_de_comptage = :site
#           AND date_et_heure_de_comptage >= :start
#           AND date_et_heure_de_comptage <= :end
#         ORDER BY date_et_heure_de_comptage ASC
#     """)
#     fc = pd.read_sql(q2, engine, params={"site": site_id, "start": start_dt, "end": end_dt})

#     df = pd.concat([hist, fc], ignore_index=True).drop_duplicates(subset=["date_et_heure_de_comptage"]).sort_values("date_et_heure_de_comptage")
#     return df.to_dict(orient="records")

# ---- Client: Correlations (returns matrix as JSON) ----
@app.get("/client/correlations")
def correlations(
    start: str,
    end: str,
    user=Depends(get_current_user),
):
    try:
        start_dt = pd.to_datetime(start).floor("h").to_pydatetime()
        end_dt = pd.to_datetime(end).floor("h").to_pydatetime()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid start/end datetime.")

    q = text("""
        SELECT comptage_horaire, nuit, vacances, heure_de_pointe, pluie, neige, vent, apparent_temperature
        FROM velib_weather_processed
        WHERE date_et_heure_de_comptage >= :start
          AND date_et_heure_de_comptage <= :end
    """)
    df = pd.read_sql(q, engine, params={"start": start_dt, "end": end_dt})
    if df.empty:
        raise HTTPException(status_code=404, detail="No data in selected range")


    num = df.copy()

    # Ensure numeric/bool
    for c in ["nuit", "vacances", "heure_de_pointe", "pluie", "neige", "vent"]:
        num[c] = num[c].astype(int)

    # drop rows with missing temp/target
    num = num.dropna(subset=["comptage_horaire", "apparent_temperature"])

    # drop constant columns (nunique <= 1)
    varying_cols = [c for c in num.columns if num[c].nunique(dropna=True) > 1]
    if len(varying_cols) < 2:
        raise HTTPException(status_code=400, detail="Not enough variation in selected window to compute correlations.")
    num = num[varying_cols]

    corr = num.corr(numeric_only=True)
    corr = corr.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # --- Plot heatmap ---
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Matrix")

    buf = BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)

    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")

    # return {"columns": corr.columns.tolist(), "matrix": corr.values.tolist()}

# ---- Admin: user management ----
@app.post("/admin/users", dependencies=[Depends(require_admin)])
def create_user(req: UserCreate):
    existing = get_user(req.username)
    if existing:
        raise HTTPException(status_code=409, detail="User already exists")

    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO users(username, password_hash, role) VALUES (:u,:p,:r)"),
            {"u": req.username, "p": pwd_context.hash(req.password), "r": req.role},
        )
    return {"status": "created", "username": req.username, "role": req.role}

@app.delete("/admin/users/{username}", dependencies=[Depends(require_admin)])
def delete_user(username: str):
    with engine.begin() as conn:
        res = conn.execute(text("DELETE FROM users WHERE username=:u"), {"u": username})
    if res.rowcount == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"status": "deleted", "username": username}

# ---- Admin: trigger pipelines (job table) ----
@app.post("/admin/pipeline/trigger", dependencies=[Depends(require_admin)])
def trigger_pipeline(req: PipelineTriggerRequest):
    with engine.begin() as conn:
        job_id = conn.execute(
            text("INSERT INTO pipeline_jobs(job_type,status) VALUES (:t,'queued') RETURNING id"),
            {"t": req.job_type},
        ).scalar()
    return {"status": "queued", "job_id": job_id, "job_type": req.job_type}

@app.get("/admin/pipeline/jobs", dependencies=[Depends(require_admin)])
def list_jobs(limit: int = 50):
    df = pd.read_sql(
        text("SELECT * FROM pipeline_jobs ORDER BY created_at DESC LIMIT :l"),
        engine,
        params={"l": limit},
    )
    return df.to_dict(orient="records")

# ---- Admin: MLflow link ----
@app.get("/admin/mlflow", dependencies=[Depends(require_admin)])
def mlflow_link():
    return {"url": MLFLOW_UI_URL}


# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import pandas as pd

# from models.predict import predict_model

# app = FastAPI(title="Velib Prediction API")


# # -------- Request schema --------
# class PredictRequest(BaseModel):
#     datetime: str


# # -------- Response schema (optionnel mais propre) --------
# class PredictionResponse(BaseModel):
#     site: str
#     latitude: float
#     longitude: float
#     comptage_horaire: float

# @app.get("/health")
# def health():
#     return {"status": "ok"}

# # Format date pour tester API : "2026-02-12 15:00:00"

# @app.post("/predict")
# def predict(req: PredictRequest):

#     try:
#         datetime_pred = pd.to_datetime(req.datetime)

#         locations = predict_model(datetime_pred)

#         locations["date_et_heure_de_comptage"] = pd.to_datetime(
#             locations["date_et_heure_de_comptage"]
#         )

#         locations = locations[
#             locations["date_et_heure_de_comptage"].dt.floor("h")
#             == datetime_pred.floor("h")
#         ]

#         result = locations[
#             [
#                 "nom_du_site_de_comptage",
#                 "latitude",
#                 "longitude",
#                 "comptage_horaire"
#             ]
#         ]

#         return result.to_dict(orient="records")

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))