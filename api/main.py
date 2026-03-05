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
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from io import BytesIO
from datetime import timezone
from fastapi.openapi.utils import get_openapi
from fastapi import Security



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
bearer_scheme = HTTPBearer(scheme_name="BearerAuth")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token", scheme_name="OAuth2Password")

tags_metadata = [
    {
        "name": "system",
        "description": "Health and system information endpoints.",
    },
    {
        "name": "auth",
        "description": "Authentication and identity endpoints.",
    },
    {
        "name": "client",
        "description": "Client-accessible prediction and analytics endpoints. Clients should use the filtered docs UI at `/client-docs`.",
    },
    {
        "name": "admin",
        "description": "Admin-only management and pipeline endpoints.",
    },
]

app = FastAPI(
    title="Velib Prediction API",
    description="""
API for Velib traffic prediction and analytics.

### Documentation
- Full documentation: `/docs`
- Client-only documentation: `/client-docs`

Clients should use `/client-docs` to access the filtered interface showing only client-accessible endpoints.
""",
    version="1.0.0",
    openapi_tags=tags_metadata,
)

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

class ClientSignup(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class PipelineTriggerRequest(BaseModel):
    job_type: Literal["data", "model", "forecast"]

class TrendPoint(BaseModel):
    date_et_heure_de_comptage: datetime
    comptage_horaire: float

# ---------------- DB bootstrap (tables) ----------------
def ensure_default_admin():
    with engine.begin() as conn:
        # Create a default admin if none exists (optional)
        res = conn.execute(text("SELECT COUNT(*) FROM api.users WHERE role='admin';")).scalar()
        if res == 0:
            admin_user = os.getenv("API_ADMIN_USER", "admin")
            admin_pass = os.getenv("API_ADMIN_PASS", "admin")
            conn.execute(
                text("INSERT INTO users(username, password_hash, role) VALUES (:u,:p,'admin')"),
                {"u": admin_user, "p": pwd_context.hash(admin_pass)}
            )

ensure_default_admin()

# ---------------- Auth helpers ----------------
def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

# def create_access_token(sub: str, role: str) -> str:
#     exp = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_MINUTES)
#     payload = {"sub": sub, "role": role, "exp": exp}
#     return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

def create_access_token(sub: str, role: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_MINUTES)
    payload = {"sub": sub, "role": role, "exp": exp}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

def get_user(username: str):
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT username, password_hash, role FROM users WHERE username=:u"),
            {"u": username},
        ).mappings().first()
    return row

# def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
#     token = credentials.credentials
#     try:
#         payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
#         username = payload.get("sub")
#         role = payload.get("role")
#         if not username or not role:
#             raise HTTPException(status_code=401, detail="Invalid token")
#         return {"username": username, "role": role}
#     except JWTError:
#         raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(
    bearer: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    oauth_token: Optional[str] = Security(oauth2_scheme),
):
    """
    Accept token from either:
      - HTTP Bearer (BearerAuth in Swagger)
      - OAuth2 Password flow (OAuth2Password in Swagger)
    """
    token = None
    if bearer and bearer.credentials:
        token = bearer.credentials
    elif oauth_token:
        token = oauth_token

    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

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

@app.get("/system/health", tags=["system"])
def health():
    return {"status": "ok"}

@app.get("/system/info", tags=["system"])
def system_info():
    """
    Basic metadata endpoint (useful for demos + debugging).
    """
    # Try a lightweight DB ping
    db_ok = True
    db_error = None
    try:
        with engine.begin() as conn:
            conn.execute(text("SELECT 1")).scalar()
    except Exception as e:
        db_ok = False
        db_error = str(e)

    return {
        "api": "velib-prediction-api",
        "version": os.getenv("API_VERSION", "1.0.0"),
        "server_time_utc": datetime.now(timezone.utc).isoformat(),
        "mlflow_ui_url": MLFLOW_UI_URL,
        "db_ok": db_ok,
        "db_error": db_error,
        "db_dsn": DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else DATABASE_URL,  # avoid leaking creds
        "forecast_horizon_hours": int(os.getenv("FORECAST_HOURS", "48")),
    }

# ---- Auth ----
# def custom_openapi():
#     if app.openapi_schema:
#         return app.openapi_schema

#     openapi_schema = get_openapi(
#         title=app.title,
#         version=app.version,
#         description=app.description,
#         routes=app.routes,
#     )

#     # Ensure both schemes exist
#     openapi_schema.setdefault("components", {}).setdefault("securitySchemes", {})
#     openapi_schema["components"]["securitySchemes"]["BearerAuth"] = {
#         "type": "http",
#         "scheme": "bearer",
#         "bearerFormat": "JWT",
#     }
#     openapi_schema["components"]["securitySchemes"]["OAuth2Password"] = {
#         "type": "oauth2",
#         "flows": {
#             "password": {
#                 "tokenUrl": "/auth/token",
#                 "scopes": {},
#             }
#         },
#     }

#     # Apply globally: user can choose either in the Authorize dialog
#     openapi_schema["security"] = [
#         {"BearerAuth": []},
#         {"OAuth2Password": []},
#     ]

#     app.openapi_schema = openapi_schema
#     return app.openapi_schema

# app.openapi = custom_openapi

# def custom_openapi():
#     if app.openapi_schema:
#         return app.openapi_schema

#     schema = get_openapi(
#         title=app.title,
#         version=app.version,
#         description=app.description,
#         routes=app.routes,
#     )

#     # Allow either Bearer token OR OAuth2 password flow
#     schema["security"] = [
#         {"BearerAuth": []},
#         {"OAuth2Password": []},
#     ]

#     app.openapi_schema = schema
#     return app.openapi_schema

# app.openapi = custom_openapi


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=tags_metadata,
    )

    # Ensure components + securitySchemes exist
    schema.setdefault("components", {}).setdefault("securitySchemes", {})

    # IMPORTANT: Do NOT add another Bearer scheme here (FastAPI already generates BearerAuth
    # because you use HTTPBearer(scheme_name="BearerAuth") in dependencies).
    # Just add OAuth2Password so Swagger can show username/password.
    schema["components"]["securitySchemes"]["OAuth2Password"] = {
        "type": "oauth2",
        "flows": {
            "password": {
                "tokenUrl": "/auth/token",
                "scopes": {}
            }
        },
    }

    # Allow either method globally in Swagger
    schema["security"] = [
        {"OAuth2Password": []},
        {"BearerAuth": []}
    ]

    app.openapi_schema = schema
    return app.openapi_schema

app.openapi = custom_openapi

@app.post("/auth/token", response_model=Token, tags=["auth"])
def login(form: OAuth2PasswordRequestForm = Depends()):
    user = get_user(form.username)
    if not user or not verify_password(form.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = create_access_token(user["username"], user["role"])
    return Token(access_token=token)

@app.get("/auth/me", tags=["auth"])
def who_am_i(user=Depends(get_current_user)):
    return user

# ---- Client: Predict (DB forecast) ----

@app.post("/client/predict", response_model=List[PredictionRow], tags=["client"])
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
@app.get("/client/trends", tags=["client"])
def trends(site_id: int, start: str, end: str, user=Depends(get_current_user)):

    try:
        start_dt = pd.to_datetime(start).floor("h").to_pydatetime()
        end_dt = pd.to_datetime(end).floor("h").to_pydatetime()
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

# ---- Client: Correlations (returns matrix as JSON) ----
@app.get("/client/correlations", tags=["client"])
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
@app.post("/admin/users", tags=["admin"], dependencies=[Depends(require_admin)])
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

@app.delete("/admin/users/{username}", tags=["admin"], dependencies=[Depends(require_admin)])
def delete_user(username: str):
    with engine.begin() as conn:
        res = conn.execute(text("DELETE FROM users WHERE username=:u"), {"u": username})
    if res.rowcount == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"status": "deleted", "username": username}

# ---- Admin: trigger pipelines (job table) ----
@app.post("/admin/pipeline/trigger", tags=["admin"], dependencies=[Depends(require_admin)])
def trigger_pipeline(req: PipelineTriggerRequest):
    with engine.begin() as conn:
        job_id = conn.execute(
            text("INSERT INTO pipeline_jobs(job_type,status) VALUES (:t,'queued') RETURNING id"),
            {"t": req.job_type},
        ).scalar()
    return {"status": "queued", "job_id": job_id, "job_type": req.job_type}

@app.get("/admin/pipeline/jobs", tags=["admin"], dependencies=[Depends(require_admin)])
def list_jobs(limit: int = 50):
    df = pd.read_sql(
        text("SELECT * FROM pipeline_jobs ORDER BY created_at DESC LIMIT :l"),
        engine,
        params={"l": limit},
    )
    return df.to_dict(orient="records")

# ---- Admin: MLflow link ----
@app.get("/admin/mlflow", tags=["admin"], dependencies=[Depends(require_admin)])
def mlflow_link():
    return {"url": MLFLOW_UI_URL}














# --- add near other schemas in main.py ---
class ForecastWindowResponse(BaseModel):
    start: str  # ISO string
    end: str    # ISO string

class ForecastRangeRow(BaseModel):
    date_et_heure_de_comptage: datetime
    identifiant_du_site_de_comptage: int
    nom_du_site_de_comptage: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    comptage_horaire: float

class HistoricalRefRow(BaseModel):
    label: str
    comptage_horaire: float


# --- add near other client routes in main.py ---
@app.get("/client/forecast-window", response_model=ForecastWindowResponse, tags=["client"])
def forecast_window(user=Depends(get_current_user)):
    """
    Window available for predictions:
    (max velib_raw dt floored to hour) + 1 hour  -> + 48 hours
    Mirrors Streamlit forecast_window() logic.
    """
    q = text("SELECT MAX(date_et_heure_de_comptage) AS dt FROM velib_raw;")
    df = pd.read_sql(q, engine)
    last_raw = df.loc[0, "dt"]

    if last_raw is None:
        raise HTTPException(status_code=404, detail="velib_raw is empty (no max datetime).")

    last_raw = pd.to_datetime(last_raw).floor("h")
    start_dt = (last_raw + timedelta(hours=1)).to_pydatetime()
    end_dt   = (last_raw + timedelta(hours=48)).to_pydatetime()

    return {"start": start_dt.isoformat(), "end": end_dt.isoformat()}


@app.get("/client/forecast-range", response_model=List[ForecastRangeRow], tags=["client"])
def forecast_range(start: str, end: str, user=Depends(get_current_user)):
    """
    Returns all forecast rows between start and end (inclusive),
    from velib_forecast_geo (view).
    """
    try:
        start_dt = pd.to_datetime(start, errors="raise").floor("h").to_pydatetime()
        end_dt   = pd.to_datetime(end, errors="raise").floor("h").to_pydatetime()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid start/end datetime. Use ISO or 'YYYY-MM-DD HH:00:00'.")

    q = text("""
        SELECT
            date_et_heure_de_comptage,
            identifiant_du_site_de_comptage,
            nom_du_site_de_comptage,
            latitude,
            longitude,
            comptage_horaire
        FROM velib_forecast_geo
        WHERE date_et_heure_de_comptage BETWEEN :start AND :end
        ORDER BY nom_du_site_de_comptage, date_et_heure_de_comptage
    """)
    df = pd.read_sql(q, engine, params={"start": start_dt, "end": end_dt})
    if df.empty:
        return []
    df["date_et_heure_de_comptage"] = pd.to_datetime(df["date_et_heure_de_comptage"])
    return df.to_dict(orient="records")


@app.get("/client/historical-reference", response_model=List[HistoricalRefRow], tags=["client"])
def historical_reference(site_name: str, datetime: str, user=Depends(get_current_user)):
    """
    For a given site name + target hour, returns:
      - Avg of last 4 weeks same ISODOW + hour
    Mirrors Streamlit load_historical_reference().
    """
    try:
        target_dt = pd.to_datetime(datetime, errors="raise").floor("h")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid datetime. Use ISO or 'YYYY-MM-DD HH:00:00'.")

    target_hour = int(target_dt.hour)
    target_isodow = int(target_dt.dayofweek) + 1  # pandas: 0..6, ISODOW: 1..7

    q = text("""
        SELECT AVG(comptage_horaire) AS val
        FROM velib_raw
        WHERE nom_du_site_de_comptage = :site
          AND EXTRACT(ISODOW FROM date_et_heure_de_comptage) = :dow
          AND EXTRACT(HOUR   FROM date_et_heure_de_comptage) = :hour
          AND date_et_heure_de_comptage BETWEEN :ws AND :we;
    """)

    params = {
        "site": site_name,
        "dow": target_isodow,
        "hour": target_hour,
        "ws": (target_dt - pd.Timedelta(weeks=4)).to_pydatetime(),
        "we": (target_dt - pd.Timedelta(hours=1)).to_pydatetime(),
    }

    df = pd.read_sql(q, engine, params=params)
    val = df.loc[0, "val"] if not df.empty else None

    if val is None:
        return []

    return [{
        "label": "Moy. 4 semaines (même jour/heure)",
        "comptage_horaire": round(float(val), 1),
    }]

# @app.post("/client/signup", tags=["client"])
@app.post("/auth/signup", tags=["auth"])
def client_signup(req: ClientSignup):
    existing = get_user(req.username)
    if existing:
        raise HTTPException(status_code=409, detail="User already exists")

    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO users(username, password_hash, role) VALUES (:u,:p,'client')"),
            {"u": req.username, "p": pwd_context.hash(req.password)},
        )
    return {"status": "created", "username": req.username, "role": "client"}

from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import JSONResponse

def client_openapi():
    base = app.openapi()  # full schema
    client_paths = {}
    for path, methods in base["paths"].items():
        # keep only /client and /auth routes (plus /health maybe)
        if path.startswith("/client") or path.startswith("/auth") or path in ["/health", "/system/info"]:
            client_paths[path] = methods

    filtered = dict(base)
    filtered["paths"] = client_paths
    return filtered

@app.get("/client-openapi.json", include_in_schema=False)
def client_openapi_json():
    return JSONResponse(client_openapi())

@app.get("/client-docs", include_in_schema=False)
def client_docs():
    return get_swagger_ui_html(openapi_url="/client-openapi.json", title="Client API Docs")