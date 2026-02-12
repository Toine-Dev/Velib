from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

from models.predict import predict_model

app = FastAPI(title="Velib Prediction API")


# -------- Request schema --------
class PredictRequest(BaseModel):
    datetime: str


# -------- Response schema (optionnel mais propre) --------
class PredictionResponse(BaseModel):
    site: str
    latitude: float
    longitude: float
    comptage_horaire: float

@app.get("/health")
def health():
    return {"status": "ok"}

# Format date pour tester API : "2026-02-12 15:00:00"

@app.post("/predict")
def predict(req: PredictRequest):

    try:
        datetime_pred = pd.to_datetime(req.datetime)

        locations = predict_model(datetime_pred)

        locations["date_et_heure_de_comptage"] = pd.to_datetime(
            locations["date_et_heure_de_comptage"]
        )

        locations = locations[
            locations["date_et_heure_de_comptage"].dt.floor("h")
            == datetime_pred.floor("h")
        ]

        result = locations[
            [
                "nom_du_site_de_comptage",
                "latitude",
                "longitude",
                "comptage_horaire"
            ]
        ]

        return result.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))