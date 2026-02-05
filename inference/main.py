from fastapi import FastAPI
import joblib
import pandas as pd
from utils.config import MODELS

app = FastAPI()

model = joblib.load(MODELS / "model.pkl")


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: dict):

    df = pd.DataFrame([payload])

    prediction = model.predict(df)

    return {
        "prediction": float(prediction[0])
    }