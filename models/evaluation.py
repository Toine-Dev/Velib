import argparse
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data.loader import load_raw_velib_data, load_raw_weather_data, load_processed_data
from utils.config import load_model
from utils.config import DATA, MODELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="comptage_velo_donnees_compteurs.csv", help="Path to raw CSV to evaluate on (default: comptage_velo_donnees_compteurs.csv). Run `python scripts/update_data.py` to fetch data if missing.")
    parser.add_argument("--model", default="artifacts/model.pkl", help="Path to trained model.pkl")
    parser.add_argument("--out", default="artifacts/eval_metrics.json", help="Where to save eval metrics")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file '{args.model}' not found. Train a model first with:\n  python models/train.py --data <raw_csv> --model-out {args.model}\nOr pass a different path via --model."
        )

    model = load_model(args.model)

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file '{args.data}' not found. Run `python scripts/update_data.py` to fetch raw data, or pass --data with the correct CSV path."
        )

    # Try reading CSV; if the file uses semicolons (common for exported CSV), try that as a fallback
    try:
        df_raw = pd.read_csv(args.data)
    except Exception as e:
        try:
            df_raw = pd.read_csv(args.data, sep=";")
        except Exception as e2:
            raise RuntimeError(
                f"Failed to read data file '{args.data}': {e}; also tried sep=';' -> {e2}"
            )

    raw_df_velib = load_raw_velib_data()
    raw_df_weather = load_raw_weather_data()
    df_encoded, features = load_processed_data(raw_df_velib, raw_df_weather)

    if "comptage_horaire" not in df_encoded.columns:
        raise ValueError(
            f"Cannot evaluate: 'comptage_horaire' not found after preprocessing. Columns present after preprocess: {list(df_encoded.columns)}"
        )

    X = df_encoded[features]
    y = df_encoded["comptage_horaire"]

    preds = model.predict(X)

    metrics = {
        "MAE": float(mean_absolute_error(y, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(y, preds))),
        "R2": float(r2_score(y, preds)),
        "n_rows": int(len(df_encoded)),
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(" Evaluation done.")
    print("Saved:", args.out)
    print(metrics)


if __name__ == "__main__":
    main()