import argparse
import os
import pandas as pd
from utils.config import load_model
from loader import


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to raw CSV to predict on")
    parser.add_argument("--model", default="artifacts/model.pkl", help="Path to trained model.pkl")
    parser.add_argument("--out", default="artifacts/predictions.csv", help="Output CSV path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    model = load_model(args.model)

    df_raw = pd.read_csv(args.data)
    df_encoded, features = preprocess_data(df_raw)

    X = df_encoded[features]
    preds = model.predict(X)

    out_df = df_encoded[["date_et_heure_de_comptage", "identifiant_du_site_de_comptage"]].copy()
    out_df["prediction_comptage_horaire"] = preds

    if "comptage_horaire" in df_encoded.columns:
        out_df["actual_comptage_horaire"] = df_encoded["comptage_horaire"].values
        out_df["abs_error"] = (out_df["actual_comptage_horaire"] - out_df["prediction_comptage_horaire"]).abs()

    out_df.to_csv(args.out, index=False)
    print(" Predictions saved to:", args.out)


if __name__ == "__main__":
    main()