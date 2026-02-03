import argparse
import json
import os
import pandas as pd

from pathlib import Path
import sys
# Ensure project root is on sys.path when running as a script so imports like `utils.my_utils` work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.my_utils import preprocess_data, train_final_model, save_model


REQUIRED_COLUMNS = {
    "date_et_heure_de_comptage",
    "comptage_horaire",
    "identifiant_du_site_de_comptage",
    "coordonnées_géographiques",
    "date_d'installation_du_site_de_comptage",
    "identifiant_technique_compteur",
    "mois_annee_comptage",
    "identifiant_du_compteur",
    "nom_du_site_de_comptage",
    "nom_du_compteur",
    "lien_vers_photo_du_site_de_comptage",
    "id_photos",
    "test_lien_vers_photos_du_site_de_comptage_",
    "id_photo_1",
    "url_sites",
    "type_dimage",
}


def validate_schema(df: pd.DataFrame):
    missing = sorted(list(REQUIRED_COLUMNS - set(df.columns)))
    if missing:
        raise ValueError(
            "Input CSV is missing required columns:\n" + "\n".join(missing)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to raw training CSV")
    parser.add_argument("--model-out", default="artifacts/model.pkl", help="Where to save trained model")
    parser.add_argument("--metrics-out", default="artifacts/metrics.json", help="Where to save metrics json")
    parser.add_argument("--test-ratio", type=float, default=0.10, help="Chronological test split ratio")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)

    df_raw = pd.read_csv(args.data)
    validate_schema(df_raw)

    df_encoded, features = preprocess_data(df_raw)

    X = df_encoded[features]
    y = df_encoded["comptage_horaire"]

    target_cols = ["identifiant_du_site_de_comptage"]
    numeric_cols = [c for c in features if c not in target_cols]

    model_params = dict(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=-1,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    pipeline, metrics = train_final_model(
        X=X,
        y=y,
        model_params=model_params,
        target_cols=target_cols,
        numeric_cols=numeric_cols,
        test_size_ratio=args.test_ratio,
    )

    save_model(pipeline, args.model_out)

    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n Training done.")
    print("Saved model:", args.model_out)
    print("Saved metrics:", args.metrics_out)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()