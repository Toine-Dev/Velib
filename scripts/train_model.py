from models.train import *
from utils.config import database_url
from sqlalchemy import create_engine
import json
import os
import mlflow
import mlflow.sklearn


if __name__ == "__main__":

    engine = create_engine(database_url())
    processed_df, feature_names = load_training_data_from_db(engine) # This function sorts data chronologically from oldest to newest
    target_cols = ["identifiant_du_site_de_comptage"]

    # Exclude timestamp column (or any other columns you never want as raw features)
    exclude_cols = ["date_et_heure_de_comptage"]

    numeric_cols, _ = infer_feature_columns(
        processed_df,
        feature_names=feature_names,
        target_encode_cols=target_cols,
        exclude_cols=exclude_cols,
    )

    # Cast booleans to int (recommended)
    for c in numeric_cols:
        if processed_df[c].dtype == bool:
            processed_df[c] = processed_df[c].astype("int8")

    # target_cols = ['identifiant_du_site_de_comptage']
    # numeric_cols = [col for col in feature_names if col not in target_cols]
    # Préprocesseur
    preprocessor = make_preprocessor(target_cols, numeric_cols)

    # Model hyperparameters
    model_params = {
        "n_estimators" : 500, 
        "learning_rate" : 0.05,
        "max_depth" : -1,
        "random_state" : 42
    }


# Optional: make an experiment name configurable
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "velib_forecast")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT)

with mlflow.start_run() as run:
    pipeline, metrics = train_final_model(processed_df[feature_names], processed_df['comptage_horaire'], model_params, target_cols, numeric_cols, test_size_ratio=0.1)
    # save_model(pipeline)

    # Log params/metrics
    mlflow.log_params(model_params)
    mlflow.log_metrics(metrics)

    # Log the full sklearn pipeline as the model artifact
    mlflow.sklearn.log_model(pipeline, artifact_path="model")

    print("MLflow run_id:", run.info.run_id)
    
    from models.model_pointer import set_model_pointer   # adjust import path to your project
    # or: from src.models.model_pointer import set_model_pointer

    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/model"

    set_model_pointer(
        engine=engine,                  # your SQLAlchemy engine
        name="production",              # convention
        run_id=run_id,
        model_uri=model_uri,
        metrics=metrics,
    )

    print("✅ Production model pointer set to:", model_uri)

    print(f"R2 Score: {metrics['R2']}")

    # with open("metrics.json", "w") as f:
    #     json.dump(metrics, f)






