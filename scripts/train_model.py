from models.train import *
from utils.config import database_url
from sqlalchemy import create_engine
import json


if __name__ == "__main__":

    engine = create_engine(database_url())
    processed_df, feature_names = load_training_data_from_db(engine) # This function sorts data chronologically from oldest to newest
    target_cols = ['identifiant_du_site_de_comptage']
    numeric_cols = [col for col in feature_names if col not in target_cols]
    # Préprocesseur
    preprocessor = make_preprocessor(target_cols, numeric_cols)

    # Model hyperparameters
    model_params = {
        "n_estimators" : 500, 
        "learning_rate" : 0.05,
        "max_depth" : -1,
        "random_state" : 42
    }

    pipeline, metrics = train_final_model(processed_df[feature_names], processed_df[['comptage_horaire']], model_params, target_cols, numeric_cols, test_size_ratio=0.1)
    save_model(pipeline)

    with open("metrics.json", "w") as f:
        json.dump(metrics, f)