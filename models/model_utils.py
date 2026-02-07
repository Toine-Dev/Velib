import pandas as pd
import numpy as np


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from category_encoders.target_encoder import TargetEncoder
from lightgbm import LGBMRegressor
import pickle
from sklearn.pipeline import Pipeline

def make_preprocessor(target_encode_cols, numeric_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ("target_enc", TargetEncoder(cols=target_encode_cols), target_encode_cols),
            # ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), onehot_cols),
            ("scale", StandardScaler(), numeric_cols)
        ],
        remainder="drop"
    )
    return preprocessor

def train_final_model(X, y, model_params, target_cols, numeric_cols,
                      test_size_ratio=0.1):
    """
    Trains the final model on train+val and evaluates on unseen test set.
    """
    # Split chronologically
    test_size = int(len(X) * test_size_ratio)
    X_trainval, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_trainval, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

    # Build preprocessor and fit on *all* trainval data
    preprocessor = make_preprocessor(target_cols, numeric_cols)
    # X_trainval_prep = preprocessor.fit_transform(X_trainval, y_trainval)
    # X_test_prep     = preprocessor.transform(X_test)

    # # Train final model
    final_model = LGBMRegressor(**model_params)
    # final_model.fit(X_trainval_prep, y_trainval)

    # Evaluate on test
    # test_preds = final_model.predict(X_test_prep)
    # test_mae = mean_absolute_error(y_test, test_preds)
    # test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    # test_r2 = r2_score(y_test, test_preds)

    final_model.target_cols_ = target_cols
    # final_model.onehot_cols_ = onehot_cols
    final_model.numeric_cols_ = numeric_cols

    # final_model.test_metrics_ = {
    #     "MAE": test_mae,
    #     "RMSE": test_rmse,
    #     "R2": test_r2
    # }
    #final_model.preprocessor_ = preprocessor  # So you can save/restore the whole pipeline -> return preprocessor puis ajouter en attribut du modèle
    print(isinstance(X_trainval,pd.DataFrame))
    print(isinstance(y_trainval,pd.DataFrame))

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', final_model)
    ])
    pipeline.fit(X_trainval, y_trainval)

    # test_preds = final_model.predict(X_test)
    # test_mae = mean_absolute_error(y_test, test_preds)
    # test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    # test_r2 = r2_score(y_test, test_preds)

    preds = pipeline.predict(X_test)
    mae_scores = mean_absolute_error(y_test, preds)
    rmse_scores = np.sqrt(mean_squared_error(y_test, preds))
    r2_scores = r2_score(y_test, preds)
    
    # pipeline.test_metrics_ = {
    #     "MAE": np.mean(mae_scores),
    #     "RMSE": np.mean(rmse_scores),
    #     "R2": np.mean(r2_scores)
    # }
    
    # print("\n=== Final Test Set Performance ===")
    # print(f"MAE={mae_scores:.4f}, RMSE={rmse_scores:.4f}, R2={r2_scores:.4f}")
    metrics = {"MAE": mae_scores, "RMSE": rmse_scores, "R2": r2_scores}

    return pipeline, metrics
    # return final_model, pipeline, pipeline.test_metrics_

def save_model(model, filename="model.pkl"):
    """
    Sauvegarde le pipeline complet (préprocesseur + modèle) dans un fichier pickle.
    """
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    print(f"Modèle sauvegardé dans {filename}")

def load_model(filename="model.pkl"):
    """
    Charge le pipeline complet depuis un fichier pickle.
    """
    with open(filename, "rb") as f:
        model = pickle.load(f)
    print(f"Modèle chargé depuis {filename}")
    return model