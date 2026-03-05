import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from category_encoders.target_encoder import TargetEncoder
from lightgbm import LGBMRegressor
import pickle
from sklearn.pipeline import Pipeline
from sqlalchemy import text
from sqlalchemy.engine import Engine
from pandas.api.types import (
    is_bool_dtype,
    is_numeric_dtype,
    is_datetime64_any_dtype,
)

def infer_feature_columns(
    df: pd.DataFrame,
    feature_names: list[str],
    target_encode_cols: list[str],
    exclude_cols: list[str] | None = None,
):
    """
    Returns (numeric_cols, passthrough_cols) while excluding:
    - target-encoded columns (handled separately)
    - datetime columns (unless you explicitly engineer them)
    - any explicitly excluded columns

    Booleans are included in numeric_cols (and should be cast to int).
    """
    exclude_cols = set(exclude_cols or [])
    target_encode_cols_set = set(target_encode_cols)

    numeric_cols: list[str] = []
    passthrough_cols: list[str] = []  # optional, for already-ready numeric or other handling

    for col in feature_names:
        if col in exclude_cols or col in target_encode_cols_set:
            continue

        s = df[col]

        # Exclude datetimes from scaling by default
        if is_datetime64_any_dtype(s):
            continue

        # Booleans → numeric
        if is_bool_dtype(s):
            numeric_cols.append(col)
            continue

        # Numeric dtypes → numeric
        if is_numeric_dtype(s):
            numeric_cols.append(col)
            continue

        # Everything else (object/category) is NOT safe to scale.
        # You can either drop it or add another transformer later.
        # For now we drop it.
        # passthrough_cols.append(col)

    return numeric_cols, passthrough_cols


def load_training_data_from_db(engine: Engine, table: str = "velib_weather_processed"):
    # Only select columns you actually train on.
    # You can also do column introspection, but explicit is safer.
    df = pd.read_sql_query(text(f"SELECT * FROM {table} ORDER BY date_et_heure_de_comptage ASC"), engine)

    target = "comptage_horaire"
    drop_cols = []  # e.g. drop IDs you don't want besides identifiant_du_site_de_comptage
    feature_names = [c for c in df.columns if c not in drop_cols + [target]]

    return df, feature_names


# Target encoding and standard scaling for numeric features in a single preprocessor to avoid data leakage and ensure consistent transformations between train and test sets.
# This cannot be done in the preprocessing step because target encoding needs to be fit on the training data and then applied to the test data without refitting.
def make_preprocessor(target_encode_cols, numeric_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ("target_enc", TargetEncoder(cols=target_encode_cols), target_encode_cols),
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

    # Ensure y is 1D
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError(f"y has {y.shape[1]} columns; expected 1.")
        y = y.iloc[:, 0]
    else:
        y = pd.Series(y)

    # Split chronologically
    test_size = int(len(X) * test_size_ratio)
    X_trainval, X_test = X.iloc[:-test_size], X.iloc[-test_size:] # Assumes data is chronologically sorted from oldest to newest
    y_trainval, y_test = y.iloc[:-test_size], y.iloc[-test_size:] # Assumes data is chronologically sorted from oldest to newest

    # Build preprocessor and fit on *all* trainval data
    preprocessor = make_preprocessor(target_cols, numeric_cols)

    # # Train final model
    final_model = LGBMRegressor(**model_params)

    final_model.target_cols_ = target_cols
    final_model.numeric_cols_ = numeric_cols

    print(isinstance(X_trainval,pd.DataFrame))
    print(isinstance(y_trainval,pd.DataFrame))

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', final_model)
    ])
    pipeline.fit(X_trainval, y_trainval)

    preds = pipeline.predict(X_test)
    mae_scores = mean_absolute_error(y_test, preds)
    rmse_scores = np.sqrt(mean_squared_error(y_test, preds))
    r2_scores = r2_score(y_test, preds)
  
    metrics = {"MAE": mae_scores, "RMSE": rmse_scores, "R2": r2_scores}

    return pipeline, metrics


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