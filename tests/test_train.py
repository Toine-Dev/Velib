"""
Tests unitaires — models/train.py & model_utils
Couvre : infer_feature_columns, make_preprocessor, train_final_model
"""
import pytest
import pandas as pd
import numpy as np
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from models.train import (
    infer_feature_columns,
    make_preprocessor,
    train_final_model,
)


# ===========================================================================
# Fixtures partagées
# ===========================================================================
@pytest.fixture
def small_df():
    """DataFrame minimal avec les types attendus par le pipeline."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "identifiant_du_site_de_comptage": np.random.choice([101, 102, 103], size=n),
        "heure_sin": np.sin(np.linspace(0, 2 * np.pi, n)),
        "heure_cos": np.cos(np.linspace(0, 2 * np.pi, n)),
        "jour_sin": np.random.uniform(-1, 1, n),
        "jour_cos": np.random.uniform(-1, 1, n),
        "mois_sin": np.random.uniform(-1, 1, n),
        "mois_cos": np.random.uniform(-1, 1, n),
        "saison_sin": np.random.uniform(-1, 1, n),
        "saison_cos": np.random.uniform(-1, 1, n),
        "vacances": np.random.choice([True, False], size=n),
        "heure_de_pointe": np.random.choice([True, False], size=n),
        "nuit": np.random.choice([True, False], size=n),
        "pluie": np.random.choice([True, False], size=n),
        "neige": np.random.choice([False], size=n),
        "vent": np.random.choice([True, False], size=n),
        "apparent_temperature": np.random.uniform(5, 30, n),
        "lag_1": np.random.randint(0, 100, n).astype(float),
        "lag_24": np.random.randint(0, 100, n).astype(float),
        "rolling_mean_24": np.random.uniform(0, 100, n),
        "date_et_heure_de_comptage": pd.date_range("2025-01-01", periods=n, freq="h"),
        "comptage_horaire": np.random.randint(0, 200, n).astype(float),
    })
    return df


# ===========================================================================
# 1. infer_feature_columns
# ===========================================================================
class TestInferFeatureColumns:
    def test_returns_two_lists(self, small_df):
        feature_names = [c for c in small_df.columns if c != "comptage_horaire"]
        num, passthrough = infer_feature_columns(
            small_df,
            feature_names=feature_names,
            target_encode_cols=["identifiant_du_site_de_comptage"],
            exclude_cols=["date_et_heure_de_comptage"],
        )
        assert isinstance(num, list)
        assert isinstance(passthrough, list)

    def test_target_encode_cols_not_in_numeric(self, small_df):
        feature_names = [c for c in small_df.columns if c != "comptage_horaire"]
        num, _ = infer_feature_columns(
            small_df,
            feature_names=feature_names,
            target_encode_cols=["identifiant_du_site_de_comptage"],
            exclude_cols=["date_et_heure_de_comptage"],
        )
        assert "identifiant_du_site_de_comptage" not in num

    def test_datetime_excluded_from_numeric(self, small_df):
        feature_names = [c for c in small_df.columns if c != "comptage_horaire"]
        num, _ = infer_feature_columns(
            small_df,
            feature_names=feature_names,
            target_encode_cols=["identifiant_du_site_de_comptage"],
            exclude_cols=["date_et_heure_de_comptage"],
        )
        assert "date_et_heure_de_comptage" not in num

    def test_bool_columns_included_as_numeric(self, small_df):
        feature_names = [c for c in small_df.columns if c != "comptage_horaire"]
        num, _ = infer_feature_columns(
            small_df,
            feature_names=feature_names,
            target_encode_cols=["identifiant_du_site_de_comptage"],
            exclude_cols=["date_et_heure_de_comptage"],
        )
        # vacances, heure_de_pointe, nuit, pluie, neige, vent are bool
        assert "vacances" in num

    def test_explicit_exclude_cols_absent(self, small_df):
        feature_names = [c for c in small_df.columns if c != "comptage_horaire"]
        num, _ = infer_feature_columns(
            small_df,
            feature_names=feature_names,
            target_encode_cols=["identifiant_du_site_de_comptage"],
            exclude_cols=["date_et_heure_de_comptage", "lag_1"],
        )
        assert "lag_1" not in num


# ===========================================================================
# 2. make_preprocessor
# ===========================================================================
class TestMakePreprocessor:
    def test_returns_column_transformer(self, small_df):
        from sklearn.compose import ColumnTransformer
        target_cols = ["identifiant_du_site_de_comptage"]
        feature_names = [c for c in small_df.columns if c != "comptage_horaire"]
        num, _ = infer_feature_columns(
            small_df, feature_names, target_cols, ["date_et_heure_de_comptage"]
        )
        prep = make_preprocessor(target_cols, num)
        assert isinstance(prep, ColumnTransformer)

    def test_preprocessor_fits_without_error(self, small_df):
        target_cols = ["identifiant_du_site_de_comptage"]
        feature_names = [c for c in small_df.columns
                         if c not in ["comptage_horaire", "date_et_heure_de_comptage"]]
        num, _ = infer_feature_columns(
            small_df, feature_names, target_cols, ["date_et_heure_de_comptage"]
        )
        # Cast booleans
        for c in num:
            if small_df[c].dtype == bool:
                small_df[c] = small_df[c].astype("int8")

        prep = make_preprocessor(target_cols, num)
        X = small_df[feature_names]
        y = small_df["comptage_horaire"]
        prep.fit(X, y)  # should not raise


# ===========================================================================
# 3. train_final_model
# ===========================================================================
class TestTrainFinalModel:
    def _prepare(self, small_df):
        target_cols = ["identifiant_du_site_de_comptage"]
        feature_names = [c for c in small_df.columns
                         if c not in ["comptage_horaire", "date_et_heure_de_comptage"]]
        num, _ = infer_feature_columns(
            small_df, feature_names, target_cols, ["date_et_heure_de_comptage"]
        )
        for c in num:
            if small_df[c].dtype == bool:
                small_df[c] = small_df[c].astype("int8")
        X = small_df[feature_names]
        y = small_df["comptage_horaire"]
        return X, y, target_cols, num

    def test_returns_pipeline_and_metrics(self, small_df):
        X, y, target_cols, num = self._prepare(small_df)
        model_params = {"n_estimators": 10, "learning_rate": 0.1,
                        "max_depth": 3, "random_state": 42}
        pipeline, metrics = train_final_model(
            X, y, model_params, target_cols, num, test_size_ratio=0.2
        )
        assert pipeline is not None
        assert isinstance(metrics, dict)

    def test_metrics_contain_mae_rmse_r2(self, small_df):
        X, y, target_cols, num = self._prepare(small_df)
        model_params = {"n_estimators": 10, "learning_rate": 0.1,
                        "max_depth": 3, "random_state": 42}
        _, metrics = train_final_model(
            X, y, model_params, target_cols, num, test_size_ratio=0.2
        )
        assert "MAE" in metrics
        assert "RMSE" in metrics
        assert "R2" in metrics

    def test_metrics_are_finite(self, small_df):
        X, y, target_cols, num = self._prepare(small_df)
        model_params = {"n_estimators": 10, "learning_rate": 0.1,
                        "max_depth": 3, "random_state": 42}
        _, metrics = train_final_model(
            X, y, model_params, target_cols, num, test_size_ratio=0.2
        )
        for k, v in metrics.items():
            assert np.isfinite(v), f"Metric {k} is not finite: {v}"

    def test_mae_is_non_negative(self, small_df):
        X, y, target_cols, num = self._prepare(small_df)
        model_params = {"n_estimators": 10, "learning_rate": 0.1,
                        "max_depth": 3, "random_state": 42}
        _, metrics = train_final_model(
            X, y, model_params, target_cols, num, test_size_ratio=0.2
        )
        assert metrics["MAE"] >= 0

    def test_rmse_is_non_negative(self, small_df):
        X, y, target_cols, num = self._prepare(small_df)
        model_params = {"n_estimators": 10, "learning_rate": 0.1,
                        "max_depth": 3, "random_state": 42}
        _, metrics = train_final_model(
            X, y, model_params, target_cols, num, test_size_ratio=0.2
        )
        assert metrics["RMSE"] >= 0

    def test_pipeline_can_predict(self, small_df):
        X, y, target_cols, num = self._prepare(small_df)
        model_params = {"n_estimators": 10, "learning_rate": 0.1,
                        "max_depth": 3, "random_state": 42}
        pipeline, _ = train_final_model(
            X, y, model_params, target_cols, num, test_size_ratio=0.2
        )
        preds = pipeline.predict(X.iloc[:5])
        assert len(preds) == 5
        assert all(np.isfinite(p) for p in preds)

    def test_dataframe_y_accepted(self, small_df):
        """train_final_model doit accepter y sous forme DataFrame 1-colonne."""
        X, y, target_cols, num = self._prepare(small_df)
        y_df = y.to_frame()
        model_params = {"n_estimators": 10, "learning_rate": 0.1,
                        "max_depth": 3, "random_state": 42}
        pipeline, metrics = train_final_model(
            X, y_df, model_params, target_cols, num, test_size_ratio=0.2
        )
        assert pipeline is not None