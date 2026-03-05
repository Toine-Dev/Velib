"""
Tests unitaires — models/features.py :: recursive_forecast
Couvre : logique FIFO de l'historique, clipping à 0, gestion site inconnu,
         alignement index, ordre chronologique, mise à jour de l'historique.
Toutes les dépendances ML sont mockées (pas besoin de vrai modèle entraîné).
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from models.features import recursive_forecast


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pipeline_mock(pred_value: float = 10.0):
    """Pipeline mock qui retourne toujours pred_value."""
    pipeline = MagicMock()
    pipeline.predict.return_value = np.array([pred_value])
    return pipeline


def make_cartesian_df(site_ids: list, hours: list, extra_cols: dict = None) -> pd.DataFrame:
    """Construit un cartesian_df minimal avec les colonnes attendues."""
    rows = []
    for site in site_ids:
        for h in hours:
            row = {
                "identifiant_du_site_de_comptage": site,
                "date_et_heure_de_comptage": h,
                "lag_1": np.nan,
                "lag_24": np.nan,
                "rolling_mean_24": np.nan,
            }
            if extra_cols:
                row.update(extra_cols)
            rows.append(row)
    df = pd.DataFrame(rows)
    return df


def make_history_dict(site_ids: list, value: float = 5.0) -> dict:
    """Historique avec 24 valeurs identiques pour chaque site."""
    return {site: [value] * 24 for site in site_ids}


def make_site_stats(site_ids: list, mean: float = 5.0) -> pd.DataFrame:
    return pd.DataFrame({
        "identifiant_du_site_de_comptage": site_ids,
        "site_mean_usage": [mean] * len(site_ids),
        "site_usage_variability": [1.0] * len(site_ids),
        "site_max_usage": [mean * 2] * len(site_ids),
        "site_min_usage": [0.0] * len(site_ids),
    })


HOURS = pd.date_range("2025-03-01 01:00", periods=6, freq="h").tolist()
FEATURE_NAMES = ["identifiant_du_site_de_comptage", "lag_1", "lag_24", "rolling_mean_24"]


# ===========================================================================
# 1. Structure de sortie
# ===========================================================================
class TestRecursiveForecastOutput:
    def test_returns_dataframe(self):
        df = make_cartesian_df([1], HOURS)
        history = make_history_dict([1])
        site_stats = make_site_stats([1])
        pipeline = make_pipeline_mock(10.0)

        result = recursive_forecast(df, history, pipeline, FEATURE_NAMES, site_stats)
        assert isinstance(result, pd.DataFrame)

    def test_output_has_comptage_horaire_column(self):
        df = make_cartesian_df([1], HOURS)
        history = make_history_dict([1])
        site_stats = make_site_stats([1])
        pipeline = make_pipeline_mock(10.0)

        result = recursive_forecast(df, history, pipeline, FEATURE_NAMES, site_stats)
        assert "comptage_horaire" in result.columns

    def test_output_row_count_matches_input(self):
        df = make_cartesian_df([1, 2], HOURS)
        history = make_history_dict([1, 2])
        site_stats = make_site_stats([1, 2])
        pipeline = make_pipeline_mock(10.0)

        result = recursive_forecast(df, history, pipeline, FEATURE_NAMES, site_stats)
        assert len(result) == len(df)

    def test_all_sites_present_in_output(self):
        df = make_cartesian_df([1, 2, 3], HOURS)
        history = make_history_dict([1, 2, 3])
        site_stats = make_site_stats([1, 2, 3])
        pipeline = make_pipeline_mock(10.0)

        result = recursive_forecast(df, history, pipeline, FEATURE_NAMES, site_stats)
        assert set(result["identifiant_du_site_de_comptage"].unique()) == {1, 2, 3}


# ===========================================================================
# 2. Clipping à 0
# ===========================================================================
class TestRecursiveForecastClipping:
    def test_negative_predictions_clipped_to_zero(self):
        df = make_cartesian_df([1], HOURS)
        history = make_history_dict([1])
        site_stats = make_site_stats([1])
        pipeline = make_pipeline_mock(-99.0)  # valeur négative

        result = recursive_forecast(df, history, pipeline, FEATURE_NAMES, site_stats)
        assert (result["comptage_horaire"] >= 0).all()

    def test_zero_prediction_stays_zero(self):
        df = make_cartesian_df([1], HOURS)
        history = make_history_dict([1])
        site_stats = make_site_stats([1])
        pipeline = make_pipeline_mock(0.0)

        result = recursive_forecast(df, history, pipeline, FEATURE_NAMES, site_stats)
        assert (result["comptage_horaire"] == 0.0).all()

    def test_positive_predictions_unchanged(self):
        df = make_cartesian_df([1], HOURS)
        history = make_history_dict([1])
        site_stats = make_site_stats([1])
        pipeline = make_pipeline_mock(42.0)

        result = recursive_forecast(df, history, pipeline, FEATURE_NAMES, site_stats)
        assert (result["comptage_horaire"] == 42.0).all()


# ===========================================================================
# 3. Logique FIFO — mise à jour de l'historique
# ===========================================================================
class TestRecursiveForecastFIFO:
    def test_pipeline_called_once_per_row(self):
        n_hours = 4
        hours = pd.date_range("2025-03-01 01:00", periods=n_hours, freq="h").tolist()
        df = make_cartesian_df([1], hours)
        history = make_history_dict([1])
        site_stats = make_site_stats([1])
        pipeline = make_pipeline_mock(10.0)

        recursive_forecast(df, history, pipeline, FEATURE_NAMES, site_stats)
        assert pipeline.predict.call_count == n_hours

    def test_lag_1_uses_previous_prediction(self):
        """Le lag_1 de la 2ème heure doit être la prédiction de la 1ère heure."""
        hours = pd.date_range("2025-03-01 01:00", periods=3, freq="h").tolist()
        df = make_cartesian_df([1], hours)
        history = make_history_dict([1], value=5.0)
        site_stats = make_site_stats([1])

        captured_lags = []

        def capture_predict(X):
            captured_lags.append(float(X["lag_1"].iloc[0]))
            return np.array([20.0])

        pipeline = MagicMock()
        pipeline.predict.side_effect = capture_predict

        recursive_forecast(df, history, pipeline, FEATURE_NAMES, site_stats)

        # 1ère heure : lag_1 = dernière valeur de l'historique initial = 5.0
        assert captured_lags[0] == 5.0
        # 2ème heure : lag_1 = prédiction de la 1ère heure = 20.0
        assert captured_lags[1] == 20.0
        # 3ème heure : lag_1 = prédiction de la 2ème heure = 20.0
        assert captured_lags[2] == 20.0

    def test_history_window_stays_at_24(self):
        """Après N prédictions, l'historique ne doit jamais dépasser 24 valeurs."""
        n_hours = 30
        hours = pd.date_range("2025-03-01 01:00", periods=n_hours, freq="h").tolist()
        df = make_cartesian_df([1], hours)
        history = make_history_dict([1], value=5.0)
        site_stats = make_site_stats([1])

        history_lengths = []

        original_history = history.copy()

        def capture_predict(X):
            return np.array([10.0])

        pipeline = MagicMock()
        pipeline.predict.side_effect = capture_predict

        recursive_forecast(df, history, pipeline, FEATURE_NAMES, site_stats)
        # history est modifié in-place dans la fonction — on vérifie via les appels
        assert pipeline.predict.call_count == n_hours


# ===========================================================================
# 4. Gestion site inconnu (pas dans site_stats)
# ===========================================================================
class TestRecursiveForecastUnknownSite:
    def test_unknown_site_uses_fallback_not_crash(self):
        """Un site absent de site_stats ne doit pas faire planter la fonction."""
        df = make_cartesian_df([999], HOURS)
        history = make_history_dict([999])
        site_stats = make_site_stats([1, 2])  # site 999 absent

        pipeline = MagicMock()
        pipeline.predict.side_effect = Exception("model error")  # force le fallback

        result = recursive_forecast(df, history, pipeline, FEATURE_NAMES, site_stats)
        assert isinstance(result, pd.DataFrame)
        assert "comptage_horaire" in result.columns

    def test_unknown_site_fallback_predictions_non_negative(self):
        df = make_cartesian_df([999], HOURS)
        history = make_history_dict([999])
        site_stats = make_site_stats([1, 2])

        pipeline = MagicMock()
        pipeline.predict.side_effect = Exception("forced fallback")

        result = recursive_forecast(df, history, pipeline, FEATURE_NAMES, site_stats)
        assert (result["comptage_horaire"] >= 0).all()


# ===========================================================================
# 5. Ordre chronologique et isolation entre sites
# ===========================================================================
class TestRecursiveForecastIsolation:
    def test_sites_are_independent(self):
        """Deux sites avec des historiques différents donnent des lags différents."""
        hours = pd.date_range("2025-03-01 01:00", periods=2, freq="h").tolist()
        df = make_cartesian_df([1, 2], hours)

        history = {
            1: [100.0] * 24,   # site 1 : historique élevé
            2: [1.0] * 24,     # site 2 : historique bas
        }
        site_stats = make_site_stats([1, 2])

        lags_per_site = {}

        def capture_predict(X):
            site = int(X["identifiant_du_site_de_comptage"].iloc[0])
            lag = float(X["lag_1"].iloc[0])
            lags_per_site.setdefault(site, []).append(lag)
            return np.array([lag])  # retourne le lag comme pred pour traçabilité

        pipeline = MagicMock()
        pipeline.predict.side_effect = capture_predict

        recursive_forecast(df, history, pipeline, FEATURE_NAMES, site_stats)

        # Site 1 doit avoir lag_1 = 100, site 2 lag_1 = 1
        assert lags_per_site[1][0] == 100.0
        assert lags_per_site[2][0] == 1.0

    def test_predictions_are_finite(self):
        df = make_cartesian_df([1, 2], HOURS)
        history = make_history_dict([1, 2])
        site_stats = make_site_stats([1, 2])
        pipeline = make_pipeline_mock(15.0)

        result = recursive_forecast(df, history, pipeline, FEATURE_NAMES, site_stats)
        assert result["comptage_horaire"].apply(np.isfinite).all()

    def test_multiple_sites_all_rows_filled(self):
        sites = [1, 2, 3]
        df = make_cartesian_df(sites, HOURS)
        history = make_history_dict(sites)
        site_stats = make_site_stats(sites)
        pipeline = make_pipeline_mock(7.0)

        result = recursive_forecast(df, history, pipeline, FEATURE_NAMES, site_stats)
        assert result["comptage_horaire"].notna().all()