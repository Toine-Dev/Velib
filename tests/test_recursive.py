import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
import numpy as np
from models.features import recursive_forecast


class DummyPipeline:
    def predict(self, X):
        return np.array([10])

# ---------------------------------------
# Test du forecast récursif pour un site unique
# Vérifie que la colonne 'comptage_horaire' est bien ajoutée et que la valeur est positive
# ---------------------------------------
def test_recursive_forecast_one_site():

    df = pd.DataFrame({
        "identifiant_du_site_de_comptage": ["A"],
        "date_et_heure_de_comptage": [pd.Timestamp("2026-01-01")],
        "feature1": [0]
    })

    history = {"A": [5]*24}

    site_stats = pd.DataFrame({
        "identifiant_du_site_de_comptage": ["A"],
        "site_mean_usage": [5]
    })

    result = recursive_forecast(
        df,
        history,
        DummyPipeline(),
        ["feature1"],
        site_stats
    )

    assert "comptage_horaire" in result.columns
    assert result["comptage_horaire"].iloc[0] >= 0