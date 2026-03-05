# conftest.py — fixtures partagées entre tous les modules de tests
import pytest
import pandas as pd
import numpy as np


@pytest.fixture(scope="session")
def sample_velib_df():
    """DataFrame Vélib minimal et propre pour les tests."""
    return pd.DataFrame({
        "identifiant_du_site_de_comptage": [100056789, 100056789, 100056790],
        "comptage_horaire": [10, 25, 5],
        "date_et_heure_de_comptage": pd.to_datetime([
            "2025-03-01 08:00:00",
            "2025-03-01 09:00:00",
            "2025-03-01 08:00:00",
        ]),
        "nom_du_site_de_comptage": ["Site A", "Site A", "Site B"],
    })


@pytest.fixture(scope="session")
def sample_weather_df():
    """DataFrame météo minimal."""
    return pd.DataFrame({
        "time": pd.to_datetime(["2025-03-01 08:00:00", "2025-03-01 09:00:00"]),
        "rain": [0.0, 3.5],
        "wind_speed_10m": [5.0, 20.0],
        "snowfall": [0.0, 0.0],
        "apparent_temperature": [12.0, 11.5],
    })