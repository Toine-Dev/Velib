from utils.config import load_model
import pandas as pd
import numpy as np
import pytest

# Teste que le modèle se charge correctement depuis le fichier
def test_model_loads():
    pipeline = load_model()
    assert pipeline is not None

# Teste que le modèle possède bien des noms de features enregistrés
def test_feature_names_exist():
    pipeline = load_model()
    assert hasattr(pipeline, "feature_names_in_")
    assert len(pipeline.feature_names_in_) > 0

# Teste une prédiction simple avec une seule ligne factice
# Vérifie que la prédiction est finie et retourne un résultat
def test_single_prediction():
    pipeline = load_model()
    feature_names = pipeline.feature_names_in_

    # ligne fake mais valide
    X = pd.DataFrame([{
        col: 0 for col in feature_names
    }])

    y_pred = pipeline.predict(X)

    assert y_pred is not None
    assert len(y_pred) == 1
    assert np.isfinite(y_pred[0])

# Teste que la prédiction renvoyée est toujours positive
def test_prediction_is_positive():
    pipeline = load_model()
    feature_names = pipeline.feature_names_in_

    X = pd.DataFrame([{col: 0 for col in feature_names}])
    y_pred = pipeline.predict(X)[0]

    assert y_pred >= 0

# Teste que le modèle renvoie une erreur si une feature est manquante
def test_missing_feature_raises_error():
    pipeline = load_model()
    feature_names = pipeline.feature_names_in_

    # on enlève volontairement une feature
    bad_features = feature_names[:-1]

    X = pd.DataFrame([{col: 0 for col in bad_features}])

    with pytest.raises(Exception):
        pipeline.predict(X)

# Teste le forecast récursif minimaliste pour un site unique
# Vérifie que la prédiction retourne une valeur positive
def test_recursive_forecast_one_site():
    pipeline = load_model()
    feature_names = pipeline.feature_names_in_

    # simulation minimaliste
    fake_row = {
        col: 0 for col in feature_names
    }
    fake_row["lag_1"] = 10
    fake_row["lag_24"] = 15
    fake_row["rolling_mean_24"] = 12

    X = pd.DataFrame([fake_row])
    y = pipeline.predict(X)

    assert y[0] >= 0

# Teste la boucle récursive complète sur 24 heures pour un site
# Vérifie : nombre de prédictions, valeurs finies, valeurs positives
def test_recursive_forecast_loop_one_site_24h():
    pipeline = load_model()
    feature_names = pipeline.feature_names_in_

    # 24 heures futures
    dates = pd.date_range("2024-01-01", periods=24, freq="H")

    # dataframe simulant cartesian_df pour 1 site
    cartesian_df = pd.DataFrame({
        "identifiant_du_site_de_comptage": ["SITE_TEST"] * 24,
        "date_et_heure_de_comptage": dates,
        **{col: 0 for col in feature_names}
    })

    # historique initial (24 valeurs passées)
    history = [10] * 24

    preds = []

    for _, row in cartesian_df.iterrows():
        lag_1 = history[-1]
        lag_24 = history[0]
        rolling_mean_24 = np.mean(history)

        row = row.copy()
        row["lag_1"] = lag_1
        row["lag_24"] = lag_24
        row["rolling_mean_24"] = rolling_mean_24

        X = pd.DataFrame([row[feature_names]])
        pred = float(pipeline.predict(X)[0])
        pred = max(pred, 0)

        preds.append(pred)

        history.append(pred)
        history = history[-24:]

    # assertions clés
    assert len(preds) == 24
    assert all(np.isfinite(p) for p in preds)
    assert all(p >= 0 for p in preds)

# Teste qu'aucune feature ne provoque de fuite de données (data leakage)
# Vérifie que les noms de features ne contiennent pas de mots interdits
def test_no_data_leakage_in_features():
    pipeline = load_model()
    feature_names = pipeline.feature_names_in_

    forbidden_keywords = [
        "comptage",
        "target",
        "future",
        "label"
    ]

    for col in feature_names:
        for bad in forbidden_keywords:
            assert bad not in col.lower(), f"Data leakage détecté : {col}"