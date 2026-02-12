import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
import numpy as np
import pytest
from models.model_utils import load_model

# ---------------------------------------
# Test que le modèle se charge correctement depuis le fichier model.pkl
# ---------------------------------------
def test_model_loads():
    pipeline = load_model()
    assert pipeline is not None

# ---------------------------------------
# Test que le modèle possède bien des noms de features
# ---------------------------------------
def test_feature_names_exist():
    pipeline = load_model()
    assert hasattr(pipeline, "feature_names_in_")
    assert len(pipeline.feature_names_in_) > 0

# ---------------------------------------
# Test d'une prédiction simple sur une seule ligne factice
# ---------------------------------------
def test_single_prediction():
    pipeline = load_model()
    features = pipeline.feature_names_in_

    X = pd.DataFrame([{c: 0 for c in features}])

    y = pipeline.predict(X)

    assert len(y) == 1
    assert np.isfinite(y[0])

# ---------------------------------------
# Test que la prédiction renvoyée est toujours positive
# ---------------------------------------
def test_prediction_positive():
    pipeline = load_model()
    features = pipeline.feature_names_in_

    X = pd.DataFrame([{c: 0 for c in features}])
    y = pipeline.predict(X)[0]

    assert y >= 0

# ---------------------------------------
# Test que le modèle renvoie une erreur si une feature est manquante
# ---------------------------------------
def test_missing_feature_raises():
    pipeline = load_model()
    features = pipeline.feature_names_in_[:-1]

    X = pd.DataFrame([{c: 0 for c in features}])

    with pytest.raises(Exception):
        pipeline.predict(X)

# ---------------------------------------
# Test qu'aucune feature sensible ne provoque de fuite de données (data leakage)
# Vérifie que les noms de features ne contiennent pas 'target', 'future', 'label' ou 'comptage_horaire'
# Exception pour les colonnes identifiant du site
# ---------------------------------------
def test_no_data_leakage():
    pipeline = load_model()
    features = pipeline.feature_names_in_

    forbidden = ["target", "future", "label", "comptage_horaire"]

    for f in features:
        if "identifiant" in f.lower():
            continue
        for bad in forbidden:
            assert bad not in f.lower()