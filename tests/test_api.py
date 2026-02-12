import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from fastapi.testclient import TestClient
from api.main import app


client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

# ---------------------------------------
# Test prédiction API pour un datetime valide
# Vérifie que l'endpoint /predict retourne bien 200 et un dictionnaire avec les colonnes attendues pour une date future
# ---------------------------------------
def test_predict_endpoint_valid():
    # ISO format strict pour Pydantic
    test_datetime = "2026-02-12T12:00:00"
    response = client.post("/predict", json={"datetime": test_datetime})
    
    # Statut HTTP
    assert response.status_code == 200
    
    # Vérifier qu'il y a des résultats et colonnes attendues
    data = response.json()
    assert isinstance(data, list)
    if len(data) > 0:
        assert "nom_du_site_de_comptage" in data[0]
        assert "latitude" in data[0]
        assert "longitude" in data[0]
        assert "comptage_horaire" in data[0]

# ---------------------------------------
# Test prédiction API pour un datetime invalide
# Vérifie que l'API renvoie 500 si le format datetime est incorrect
# ---------------------------------------
def test_predict_endpoint_invalid_datetime():
    test_datetime = "invalid-date-format"
    response = client.post("/predict", json={"datetime": test_datetime})
    
    assert response.status_code == 500