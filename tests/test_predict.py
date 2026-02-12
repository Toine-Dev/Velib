import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
from models.predict import predict_model

# ---------------------------------------
# Test que la fonction predict_model fonctionne
# Vérifie que le dataframe retourné n'est pas vide et contient la colonne 'comptage_horaire'
# ---------------------------------------
def test_predict_model_runs():

    dt = pd.Timestamp("2026-02-12 12:00")

    df = predict_model(dt)

    assert df is not None
    assert len(df) > 0
    assert "comptage_horaire" in df.columns