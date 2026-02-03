VENV_PY := .venv/bin/python
DATA := comptage_velo_donnees_compteurs.csv
MODEL := artifacts/model.pkl
PRED_OUT := artifacts/predictions.csv

.PHONY: update train eval predict

update:
	$(VENV_PY) scripts/update_data.py

train:
	$(VENV_PY) models/train.py --data $(DATA) --model-out $(MODEL)

eval:
	$(VENV_PY) models/evaluation.py --data $(DATA) --model $(MODEL)

predict:
	$(VENV_PY) models/predict.py --data $(DATA) --model $(MODEL) --out $(PRED_OUT)
