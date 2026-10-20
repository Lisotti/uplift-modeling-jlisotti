# Uplift Modeling — Proyecto Final CAECE
# Uso: make <target> [DATASET=hillstrom|criteo]   (requiere Poetry: https://python-poetry.org)
# Cada dataset se declara en configs/<DATASET>.yaml; los artefactos van a artifacts/<DATASET>/.

PY      = poetry run python
DATASET ?= hillstrom
export PYTHONPATH := src:.

.PHONY: help install download eda train monitor figures pipeline api dashboard test notebooks report pdf all mlflow-ui clean

help:
	@echo "Datasets disponibles: $$(ls configs | sed 's/.yaml//' | tr '\n' ' ')"
	@echo ""
	@echo "install     Instala dependencias (poetry install)"
	@echo "download    Descarga los datos del dataset (Criteo: desde Kaggle, ver README)"
	@echo "            Criteo a mano: https://www.kaggle.com/datasets/arashnic/uplift-modeling -> data/raw/"
	@echo "eda         EDA genérico -> artifacts/\$$DATASET/reports"
	@echo "train       CV, entrenamiento, evaluación y MLflow -> artifacts/\$$DATASET"
	@echo "monitor     Ventanas simuladas y reportes de drift"
	@echo "figures     Figuras de resultados"
	@echo "pipeline    eda + train + monitor + figures para \$$DATASET"
	@echo "notebooks   Genera y ejecuta los notebooks de \$$DATASET"
	@echo "api         API REST multi-dataset en http://localhost:8000/docs"
	@echo "dashboard   Dashboard Streamlit en http://localhost:8501"
	@echo "mlflow-ui   MLflow en http://127.0.0.1:8080"
	@echo "test        Suite de tests"
	@echo ""
	@echo "Ejemplo: make pipeline DATASET=criteo"

install:
	poetry install

download:
	$(PY) -m uplift.download $(DATASET)

eda:
	$(PY) -m uplift.eda --dataset $(DATASET)

train:
	$(PY) -m uplift.train --dataset $(DATASET)

monitor:
	$(PY) -m uplift.monitoring --dataset $(DATASET)

figures:
	$(PY) -m uplift.figures --dataset $(DATASET)

pipeline: eda train monitor figures

notebooks:
	$(PY) scripts/make_notebooks.py --dataset $(DATASET)
	poetry run jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=7200 notebooks/$(DATASET)/*.ipynb

api:
	poetry run uvicorn api.main:app --reload --port 8000

dashboard:
	poetry run streamlit run dashboard/app.py

test:
	poetry run pytest -q

report:
	$(PY) scripts/build_report.py

pdf:
	python3 scripts/docx_to_pdf.py "$(CURDIR)/entregas/Proyecto_Final_Jlisotti.docx" "$(CURDIR)/entregas/Proyecto_Final_Jlisotti.pdf"

mlflow-ui:
	poetry run mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 8080 --workers 1 --gunicorn-opts "--timeout 300"

all:
	$(MAKE) pipeline DATASET=hillstrom
	$(MAKE) pipeline DATASET=criteo
	$(MAKE) test

clean:
	rm -rf artifacts/$(DATASET)
