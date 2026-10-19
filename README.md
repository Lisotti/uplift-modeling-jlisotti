# Uplift Modeling — framework reproducible para experimentos aleatorizados

**Proyecto Final · Licenciatura en Ciencia de Datos · Universidad CAECE (2026)**
Autor: Joaquín Tomás Lisotti · Docente: Dra. Sandra González Císaro

Sistema end-to-end que estima el **efecto causal incremental (CATE)** de uno o más
tratamientos sobre uno o más outcomes y recomienda **a quién contactar y con qué
tratamiento** bajo una restricción de presupuesto.

El pipeline es **genérico**: cada experimento se declara en un YAML
(`configs/<dataset>.yaml`) y con eso se generan EDA, modelos, métricas, API,
dashboard y monitoreo. Incluye dos experimentos:

| Dataset | Filas | Tratamientos | Outcomes |
|---|---|---|---|
| `hillstrom` — MineThatData (Hillstrom, 2008) | 64.000 | Email Mens / Email Womens vs. control | visita, conversión, gasto |
| `criteo` — Criteo Uplift v2.1 (Diemert et al., 2021) | 13.979.592 | Publicidad vs. control | visita, conversión |

## Estructura

```
├── configs/
│   ├── hillstrom.yaml              # declaración del experimento 1
│   └── criteo.yaml                 # declaración del experimento 2
├── data/raw/                       # datos originales (criteo se descarga aparte)
├── src/uplift/
│   ├── dataset.py                  # carga y acceso a la config de un dataset
│   ├── data.py                     # ingesta, contrato de datos, muestreo, split estratificado
│   ├── features.py                 # feature engineering declarativo + segmentos derivados
│   ├── eda.py                      # EDA genérico (calidad, balance RCT, ATE, heterogeneidad, viabilidad)
│   ├── learners.py                 # baseline + S/T/X-learners multi-tratamiento, Hurdle
│   ├── metrics.py                  # Qini, AUUC, uplift@k, deciles, bootstrap
│   ├── policy.py                   # política bajo presupuesto + evaluación IPW con IC 95 %
│   ├── train.py                    # pipeline reproducible + MLflow
│   ├── predict.py                  # scoring y recomendación
│   ├── monitoring.py               # drift (Evidently + PSI) y performance por ventana
│   ├── figures.py                  # figuras de resultados
│   └── download.py                 # descarga de datos desde source.url
├── api/main.py                     # API REST (FastAPI), un endpoint por dataset
├── dashboard/app.py                # simulador de campañas (Streamlit), selector de dataset
├── notebooks/EDA/                  # Entrega 2 — EDA original de Hillstrom
├── notebooks/<dataset>/            # 01_eda.ipynb y 02_modelado.ipynb generados por dataset
├── scripts/make_notebooks.py       # genera los notebooks de un dataset
├── tests/                          # pytest (pipeline, API y dataset sintético end-to-end)
└── artifacts/<dataset>/            # datos procesados, modelos, reportes y figuras (generados)
```

## Cómo correrlo

```bash
poetry install                       # Python 3.11
make pipeline DATASET=hillstrom      # EDA + entrenamiento + monitoreo + figuras (~5 min)
make pipeline DATASET=criteo         # requiere data/raw/criteo.csv.gz (ver abajo)
make test                            # suite de tests
make api                             # http://localhost:8000/docs
make dashboard                       # http://localhost:8501
make mlflow-ui                       # http://127.0.0.1:8080
make notebooks DATASET=criteo        # genera y ejecuta los notebooks del dataset
```

Pasos sueltos: `make eda|train|monitor|figures DATASET=<nombre>`. Sin `make`:
`PYTHONPATH=src poetry run python -m uplift.train --dataset criteo`.

### Datos de Criteo

Los datos de Criteo **no están en el repositorio** (≈ 300 MB). Antes de correr el pipeline hay que
descargarlos a `data/raw/` por alguna de estas vías:

1. `make download DATASET=criteo` (descarga automática desde Hugging Face).
2. A mano desde Kaggle: https://www.kaggle.com/datasets/arashnic/uplift-modeling (requiere cuenta).
   Descomprimí el archivo y copiá el CSV a `data/raw/`.
3. A mano desde el repositorio oficial de Criteo: https://huggingface.co/datasets/criteo/criteo-uplift

Se acepta cualquier archivo `data/raw/criteo*.csv` o `criteo*.csv.gz`
(por ejemplo `criteo-research-uplift-v2.1.csv`). Para modelar se usa una muestra
estratificada (`modeling.sample_size` en la config, por defecto 2.000.000 filas;
`--sample-size` la pisa desde la línea de comandos). El EDA usa el dataset completo.

## Agregar un dataset nuevo

1. Copiar un YAML de `configs/` y declarar: archivo fuente, columna de tratamiento y
   sus brazos, outcomes (`binary` o `continuous`), features (`numeric`, `binary`,
   `categorical`, `interaction`), columnas excluidas y por qué, reglas de validación,
   supuestos de negocio (`business.values`: valor por unidad de cada outcome) y
   escenario de drift para el monitoreo.
2. `make pipeline DATASET=<nombre>` y `make notebooks DATASET=<nombre>`.
3. El dataset aparece automáticamente en la API (`/<nombre>/predict`) y en el dashboard.

### Ejemplo de scoring

```bash
curl -X POST localhost:8000/hillstrom/predict -H "Content-Type: application/json" -d '{
  "customers": [{"recency": 2, "history": 350, "mens": 1, "womens": 1,
                 "zip_code": "Surburban", "newbie": 0, "channel": "Web"}],
  "objective": "profit", "cost_per_contact": 0.10, "value_spend": 0.40}'
```

## Metodología (resumen)

* **CRISP-DM**. Entrega 1: comprensión del negocio. Entrega 2: EDA y diccionario de datos.
* **Modelos**: baseline de respuesta (no causal) vs. S-, T- y X-learner (Künzel et al., 2019)
  con base learners LightGBM y lineales regularizados; T-learner Hurdle para outcomes
  continuos inflados de ceros. Multi-tratamiento: un CATE por tratamiento frente al control.
* **Selección**: K-fold estratificado repetido sobre train, criterio Qini relativo.
* **Evaluación**: Qini/AUUC normalizados (validados contra scikit-uplift), uplift@k,
  uplift por decil, IC bootstrap y evaluación de políticas por IPW con IC 95 %.
* **Negocio**: beneficio = Σ valor × incremental − costo por contacto (parametrizable).
