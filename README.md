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
│   └── download.py                 # descarga de datos (Criteo: desde Kaggle)
├── api/main.py                     # API REST (FastAPI), un endpoint por dataset
├── dashboard/app.py                # simulador de campañas (Streamlit), selector de dataset
├── notebooks/EDA/                  # Entrega 2 — EDA original de Hillstrom
├── notebooks/<dataset>/            # 01_eda.ipynb y 02_modelado.ipynb generados por dataset
├── scripts/make_notebooks.py       # genera los notebooks de un dataset
├── tests/                          # pytest (pipeline, API y dataset sintético end-to-end)
├── pyproject.toml / poetry.lock    # dependencias (Poetry)
├── requirements.txt                # mismas versiones, para quien prefiera pip
└── artifacts/<dataset>/            # datos procesados, modelos, reportes y figuras (generados)
```

## 1. Requisitos previos

| Herramienta | Versión | Cómo instalarla |
|---|---|---|
| **Python** | 3.11 o 3.12 (3.13 no está soportado) | https://www.python.org/downloads/ · macOS: `brew install python@3.11` |
| **Poetry** | 2.x | `pipx install poetry` (recomendado) o `curl -sSL https://install.python-poetry.org \| python3 -` |
| **make** | cualquiera | ya viene en macOS y Linux (en macOS, si falta: `xcode-select --install`) |
| **libomp** | sólo macOS | `brew install libomp` (la necesita LightGBM) |

Para instalar Poetry con `pipx`:

```bash
python3 -m pip install --user pipx    # macOS: brew install pipx
python3 -m pipx ensurepath            # cerrar y volver a abrir la terminal
pipx install poetry
poetry --version                      # debería mostrar "Poetry (version 2.x)"
```

Documentación oficial: https://python-poetry.org/docs/#installation. En Windows se recomienda
usar WSL2.

## 2. Instalación

```bash
git clone https://github.com/Lisotti/uplift-modeling-jlisotti.git
cd uplift-modeling-jlisotti
poetry env use python3.11      # le indica a Poetry qué Python usar (o la ruta completa al ejecutable)
make install                   # = poetry install: crea .venv/ en el repo con las versiones de poetry.lock
make test                      # verificación: tienen que pasar todos los tests
```

Sin Poetry también se puede instalar con pip, con las mismas versiones:
`python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
En ese caso los comandos de `make` (que usan `poetry run`) se reemplazan por el comando directo,
por ejemplo `PYTHONPATH=src:. python -m uplift.train --dataset hillstrom`.

## 3. Datos

* **Hillstrom** viene incluido en el repositorio (`data/raw/hillstrom.csv`): no hay que hacer nada.
* **Criteo** se descarga aparte (ver más abajo).

### Datos de Criteo (sólo para el experimento 2)

Los datos de Criteo **no están en el repositorio** (≈ 300 MB) y **la única fuente disponible es Kaggle**:
https://www.kaggle.com/datasets/arashnic/uplift-modeling (requiere una cuenta gratuita).

**A mano:**

1. Entrar a la página del dataset con tu cuenta y hacer clic en **Download**.
2. Descomprimir el `.zip`.
3. Copiar el CSV a `data/raw/`. El nombre tiene que empezar con `criteo`
   (por ejemplo `data/raw/criteo-research-uplift-v2.1.csv`; también sirve `.csv.gz`).

**Con el CLI de Kaggle** (opcional): instalarlo (`pipx install kaggle`), configurar el token en
`~/.kaggle/kaggle.json` (https://www.kaggle.com/docs/api) y correr `make download DATASET=criteo`.
Sin el CLI configurado, ese comando sólo muestra los pasos manuales.

Para modelar se usa una muestra estratificada (`modeling.sample_size` en la config, por defecto
2.000.000 filas; `--sample-size` la pisa desde la línea de comandos). El EDA usa el dataset completo.

## 4. Paso a paso

El repositorio trae los modelos entrenados y las figuras, pero **no** los datos procesados
(`artifacts/<dataset>/data/`) ni los runs de MLflow (`mlruns/`). Por eso, antes de usar los
notebooks, el dashboard o MLflow hay que **correr el pipeline de cada experimento una vez**.

### 4.1. Correr los experimentos

```bash
make pipeline DATASET=hillstrom      # ~5 min
# antes: copiar los datos de Criteo desde Kaggle a data/raw/ (ver "Datos de Criteo")
make pipeline DATASET=criteo         # puede superar los 20 min: 14 M de filas para el EDA, 2 M para modelar
```

`make pipeline` ejecuta, en orden, cuatro pasos que también se pueden correr por separado:

| Paso | Comando | Qué genera (en `artifacts/<dataset>/`) |
|---|---|---|
| EDA | `make eda DATASET=<nombre>` | calidad de datos, balance del RCT, ATE, heterogeneidad → `reports/` |
| Entrenamiento | `make train DATASET=<nombre>` | split train/test, validación cruzada, modelos, métricas, run de MLflow → `data/`, `models/`, `reports/` |
| Monitoreo | `make monitor DATASET=<nombre>` | ventanas simuladas y reportes de drift → `reports/monitoring/` |
| Figuras | `make figures DATASET=<nombre>` | figuras de resultados → `reports/figures/` |

### 4.2. Correr los notebooks

Cada experimento tiene dos notebooks que leen lo que generó el pipeline (por eso hay que correr
el paso 4.1 antes):

* `notebooks/<dataset>/01_eda.ipynb` — análisis exploratorio.
* `notebooks/<dataset>/02_modelado.ipynb` — modelos, política, scoring vía API y monitoreo.
* `notebooks/EDA/` — EDA original de Hillstrom (Entrega 2); sólo necesita `data/raw/hillstrom.csv`.

Para abrirlos y ejecutarlos:

```bash
poetry run jupyter lab               # se abre en el navegador; navegar a notebooks/
```

Desde VS Code: abrir el notebook y elegir como kernel el Python de `.venv/` del repo.

### 4.3. Dashboard (simulador de campañas)

```bash
make dashboard                       # http://localhost:8501
```

En la barra lateral se elige el experimento (Hillstrom o Criteo) y se ajustan objetivo,
presupuesto, costo por contacto y valor de cada outcome. **Requiere haber corrido el pipeline de
los dos experimentos** (lee el split de test de cada uno).

### 4.4. API REST

```bash
make api                             # http://localhost:8000/docs (documentación interactiva)
```

Desde otra terminal, o desde `/docs` → *Try it out*:

```bash
# Hillstrom: ¿a qué cliente mandarle qué e-mail?
curl -X POST localhost:8000/hillstrom/predict -H "Content-Type: application/json" -d '{
  "customers": [{"recency": 2, "history": 350, "mens": 1, "womens": 1,
                 "zip_code": "Surburban", "newbie": 0, "channel": "Web"}],
  "objective": "profit", "cost_per_contact": 0.10, "value_spend": 0.40}'

# Criteo: un usuario al que conviene mostrarle publicidad ("ads") …
curl -X POST localhost:8000/criteo/predict -H "Content-Type: application/json" -d '{
  "customers": [{"f0": 12.846, "f1": 10.06, "f2": 8.334, "f3": 0.604, "f4": 12.31, "f5": 0.026,
                 "f6": -21.373, "f7": 10.493, "f8": 3.796, "f9": 32.678, "f10": 5.893, "f11": -0.267}],
  "objective": "profit", "cost_per_contact": 0.005, "value_conversion": 20}'

# … y uno al que no ("no_ads")
curl -X POST localhost:8000/criteo/predict -H "Content-Type: application/json" -d '{
  "customers": [{"f0": 25.699, "f1": 10.68, "f2": 8.214, "f3": 4.68, "f4": 10.281, "f5": 0.026,
                 "f6": -1.288, "f7": 8.788, "f8": 3.972, "f9": 13.19, "f10": 5.3, "f11": -0.169}],
  "objective": "profit", "cost_per_contact": 0.005, "value_conversion": 20}'
```

La respuesta trae, por cliente, el uplift estimado de cada outcome y tratamiento, el tratamiento
recomendado y el beneficio esperado. `objective` acepta `profit` o el nombre de un outcome;
`cost_per_contact` y `value_<outcome>` pisan los supuestos de negocio de la config.
Otras rutas: `GET /datasets`, `GET /<dataset>/model/info`, `GET /health`. La API usa sólo los
modelos (`artifacts/<dataset>/models/`), que ya vienen en el repo.

### 4.5. MLflow (seguimiento de experimentos)

```bash
make mlflow-ui                       # http://127.0.0.1:8080
```

Muestra un experimento por dataset (`uplift-hillstrom`, `uplift-criteo`) con parámetros,
métricas y modelos de cada corrida de `make train`. Aparece vacío hasta entrenar al menos una vez.

## 5. Agregar un experimento nuevo

El código no cambia: un experimento nuevo es un archivo YAML en `configs/`. Flujo de trabajo:

1. **Datos.** Copiar el CSV (o `.csv.gz`) a `data/raw/`. Tiene que venir de un experimento
   aleatorizado: una columna de tratamiento (control y uno o más brazos), uno o más outcomes y
   features medidas **antes** del tratamiento.
2. **Config.** Copiar `configs/hillstrom.yaml` (multi-tratamiento, outcome continuo) o
   `configs/criteo.yaml` (un tratamiento, features numéricas) a `configs/<nombre>.yaml` y completar:

   | Bloque | Qué se declara |
   |---|---|
   | `name`, `title`, `description` | identificador (igual al nombre del archivo) y textos para dashboard y API |
   | `source` | `path` del archivo (opcional: `kaggle: <owner>/<dataset>` o `url` para `make download`, `dtypes`) |
   | `treatment` | columna, valor del control y de cada brazo (`value`, `label`, `name`) |
   | `outcomes` | cada target con `type: binary` o `continuous` (`hurdle: true` si es gasto con muchos ceros); `primary_outcome` y `stratify_outcome` |
   | `features` | `numeric` (opcional `log1p`), `binary`, `categorical` (con `categories`) o `interaction` |
   | `excluded_columns` | columnas que no se usan y por qué (por ejemplo, variables post-tratamiento) |
   | `validation` | rangos válidos y reglas de consistencia entre outcomes |
   | `modeling` | tamaño de muestra, % de test, folds de CV, réplicas bootstrap |
   | `business` | costo por contacto y valor monetario de cada outcome (`values`) |
   | `segments`, `figures`, `monitoring` | segmentos para el dashboard, heatmap y escenario de drift simulado |

3. **Correr el pipeline:** `make pipeline DATASET=<nombre>`. Revisar el EDA en
   `artifacts/<nombre>/reports/` (balance del RCT y ATE) antes de mirar los modelos.
4. **Generar los notebooks:** `make notebooks DATASET=<nombre>` crea y ejecuta
   `notebooks/<nombre>/01_eda.ipynb` y `02_modelado.ipynb`. (Ojo: sobrescribe los notebooks de
   ese dataset si ya existen.)
5. **Listo:** el experimento aparece solo en el dashboard (selector de la barra lateral), en la
   API (`/<nombre>/predict`) y en MLflow (`uplift-<nombre>`). `make test` debería seguir pasando.

## 6. Problemas frecuentes

* `poetry: command not found` → falta instalar Poetry (paso 1) o reabrir la terminal después de `pipx ensurepath`.
* Poetry usa otro Python (por ejemplo 3.13) → `poetry env use python3.11` y otra vez `make install`.
* `Library not loaded: libomp.dylib` al importar LightGBM (macOS) → `brew install libomp`.
* `No existe test para <dataset>. Ejecutá make train ...` en el dashboard o en un notebook → falta
  correr el pipeline de ese dataset (paso 4.1).
* MLflow no muestra nada → todavía no se corrió `make train` en esa máquina.
* Un puerto ocupado → cerrar el proceso que lo usa o cambiar el `--port` en el `Makefile`.

## Metodología (resumen)

* **CRISP-DM**. Entrega 1: comprensión del negocio. Entrega 2: EDA y diccionario de datos.
* **Modelos**: baseline de respuesta (no causal) vs. S-, T- y X-learner (Künzel et al., 2019)
  con base learners LightGBM y lineales regularizados; T-learner Hurdle para outcomes
  continuos inflados de ceros. Multi-tratamiento: un CATE por tratamiento frente al control.
* **Selección**: K-fold estratificado repetido sobre train, criterio Qini relativo.
* **Evaluación**: Qini/AUUC normalizados (validados contra scikit-uplift), uplift@k,
  uplift por decil, IC bootstrap y evaluación de políticas por IPW con IC 95 %.
* **Negocio**: beneficio = Σ valor × incremental − costo por contacto (parametrizable).
