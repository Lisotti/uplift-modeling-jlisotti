---
name: uplift-setup
description: Instala y levanta el repo de Uplift Modeling (dependencias con Poetry, API, dashboard y MLflow) y, si el usuario lo pide, corre los pipelines (descarga de datos, EDA, entrenamiento, monitoreo, figuras) o los tests. Usar cuando pidan instalar, configurar, levantar, probar, entrenar o "dejar andando" el proyecto. No usar para modificar código, notebooks o configs.
---

# Setup y operación del repo de Uplift Modeling

Toda la operación pasa por un script multiplataforma (macOS, Linux, Windows) que sólo usa la
librería estándar de Python. Correrlo siempre **desde la raíz del repo**:

```bash
python .agents/skills/uplift-setup/scripts/uplift_env.py <comando>
```

En Windows usar `py -3.11` en lugar de `python` si `python` no es 3.11/3.12.

## Reglas

* **No modificar archivos del repo** (código, notebooks, configs, README, Makefile). Esta skill
  sólo instala, levanta servicios y ejecuta pipelines. Si algo falla por un bug, reportarlo al
  usuario en vez de "arreglarlo".
* **No hacer operaciones de git** (commit, push, checkout, etc.).
* **Pedir confirmación antes de**: instalar software en el sistema (Python, Poetry, libomp),
  descargar datos grandes (Criteo ≈ 300 MB) y correr pasos largos (pipeline de Criteo: 20+ min).
* **No pedir ni escribir credenciales** (por ejemplo el token de Kaggle): si hacen falta, las
  configura el usuario.
* Los pipelines **sólo se corren si el usuario lo pide**. El setup normal no entrena nada.

## Flujo de setup (lo que se hace por defecto)

1. **Diagnóstico:** `uplift_env.py check`. Devuelve un JSON con versión de Python, Poetry, `.venv`,
   libomp (macOS) y, por dataset, si hay datos crudos, modelo, split de test, EDA y monitoreo.
2. **Requisitos faltantes** (pedir confirmación y, con el OK del usuario, instalar):
   * Python 3.11 o 3.12 (3.13 no está soportado). Si el Python actual no sirve, mirar
     `otros_python` en el diagnóstico.
   * Poetry 2.x — ver "Instalar Poetry".
   * macOS: `brew install libomp` si `libomp` es `false` (lo necesita LightGBM).
3. **Dependencias:** `uplift_env.py install --python <python3.11 | ruta al ejecutable>`
   (hace `poetry env use` + `poetry install`). Si `.venv` ya existe y funciona, se puede omitir `--python`.
4. **Verificación:** `uplift_env.py test` (deben pasar todos los tests).
5. **Servicios:** `uplift_env.py up all` levanta en segundo plano:

   | Servicio | URL | Requisito |
   |---|---|---|
   | API (FastAPI) | http://localhost:8000/docs | sólo los modelos, que vienen en el repo |
   | Dashboard (Streamlit) | http://localhost:8501 | split de test de **todos** los datasets con modelo |
   | MLflow | http://127.0.0.1:8080 | abre vacío hasta que se corra al menos un `train` |

   El script no levanta el dashboard si falta algún split de test y avisa qué dataset falta.
   En ese caso, explicarle al usuario que hay que entrenar ese dataset (paso "Pipelines") y
   preguntarle si quiere hacerlo. En un clon nuevo faltan los splits de **todos** los datasets.
6. **Informar al usuario** las URLs que respondieron OK y cualquier aviso. Logs en `.uplift/logs/<servicio>.log`.

Otros comandos: `uplift_env.py status` (qué está corriendo) y `uplift_env.py down [api|dashboard|mlflow|all]`.

## Pipelines (sólo a pedido del usuario)

Datasets disponibles: un YAML por dataset en `configs/` (hoy `hillstrom` y `criteo`).

| Pedido | Comando |
|---|---|
| Descargar datos (Criteo) | `uplift_env.py download criteo` (ver "Datos de Criteo") |
| Pipeline completo | `uplift_env.py pipeline <dataset>` (eda → train → monitor → figures) |
| Sólo entrenar (lo mínimo para el dashboard) | `uplift_env.py pipeline <dataset> --steps train` |
| Pasos sueltos | `--steps eda`, `--steps monitor,figures`, etc. |
| Entrenar con una muestra más chica | `uplift_env.py pipeline criteo --steps train -- --sample-size 500000` |
| Tests | `uplift_env.py test` |

* Hillstrom trae los datos en `data/raw/` y su pipeline tarda ~5 min.
* Criteo necesita sus datos en `data/raw/` antes (ver "Datos de Criteo"); el pipeline completo tarda 20+ min (EDA sobre 14 M de
  filas, modelado sobre 2 M). Para que el dashboard funcione alcanza con `--steps train`.
* Después de entrenar, si el dashboard o MLflow ya estaban levantados, reiniciarlos
  (`down dashboard` + `up dashboard`) para que tomen los nuevos artefactos.
* Notebooks: `poetry run jupyter lab` y abrir `notebooks/<dataset>/`. Leen lo que generó el
  pipeline, así que necesitan haberlo corrido antes. No regenerar notebooks (`make notebooks`)
  salvo pedido explícito: sobrescribe los existentes.

## Datos de Criteo

La **única fuente que funciona es Kaggle**: https://www.kaggle.com/datasets/arashnic/uplift-modeling
(requiere cuenta). No buscar otras fuentes.

* `uplift_env.py download criteo` usa el CLI de Kaggle **si el usuario ya lo tiene instalado y con
  el token configurado** (`~/.kaggle/kaggle.json`). Si no, imprime los pasos manuales y termina con
  código 2: mostrárselos al usuario y esperar a que avise que copió el archivo.
* Pasos manuales: descargar el .zip desde la página, descomprimirlo y copiar el CSV a `data/raw/`
  con un nombre que empiece con `criteo` (por ejemplo `criteo-research-uplift-v2.1.csv`; también
  sirve `.csv.gz`).
* Confirmar con `uplift_env.py check` (`datasets.criteo.raw_data: true`).

## Instalar Poetry

Con confirmación del usuario:

* macOS / Linux: `python3 -m pip install --user pipx && python3 -m pipx ensurepath && pipx install poetry`
  (macOS con Homebrew: `brew install pipx && pipx ensurepath && pipx install poetry`).
* Windows (PowerShell): `py -3.11 -m pip install --user pipx; py -3.11 -m pipx ensurepath; pipx install poetry`.

Después de `ensurepath` puede hacer falta abrir una terminal nueva para que `poetry` esté en el PATH.
Documentación oficial: https://python-poetry.org/docs/#installation

## Problemas frecuentes

| Síntoma | Causa / solución |
|---|---|
| `Library not loaded: libomp.dylib` | macOS sin libomp → `brew install libomp` |
| Poetry usa Python 3.13 | `uplift_env.py install --python python3.11` |
| Dashboard: `No existe test para <dataset>` | falta entrenar ese dataset → `pipeline <dataset> --steps train` |
| MLflow sin experimentos | no se corrió `train` en esta máquina |
| `NO RESPONDE` en `up` | leer `.uplift/logs/<servicio>.log`; suele ser un puerto ocupado (8000, 8501, 8080) |
| `faltan los datos crudos` | Criteo: ver "Datos de Criteo" |
