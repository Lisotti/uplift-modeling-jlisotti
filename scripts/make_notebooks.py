"""Genera los notebooks ejecutables de un dataset: notebooks/<dataset>/01_eda.ipynb y 02_modelado.ipynb.

Uso:  python scripts/make_notebooks.py --dataset criteo
Los notebooks orquestan el paquete ``uplift`` y presentan resultados; con
``RUN = True`` re-ejecutan el paso correspondiente del pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from uplift.dataset import load_config  # noqa: E402

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

SETUP = """import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path.cwd().resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import numpy as np, pandas as pd
from IPython.display import Image, display
from uplift.dataset import load_config
pd.set_option("display.float_format", "{:.4f}".format)
cfg = load_config("%s")
FIG = cfg.figures_dir
def show(name):
    p = FIG / f"{name}.png"
    if p.exists():
        display(Image(str(p)))"""


def eda_notebook(cfg):
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(f"# EDA — {cfg.title}\n**Proyecto Final · Lic. en Ciencia de Datos · Universidad CAECE**\n\n"
           f"{cfg.description}\n\nNotebook generado por `scripts/make_notebooks.py` a partir de "
           f"`configs/{cfg.name}.yaml`. El análisis lo hace `uplift.eda` (genérico para cualquier dataset)."),
        code(SETUP % cfg.name + "\nRUN = not (cfg.reports_dir / 'eda_summary.json').exists()"),
        code("from uplift import eda\nsummary = eda.run(cfg) if RUN else json.load(open(cfg.reports_dir / 'eda_summary.json'))\n"
             "pd.Series({k: v for k, v in summary['quality'].items() if not isinstance(v, dict)})"),
        md("## 1. Diseño experimental y balance de covariables\nEn un experimento aleatorizado las covariables "
           "deben estar balanceadas entre brazos (|SMD| < 0,1)."),
        code("display(pd.read_csv(cfg.tables_dir / 'eda_arm_counts.csv'))\nshow('eda_balance')\n"
             "print('|SMD| máximo:', round(summary['max_abs_smd'], 4))"),
        md("## 2. Efecto promedio de cada tratamiento (ATE)"),
        code("display(pd.read_csv(cfg.tables_dir / 'eda_ate.csv'))\nshow('eda_rates_by_group')"),
        md("## 3. Distribuciones y correlaciones"),
        code("display(pd.read_csv(cfg.tables_dir / 'eda_numeric_summary.csv'))\nshow('eda_distributions')\n"
             "show('eda_outcome_distribution')\nshow('eda_correlation')\nsummary['strong_correlations']"),
        md("## 4. Heterogeneidad del efecto\nSi el efecto del tratamiento varía entre tramos de las features, "
           "el CATE individual es estimable (test Q de Cochran sobre los ATE por tramo)."),
        code(f"display(pd.read_csv(cfg.tables_dir / 'eda_heterogeneity_test_{cfg.primary_outcome}.csv').head(10))\n"
             "show('eda_heterogeneity')\nshow('eda_heterogeneity_features')"),
        md("## 5. Veredicto de viabilidad"),
        code("display(pd.DataFrame(summary['viability']))\nprint('VIABLE' if summary['viable'] else 'REVISAR')"),
    ]
    return nb


def modeling_notebook(cfg):
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(f"# Modelado de uplift — {cfg.title}\nFases 3 a 6 de CRISP-DM: preparación, modelado "
           "(meta-learners S/T/X), evaluación (Qini, AUUC, IPW) y despliegue (API, dashboard, drift).\n\n"
           f"Con `RUN = True` se re-ejecuta el pipeline completo para `{cfg.name}`."),
        code(SETUP % cfg.name + "\nRUN = not cfg.bundle_path.exists()\nprint(cfg.modeling)"),
        md("## 1. Entrenamiento y selección por validación cruzada"),
        code("if RUN:\n    from uplift import train\n    train.run(cfg)\n"
             "metrics = json.load(open(cfg.reports_dir / 'metrics.json'))\nprint('Filas:', metrics.get('n_rows'))\n"
             "cv = pd.DataFrame(metrics['cv_summary'])\ndisplay(cv.pivot(index='learner', columns='outcome', values='mean'))\n"
             "print('Seleccionados:', metrics['selected'])\nshow('cv_model_selection')"),
        md("## 2. Evaluación en test"),
        code("tm = pd.read_csv(cfg.tables_dir / 'test_metrics.csv')\n"
             "display(tm.pivot_table(index=['outcome','learner'], columns='treatment', values=['qini_relative','auuc','uplift_at_10']))\n"
             "for o, v in metrics['results'].items():\n    sel = metrics['selected'][o]\n"
             "    for a in [t.label for t in cfg.treatments]:\n        for key in ('baseline', sel):\n"
             "            print(f\"{o:10s} {a:7s} {key:18s} Qini rel={v[key][a]['qini_relative']:.3f} IC95%={np.round(v[key][a]['qini_relative_ci95'], 3)}\")\n"
             "show('qini_curves')\nshow('uplift_by_decile')"),
        md("## 3. Interpretación del CATE"),
        code("if RUN or not (FIG / 'surrogate_tree.png').exists():\n    from uplift import figures\n    figures.run(cfg)\n"
             "show('cate_distribution')\nshow('cate_segments')\nshow('surrogate_tree')"),
        md("## 4. Política bajo presupuesto (IPW, con IC 95 %)"),
        code("rank = list(cfg.business['values'])[0] if cfg.business['values'] else cfg.primary_outcome\n"
             "display(pd.read_csv(cfg.tables_dir / f'policy_benchmark_{rank}.csv'))\nshow('budget_curves')\nshow('policy_benchmark')"),
        md("## 5. Despliegue: scoring vía API"),
        code("from fastapi.testclient import TestClient\nfrom api.main import app\nfrom uplift.data import load_split\n"
             "client = TestClient(app)\ncust = load_split('test', cfg)[cfg.raw_features].head(2).to_dict(orient='records')\n"
             "client.post(f'/{cfg.name}/predict', json={'customers': cust}).json()"),
        md("## 6. Monitoreo de drift (ventanas simuladas)"),
        code("from uplift import monitoring\nmonitoring.run(cfg)"),
    ]
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hillstrom")
    cfg = load_config(ap.parse_args().dataset)
    out = ROOT / "notebooks" / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    for fname, nb in (("01_eda.ipynb", eda_notebook(cfg)), ("02_modelado.ipynb", modeling_notebook(cfg))):
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nbf.write(nb, out / fname)
        print("Notebook:", out / fname)


if __name__ == "__main__":
    main()
