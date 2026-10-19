"""API REST de scoring de uplift, multi-dataset (Objetivo específico 6).

Ejecutar:
    make api            # equivale a: uvicorn api.main:app --reload --port 8000
Documentación interactiva: http://localhost:8000/docs

Por cada dataset de ``configs/`` con un modelo entrenado se publican:
    GET  /{dataset}/model/info
    POST /{dataset}/predict
El esquema de entrada (features, rangos y categorías) se genera a partir de la
config del dataset. ``/predict`` y ``/model/info`` sin prefijo usan el dataset
por defecto (variable de entorno ``UPLIFT_DATASET``, por defecto ``hillstrom``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, create_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uplift import __version__  # noqa: E402
from uplift.dataset import DEFAULT_DATASET, DatasetConfig, available_configs, load_config  # noqa: E402
from uplift.predict import load_bundle, objectives, score  # noqa: E402

app = FastAPI(
    title="Uplift Modeling API",
    description=(
        "Estima el efecto causal incremental (CATE) de cada tratamiento sobre cada outcome "
        "y recomienda a quién contactar y con qué tratamiento. Un endpoint por dataset."
    ),
    version=__version__,
)


def _trained() -> list[str]:
    return [d for d in available_configs() if load_config(d).bundle_path.exists()]


def customer_model(cfg: DatasetConfig) -> type[BaseModel]:
    """Modelo Pydantic del cliente generado desde las features de la config."""
    ranges = cfg.validation.get("ranges") or {}
    fields = {}
    example = {}
    for s in cfg.feature_specs:
        name, kind = s["name"], s["type"]
        if kind == "interaction":
            continue
        if kind == "binary":
            fields[name] = (Literal[0, 1], Field(..., description=s.get("description", name)))
            example[name] = 1
        elif kind == "categorical":
            fields[name] = (Literal[tuple(s["categories"])], Field(...))
            example[name] = s["categories"][0]
        else:
            lo, hi = (ranges.get(name) or [None, None])
            fields[name] = (float, Field(..., ge=lo, le=hi, description=s.get("description", name)))
            example[name] = lo if lo is not None else 0.0
    model = create_model(f"Customer_{cfg.name}", **fields)
    model.model_config["json_schema_extra"] = {"example": example}
    return model


def request_model(cfg: DatasetConfig, customer: type[BaseModel]) -> type[BaseModel]:
    objs = tuple(objectives(cfg))
    fields = {
        "customers": (list[customer], Field(..., min_length=1, max_length=10_000)),
        "objective": (Literal[objs], objs[0]),
        "cost_per_contact": (Optional[float], Field(None, ge=0, description="Costo por contacto (USD)")),
    }
    for o, v in cfg.business["values"].items():
        fields[f"value_{o}"] = (Optional[float], Field(None, ge=0, description=v["label"]))
    return create_model(f"ScoringRequest_{cfg.name}", **fields)


class Prediction(BaseModel):
    uplift: dict[str, dict[str, float]]
    recommended_treatment: str
    expected_value: float
    expected_profit_usd: Optional[float] = None


class ScoringResponse(BaseModel):
    dataset: str
    model_version: str
    objective: str
    predictions: list[Prediction]


def _bundle(cfg: DatasetConfig):
    try:
        return load_bundle(cfg)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


def _predict(cfg: DatasetConfig, req) -> ScoringResponse:
    b = _bundle(cfg)
    df = pd.DataFrame([c.model_dump() for c in req.customers])
    values = {o: getattr(req, f"value_{o}") for o in cfg.business["values"]}
    values = {o: (v if v is not None else cfg.value_defaults[o]) for o, v in values.items()}
    res = score(df, cfg, b, req.objective, values, req.cost_per_contact)
    preds = []
    for _, r in res.iterrows():
        preds.append(Prediction(
            uplift={o: {a.label: float(r[f"uplift_{o}_{a.label}"]) for a in cfg.treatments}
                    for o in cfg.outcome_names},
            recommended_treatment=r["recommended_treatment"],
            expected_value=float(r["expected_value"]),
            expected_profit_usd=float(r["expected_profit_usd"]) if "expected_profit_usd" in r else None,
        ))
    return ScoringResponse(dataset=cfg.name, model_version=b["version"], objective=req.objective, predictions=preds)


def _info(cfg: DatasetConfig) -> dict:
    b = _bundle(cfg)
    return {
        "dataset": cfg.name, "title": cfg.title, "version": b["version"], "trained_at": b["trained_at"],
        "rows": b.get("n_rows"), "selected_learners": b["selected"], "features": b["feature_names"],
        "treatments": [a.label for a in cfg.treatments], "outcomes": cfg.outcome_names,
        "objectives": objectives(cfg), "test_metrics": b["test_metrics"],
    }


def _register(cfg: DatasetConfig, prefix: str, tag: str) -> None:
    Customer = customer_model(cfg)
    Req = request_model(cfg, Customer)

    def predict(req):
        return _predict(cfg, req)

    # anotación explícita: el modelo de la request se crea dinámicamente por dataset
    predict.__annotations__ = {"req": Req, "return": ScoringResponse}

    def info() -> dict:
        return _info(cfg)

    app.post(f"{prefix}/predict", response_model=ScoringResponse, tags=[tag],
             summary=f"Scoring de uplift — {cfg.title}")(predict)
    app.get(f"{prefix}/model/info", tags=[tag], summary=f"Modelo — {cfg.name}")(info)


@app.get("/health")
def health():
    trained = _trained()
    return {"status": "ok" if trained else "degraded", "model_loaded": bool(trained), "datasets": trained}


@app.get("/datasets")
def datasets():
    return [{"name": d, "title": load_config(d).title, "trained": load_config(d).bundle_path.exists(),
             "endpoint": f"/{d}/predict"} for d in available_configs()]


for _name in available_configs():
    _register(load_config(_name), f"/{_name}", _name)
_register(load_config(DEFAULT_DATASET), "", f"default ({DEFAULT_DATASET})")
