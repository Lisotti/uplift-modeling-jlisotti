"""Tests de integración de la API (requieren el modelo entrenado: `make train`)."""

from __future__ import annotations

import pytest

from uplift.dataset import load_config

CFG = load_config("hillstrom")
pytestmark = pytest.mark.skipif(not CFG.bundle_path.exists(), reason="modelo no entrenado")

CUSTOMER = {"recency": 2, "history": 350.0, "mens": 1, "womens": 1,
            "zip_code": "Surburban", "newbie": 0, "channel": "Web"}


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from api.main import app

    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_model_info(client):
    r = client.get("/model/info").json()
    assert set(r["selected_learners"]) == set(CFG.outcome_names)


def test_predict_single(client):
    r = client.post("/predict", json={"customers": [CUSTOMER]})
    assert r.status_code == 200
    p = r.json()["predictions"][0]
    assert p["recommended_treatment"] in {"mens", "womens", "no_email"}
    assert set(p["uplift"]["visit"]) == {"mens", "womens"}


def test_predict_batch_and_objective(client):
    body = {"customers": [CUSTOMER, {**CUSTOMER, "mens": 0, "channel": "Phone"}], "objective": "visit"}
    r = client.post("/predict", json=body)
    assert r.status_code == 200 and len(r.json()["predictions"]) == 2


def test_predict_validation_error(client):
    r = client.post("/predict", json={"customers": [{**CUSTOMER, "recency": 40}]})
    assert r.status_code == 422


def test_datasets_and_prefixed_routes(client):
    names = {d["name"] for d in client.get("/datasets").json()}
    assert {"hillstrom", "criteo"} <= names
    r = client.post("/hillstrom/predict", json={"customers": [CUSTOMER], "objective": "visit",
                                                "value_spend": 0.75, "cost_per_contact": 0.12})
    assert r.status_code == 200 and r.json()["objective"] == "visit"
