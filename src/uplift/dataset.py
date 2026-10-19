"""Configuración declarativa por dataset (``configs/<nombre>.yaml``).

Todo lo específico de un experimento —columnas, brazos del tratamiento, outcomes y
su tipo, feature engineering, reglas de validación, supuestos de negocio y
escenario de drift— vive en un YAML. El resto del pipeline es genérico: dado un
dataset X con su config se generan EDA, modelos, API y dashboard.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from pathlib import Path

import yaml

from .config import ARTIFACTS_DIR, CONFIGS_DIR, ROOT_DIR

DEFAULT_DATASET = os.environ.get("UPLIFT_DATASET", "hillstrom")


@dataclass(frozen=True)
class Arm:
    value: object      # valor en la columna de tratamiento del dataset
    label: str         # etiqueta corta (columnas, API)
    name: str          # nombre legible (tablas, gráficos)


@dataclass(frozen=True)
class Outcome:
    name: str
    type: str          # "binary" | "continuous"
    label: str
    hurdle: bool = False

    @property
    def binary(self) -> bool:
        return self.type == "binary"


@dataclass
class DatasetConfig:
    raw: dict = field(repr=False)

    # ── Identidad ──────────────────────────────────────────────────
    @property
    def name(self) -> str:
        return self.raw["name"]

    @property
    def title(self) -> str:
        return self.raw.get("title", self.name)

    @property
    def description(self) -> str:
        return self.raw.get("description", "").strip()

    # ── Fuente ─────────────────────────────────────────────────────
    @property
    def source_path(self) -> Path:
        """Archivo de datos; si no está en ``path`` busca alternativas con ``source.glob``."""
        p = ROOT_DIR / self.raw["source"]["path"]
        pattern = self.raw["source"].get("glob")
        if not p.exists() and pattern:
            found = sorted((ROOT_DIR / "data" / "raw").rglob(pattern))
            found = [f for f in found if f.is_file() and not f.name.endswith((".part", ".crdownload"))]
            if found:
                return found[0]
        return p

    @property
    def source_url(self) -> str | None:
        return self.raw["source"].get("url")

    @property
    def manual_url(self) -> str | None:
        """Página para descargar los datos a mano (p. ej. Kaggle) si la descarga automática falla."""
        return self.raw["source"].get("manual_url")

    @property
    def dtypes(self) -> dict | None:
        return self.raw["source"].get("dtypes")

    # ── Tratamiento ────────────────────────────────────────────────
    @property
    def treatment_col(self) -> str:
        return self.raw["treatment"]["column"]

    @cached_property
    def control(self) -> Arm:
        return Arm(**self.raw["treatment"]["control"])

    @cached_property
    def treatments(self) -> list[Arm]:
        return [Arm(**a) for a in self.raw["treatment"]["arms"]]

    @property
    def arms(self) -> list[Arm]:
        """Control primero: el índice en esta lista es el código del tratamiento (0 = control)."""
        return [self.control, *self.treatments]

    @property
    def n_treatments(self) -> int:
        return len(self.treatments)

    @property
    def none_action(self) -> str:
        return self.raw["treatment"].get("none_action", "no_contact")

    @property
    def contact_noun(self) -> str:
        return self.raw["treatment"].get("contact_noun", "contacto")

    # ── Outcomes ───────────────────────────────────────────────────
    @cached_property
    def outcomes(self) -> list[Outcome]:
        return [Outcome(**o) for o in self.raw["outcomes"]]

    @property
    def outcome_names(self) -> list[str]:
        return [o.name for o in self.outcomes]

    def outcome(self, name: str) -> Outcome:
        return next(o for o in self.outcomes if o.name == name)

    @property
    def primary_outcome(self) -> str:
        return self.raw.get("primary_outcome", self.outcome_names[0])

    @property
    def stratify_outcome(self) -> str:
        return self.raw.get("stratify_outcome", self.primary_outcome)

    # ── Features ───────────────────────────────────────────────────
    @property
    def feature_specs(self) -> list[dict]:
        return self.raw["features"]

    @property
    def raw_features(self) -> list[str]:
        """Columnas del dataset que necesita el modelo (entrada de la API)."""
        return [f["name"] for f in self.feature_specs if f["type"] != "interaction"]

    @property
    def excluded_columns(self) -> dict:
        return self.raw.get("excluded_columns", {}) or {}

    def spec(self, name: str) -> dict:
        return next(f for f in self.feature_specs if f["name"] == name)

    # ── Validación, modelado, negocio ──────────────────────────────
    @property
    def validation(self) -> dict:
        return self.raw.get("validation", {}) or {}

    @property
    def modeling(self) -> dict:
        defaults = dict(sample_size=None, test_size=0.3, cv_folds=5, cv_repeats=3, n_boot=200, n_jobs=2)
        return {**defaults, **(self.raw.get("modeling") or {})}

    @property
    def business(self) -> dict:
        b = self.raw.get("business") or {}
        return {"cost_per_contact": 0.0, "cost_label": "Costo por contacto (USD)", "values": {},
                "default_base_size": 100_000, **b}

    @property
    def value_defaults(self) -> dict[str, float]:
        return {o: v["default"] for o, v in self.business["values"].items()}

    @property
    def segments(self) -> dict:
        return self.raw.get("segments") or {}

    @property
    def figures(self) -> dict:
        return self.raw.get("figures") or {}

    @property
    def monitoring(self) -> dict:
        return {"n_windows": 4, "drift_scenario": [], **(self.raw.get("monitoring") or {})}

    # ── Paths de artefactos ────────────────────────────────────────
    @property
    def artifacts_dir(self) -> Path:
        return ARTIFACTS_DIR / self.name

    @property
    def data_dir(self) -> Path:
        return self.artifacts_dir / "data"

    @property
    def models_dir(self) -> Path:
        return self.artifacts_dir / "models"

    @property
    def reports_dir(self) -> Path:
        return self.artifacts_dir / "reports"

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.reports_dir / "tables"

    @property
    def monitoring_dir(self) -> Path:
        return self.reports_dir / "monitoring"

    @property
    def bundle_path(self) -> Path:
        return self.models_dir / "uplift_bundle.joblib"

    @property
    def mlflow_experiment(self) -> str:
        return f"uplift-{self.name}"

    def make_dirs(self) -> None:
        for d in (self.data_dir, self.models_dir, self.figures_dir, self.tables_dir, self.monitoring_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=16)
def load_config(name_or_path: str | Path = DEFAULT_DATASET) -> DatasetConfig:
    """Carga ``configs/<nombre>.yaml`` (o un path explícito a un YAML)."""
    p = Path(name_or_path)
    if p.suffix not in (".yaml", ".yml"):
        p = CONFIGS_DIR / f"{name_or_path}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"No existe la configuración {p}")
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return DatasetConfig(raw=raw)


def available_configs() -> list[str]:
    return sorted(p.stem for p in CONFIGS_DIR.glob("*.yaml"))
