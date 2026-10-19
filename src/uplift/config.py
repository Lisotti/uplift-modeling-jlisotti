"""Configuración global del proyecto (paths y semillas).

Todo lo específico de cada dataset (columnas, tratamiento, outcomes, features,
supuestos de negocio) está en ``configs/<dataset>.yaml``; ver ``uplift.dataset``.
"""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIGS_DIR = ROOT_DIR / "configs"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MLRUNS_DIR = ROOT_DIR / "mlruns"

RANDOM_STATE = 42
