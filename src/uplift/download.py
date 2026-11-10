"""Obtiene el archivo de datos de un dataset según ``source`` de su config.

* ``source.kaggle: <owner>/<dataset>`` (Criteo): descarga con el CLI de Kaggle si está
  instalado y con el token configurado; si no, imprime los pasos para descargarlo a mano.
* ``source.url``: descarga directa por HTTP.

Uso:
    python -m uplift.download criteo
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

from .config import ROOT_DIR
from .dataset import load_config


def _manual_steps(slug: str, cfg) -> str:
    pattern = cfg.raw["source"].get("glob", Path(cfg.raw["source"]["path"]).name)
    return (
        f"Descarga manual desde Kaggle (requiere cuenta):\n"
        f"  1. Entrar a https://www.kaggle.com/datasets/{slug} y hacer clic en \"Download\".\n"
        f"  2. Descomprimir el .zip.\n"
        f"  3. Copiar el CSV a data/raw/ con un nombre que coincida con '{pattern}' "
        f"(por ejemplo {cfg.raw['source']['path']}).\n"
        f"Alternativa automática: instalar el CLI de Kaggle (pipx install kaggle), configurar el token "
        f"en ~/.kaggle/kaggle.json (https://www.kaggle.com/docs/api) y volver a correr este comando."
    )


def _from_kaggle(slug: str, cfg) -> None:
    kaggle = shutil.which("kaggle")
    has_token = (Path.home() / ".kaggle" / "kaggle.json").exists() or os.environ.get("KAGGLE_USERNAME")
    if not (kaggle and has_token):
        print(_manual_steps(slug, cfg))
        raise SystemExit(2)
    raw = ROOT_DIR / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    print(f"Descargando {slug} desde Kaggle → {raw} …")
    if subprocess.call([kaggle, "datasets", "download", "-d", slug, "-p", str(raw), "--unzip"]):
        raise SystemExit("Falló la descarga con el CLI de Kaggle.\n\n" + _manual_steps(slug, cfg))
    if not cfg.source_path.exists():
        raise SystemExit("La descarga no dejó un archivo con el nombre esperado en data/raw/.\n\n"
                         + _manual_steps(slug, cfg))
    print(f"Listo: {cfg.source_path}")


def _from_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Descargando {url} → {dest} …")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:  # noqa: S310 (URL de la config)
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.rename(dest)
    print("Listo.")


def download(dataset: str) -> None:
    cfg = load_config(dataset)
    if cfg.source_path.exists():
        print(f"Ya existe {cfg.source_path}")
        return
    src = cfg.raw["source"]
    if src.get("kaggle"):
        _from_kaggle(src["kaggle"], cfg)
    elif cfg.source_url:
        _from_url(cfg.source_url, ROOT_DIR / src["path"])
    else:
        raise SystemExit(f"La config de {dataset} no define source.kaggle ni source.url: "
                         f"copiar el archivo a mano en {src['path']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    download(ap.parse_args().dataset)


if __name__ == "__main__":
    main()
