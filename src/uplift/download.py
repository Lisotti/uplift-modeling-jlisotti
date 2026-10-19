"""Descarga el archivo de datos de un dataset desde ``source.url`` de su config.

Uso:
    python -m uplift.download criteo
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request

from .dataset import load_config


def download(dataset: str) -> None:
    cfg = load_config(dataset)
    dest = cfg.raw["source"]["path"]
    from .config import ROOT_DIR

    dest = ROOT_DIR / dest
    if dest.exists():
        print(f"Ya existe {dest}")
        return
    url = cfg.source_url
    if not url:
        raise SystemExit(f"La config de {dataset} no tiene source.url")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Descargando {url} → {dest} …")
    try:
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:  # noqa: S310 (URL de la config)
            shutil.copyfileobj(r, f, length=1 << 20)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        manual = cfg.raw["source"].get("manual_url")
        hint = (f"\nDescargalo a mano desde {manual}, descomprimí el CSV y guardalo en {dest.parent}/"
                if manual else "")
        raise SystemExit(f"No se pudo descargar {url}: {e}{hint}") from e
    tmp.rename(dest)
    print("Listo.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    download(ap.parse_args().dataset)


if __name__ == "__main__":
    main()
