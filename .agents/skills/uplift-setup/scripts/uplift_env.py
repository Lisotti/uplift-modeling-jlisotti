#!/usr/bin/env python3
"""Setup y operación local del repo de Uplift Modeling (macOS, Linux y Windows).

Sólo usa la librería estándar: se puede correr antes de instalar las dependencias.

    python .agents/skills/uplift-setup/scripts/uplift_env.py <comando> [opciones]

Comandos:
    check                          diagnóstico: Python, Poetry, libomp, datos y artefactos por dataset
    install [--python RUTA]        poetry env use + poetry install (Poetry tiene que estar instalado)
    up [api|dashboard|mlflow|all]  levanta servicios en segundo plano (logs en .uplift/logs/)
    status                         qué servicios están corriendo y si responden
    down [api|dashboard|mlflow|all] detiene servicios levantados con `up`
    download criteo                datos de Criteo desde Kaggle (CLI de Kaggle o instrucciones manuales)
    pipeline DATASET [--steps eda,train,monitor,figures] [-- ARGS de train]  corre pasos del pipeline
    test                           corre la suite de tests

Todo se ejecuta desde la raíz del repo, con ``poetry run`` y PYTHONPATH armado según el sistema operativo.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
STATE = ROOT / ".uplift"
LOGS = STATE / "logs"
IS_WIN = os.name == "nt"

SERVICES = {
    "api": {
        "cmd": ["uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000"],
        "url": "http://127.0.0.1:8000/health",
        "open": "http://localhost:8000/docs",
    },
    "dashboard": {
        "cmd": ["streamlit", "run", "dashboard/app.py", "--server.headless", "true",
                "--server.port", "8501", "--browser.gatherUsageStats", "false"],
        "url": "http://127.0.0.1:8501/_stcore/health",
        "open": "http://localhost:8501",
    },
    "mlflow": {
        # --workers sólo existe en el servidor gunicorn (no en Windows)
        "cmd": ["mlflow", "ui", "--backend-store-uri", "./mlruns", "--host", "127.0.0.1", "--port", "8080"]
        + ([] if IS_WIN else ["--workers", "1", "--gunicorn-opts", "--timeout 300"]),
        "url": "http://127.0.0.1:8080/health",
        "open": "http://127.0.0.1:8080",
    },
}
STEPS = {"eda": "uplift.eda", "train": "uplift.train", "monitor": "uplift.monitoring", "figures": "uplift.figures"}


# ── utilidades ────────────────────────────────────────────────────────────────

def env() -> dict:
    e = os.environ.copy()
    e["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT)])
    e.setdefault("PYTHONUTF8", "1")
    return e


def poetry() -> str:
    exe = shutil.which("poetry")
    if not exe:
        sys.exit("ERROR: Poetry no está instalado o no está en el PATH. Ver la sección 'Instalar Poetry' del SKILL.md.")
    return exe


def run(cmd: list[str]) -> int:
    print("$", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=ROOT, env=env())


def datasets() -> list[str]:
    return sorted(p.stem for p in (ROOT / "configs").glob("*.yaml"))


def dataset_status(name: str) -> dict:
    art = ROOT / "artifacts" / name
    return {
        "raw_data": bool(list((ROOT / "data" / "raw").glob(f"{name}*"))),
        "model": (art / "models" / "uplift_bundle.joblib").exists(),
        "test_split": any((art / "data").glob("test.*")),
        "eda": (art / "reports" / "eda_summary.json").exists(),
        "monitoring": (art / "reports" / "monitoring" / "monitoring_summary.csv").exists(),
    }


def responds(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 (localhost)
            return r.status < 500
    except Exception:
        return False


def pid_file(svc: str) -> Path:
    return STATE / f"{svc}.pid"


def pid_alive(pid: int) -> bool:
    if IS_WIN:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_pid(svc: str) -> int | None:
    p = pid_file(svc)
    if not p.exists():
        return None
    pid = int(p.read_text().strip() or 0)
    return pid if pid and pid_alive(pid) else None


def which_services(arg: str) -> list[str]:
    return list(SERVICES) if arg == "all" else [arg]


# ── comandos ──────────────────────────────────────────────────────────────────

def cmd_check(_):
    report = {
        "os": f"{platform.system()} {platform.release()}",
        "python_actual": platform.python_version(),
        "python_ok": (3, 11) <= sys.version_info[:2] <= (3, 12),
        "poetry": None,
        "venv": (ROOT / ".venv").exists(),
        "datasets": {d: dataset_status(d) for d in datasets()},
        "servicios_corriendo": {s: bool(read_pid(s)) for s in SERVICES},
    }
    exe = shutil.which("poetry")
    if exe:
        report["poetry"] = subprocess.run([exe, "--version"], capture_output=True, text=True).stdout.strip()
    if platform.system() == "Darwin":
        report["libomp"] = any(Path(p).exists() for p in (
            "/opt/homebrew/opt/libomp/lib/libomp.dylib", "/usr/local/opt/libomp/lib/libomp.dylib"))
    others = [f"python{v}" for v in ("3.11", "3.12")] + (["py"] if IS_WIN else [])
    report["otros_python"] = [o for o in others if shutil.which(o)]
    print(json.dumps(report, indent=2, ensure_ascii=False))
    missing = [d for d, s in report["datasets"].items() if s["model"] and not s["test_split"]]
    if missing:
        print(f"\nAVISO: el dashboard necesita el split de test de todos los datasets con modelo. "
              f"Falta en: {', '.join(missing)} → correr `pipeline <dataset> --steps train`.")


def cmd_install(a):
    exe = poetry()
    if a.python:
        if run([exe, "env", "use", a.python]):
            sys.exit("ERROR en `poetry env use`.")
    sys.exit(run([exe, "install", "--no-interaction"]))


def cmd_up(a):
    exe = poetry()
    LOGS.mkdir(parents=True, exist_ok=True)
    targets = which_services(a.service)
    if "dashboard" in targets:
        missing = [d for d in datasets() if dataset_status(d)["model"] and not dataset_status(d)["test_split"]]
        if missing and not a.force:
            print(f"NO SE LEVANTA el dashboard: falta el split de test de {', '.join(missing)}. "
                  f"Correr antes `pipeline <dataset> --steps train` (o usar --force).")
            targets = [t for t in targets if t != "dashboard"]
    if "mlflow" in targets and not any((ROOT / "mlruns").glob("*/meta.yaml")):
        print("AVISO: mlruns/ está vacío; MLflow va a abrir sin experimentos hasta correr un `train`.")
    started = []
    for svc in targets:
        if read_pid(svc):
            print(f"{svc}: ya estaba corriendo → {SERVICES[svc]['open']}")
            continue
        log = open(LOGS / f"{svc}.log", "w")
        kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008} if IS_WIN else {"start_new_session": True}
        proc = subprocess.Popen([exe, "run", *SERVICES[svc]["cmd"]], cwd=ROOT, env=env(),
                                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **kw)
        pid_file(svc).write_text(str(proc.pid))
        started.append((svc, proc))
    for svc, proc in started:
        deadline = time.time() + a.timeout
        while time.time() < deadline and proc.poll() is None and not responds(SERVICES[svc]["url"]):
            time.sleep(1)
        if responds(SERVICES[svc]["url"]):
            print(f"{svc}: OK → {SERVICES[svc]['open']}")
        else:
            print(f"{svc}: NO RESPONDE (ver {LOGS / (svc + '.log')})")
            tail = (LOGS / f"{svc}.log").read_text(errors="replace").splitlines()[-15:]
            print("\n".join("    " + t for t in tail))


def cmd_status(_):
    for svc, s in SERVICES.items():
        pid = read_pid(svc)
        state = "corriendo" if pid else "detenido"
        ok = responds(s["url"])
        print(f"{svc:10s} {state:10s} {'responde' if ok else 'no responde':12s} {s['open']}")


def cmd_down(a):
    for svc in which_services(a.service):
        pid = read_pid(svc)
        if not pid:
            print(f"{svc}: no estaba corriendo")
        elif IS_WIN:
            subprocess.call(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.DEVNULL)
            print(f"{svc}: detenido")
        else:
            os.killpg(pid, signal.SIGTERM)
            print(f"{svc}: detenido")
        pid_file(svc).unlink(missing_ok=True)


def cmd_download(a):
    # Criteo: Kaggle es la única fuente; uplift.download usa el CLI de Kaggle si está configurado
    # o imprime los pasos manuales y termina con código 2.
    sys.exit(run([poetry(), "run", "python", "-m", "uplift.download", a.dataset]))


def cmd_pipeline(a):
    if a.dataset not in datasets():
        sys.exit(f"ERROR: no existe configs/{a.dataset}.yaml. Disponibles: {', '.join(datasets())}")
    steps = a.steps.split(",")
    bad = [s for s in steps if s not in STEPS]
    if bad:
        sys.exit(f"ERROR: pasos desconocidos {bad}. Válidos: {', '.join(STEPS)}")
    if not dataset_status(a.dataset)["raw_data"] and ({"eda", "train"} & set(steps)):
        sys.exit(f"ERROR: faltan los datos crudos de {a.dataset} en data/raw/. Ver `download {a.dataset}`.")
    exe = poetry()
    for step in steps:
        t0 = time.time()
        extra = a.extra if step == "train" else []  # --sample-size, --folds, etc. son de train
        if run([exe, "run", "python", "-m", STEPS[step], "--dataset", a.dataset, *extra]):
            sys.exit(f"ERROR en el paso {step}.")
        print(f"[{step}] OK en {time.time() - t0:.0f}s", flush=True)


def cmd_test(_):
    sys.exit(run([poetry(), "run", "pytest", "-q"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    p = sub.add_parser("install"); p.add_argument("--python"); p.set_defaults(fn=cmd_install)
    choices = [*SERVICES, "all"]
    p = sub.add_parser("up"); p.add_argument("service", nargs="?", default="all", choices=choices)
    p.add_argument("--timeout", type=int, default=90); p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_up)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    p = sub.add_parser("down"); p.add_argument("service", nargs="?", default="all", choices=choices)
    p.set_defaults(fn=cmd_down)
    p = sub.add_parser("download"); p.add_argument("dataset"); p.set_defaults(fn=cmd_download)
    p = sub.add_parser("pipeline"); p.add_argument("dataset")
    p.add_argument("--steps", default="eda,train,monitor,figures")
    p.set_defaults(fn=cmd_pipeline)
    sub.add_parser("test").set_defaults(fn=cmd_test)
    # todo lo que va después de "--" se pasa tal cual a train (p. ej. -- --sample-size 500000)
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    a = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
    a.extra = extra
    a.fn(a)


if __name__ == "__main__":
    main()
