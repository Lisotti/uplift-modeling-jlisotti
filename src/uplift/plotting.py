"""Estilo común de las figuras (EDA, resultados e informe).

Paleta categórica validada (CVD ΔE ≥ 12,9; contraste y banda de luminosidad OK):
teal #00897B · oro #C08A00 · azul #3D6CC9 · coral #D9472B. Gris para control/azar.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TEAL, GOLD, BLUE, CORAL = "#00897B", "#C08A00", "#3D6CC9", "#D9472B"
GRAY, LIGHT, INK, MUTED = "#8A8A8A", "#C9CED3", "#1F2933", "#5F6B76"
TREATMENT_COLORS = [TEAL, GOLD, BLUE, CORAL]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "axes.edgecolor": "#B0B7BE", "axes.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#E6E9EC", "grid.linewidth": 0.7, "xtick.color": MUTED,
    "ytick.color": MUTED, "legend.frameon": False, "figure.dpi": 110,
    "savefig.dpi": 200, "savefig.bbox": "tight", "lines.linewidth": 2,
})


def arm_colors(cfg) -> dict[str, str]:
    """Color por etiqueta de brazo: control gris y tratamientos en orden fijo."""
    out = {cfg.control.label: GRAY}
    for i, a in enumerate(cfg.treatments):
        out[a.label] = TREATMENT_COLORS[i % len(TREATMENT_COLORS)]
    return out


def save(fig, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
