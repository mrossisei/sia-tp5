"""EJ1.a.2 — Variabilidad entre seeds (error bars) de las configs clave.

El estudio principal (optimization_study.py) corre cada config con UNA seed;
acá repetimos las configs clave con 5 seeds para reportar media ± desvío
(estilo "sweep_seeds" de TP4): la conclusión no puede depender de la suerte
de la inicialización (los pesos iniciales salen de una normal con escala
Glorot, ver shared/initializers.py).

Configs: arquitectura ganadora [35,60,40,20,2,...], BCE, cuello identity.
  - Panel optimizadores: adam@1e-3, momentum@1e-2, gd@1e-2
  - Panel learning rates (adam): 1e-4, 5e-4, 1e-3, 5e-3

Genera (en ej1/results/basic/):
  - exp_multiseed.png : 2x2 paneles (max-pixel y época de convergencia,
    media ± std sobre seeds, anotando cuántas seeds convergen)
  - exp_multiseed.csv : tabla config x seed
"""

import os
import sys
import csv
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.fonts import load_font
from shared.plotting import save_fig, PALETTE
from ej1.experiments.optimization_study import train  # reusa el runner (6000 ép.)

OUT = os.path.join(REPO_ROOT, "ej1", "results", "basic")
ARCH = [35, 60, 40, 20, 2, 20, 40, 60, 35]
SEEDS = [0, 1, 2, 3, 42]
EPOCHS = 6000

# (etiqueta, optimizador, lr) — adam@1e-3 se comparte entre ambos paneles.
CONFIGS = [
    ("adam lr=1e-3", "adam", 1e-3),
    ("momentum lr=1e-2", "momentum", 1e-2),
    ("gd lr=1e-2", "gd", 1e-2),
    ("adam lr=1e-4", "adam", 1e-4),
    ("adam lr=5e-4", "adam", 5e-4),
    ("adam lr=5e-3", "adam", 5e-3),
]
PANEL_OPT = ["adam lr=1e-3", "momentum lr=1e-2", "gd lr=1e-2"]
PANEL_LR = ["adam lr=1e-4", "adam lr=5e-4", "adam lr=1e-3", "adam lr=5e-3"]


def run_all(X):
    """dict etiqueta -> lista de (max_pixel, conv_epoch|nan) por seed."""
    results = {lab: [] for lab, _, _ in CONFIGS}
    for lab, opt_name, lr in CONFIGS:
        for seed in SEEDS:
            t = time.time()
            _, _, s, ce = train(X, ARCH, "tanh", "identity", "bce",
                                opt_name, lr, epochs=EPOCHS, seed=seed)
            ce_val = float(ce) if ce is not None else np.nan
            results[lab].append((s["max"], ce_val))
            print(f"[{lab} seed={seed}] max={s['max']} conv={ce} "
                  f"({time.time()-t:.0f}s)", flush=True)
    return results


def _bars(ax, labels, results, idx, ylabel, title):
    """Barra media ± std de la métrica idx (0=max_pixel, 1=conv_epoch).

    OJO (sesgo de censura): para conv_epoch la media/std se calculan SOLO sobre
    las seeds que convergieron dentro de EPOCHS; una config parcialmente
    censurada (p.ej. adam@1e-4 a 6000 ép) subestima el tiempo real. La anotación
    'n/len(SEEDS)' al lado de cada barra expone cuántas convergieron; la versión
    des-censurada (30000 ép) está en grid_full.py.
    """
    means, stds, anns = [], [], []
    for lab in labels:
        vals = np.array([r[idx] for r in results[lab]], dtype=float)
        ok = vals[~np.isnan(vals)]
        n_conv = int(np.sum(~np.isnan(vals))) if idx == 1 else len(vals)
        m = float(np.mean(ok)) if len(ok) else np.nan
        sd = float(np.std(ok)) if len(ok) else 0.0
        means.append(m)
        stds.append(sd)
        anns.append(f"{n_conv}/{len(SEEDS)}" if idx == 1 else "")
    y = np.arange(len(labels))
    bars = ax.barh(y, means, xerr=stds, color=PALETTE["primary"],
                   edgecolor="k", alpha=0.85, capsize=4)
    for yi, (b, ann) in enumerate(zip(bars, anns)):
        if ann:
            ax.text(b.get_width() + stds[yi] + ax.get_xlim()[1] * 0.01, yi,
                    ann, va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()


def main():
    X, _ = load_font()
    t0 = time.time()
    results = run_all(X)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    _bars(axes[0, 0], PANEL_OPT, results, 0, "max pixel-error (media ± std)",
          "Optimizador: error final")
    _bars(axes[0, 1], PANEL_OPT, results, 1, "época de convergencia (max≤1)",
          "Optimizador: velocidad (n seeds que convergen)")
    _bars(axes[1, 0], PANEL_LR, results, 0, "max pixel-error (media ± std)",
          "Learning rate (Adam): error final")
    _bars(axes[1, 1], PANEL_LR, results, 1, "época de convergencia (max≤1)",
          "Learning rate (Adam): velocidad (n seeds que convergen)")
    fig.suptitle(f"Variabilidad entre {len(SEEDS)} seeds "
                 f"(arquitectura ganadora, BCE, {EPOCHS} épocas)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, os.path.join(OUT, "exp_multiseed.png"))

    csv_path = os.path.join(OUT, "exp_multiseed.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "seed", "max_pixel", "conv_epoch"])
        for lab, _, _ in CONFIGS:
            for seed, (mx, ce) in zip(SEEDS, results[lab]):
                w.writerow([lab, seed, mx, "" if np.isnan(ce) else int(ce)])

    print(f"[ok] multiseed_study completo en {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
