"""EJ1.a.2 — Interacción optimizador × learning rate (heatmap, estilo TP3).

El learning rate óptimo DEPENDE del optimizador (Adam adapta el paso por
parámetro; GD no), así que barrer cada dimensión por separado puede engañar.
Acá la grilla completa 3 optimizadores × 5 learning rates (seed fija, 6000
épocas, arquitectura ganadora, BCE) con dos métricas:
  - max pixel-error final
  - época de convergencia (primera con max<=1; ✗ = no converge en 6000)

Genera (en ej1/results/basic/):
  - exp_opt_lr_heatmap.png
  - exp_opt_lr.csv
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
from shared.plotting import save_fig
from ej1.experiments.optimization_study import train  # reusa el runner

OUT = os.path.join(REPO_ROOT, "ej1", "results", "basic")
ARCH = [35, 60, 40, 20, 2, 20, 40, 60, 35]
OPTS = ["adam", "momentum", "gd"]
LRS = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
EPOCHS = 6000
SEED = 0


def _annotated_heatmap(ax, M, fmt, title, cbar_label, fig, mask_nan="✗"):
    masked = np.ma.masked_invalid(M)
    im = ax.imshow(masked, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(LRS)))
    ax.set_xticklabels([f"{lr:g}" for lr in LRS])
    ax.set_yticks(range(len(OPTS)))
    ax.set_yticklabels(OPTS)
    ax.set_xlabel("learning rate")
    ax.set_ylabel("optimizador")
    ax.set_title(title, fontsize=10)
    vmax = np.nanmax(M) if np.isfinite(M).any() else 1.0
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                ax.text(j, i, mask_nan, ha="center", va="center",
                        fontsize=10, color="red", fontweight="bold")
            else:
                color = "white" if v > vmax * 0.5 else "black"
                ax.text(j, i, fmt(v), ha="center", va="center",
                        fontsize=8, color=color)
    fig.colorbar(im, ax=ax, label=cbar_label)


def main():
    X, _ = load_font()
    max_px = np.full((len(OPTS), len(LRS)), np.nan)
    conv_ep = np.full((len(OPTS), len(LRS)), np.nan)

    t0 = time.time()
    rows = []
    for i, opt_name in enumerate(OPTS):
        for j, lr in enumerate(LRS):
            t = time.time()
            _, _, s, ce = train(X, ARCH, "tanh", "identity", "bce",
                                opt_name, lr, epochs=EPOCHS, seed=SEED)
            max_px[i, j] = s["max"]
            conv_ep[i, j] = float(ce) if ce is not None else np.nan
            rows.append([opt_name, lr, s["max"], s["n_exact"],
                         "" if ce is None else int(ce)])
            print(f"[{opt_name} lr={lr:g}] max={s['max']} conv={ce} "
                  f"({time.time()-t:.0f}s)", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    _annotated_heatmap(axes[0], max_px, lambda v: f"{int(v)}",
                       "max pixel-error final", "max pixel-error", fig)
    _annotated_heatmap(axes[1], conv_ep, lambda v: f"{int(v)}",
                       "época de convergencia (max≤1; ✗ = no converge)",
                       "época", fig)
    fig.suptitle(f"Interacción optimizador × learning rate "
                 f"({EPOCHS} épocas, seed={SEED}, BCE)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, os.path.join(OUT, "exp_opt_lr_heatmap.png"))

    with open(os.path.join(OUT, "exp_opt_lr.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["optimizador", "lr", "max_pixel", "n_exact", "conv_epoch"])
        w.writerows(rows)

    print(f"[ok] opt_lr_heatmap completo en {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
