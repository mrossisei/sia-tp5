"""Re-grafica la grilla optimizador x lr como UN solo heatmap (max pixel-error,
media +- std) desde exp_grid_full_runs.npz, sin reentrenar. Mas legible que el
doble panel para la presentacion; el panel de epoca de convergencia se omite.

Uso: python3 -m ej1.experiments.replot_grid_maxpix
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.plotting import save_fig

OUT = os.path.join(REPO_ROOT, "ej1", "results", "basic")
OPTS = ["adam", "momentum", "gd"]
LRS = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]


def main():
    d = np.load(os.path.join(OUT, "exp_grid_full_runs.npz"), allow_pickle=True)
    opt = d["opt"]; lr = d["lr"]; mx = d["max_pixel"].astype(float)

    M = np.full((len(OPTS), len(LRS)), np.nan)
    ann = [["" for _ in LRS] for _ in OPTS]
    for i, o in enumerate(OPTS):
        for j, l in enumerate(LRS):
            m = (opt == o) & (np.isclose(lr, l))
            vals = mx[m]
            M[i, j] = float(vals.mean())
            ann[i][j] = f"{vals.mean():.1f}$\\pm${vals.std():.1f}"

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    im = ax.imshow(M, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(LRS))); ax.set_xticklabels([f"{l:g}" for l in LRS])
    ax.set_yticks(range(len(OPTS))); ax.set_yticklabels(OPTS)
    ax.set_xlabel("learning rate"); ax.set_ylabel("optimizador")
    ax.set_title("max pixel-error final (media $\\pm$ std sobre 5 seeds, 30000 ep)",
                 fontsize=11)
    vmax = np.nanmax(M)
    for i in range(len(OPTS)):
        for j in range(len(LRS)):
            color = "white" if M[i, j] > vmax * 0.5 else "black"
            ax.text(j, i, ann[i][j], ha="center", va="center", fontsize=9, color=color)
    fig.colorbar(im, ax=ax, label="max pixel-error")
    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, "exp_grid_full_maxpix.png"))
    print("[ok] exp_grid_full_maxpix.png")


if __name__ == "__main__":
    main()
