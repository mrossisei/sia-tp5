"""Lee exp_latent_dim.csv y regenera los gráficos sin reentrenar.

Solo el gráfico de barras MSE es posible desde el CSV.
La figura de reconstrucciones necesita los VAEs entrenados (no se guardan).

Uso:
    python3 ej2/experiments/latent_dim_2.py
"""

import os, sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from shared.plotting import save_fig, PALETTE
from ej2.experiments._common import RESULTS_DIR

CSV_PATH = os.path.join(RESULTS_DIR, "exp_latent_dim.csv")
OUT_MSE = os.path.join(RESULTS_DIR, "exp_latent_dim_mse.png")


def main():
    rows = []
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    DIMS = sorted(set(int(r["dim"]) for r in rows))
    SEEDS = sorted(set(int(r["seed"]) for r in rows))

    groups = {d: [] for d in DIMS}
    for r in rows:
        groups[int(r["dim"])].append(float(r["rec_mse"]))

    means = {d: float(np.mean(groups[d])) for d in DIMS}
    stds  = {d: float(np.std(groups[d])) for d in DIMS}

    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(DIMS))
    colors = list(PALETTE.values())[:len(DIMS)]

    bars = ax.bar(x, [means[d] for d in DIMS],
                  yerr=[stds[d] for d in DIMS],
                  color=colors, capsize=6, width=0.5,
                  error_kw=dict(elinewidth=1.6, capthick=1.6))

    ax.set_xticks(x)
    ax.set_xticklabels([f"dim={d}" for d in DIMS])
    ax.set_ylabel("rec MSE (media ± std)")
    ax.set_title(f"MSE de reconstrucción por dim latente ({len(SEEDS)} seeds)",
                 fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    for rect, d in zip(bars, DIMS):
        ax.text(rect.get_x() + rect.get_width() / 2,
                rect.get_height() + stds[d] + 0.05,
                f"{means[d]:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    save_fig(fig, OUT_MSE)
    print(f"[ok] {OUT_MSE}")


if __name__ == "__main__":
    main()
