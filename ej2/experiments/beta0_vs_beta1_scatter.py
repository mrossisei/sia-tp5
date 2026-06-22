"""Comparacion de espacio latente: beta=0 vs beta=1.

Mismo dataset y misma arquitectura; cambia SOLO el peso del KL. La idea es
mostrar visualmente que, sin regularizacion (beta=0), el encoder aprende
codigos utiles para reconstruir pero no tiene incentivo a concentrarlos cerca
del prior. Con beta=1, el KL organiza el latente y lo vuelve sampleable.

Es un experimento VISUAL para presentacion: usamos una seed representativa y la
misma cantidad de epocas que el barrido de beta principal (400), asi la figura
sale rapido y es comparable con el resto de EJ2.

Salidas:
  - ej2/results/exp_beta0_vs_beta1.png
  - ej2/results/exp_beta0_vs_beta1.csv
"""

import csv
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from shared.plotting import save_fig
from ej2.experiments._common import load_emojis, train_vae, RESULTS_DIR


BETA_SETTINGS = [
    (0.0, "sin KL"),
    (1.0, "con KL"),
]
SEED = 42
EPOCHS = 400


def main():
    X, y, labels, _ = load_emojis()
    n_classes = len(labels)
    cmap = plt.get_cmap("tab20", n_classes)
    results = {}
    for beta, tag in BETA_SETTINGS:
        vae, hist = train_vae(X, latent_dim=2, beta=beta, epochs=EPOCHS, seed=SEED)
        X_hat = vae.reconstruct(X)
        rec_mse = float(np.mean(np.sum((X - X_hat) ** 2, axis=1)))
        results[beta] = {
            "tag": tag,
            "vae": vae,
            "rec_final": hist["rec"][-1],
            "kl_final": hist["kl"][-1],
            "rec_mse": rec_mse,
        }
        print(
            f"[beta={beta:.1f}] rec={hist['rec'][-1]:.3f}  "
            f"KL={hist['kl'][-1]:.3f}  MSE={rec_mse:.4f}",
            flush=True,
        )

    csv_path = os.path.join(RESULTS_DIR, "exp_beta0_vs_beta1.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["beta", "seed", "rec_final", "kl_final", "rec_mse"])
        for beta, _ in BETA_SETTINGS:
            r = results[beta]
            w.writerow([beta, SEED, f"{r['rec_final']:.4f}", f"{r['kl_final']:.4f}", f"{r['rec_mse']:.4f}"])
    print(f"[csv] {csv_path}")

    scatters = {beta: results[beta]["vae"].encode(X) for beta, _ in BETA_SETTINGS}

    all_mu = np.vstack([scatters[b] for b, _ in BETA_SETTINGS])
    mins = all_mu.min(axis=0)
    maxs = all_mu.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    pad = 0.08 * span
    xlim = (mins[0] - pad[0], maxs[0] + pad[0])
    ylim = (mins[1] - pad[1], maxs[1] + pad[1])

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), sharex=True, sharey=True)
    for ax, (beta, tag) in zip(axes, BETA_SETTINGS):
        mu = scatters[beta]
        rec = results[beta]["rec_final"]
        kl = results[beta]["kl_final"]
        mse = results[beta]["rec_mse"]
        for c in range(n_classes):
            m = y == c
            ax.scatter(mu[m, 0], mu[m, 1], s=10, color=cmap(c), alpha=0.65, edgecolors="none")
        ax.set_title(
            f"beta={beta:.0f} ({tag})\nrec={rec:.1f}  KL={kl:.2f}  MSE={mse:.2f}",
            fontsize=10,
        )
        ax.set_xlabel("z[0]")
        ax.grid(alpha=0.3)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
    axes[0].set_ylabel("z[1]")

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=5,
                   markerfacecolor=cmap(c), markeredgecolor="none", label=labels[c])
        for c in range(n_classes)
    ]
    axes[1].legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
                   fontsize=7, frameon=False, ncol=1)

    fig.suptitle(
        "Mismo VAE sobre emojis: cambia solo el peso del KL\n"
        "beta=0 reconstruye sin ordenar el latente; beta=1 lo empuja hacia el prior",
        y=1.03,
    )
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    png_path = os.path.join(RESULTS_DIR, "exp_beta0_vs_beta1.png")
    save_fig(fig, png_path)
    print(f"[png] {png_path}")
    print("[ok] beta0_vs_beta1 completo")


if __name__ == "__main__":
    main()
