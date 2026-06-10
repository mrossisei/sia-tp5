"""Experimento: β-VAE (varios β) — efecto en el latente.

β bajo -> mejor reconstrucción, latente menos regularizado.
β alto -> latente más cercano al prior N(0,I) (KL menor), reconstrucción peor.
Figura: ej2/results/exp_beta.png (scatter latente 2D por β + métricas).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load_emojis, train_vae, RESULTS_DIR  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from shared.plotting import save_fig  # noqa: E402


def run(betas=(0.1, 1.0, 4.0), epochs=400, seed=42):
    X, y, labels, image_shape = load_emojis()
    n_classes = len(labels)
    cmap = plt.get_cmap("tab20", n_classes)

    fig, axes = plt.subplots(1, len(betas), figsize=(len(betas) * 4.2, 4.2))
    axes = np.atleast_1d(axes)
    summary = []
    for ax, b in zip(axes, betas):
        vae, hist = train_vae(X, latent_dim=2, beta=b, epochs=epochs, seed=seed)
        mu = vae.encode(X)
        X_hat = vae.reconstruct(X)
        rec_mse = float(np.mean(np.sum((X - X_hat) ** 2, axis=1)))
        for c in range(n_classes):
            m = (y == c)
            ax.scatter(mu[m, 0], mu[m, 1], s=10, color=cmap(c), alpha=0.6,
                       edgecolors="none")
        ax.set_title(f"β={b}\nrec={hist['rec'][-1]:.1f}  KL={hist['kl'][-1]:.2f}",
                     fontsize=10)
        ax.set_xlabel("z[0]"); ax.set_ylabel("z[1]"); ax.grid(alpha=0.3)
        summary.append({"beta": b, "rec": hist["rec"][-1], "kl": hist["kl"][-1],
                        "rec_mse": rec_mse})
        print(f"β={b:5.2f}  rec_loss={hist['rec'][-1]:.3f}  "
              f"KL={hist['kl'][-1]:.3f}  rec_MSE={rec_mse:.3f}")

    fig.suptitle("β-VAE: efecto del peso del KL en el espacio latente", y=1.03)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp_beta.png")
    save_fig(fig, path)
    print(f"-> {path}")
    return summary, path


if __name__ == "__main__":
    run()
