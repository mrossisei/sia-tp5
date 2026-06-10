"""Experimento: latent_dim 2 vs >2 (8, 16).

Compara loss de reconstrucción final y calidad de samples z~N(0,I).
Figura: ej2/results/exp_latent_dim.png
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load_emojis, train_vae, img, RESULTS_DIR  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from shared.plotting import save_fig  # noqa: E402


def run(dims=(2, 8, 16), epochs=400, n_samples=8, seed=42):
    X, y, labels, image_shape = load_emojis()
    results = []
    for d in dims:
        vae, hist = train_vae(X, latent_dim=d, epochs=epochs, seed=seed)
        X_hat = vae.reconstruct(X)
        rec_mse = float(np.mean(np.sum((X - X_hat) ** 2, axis=1)))
        samples, _ = vae.generate(n_samples, seed=7)
        results.append({
            "dim": d, "rec_final": hist["rec"][-1], "kl_final": hist["kl"][-1],
            "rec_mse": rec_mse, "samples": samples,
        })
        print(f"latent_dim={d:2d}  rec_loss={hist['rec'][-1]:.3f}  "
              f"KL={hist['kl'][-1]:.3f}  rec_MSE={rec_mse:.3f}")

    # Figura: filas = dims, columnas = muestras
    fig, axes = plt.subplots(len(dims), n_samples,
                             figsize=(n_samples * 1.2, len(dims) * 1.3))
    axes = np.atleast_2d(axes)
    for i, r in enumerate(results):
        for j in range(n_samples):
            axes[i, j].imshow(img(r["samples"][j], image_shape), cmap="gray",
                              interpolation="nearest")
            axes[i, j].axis("off")
        axes[i, 0].set_ylabel(f"dim={r['dim']}", fontsize=9)
        axes[i, 0].axis("on")
        axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])
        axes[i, 0].set_title(f"dim={r['dim']}  rec={r['rec_final']:.1f}",
                             fontsize=8, loc="left")
    fig.suptitle("latent_dim: muestras z~N(0,I) (rec_loss menor = mejor reconstrucción)",
                 y=1.01)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp_latent_dim.png")
    save_fig(fig, path)
    print(f"-> {path}")
    return results, path


if __name__ == "__main__":
    run()
