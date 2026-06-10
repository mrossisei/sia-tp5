"""Experimento: variabilidad entre seeds del VAE principal (latent_dim=2).

El entrenamiento del VAE es doblemente estocástico (init de pesos + sampleo
de eps en el reparametrization trick), así que una sola corrida no alcanza
para afirmar estabilidad. Acá: 3 seeds × 600 épocas con la config principal
y curvas total/rec/KL con media ± std entre seeds.

Figura: ej2/results/exp_seeds.png  (+ exp_seeds.csv con los finales)
"""

import os
import sys
import csv
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load_emojis, train_vae, RESULTS_DIR  # noqa: E402

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from shared.plotting import save_fig, PALETTE  # noqa: E402

SEEDS = [42, 7, 123]
EPOCHS = 600  # misma config que el run principal (ej2/config.yaml)


def main():
    X, y, labels, image_shape = load_emojis()
    hists, finals = [], []
    t0 = time.time()
    for seed in SEEDS:
        t = time.time()
        vae, hist = train_vae(X, latent_dim=2, epochs=EPOCHS, seed=seed)
        X_hat = vae.reconstruct(X)
        rec_mse = float(np.mean(np.sum((X - X_hat) ** 2, axis=1)))
        hists.append(hist)
        finals.append({"seed": seed, "total": hist["total"][-1],
                       "rec": hist["rec"][-1], "kl": hist["kl"][-1],
                       "rec_mse": rec_mse})
        print(f"[seed={seed}] total={hist['total'][-1]:.3f} "
              f"rec={hist['rec'][-1]:.3f} KL={hist['kl'][-1]:.3f} "
              f"recMSE={rec_mse:.3f} ({time.time()-t:.0f}s)", flush=True)

    epochs_axis = np.arange(1, EPOCHS + 1)
    keys = [("total", "Loss total (rec + β·KL)", PALETTE["primary"]),
            ("rec", "Reconstrucción", PALETTE["positive"]),
            ("kl", "KL", PALETTE["highlight"])]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (key, title, color) in zip(axes, keys):
        curves = np.array([h[key] for h in hists])  # (n_seeds, epochs)
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        for c in curves:  # corridas individuales, finitas
            ax.plot(epochs_axis, c, color=color, alpha=0.25, lw=0.8)
        ax.plot(epochs_axis, mean, color=color, lw=1.8,
                label=f"media ({len(SEEDS)} seeds)")
        ax.fill_between(epochs_axis, mean - std, mean + std,
                        color=color, alpha=0.2, label="± std")
        ax.set_title(title)
        ax.set_xlabel("Época")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"VAE (latent=2): variabilidad entre seeds {SEEDS}", y=1.02)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp_seeds.png")
    save_fig(fig, path)

    with open(os.path.join(RESULTS_DIR, "exp_seeds.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "total", "rec", "kl", "rec_mse"])
        w.writeheader()
        for r in finals:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items()})

    print(f"[ok] seeds completo en {time.time()-t0:.0f}s -> {path}")


if __name__ == "__main__":
    main()
