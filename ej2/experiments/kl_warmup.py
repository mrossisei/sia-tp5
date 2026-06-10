"""Experimento: KL warmup (annealing del término KL) — estabilidad.

Sin warmup: β=1 desde la época 0 (el KL puede colapsar la varianza temprano).
Con warmup: β sube linealmente de 0 a 1 en `warmup` épocas, dando tiempo al
decoder a aprender a reconstruir antes de regularizar.
Figura: ej2/results/exp_kl_warmup.png (curvas rec y KL por época).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load_emojis, train_vae, RESULTS_DIR  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from shared.plotting import save_fig, PALETTE  # noqa: E402


def make_schedule(warmup, beta_target=1.0):
    if warmup <= 0:
        return None
    return lambda ep: beta_target * min(1.0, (ep + 1) / warmup)


def run(warmups=(0, 100), epochs=400, seed=42):
    X, y, labels, image_shape = load_emojis()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"]]
    summary = []
    for k, w in enumerate(warmups):
        sched = make_schedule(w, beta_target=1.0)
        vae, hist = train_vae(X, latent_dim=2, beta=1.0, beta_schedule=sched,
                              epochs=epochs, seed=seed)
        epochs_ax = np.arange(1, epochs + 1)
        lab = "sin warmup" if w == 0 else f"warmup={w}"
        axes[0].plot(epochs_ax, hist["rec"], color=colors[k % len(colors)], label=lab)
        axes[1].plot(epochs_ax, hist["kl"], color=colors[k % len(colors)], label=lab)
        summary.append({"warmup": w, "rec": hist["rec"][-1], "kl": hist["kl"][-1]})
        print(f"warmup={w:4d}  rec_loss_final={hist['rec'][-1]:.3f}  "
              f"KL_final={hist['kl'][-1]:.3f}")

    axes[0].set_title("Reconstrucción por época"); axes[0].set_xlabel("Época")
    axes[0].set_ylabel("rec loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_title("KL por época"); axes[1].set_xlabel("Época")
    axes[1].set_ylabel("KL"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.suptitle("KL warmup (annealing): estabilidad del entrenamiento", y=1.02)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp_kl_warmup.png")
    save_fig(fig, path)
    print(f"-> {path}")
    return summary, path


if __name__ == "__main__":
    run()
