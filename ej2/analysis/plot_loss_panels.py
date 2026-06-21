"""Genera vae_loss_panels.png: 2 paneles (rec y KL) desde loss_history.csv."""

import os
import csv
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.plotting import save_fig, PALETTE

SRC = os.path.join(REPO_ROOT, "ej2", "results", "long_16px", "loss_history.csv")
OUT = os.path.join(REPO_ROOT, "ej2", "results", "long_16px", "vae_loss_panels.png")


def main():
    epochs, rec, kl = [], [], []
    with open(SRC) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            rec.append(float(row["rec"]))
            kl.append(float(row["kl"]))

    epochs = np.array(epochs)
    rec = np.array(rec)
    kl = np.array(kl)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(epochs, rec, color=PALETTE["positive"], lw=1.0)
    ax1.set_xscale("log")
    ax1.set_xlabel("Época (log)")
    ax1.set_ylabel("Loss de reconstrucción (BCE)")
    ax1.set_title("Reconstrucción — el decoder aprende")
    ax1.grid(alpha=0.3)
    ax1.annotate(f"final: {rec[-1]:.1f}", xy=(epochs[-1], rec[-1]),
                 xytext=(-60, 12), textcoords="offset points",
                 fontsize=9, color=PALETTE["positive"],
                 arrowprops=dict(arrowstyle="-", color=PALETTE["positive"], lw=0.8))

    ax2.plot(epochs, kl, color=PALETTE["highlight"], lw=1.0)
    ax2.set_xscale("log")
    ax2.set_xlabel("Época (log)")
    ax2.set_ylabel("KL  (nats)")
    ax2.set_title("KL — el latente se organiza")
    ax2.grid(alpha=0.3)
    ax2.annotate(f"final: {kl[-1]:.1f}", xy=(epochs[-1], kl[-1]),
                 xytext=(-60, -18), textcoords="offset points",
                 fontsize=9, color=PALETTE["highlight"],
                 arrowprops=dict(arrowstyle="-", color=PALETTE["highlight"], lw=0.8))

    fig.suptitle("Curvas de aprendizaje del VAE (50.000 épocas)", y=1.01)
    fig.tight_layout()
    save_fig(fig, OUT)
    print(f"[ok] -> {OUT}")


if __name__ == "__main__":
    main()
