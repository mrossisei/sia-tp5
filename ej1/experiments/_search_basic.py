"""Busqueda rapida de la mejor config/seed para el AE basico (objetivo max<=1).

Entrena unos pocos miles de epocas por seed y reporta el pixel-error maximo.
Sirve para decidir la config del main antes de la corrida larga.
"""

import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import numpy as np

from shared.fonts import load_font
from shared.optimizers import Adam
from shared.metrics import pixel_error_summary
from ej1.models.autoencoder import Autoencoder


def train_quick(X, arch, hidden, bottleneck_act, loss_name, seed, epochs, lr=1e-3):
    rng = np.random.default_rng(seed)
    ae = Autoencoder(
        architecture=arch, hidden_activation=hidden,
        output_activation="logistic", bottleneck_activation=bottleneck_act,
        initializer="glorot", seed=seed, loss_name=loss_name,
    )
    opt = Adam(lr=lr)
    best_max = 99
    best_n_exact = 0
    for ep in range(epochs):
        ae.train_epoch(X, X, opt, batch_size=0, shuffle=False, rng=rng)
        if ep % 500 == 0 or ep == epochs - 1:
            rec = ae.reconstruct(X)
            s = pixel_error_summary(X, rec)
            if (s["max"], -s["n_exact"]) < (best_max, -best_n_exact):
                best_max = s["max"]
                best_n_exact = s["n_exact"]
            if s["max"] <= 1:
                return s["max"], s["n_exact"], s["n_le1"], ep
    return best_max, best_n_exact, None, epochs


def main():
    X, labels = load_font()
    configs = [
        # (arch, hidden, bottleneck_act, loss)
        ([35, 60, 40, 20, 2, 20, 40, 60, 35], "tanh", "identity", "bce"),
        ([35, 60, 40, 20, 2, 20, 40, 60, 35], "tanh", "tanh", "bce"),
        ([35, 60, 40, 20, 2, 20, 40, 60, 35], "tanh", "identity", "mse"),
        ([35, 50, 30, 15, 2, 15, 30, 50, 35], "relu", "identity", "bce"),
    ]
    epochs = 8000
    seeds = [0, 1, 2, 3, 42]
    for arch, hidden, bact, loss in configs:
        print(f"\n=== arch={arch} hidden={hidden} bottleneck={bact} loss={loss} ===")
        for seed in seeds:
            t0 = time.time()
            mx, nex, nle1, ep = train_quick(X, arch, hidden, bact, loss, seed, epochs)
            dt = time.time() - t0
            tag = "  <-- SUCCESS (max<=1)" if mx <= 1 else ""
            print(f"  seed={seed}: max={mx} n_exact={nex} reached_le1_at={ep} "
                  f"({dt:.1f}s){tag}")


if __name__ == "__main__":
    main()
