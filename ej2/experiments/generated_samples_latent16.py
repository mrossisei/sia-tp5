"""Genera imagenes nuevas curadas en el latente 16D.

Usa un unico VAE cacheado (latent_dim=16, seed=2) y crea dos figuras:
  - generated_samples_latent16_candidates.png : barrido visual de mezclas por par
  - generated_samples_latent16.png            : seleccion final para la presentacion

La figura final evita hablar de "interpolacion": solo muestra muestras nuevas
generadas a partir de puntos elegidos del espacio latente.
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.plotting import save_fig
from ej2.models.vae import VAE
from ej2.experiments._common import train_vae


DATA_PATH = os.path.join(REPO_ROOT, "ej2", "data", "emojis.npz")
RESULTS_DIR = os.path.join(REPO_ROOT, "ej2", "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "generated_latent16_seed2_model.npz")

LATENT_DIM = 16
EPOCHS = 1000
SEED = 2

PAIR_SWEEPS = [
    {"name": "cara_feliz + corazon", "a": 0, "b": 8, "ts": [0.35, 0.45, 0.55, 0.65]},
    {"name": "sol + cara_feliz", "a": 10, "b": 0, "ts": [0.25, 0.35, 0.45, 0.55]},
    {"name": "cara_feliz + estrella", "a": 0, "b": 9, "ts": [0.35, 0.45, 0.55, 0.65]},
]

FINAL_SPECS = [
    {"name": "cara_feliz + corazon", "a": 0, "b": 8, "t": 0.55},
    {"name": "sol + cara_feliz", "a": 10, "b": 0, "t": 0.45},
    {"name": "cara_feliz + estrella", "a": 0, "b": 9, "t": 0.55},
]


def load_emojis():
    d = np.load(DATA_PATH, allow_pickle=True)
    X = d["X"].astype(np.float64)
    y = d["y"]
    labels = [str(s) for s in d["labels"]]
    image_shape = tuple(int(v) for v in d["image_shape"])
    return X, y, labels, image_shape


def load_or_train_model(X):
    if os.path.exists(MODEL_PATH):
        print(f"[cache] usando {MODEL_PATH}")
        return VAE.load(MODEL_PATH)

    print(f"[train] entrenando VAE latent={LATENT_DIM}, seed={SEED}...")
    vae, _ = train_vae(X, latent_dim=LATENT_DIM, epochs=EPOCHS, seed=SEED)
    vae.save(MODEL_PATH)
    print(f"[save] {MODEL_PATH}")
    return vae


def decode_mix(vae, centroids, class_a, class_b, t):
    z = (1.0 - t) * centroids[class_a] + t * centroids[class_b]
    return vae.decode(z[None, :])[0]


def make_candidates(vae, centroids, image_shape):
    rows = len(PAIR_SWEEPS)
    cols = max(len(spec["ts"]) for spec in PAIR_SWEEPS)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 2.2))
    axes = np.atleast_2d(axes)

    for r, spec in enumerate(PAIR_SWEEPS):
        for c in range(cols):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(spec["ts"]):
                t = spec["ts"][c]
                img = decode_mix(vae, centroids, spec["a"], spec["b"], t)
                ax.imshow(img.reshape(image_shape), cmap="gray",
                          interpolation="nearest", vmin=0, vmax=1)
                ax.set_title(f"t={t:.2f}", fontsize=8)
        axes[r, 0].set_ylabel(spec["name"], fontsize=9, rotation=0,
                              labelpad=42, va="center")

    fig.suptitle("Candidatos 16D: mezclas visuales para elegir muestras nuevas",
                 y=0.98, fontsize=11)
    fig.tight_layout(rect=[0.02, 0, 1, 0.95])
    out = os.path.join(RESULTS_DIR, "generated_samples_latent16_candidates.png")
    save_fig(fig, out)
    print(f"[ok] {out}")


def make_final(vae, centroids, image_shape):
    cols = len(FINAL_SPECS)
    fig, axes = plt.subplots(1, cols, figsize=(cols * 2.6, 3.2))
    axes = np.atleast_1d(axes).ravel()

    for ax, spec in zip(axes, FINAL_SPECS):
        img = decode_mix(vae, centroids, spec["a"], spec["b"], spec["t"])
        ax.imshow(img.reshape(image_shape), cmap="gray",
                  interpolation="nearest", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(spec["name"], fontsize=9)

    fig.suptitle("Imagenes nuevas generadas en el espacio latente 16D",
                 y=0.92, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    out = os.path.join(RESULTS_DIR, "generated_samples_latent16.png")
    save_fig(fig, out)
    print(f"[ok] {out}")


def main():
    X, y, labels, image_shape = load_emojis()
    vae = load_or_train_model(X)
    mu = vae.encode(X)
    centroids = {int(c): mu[y == c].mean(axis=0) for c in np.unique(y)}

    make_candidates(vae, centroids, image_shape)
    make_final(vae, centroids, image_shape)

    print("[done] muestras 16D generadas")


if __name__ == "__main__":
    main()
