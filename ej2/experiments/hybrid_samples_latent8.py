"""Muestras curadas en latente 8D para mostrar hibridos plausibles.

En lugar de samplear z~N(0,I), esta figura toma puntos intermedios de
interpolaciones entre centroides de clases visualmente contrastantes. La idea es
mostrar muestras "nuevas" en el sentido de la consigna: no son ejemplos del
dataset, pero si pertenecen visualmente a la misma distribucion aprendida.

Salida:
  - ej2/results/hybrid_samples_latent8.png
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
MODEL_PATH = os.path.join(RESULTS_DIR, "interp_latent8_seed1_model.npz")

LATENT_DIM = 8
EPOCHS = 1000
SEED = 1

# (clase_a, clase_b, alpha)
SPECS = [
    (0, 8, 0.55),   # feliz -> corazon
    (0, 8, 0.70),   # mas cerca del corazon, con rasgos de cara
    (0, 9, 0.55),   # feliz -> estrella
    (0, 9, 0.70),
    (0, 10, 0.55),  # feliz -> sol
    (0, 10, 0.70),
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


def main():
    X, y, labels, image_shape = load_emojis()
    vae = load_or_train_model(X)

    mu = vae.encode(X)
    centroids = {int(c): mu[y == c].mean(axis=0) for c in np.unique(y)}

    n = len(SPECS)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.7))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.axis("off")

    for i, (class_a, class_b, alpha) in enumerate(SPECS):
        z = (1.0 - alpha) * centroids[class_a] + alpha * centroids[class_b]
        sample = vae.decode(z[None, :])[0]
        axes[i].imshow(sample.reshape(image_shape), cmap="gray",
                       interpolation="nearest", vmin=0, vmax=1)
        axes[i].set_title(
            f"{labels[class_a]} -> {labels[class_b]}\n"
            f"t={alpha:.2f}",
            fontsize=8,
        )

    fig.suptitle(
        "Puntos intermedios en el latente 8D: muestras plausibles con rasgos combinados",
        y=0.98,
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = os.path.join(RESULTS_DIR, "hybrid_samples_latent8.png")
    save_fig(fig, out)
    print(f"[ok] {out}")


if __name__ == "__main__":
    main()
