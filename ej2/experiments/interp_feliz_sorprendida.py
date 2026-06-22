"""Interpolacion fija entre cara_feliz y corazon.

La figura original de EJ2 elige automaticamente el par de centroides mas
separados del latente, lo que sirve para mostrar continuidad pero no siempre da
una transicion visual agradable. Este script fija un par con contraste visual
fuerte para que el morph sea mas impactante:

    cara_feliz (idx 0) -> corazon (idx 8)

Para probar si una dimension latente mayor mejora el morph, este script usa un
VAE con ``latent_dim=8``. Si ya existe cacheado, lo reutiliza; si no, lo
entrena con la misma receta base de EJ2.

Salida:
  - ej2/results/interpolation_feliz_corazon_latent8.png
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
MODEL_PATH = os.path.join(RESULTS_DIR, "interp_latent8_model.npz")

CLASS_A = 0
CLASS_B = 8
STEPS = 10
LATENT_DIM = 8
EPOCHS = 600
SEED = 42


def load_emojis():
    d = np.load(DATA_PATH, allow_pickle=True)
    X = d["X"].astype(np.float64)
    y = d["y"]
    labels = [str(s) for s in d["labels"]]
    image_shape = tuple(int(v) for v in d["image_shape"])
    return X, y, labels, image_shape


def main():
    X, y, labels, image_shape = load_emojis()

    if os.path.exists(MODEL_PATH):
        vae = VAE.load(MODEL_PATH)
        print(f"[cache] usando {MODEL_PATH}")
    else:
        print(f"[train] entrenando VAE latent={LATENT_DIM} para interpolacion...")
        vae, _ = train_vae(X, latent_dim=LATENT_DIM, epochs=EPOCHS, seed=SEED)
        vae.save(MODEL_PATH)
        print(f"[save] {MODEL_PATH}")

    mu = vae.encode(X)
    za = mu[y == CLASS_A].mean(axis=0)
    zb = mu[y == CLASS_B].mean(axis=0)

    alphas = np.linspace(0.0, 1.0, STEPS)
    imgs = []
    for a in alphas:
        z = (1.0 - a) * za + a * zb
        imgs.append(vae.decode(z[None, :])[0])
    imgs = np.asarray(imgs)

    fig, axes = plt.subplots(1, STEPS, figsize=(STEPS * 1.2, 2.2))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i].reshape(image_shape), cmap="gray", interpolation="nearest", vmin=0, vmax=1)
        ax.axis("off")
        if i == 0:
            ax.set_title(labels[CLASS_A], fontsize=8)
        elif i == STEPS - 1:
            ax.set_title(labels[CLASS_B], fontsize=8)
        else:
            ax.set_title(f"t={alphas[i]:.2f}", fontsize=7)

    fig.suptitle(
        f"Interpolacion fija en el latente: {labels[CLASS_A]} -> {labels[CLASS_B]}",
        y=1.03,
    )
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "interpolation_feliz_corazon_latent8.png")
    save_fig(fig, out)
    print(f"[ok] {out}")


if __name__ == "__main__":
    main()
