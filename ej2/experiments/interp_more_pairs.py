"""Genera varias interpolaciones fijas para comparar visualmente.

No toca la presentacion: solo escribe figuras en ej2/results/ para inspeccion.
Usa el VAE principal entrenado y recorre rectas entre centroides de clases.
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


MODEL_PATH = os.path.join(REPO_ROOT, "ej2", "results", "vae_model.npz")
DATA_PATH = os.path.join(REPO_ROOT, "ej2", "data", "emojis.npz")
RESULTS_DIR = os.path.join(REPO_ROOT, "ej2", "results")
STEPS = 10

PAIRS = [
    (6, 2),   # asustada -> gafas
    (6, 8),   # asustada -> corazon
    (4, 2),   # enojada -> gafas
    (3, 9),   # triste -> estrella
    (0, 10),  # feliz -> sol
    (0, 9),   # feliz -> estrella
]


def load_emojis():
    d = np.load(DATA_PATH, allow_pickle=True)
    X = d["X"].astype(np.float64)
    y = d["y"]
    labels = [str(s) for s in d["labels"]]
    image_shape = tuple(int(v) for v in d["image_shape"])
    return X, y, labels, image_shape


def render_pair(vae, X, y, labels, image_shape, class_a, class_b, steps=STEPS):
    mu = vae.encode(X)
    za = mu[y == class_a].mean(axis=0)
    zb = mu[y == class_b].mean(axis=0)
    alphas = np.linspace(0.0, 1.0, steps)
    imgs = []
    for a in alphas:
        z = (1.0 - a) * za + a * zb
        imgs.append(vae.decode(z[None, :])[0])
    imgs = np.asarray(imgs)

    fig, axes = plt.subplots(1, steps, figsize=(steps * 1.2, 2.2))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i].reshape(image_shape), cmap="gray", interpolation="nearest", vmin=0, vmax=1)
        ax.axis("off")
        if i == 0:
            ax.set_title(labels[class_a], fontsize=7)
        elif i == steps - 1:
            ax.set_title(labels[class_b], fontsize=7)
        else:
            ax.set_title(f"t={alphas[i]:.2f}", fontsize=7)
    fig.suptitle(f"Interpolacion: {labels[class_a]} -> {labels[class_b]}", y=1.03)
    fig.tight_layout()

    out_name = f"interpolation_{labels[class_a]}_{labels[class_b]}.png"
    out_name = out_name.replace("/", "_")
    out_path = os.path.join(RESULTS_DIR, out_name)
    save_fig(fig, out_path)
    return out_path


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No existe {MODEL_PATH}. Corre primero: python3 ej2/main_vae.py"
        )

    X, y, labels, image_shape = load_emojis()
    vae = VAE.load(MODEL_PATH)

    outputs = []
    for class_a, class_b in PAIRS:
        out = render_pair(vae, X, y, labels, image_shape, class_a, class_b)
        outputs.append(out)
        print(out)

    print(f"[ok] generadas {len(outputs)} interpolaciones")


if __name__ == "__main__":
    main()
