"""Experimento: muestreo dentro y fuera del prior del VAE.

Pregunta: el VAE genera bien cuando sampleamos desde el prior p(z)=N(0,I),
pero, que pasa si nos alejamos de esa region y hacemos que z tenga una norma
mucho mayor?

Idea: cargar el modelo principal ya entrenado (latente 2D) y decodificar los
MISMOS z base escalados por distintos factores sigma. Asi aislamos el efecto de
la distancia al origen:

    z_base ~ N(0, I)
    z = sigma * z_base,   sigma in {1, 2, 4}

Con sigma=1 sampleamos exactamente desde el prior. Con sigma mayores
extrapolamos fuera de la zona donde el decoder fue regularizado por el KL.

Salida: ej2/results/exp_prior_extrapolation.png
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load_emojis, img, RESULTS_DIR  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from shared.plotting import save_fig  # noqa: E402
from ej2.models.vae import VAE  # noqa: E402


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(REPO_ROOT, "ej2", "results", "vae_model.npz")


def run(sigmas=(1.0, 2.0, 4.0), n_samples=10, seed=7):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No existe {MODEL_PATH}. Corre primero: python3 ej2/main_vae.py"
        )

    _, _, _, image_shape = load_emojis()
    vae = VAE.load(MODEL_PATH)
    if vae.latent_dim != 2:
        raise ValueError(
            f"Este experimento asume latent_dim=2; el modelo cargado tiene {vae.latent_dim}."
        )

    rng = np.random.default_rng(seed)
    base_z = rng.standard_normal(size=(n_samples, vae.latent_dim))

    fig, axes = plt.subplots(
        len(sigmas), n_samples, figsize=(n_samples * 1.2, len(sigmas) * 1.5)
    )
    axes = np.atleast_2d(axes)

    summaries = []
    for i, sigma in enumerate(sigmas):
        z = sigma * base_z
        x_hat = vae.decode(z)
        mean_abs = float(np.mean(np.abs(x_hat - 0.5)))
        summaries.append({"sigma": sigma, "sharpness": mean_abs})

        for j in range(n_samples):
            axes[i, j].imshow(
                img(x_hat[j], image_shape),
                cmap="gray",
                interpolation="nearest",
                vmin=0,
                vmax=1,
            )
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
        axes[i, 0].set_ylabel(
            f"sigma={sigma:g}\n|x-0.5|={mean_abs:.2f}",
            fontsize=9,
            rotation=0,
            labelpad=28,
            va="center",
        )

    fig.suptitle(
        "Extrapolacion del prior: mismos z base escalados antes de decodificar\n"
        "sigma=1 samplea desde N(0,I); sigma altos fuerzan extrapolacion del decoder",
        y=1.03,
    )
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp_prior_extrapolation.png")
    save_fig(fig, path)
    print(f"-> {path}")
    for s in summaries:
        print(f"sigma={s['sigma']:>3g}  mean|x-0.5|={s['sharpness']:.3f}")
    return summaries, path


if __name__ == "__main__":
    run()
