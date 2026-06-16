"""EXTRA-A — El VAE sobre el dataset del EJ1 (las 32 letras 7x5).

Puente AE -> VAE sobre los MISMOS datos: antes de saltar a los emojis, mostramos
cómo se comporta un VAE sobre las letras del EJ1 y lo comparamos con el
Autoencoder común (EJ1) entrenado con arquitectura equivalente.

La idea central: el AE básico MEMORIZA (reconstrucción casi perfecta) pero su
espacio latente no está regularizado (tiene "huecos": interpolar entre dos
letras puede pasar por basura). El VAE reconstruye un poco peor pero su latente
es CONTINUO y se puede muestrear/interpolar -> generación. Eso es lo que compra
el término KL.

FIGURAS (extra/results/):
  - letters_vae_loss.png       : curvas total/rec/KL del VAE sobre letras
  - letters_vae_recon.png      : original vs reconstrucción (las 32 letras)
  - letters_vae_manifold.png   : manifold 2D (grilla de z decodificada)
  - letters_vae_samples.png    : muestras z~N(0,I) decodificadas
  - letters_ae_vs_vae_scatter.png : latente AE vs VAE (anotado con cada letra)
  - letters_ae_vs_vae_interp.png  : interpolación AE vs VAE entre dos letras
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import RESULTS, ensure_dirs, make_vae, fit_simple  # noqa: E402

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from shared.fonts import load_font, ROWS, COLS  # noqa: E402
from shared.optimizers import Adam  # noqa: E402
from shared.metrics import pixel_error_summary  # noqa: E402
from shared.plotting import save_fig, PALETTE  # noqa: E402
from ej1.models.autoencoder import Autoencoder  # noqa: E402
from ej2.analysis.vae import (plot_loss_curves, plot_reconstructions,  # noqa: E402
                              plot_manifold_grid, plot_samples)

IMAGE_SHAPE = (ROWS, COLS)
LATENT = 2
EPOCHS = 6000
SEED = 42


def train_vae_letters(X):
    """VAE latente 2 sobre las letras (encoder/decoder chicos, acorde a 35 px)."""
    vae = make_vae(X.shape[1], LATENT, [24, 12], [12, 24], seed=SEED)
    hist = fit_simple(vae, X, epochs=EPOCHS, batch_size=X.shape[0], lr=1e-3,
                      seed=SEED)
    return vae, hist


def train_ae_letters(X):
    """AE común (EJ1) con anchos equivalentes al VAE, para comparar el latente."""
    ae = Autoencoder(architecture=[35, 24, 12, 2, 12, 24, 35],
                     hidden_activation="tanh", output_activation="logistic",
                     bottleneck_activation="identity", initializer="glorot",
                     seed=SEED, loss_name="bce")
    opt = Adam(lr=1e-3)
    rng = np.random.default_rng(SEED)
    for _ in range(EPOCHS):
        ae.train_epoch(X, X, opt, batch_size=0, shuffle=False, rng=rng)
    return ae


def _annotated_scatter(ax, Z, labels, title, color):
    ax.scatter(Z[:, 0], Z[:, 1], s=14, color=color, alpha=0.45, edgecolors="none")
    for i, lab in enumerate(labels):
        ax.annotate(lab, (Z[i, 0], Z[i, 1]), fontsize=7, ha="center", va="center")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("z[0]"); ax.set_ylabel("z[1]"); ax.grid(alpha=0.3)


def plot_scatters(ae_Z, vae_Z, labels, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ae_sd = ae_Z.std(axis=0).mean()
    vae_sd = vae_Z.std(axis=0).mean()
    _annotated_scatter(axes[0], ae_Z, labels,
                       f"AE (EJ1): latente sin regularizar (std≈{ae_sd:.1f})",
                       PALETTE["secondary"])
    _annotated_scatter(axes[1], vae_Z, labels,
                       f"VAE: latente empujado a N(0,I) (std≈{vae_sd:.1f})",
                       PALETTE["primary"])
    fig.suptitle("Espacio latente 2D sobre las 32 letras: AE vs VAE", y=1.01)
    fig.tight_layout()
    save_fig(fig, path)


def _nearest_letter_dist(pattern_bin, X_bin):
    """Distancia (píxeles distintos) al patrón real más cercano del dataset."""
    return int(np.min(np.sum(np.abs(X_bin - pattern_bin[None, :]), axis=1)))


def plot_sample_comparison(ae, vae, X, labels, path, n=12, seed=3):
    """LA comparación clave: muestrear z~N(0,I) y decodificar con AMBOS modelos.

    El decoder del VAE fue entrenado para que TODO el N(0,I) produzca salidas
    válidas; el del AE solo vio los códigos puntuales de las 32 letras, así que
    z~N(0,I) le cae 'fuera de distribución' -> tiende a basura. Métrica:
    distancia media (px) de cada muestra a la letra real más cercana.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, LATENT))
    ae_imgs = ae.decode(Z)
    vae_imgs = vae.decode(Z)
    X_bin = (X >= 0.5).astype(int)

    def mean_dist(imgs):
        return float(np.mean([_nearest_letter_dist((imgs[k] >= 0.5).astype(int),
                                                    X_bin) for k in range(n)]))
    ae_d, vae_d = mean_dist(ae_imgs), mean_dist(vae_imgs)

    fig, axes = plt.subplots(2, n, figsize=(n * 1.05, 3.0))
    for r, (name, imgs) in enumerate([("AE", ae_imgs), ("VAE", vae_imgs)]):
        for j in range(n):
            axes[r, j].imshow(imgs[j].reshape(IMAGE_SHAPE), cmap="gray",
                              vmin=0, vmax=1, interpolation="nearest")
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
        axes[r, 0].set_ylabel(name, fontsize=11, rotation=0, labelpad=16,
                              va="center")
    fig.suptitle("Mismos z~N(0,I) decodificados: el VAE da formas más cercanas a "
                 "letras válidas que el AE\n"
                 f"(dist. media a letra real: AE {ae_d:.1f}px vs VAE {vae_d:.1f}px; "
                 "el latente del AE no es N(0,I))", y=1.06, fontsize=10)
    fig.tight_layout(rect=[0.03, 0, 1, 1])
    save_fig(fig, path)
    return ae_d, vae_d


def plot_interpolation(ae, vae, ae_Z, vae_Z, X, labels, path, steps=9):
    """Interpola entre las DOS letras más separadas en el latente del VAE y
    decodifica el camino con ambos modelos. Métrica de CONTINUIDAD: el salto
    máximo entre pasos consecutivos (px). Más chico = transición más gradual,
    sin saltos bruscos (la propiedad que regulariza el KL)."""
    # par más separado según el VAE (camino largo, más ilustrativo)
    D = np.sqrt(((vae_Z[:, None, :] - vae_Z[None, :, :]) ** 2).sum(-1))
    ia, ib = np.unravel_index(np.argmax(D), D.shape)
    ts = np.linspace(0.0, 1.0, steps)

    def path_imgs(Z, decode):
        za, zb = Z[ia], Z[ib]
        Zs = np.array([(1 - t) * za + t * zb for t in ts])
        imgs = decode(Zs)
        steps_change = [float(np.sum(np.abs(imgs[k + 1] - imgs[k])))
                        for k in range(steps - 1)]
        return imgs, max(steps_change)   # salto máximo (continuidad)

    ae_imgs, ae_j = path_imgs(ae_Z, ae.decode)
    vae_imgs, vae_j = path_imgs(vae_Z, vae.decode)

    fig, axes = plt.subplots(2, steps, figsize=(steps * 1.15, 3.0))
    for r, (name, imgs) in enumerate([("AE", ae_imgs), ("VAE", vae_imgs)]):
        for j in range(steps):
            axes[r, j].imshow(imgs[j].reshape(IMAGE_SHAPE), cmap="gray",
                              vmin=0, vmax=1, interpolation="nearest")
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
        axes[r, 0].set_ylabel(name, fontsize=11, rotation=0, labelpad=16,
                              va="center")
    axes[0, 0].set_title(labels[ia], fontsize=9)
    axes[0, -1].set_title(labels[ib], fontsize=9)
    fig.suptitle(f"Interpolación en el latente '{labels[ia]}' -> '{labels[ib]}'  "
                 f"(salto máx entre pasos: AE {ae_j:.1f}px vs VAE {vae_j:.1f}px)",
                 y=1.04, fontsize=10)
    fig.tight_layout(rect=[0.03, 0, 1, 1])
    save_fig(fig, path)
    return labels[ia], labels[ib], ae_j, vae_j


def main():
    ensure_dirs()
    print("=== EXTRA-A: VAE sobre las letras del EJ1 ===")
    X, labels = load_font()
    y = np.arange(len(X))   # cada letra = su propia 'clase' (para el scatter)

    vae, hist = train_vae_letters(X)
    ae = train_ae_letters(X)

    # métricas de reconstrucción (criterio EJ1: píxeles, binarizado a 0.5)
    s_vae = pixel_error_summary(X, vae.reconstruct(X))
    s_ae = pixel_error_summary(X, ae.reconstruct(X))
    print(f"[VAE] max_pixel={s_vae['max']}  exactas={s_vae['n_exact']}/32")
    print(f"[AE ] max_pixel={s_ae['max']}  exactas={s_ae['n_exact']}/32")

    # figuras del VAE sobre letras (reutiliza el toolkit del EJ2)
    plot_loss_curves(hist, os.path.join(RESULTS, "letters_vae_loss.png"))
    plot_reconstructions(vae, X, y, labels, IMAGE_SHAPE,
                         os.path.join(RESULTS, "letters_vae_recon.png"), per_class=1)
    plot_manifold_grid(vae, IMAGE_SHAPE,
                       os.path.join(RESULTS, "letters_vae_manifold.png"), n=13)
    plot_samples(vae, IMAGE_SHAPE,
                 os.path.join(RESULTS, "letters_vae_samples.png"), n=16)

    # comparación AE vs VAE
    ae_Z = ae.encode(X)
    vae_Z = vae.encode(X)
    plot_scatters(ae_Z, vae_Z, labels,
                  os.path.join(RESULTS, "letters_ae_vs_vae_scatter.png"))
    s_ae_d, s_vae_d = plot_sample_comparison(
        ae, vae, X, labels,
        os.path.join(RESULTS, "letters_ae_vs_vae_samples.png"))
    la, lb, ae_j, vae_j = plot_interpolation(
        ae, vae, ae_Z, vae_Z, X, labels,
        os.path.join(RESULTS, "letters_ae_vs_vae_interp.png"))

    print(f"[samples z~N(0,I)] dist media a letra real: "
          f"AE={s_ae_d:.1f}px  VAE={s_vae_d:.1f}px")
    print(f"[interp '{la}'->'{lb}'] salto máx: AE={ae_j:.1f}px  VAE={vae_j:.1f}px")
    print(f"[ok] EXTRA-A -> {RESULTS}")


if __name__ == "__main__":
    main()
