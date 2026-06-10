"""Figuras del VAE (EJ2). SOLO acá se importa matplotlib (convención TP3/TP4).

Figuras producidas en ej2/results/:
  - loss_curves.png     : loss total, reconstrucción y KL por separado.
  - latent_scatter.png  : scatter de mu(x) en 2D coloreado por clase.
  - manifold_grid.png    : grilla en el latente 2D vía la inversa de la CDF normal
                          (ppf con erfinv aproximado en numpy) -> emojis generados.
  - samples.png         : muestras z~N(0,I) decodificadas (consigna 2.c).
  - interpolation.png   : recta entre dos códigos mu, decodificada (morphing).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from shared.plotting import save_fig, PALETTE


# ---------------------------------------------------------------------------
# Inversa de la CDF normal (ppf) sin scipy: ppf(p) = sqrt(2) * erfinv(2p - 1).
# erfinv vía aproximación de Winitzki (error relativo < ~1e-3, suficiente para
# elegir puntos de una grilla en el latente).
#
# Es el "Inverse Sampling Theorem" de la teoría (Autoencoders.pdf, slides
# 39-40): con Y ~ U[0,1] y F la CDF de la pdf objetivo, X = F^-1(Y) distribuye
# según esa pdf. Acá lo usamos al revés: percentiles equiespaciados de U[0,1]
# -> puntos del latente que cubren equitativamente la DENSIDAD del prior
# N(0,I), en vez de cubrir el plano con una grilla lineal ingenua.
# ---------------------------------------------------------------------------
def _erfinv(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -0.999999, 0.999999)
    a = 0.147
    ln = np.log(1.0 - x ** 2)
    t1 = 2.0 / (np.pi * a) + ln / 2.0
    inner = t1 ** 2 - ln / a
    return np.sign(x) * np.sqrt(np.sqrt(inner) - t1)


def norm_ppf(p):
    """Inversa de la CDF de N(0,1)."""
    return np.sqrt(2.0) * _erfinv(2.0 * np.asarray(p, dtype=np.float64) - 1.0)


def _img(vec, image_shape):
    return np.asarray(vec, dtype=np.float64).reshape(image_shape)


# ===========================================================================
# Figuras
# ===========================================================================
def plot_loss_curves(hist, path):
    """Total, reconstrucción y KL por separado (3 paneles)."""
    epochs = np.arange(1, len(hist["total"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    axes[0].plot(epochs, hist["total"], color=PALETTE["primary"])
    axes[0].set_title("Loss total (rec + β·KL)")
    axes[1].plot(epochs, hist["rec"], color=PALETTE["positive"])
    axes[1].set_title("Reconstrucción")
    axes[2].plot(epochs, hist["kl"], color=PALETTE["highlight"])
    axes[2].set_title("KL")
    for ax in axes:
        ax.set_xlabel("Época")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)
    fig.suptitle("Curvas de aprendizaje del VAE", y=1.02)
    fig.tight_layout()
    save_fig(fig, path)
    return path


def plot_latent_scatter(vae, X, y, labels, path):
    """Scatter de mu(x) en 2D coloreado por clase (con leyenda)."""
    mu = vae.encode(X)
    if mu.shape[1] != 2:
        # Si latent_dim>2, no hay scatter 2D directo: tomar las 2 primeras dims.
        mu = mu[:, :2]
    n_classes = len(labels)
    cmap = plt.get_cmap("tab20", n_classes)

    fig, ax = plt.subplots(figsize=(8, 7))
    for c in range(n_classes):
        m = (y == c)
        ax.scatter(mu[m, 0], mu[m, 1], s=18, color=cmap(c),
                   label=labels[c], alpha=0.75, edgecolors="none")
    ax.set_xlabel("z[0]")
    ax.set_ylabel("z[1]")
    ax.set_title("Espacio latente: μ(x) coloreado por clase")
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=8, ncol=1, frameon=False)
    fig.tight_layout()
    save_fig(fig, path)
    return path


def plot_manifold_grid(vae, image_shape, path, n=15, p_lo=0.05, p_hi=0.95):
    """Grilla en el latente 2D usando la inversa de la CDF normal.

    "La conquista del espacio latente" (Autoencoders.pdf, slides 36-37): el
    VAE le dio estructura estadística al latente, así que RECORRERLO produce
    muestras válidas en todo punto — exactamente lo que esta figura muestra.
    Decodifica una grilla n×n de puntos (cubriendo la densidad del prior) ->
    mapa de emojis generados (el plot clásico del manifold del VAE).
    Sólo aplica a latent_dim==2.
    """
    if vae.latent_dim != 2:
        return None
    grid = norm_ppf(np.linspace(p_lo, p_hi, n))
    H, W = image_shape
    canvas = np.zeros((n * H, n * W))
    # eje y invertido (de arriba a abajo) para que crezca hacia arriba
    for i, yi in enumerate(grid[::-1]):
        for j, xj in enumerate(grid):
            z = np.array([[xj, yi]])
            img = _img(vae.decode(z)[0], image_shape)
            canvas[i * H:(i + 1) * H, j * W:(j + 1) * W] = img

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(canvas, cmap="gray", interpolation="nearest")
    ax.set_title("Manifold generativo (grilla en el latente 2D, ppf normal)")
    # ticks con valores de z
    tick_pos_x = np.linspace(W / 2, n * W - W / 2, min(n, 7))
    tick_val = np.linspace(grid[0], grid[-1], min(n, 7))
    ax.set_xticks(tick_pos_x)
    ax.set_xticklabels([f"{v:.1f}" for v in tick_val])
    tick_pos_y = np.linspace(H / 2, n * H - H / 2, min(n, 7))
    ax.set_yticks(tick_pos_y)
    ax.set_yticklabels([f"{v:.1f}" for v in tick_val[::-1]])
    ax.set_xlabel("z[0]")
    ax.set_ylabel("z[1]")
    fig.tight_layout()
    save_fig(fig, path)
    return path


def plot_samples(vae, image_shape, path, n=16, seed=7):
    """Varias muestras z~N(0,I) decodificadas (consigna 2.c).

    Generative Autoencoder (Autoencoders.pdf, slides 34 y 79): se descarta el
    encoder y se samplea z directamente del prior p(z)=N(0,I) (slide 74).
    """
    samples, _ = vae.generate(n, seed=seed)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.4))
    axes = np.atleast_1d(axes).ravel()
    for k in range(len(axes)):
        axes[k].axis("off")
        if k < n:
            axes[k].imshow(_img(samples[k], image_shape), cmap="gray",
                           interpolation="nearest")
    fig.suptitle("Muestras nuevas z ~ N(0, I) decodificadas", y=1.0)
    fig.tight_layout()
    save_fig(fig, path)
    return path


def plot_interpolation(vae, X, y, labels, image_shape, path,
                       class_a=None, class_b=None, steps=10):
    """Recorre una recta entre dos códigos μ y decodifica (morphing).

    Movernos por un "concept vector" del espacio latente (Autoencoders.pdf,
    slide 33): los puntos intermedios decodifican a muestras válidas que
    transicionan suavemente entre las dos clases.

    Si no se especifican clases, elige automáticamente el par de centroides MÁS
    SEPARADOS en el latente (morph más ilustrativo de la continuidad).
    """
    mu = vae.encode(X)
    classes = np.unique(y)
    centroids = {int(c): mu[y == c].mean(axis=0) for c in classes}

    if class_a is None or class_b is None:
        best = (-1.0, None, None)
        cl = list(centroids.keys())
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                d = float(np.linalg.norm(centroids[cl[i]] - centroids[cl[j]]))
                if d > best[0]:
                    best = (d, cl[i], cl[j])
        class_a, class_b = best[1], best[2]

    za = centroids[int(class_a)]
    zb = centroids[int(class_b)]

    alphas = np.linspace(0.0, 1.0, steps)
    fig, axes = plt.subplots(1, steps, figsize=(steps * 1.3, 1.8))
    axes = np.atleast_1d(axes).ravel()
    for k, a in enumerate(alphas):
        z = (1 - a) * za + a * zb
        axes[k].imshow(_img(vae.decode(z[None, :])[0], image_shape),
                       cmap="gray", interpolation="nearest")
        axes[k].axis("off")
    fig.suptitle(f"Interpolación en el latente: {labels[class_a]} → {labels[class_b]}",
                 y=1.05)
    fig.tight_layout()
    save_fig(fig, path)
    return path


def plot_reconstructions(vae, X, y, labels, image_shape, path, per_class=1):
    """Original vs reconstrucción (z=mu) para una muestra de cada clase."""
    classes = np.unique(y)
    rows = 2
    cols = len(classes)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.3))
    for j, c in enumerate(classes):
        idx = np.where(y == c)[0][0]
        x = X[idx:idx + 1]
        x_hat = vae.reconstruct(x)[0]
        axes[0, j].imshow(_img(x[0], image_shape), cmap="gray", interpolation="nearest")
        axes[1, j].imshow(_img(x_hat, image_shape), cmap="gray", interpolation="nearest")
        axes[0, j].axis("off"); axes[1, j].axis("off")
        axes[0, j].set_title(labels[c], fontsize=6)
    axes[0, 0].set_ylabel("orig", fontsize=8)
    axes[1, 0].set_ylabel("recon", fontsize=8)
    fig.suptitle("Reconstrucción (arriba: original, abajo: x̂ con z=μ)", y=1.03)
    fig.tight_layout()
    save_fig(fig, path)
    return path


def make_all_figures(vae, X, y, labels, image_shape, hist, results_dir):
    """Produce todas las figuras y devuelve la lista de rutas escritas."""
    paths = []
    paths.append(plot_loss_curves(hist, os.path.join(results_dir, "loss_curves.png")))
    paths.append(plot_latent_scatter(vae, X, y, labels,
                                     os.path.join(results_dir, "latent_scatter.png")))
    mg = plot_manifold_grid(vae, image_shape, os.path.join(results_dir, "manifold_grid.png"))
    if mg:
        paths.append(mg)
    paths.append(plot_samples(vae, image_shape, os.path.join(results_dir, "samples.png")))
    paths.append(plot_interpolation(vae, X, y, labels, image_shape,
                                    os.path.join(results_dir, "interpolation.png")))
    paths.append(plot_reconstructions(vae, X, y, labels, image_shape,
                                      os.path.join(results_dir, "reconstructions.png")))
    return paths
