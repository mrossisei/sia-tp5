"""Figuras post-overnight del VAE largo (consume los datos crudos guardados).

No re-entrena nada: usa los snapshots (modelos intermedios), los historiales
de loss por época y los modelos finales que dejó run_overnight.sh.

Genera (en ej2/results/):
  - evolution_16px.png / evolution_24px.png : muestras decodificadas de un
    set FIJO de z a lo largo del entrenamiento (filas = época del snapshot)
  - comparison_600_vs_50k.png : curva de reconstrucción de 50k épocas con la
    marca de las 600 originales + reconstrucciones lado a lado

Uso:
    python3 ej2/analysis/long_run.py
"""

import csv
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.plotting import save_fig, PALETTE
from ej2.models.vae import VAE

RESULTS = os.path.join(REPO_ROOT, "ej2", "results")


def _load_dataset(path):
    d = np.load(path, allow_pickle=True)
    return (d["X"].astype(np.float64), d["y"], [str(s) for s in d["labels"]],
            tuple(int(v) for v in d["image_shape"]))


def _snapshot_path(tag, ep):
    return os.path.join(RESULTS, f"long_{tag}", "snapshots", f"model_ep{ep:06d}.npz")


def plot_evolution(tag, epochs_list, image_shape, path, n_samples=8, z_seed=7):
    """Decodifica los MISMOS z con cada snapshot: cómo madura el decoder."""
    rows = [ep for ep in epochs_list if os.path.exists(_snapshot_path(tag, ep))]
    if not rows:
        return
    # Dimensión latente leída del propio snapshot (no hardcodeada): el z fijo
    # debe tener la dim del modelo (los long runs usan 2, pero esto lo soporta
    # para cualquier latente).
    latent_dim = VAE.load(_snapshot_path(tag, rows[0])).latent_dim
    rng = np.random.default_rng(z_seed)
    # z fijos (los mismos para todas las filas -> la evolución es del modelo)
    Z = rng.standard_normal(size=(n_samples, latent_dim))
    fig, axes = plt.subplots(len(rows), n_samples,
                             figsize=(n_samples * 1.25, len(rows) * 1.35))
    axes = np.atleast_2d(axes)
    for i, ep in enumerate(rows):
        vae = VAE.load(_snapshot_path(tag, ep))
        imgs = vae.decode(Z)
        for j in range(n_samples):
            ax = axes[i, j]
            ax.imshow(imgs[j].reshape(image_shape), cmap="gray",
                      interpolation="nearest", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"ép {ep}", fontsize=9, rotation=0,
                              labelpad=28, va="center")
    fig.suptitle(f"Evolución del decoder ({tag}): mismos z ~ N(0,I) "
                 "decodificados por snapshots sucesivos", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_fig(fig, path)
    return rows


def _load_history(tag):
    eps, rec = [], []
    with open(os.path.join(RESULTS, f"long_{tag}", "loss_history.csv")) as f:
        for row in csv.DictReader(f):
            eps.append(int(row["epoch"]))
            rec.append(float(row["rec"]))
    return np.array(eps), np.array(rec)


def plot_comparison_600_vs_50k(X, y, labels, image_shape, path, n_classes=8):
    """Curva de rec (50k ép) con la marca de las 600 originales + recons."""
    vae_600 = VAE.load(os.path.join(RESULTS, "vae_model.npz"))
    vae_50k = VAE.load(os.path.join(RESULTS, "long_16px", "vae_model.npz"))
    eps, rec = _load_history("16px")

    fig = plt.figure(figsize=(13.5, 4.2))
    gs = fig.add_gridspec(3, n_classes + 4,
                          width_ratios=[1.15] * 4 + [1] * n_classes,
                          hspace=0.05, wspace=0.05)

    # Panel izquierdo: curva de reconstrucción completa (log-log)
    ax = fig.add_subplot(gs[:, :4])
    ax.plot(eps, rec, color=PALETTE["primary"], lw=1.0)
    ax.axvline(600, color=PALETTE["highlight"], ls="--", lw=1.4,
               label="corte del run principal (600 ép)")
    ax.set_xscale("log")
    ax.set_xlabel("Época (log)")
    ax.set_ylabel("Loss de reconstrucción")
    ax.set_title("50.000 épocas: la reconstrucción seguía bajando", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # Paneles derecha: original / recon 600 / recon 50k para una muestra por clase
    classes = np.unique(y)[:n_classes]
    row_titles = ["original", "600 ép", "50.000 ép"]
    mats = []
    for c in classes:
        x = X[np.where(y == c)[0][0]][None, :]
        mats.append([x[0], vae_600.reconstruct(x)[0], vae_50k.reconstruct(x)[0]])
    for r in range(3):
        for ci, c in enumerate(classes):
            ax = fig.add_subplot(gs[r, 4 + ci])
            ax.imshow(mats[ci][r].reshape(image_shape), cmap="gray",
                      interpolation="nearest", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if ci == 0:
                ax.set_ylabel(row_titles[r], fontsize=8, rotation=0,
                              labelpad=22, va="center")
            if r == 0:
                ax.set_title(labels[int(c)], fontsize=6)
    fig.suptitle("VAE 16×16 (latente 2D): efecto de entrenar 600 vs 50.000 épocas",
                 y=1.02)
    save_fig(fig, path)


def main():
    X16, y16, lab16, shape16 = _load_dataset(
        os.path.join(REPO_ROOT, "ej2", "data", "emojis.npz"))

    rows = plot_evolution(
        "16px", [50, 100, 250, 500, 1000, 5000, 15000, 50000], shape16,
        os.path.join(RESULTS, "evolution_16px.png"))
    print(f"[evolution 16px] snapshots usados: {rows}")

    data24 = os.path.join(REPO_ROOT, "ej2", "data", "emojis_24.npz")
    if os.path.exists(data24):
        _, _, _, shape24 = _load_dataset(data24)
        rows = plot_evolution(
            "24px", [50, 100, 250, 500, 1000, 5000, 15000, 30000], shape24,
            os.path.join(RESULTS, "evolution_24px.png"))
        print(f"[evolution 24px] snapshots usados: {rows}")

    plot_comparison_600_vs_50k(
        X16, y16, lab16, shape16,
        os.path.join(RESULTS, "comparison_600_vs_50k.png"))
    print("[comparison] 600 vs 50k listo")
    print("[ok] long_run: figuras en", RESULTS)


if __name__ == "__main__":
    main()
