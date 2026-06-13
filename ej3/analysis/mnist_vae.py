"""Figuras del EJ3 (VAE sobre MNIST) — análisis del barrido de PROFUNDIDAD.

Lee los .npz que deja ej3/run_resumable.py (modelos + historiales) y produce
las figuras en ej3/results/. NO re-entrena: re-grafica a partir de datos crudos
(regla del repo). matplotlib se usa SÓLO acá.

Uso:
    python3 ej3/analysis/mnist_vae.py            # todas las figuras disponibles
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.config_loader import load_yaml
from ej2.models.vae import VAE
from ej3.experiments.depth_study import make_jobs
import ej3.run_resumable as R

EJ3_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(EJ3_DIR, "results")
IMG = (28, 28)


def _available(jobs):
    """Jobs que ya tienen modelo + historial guardados."""
    out = []
    for j in jobs:
        mp = os.path.join(RESULTS_DIR, f"{j['id']}_model.npz")
        hp = os.path.join(RESULTS_DIR, f"{j['id']}_hist.npz")
        if os.path.exists(mp) and os.path.exists(hp):
            out.append(j)
    return out


def _grid(ax, images, n_cols, title=None):
    """Apila imágenes (cada una vector 784) en una grilla y la dibuja."""
    n = len(images)
    n_rows = int(np.ceil(n / n_cols))
    canvas = np.ones((n_rows * IMG[0], n_cols * IMG[1]))
    for i, vec in enumerate(images):
        r, c = divmod(i, n_cols)
        canvas[r*IMG[0]:(r+1)*IMG[0], c*IMG[1]:(c+1)*IMG[1]] = np.asarray(vec).reshape(IMG)
    ax.imshow(canvas, cmap="gray_r", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)


def fig_reconstructions(job, Xte, yte, n=10):
    vae = VAE.load(os.path.join(RESULTS_DIR, f"{job['id']}_model.npz"))
    # un ejemplo de cada dígito 0..9 si se puede
    picks = []
    for d in range(10):
        idx = np.where(yte == d)[0]
        if len(idx):
            picks.append(idx[0])
    Xs = Xte[picks]
    Xr = vae.reconstruct(Xs)
    fig, axes = plt.subplots(2, 1, figsize=(n, 2.4))
    _grid(axes[0], Xs, n, "originales (test)")
    _grid(axes[1], Xr, n, f"reconstruidas — {job['id']} ({job['n_hidden']} capas ocultas)")
    fig.tight_layout()
    p = os.path.join(RESULTS_DIR, f"recon_{job['id']}.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    return p


def fig_samples(job, n=40, seed=0):
    vae = VAE.load(os.path.join(RESULTS_DIR, f"{job['id']}_model.npz"))
    samples, _ = vae.generate(n, seed=seed)
    fig, ax = plt.subplots(figsize=(8, 0.8 * (n / 10)))
    _grid(ax, samples, 10, f"muestras generadas z~N(0,I) — {job['id']} "
                           f"({job['n_hidden']} capas ocultas)")
    fig.tight_layout()
    p = os.path.join(RESULTS_DIR, f"samples_{job['id']}.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    return p


def fig_curves(jobs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for j in jobs:
        h = np.load(os.path.join(RESULTS_DIR, f"{j['id']}_hist.npz"))
        lab = f"{j['id']} ({j['n_hidden']} oc.)"
        axes[0].plot(h["rec"], label=lab)
        axes[1].plot(h["test_rec"], label=lab)
    axes[0].set_title("Reconstrucción en TRAIN (BCE)")
    axes[1].set_title("Reconstrucción en TEST / held-out (BCE)")
    for ax in axes:
        ax.set_xlabel("época"); ax.set_ylabel("BCE por muestra")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("EJ3 — efecto de la PROFUNDIDAD sobre la reconstrucción", fontsize=12)
    fig.tight_layout()
    p = os.path.join(RESULTS_DIR, "depth_curves.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    return p


def fig_depth_summary(jobs):
    """El resultado central del experimento: métrica final vs cantidad de capas."""
    nh, test_rec, train_rec, mins = [], [], [], []
    for j in jobs:
        h = np.load(os.path.join(RESULTS_DIR, f"{j['id']}_hist.npz"))
        nh.append(int(h["n_hidden"]))
        test_rec.append(float(h["test_rec"][-1]))
        train_rec.append(float(h["rec"][-1]))
        mins.append(float(h["elapsed_sec"]) / 60.0)
    order = np.argsort(nh)
    nh = np.array(nh)[order]; test_rec = np.array(test_rec)[order]
    train_rec = np.array(train_rec)[order]; mins = np.array(mins)[order]

    fig, ax1 = plt.subplots(figsize=(7.5, 4.8))
    ax1.plot(nh, test_rec, "o-", color="C0", label="test rec (BCE)")
    ax1.plot(nh, train_rec, "s--", color="C2", label="train rec (BCE)")
    ax1.set_xlabel("cantidad de capas ocultas del encoder")
    ax1.set_ylabel("BCE de reconstrucción (menor = mejor)")
    ax1.set_xticks(nh); ax1.grid(alpha=0.3); ax1.legend(loc="upper right")
    ax2 = ax1.twinx()
    ax2.bar(nh, mins, width=0.25, alpha=0.25, color="C3")
    ax2.set_ylabel("tiempo de entrenamiento [min]  (barras)")
    ax1.set_title("EJ3 — ¿ayuda agregar capas? (reconstrucción y costo vs profundidad)")
    fig.tight_layout()
    p = os.path.join(RESULTS_DIR, "depth_summary.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    return p


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(EJ3_DIR, "config.yaml")
    cfg = load_yaml(cfg_path)
    jobs = _available(make_jobs(cfg))
    if not jobs:
        print("No hay jobs terminados todavía. Corré primero: python3 ej3/run_resumable.py")
        return
    print(f"Jobs con resultados: {[j['id'] for j in jobs]}")

    # dataset de test (para reconstrucciones), mismo subset determinista
    _, _, Xte, yte = R.load_mnist(cfg)

    paths = []
    for j in jobs:
        paths.append(fig_reconstructions(j, Xte, yte))
        paths.append(fig_samples(j))
    paths.append(fig_curves(jobs))
    if len(jobs) >= 2:
        paths.append(fig_depth_summary(jobs))

    print("\nfiguras generadas:")
    for p in paths:
        print(f"  -> {p}")


if __name__ == "__main__":
    main()
