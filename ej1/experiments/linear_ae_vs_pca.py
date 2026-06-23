"""EJ1.a — ¿Alcanza con un Autoencoder LINEAL? (y por qué no: PCA)

Idea narrativa (va PRIMERO en la presentacion):
  1. Probamos lo mas simple: un AE *lineal* (todas las activaciones identidad,
     loss MSE), barriendo 5 seeds. ¿Llega al objetivo <=1 pixel?
  2. NO llega. ¿Por que? Porque un AE lineal es equivalente a PCA
     (Autoencoders.pdf, slides 10-16): la mejor reconstruccion lineal de rango 2
     retiene poca varianza. Lo demostramos empiricamente:
       - el AE lineal alcanza el MISMO error de reconstruccion (MSE) que PCA
         => esta bien entrenado; su limite es la LINEALIDAD, no el entrenamiento.
       - sus reconstrucciones coinciden pixel a pixel con las de PCA.
  3. El mismo cuello 2D, con activaciones NO lineales (el AE del 1.a), llega a
     max=0 => la no linealidad es lo que rompe el limite.

Nota teorica: un AE lineal colapsa a una unica transformacion lineal sin importar
la profundidad, asi que [35, 2, 35] no pierde generalidad como "AE lineal".

Genera (en ej1/results/basic/):
  - exp_linear_ae.png       : max pixel-error del AE lineal por seed (panel unico)
  - exp_linear_ae_worst.png : las 6 letras peor reconstruidas (error en rojo)
  - exp_linear_ae_latent.png: latente 2D aprendido por el AE lineal (seed 0)
  - exp_linear_ae_vs_pca_latent.png: AE lineal vs PCA, lado a lado
  - exp_linear_ae.csv       : tabla seed -> max/mean pixel-error, rec_MSE (incluye
                              fila PCA; rec_MSE coincide => el AE lineal == PCA)

El scatter del latente lineal usado en la slide se reusa de latent_scatter_pca.png
(generado por main_autoencoder.py).
"""

import os
import sys
import csv
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.fonts import load_font
from shared.plotting import save_fig, PALETTE
from shared.optimizers import Adam
from ej1.models.autoencoder import Autoencoder
from shared.metrics import pixel_error_summary, pixel_errors

OUT = os.path.join(REPO_ROOT, "ej1", "results", "basic")
# MISMA arquitectura que el AE no lineal: el unico cambio es la activacion
# (todas identidad vs tanh ocultas). Asi el experimento aisla la NO LINEALIDAD.
# Nota: un AE lineal colapsa a una unica transformacion lineal sin importar la
# profundidad, asi que esta red profunda es igualmente equivalente a PCA.
ARCH = [35, 60, 40, 20, 2, 20, 40, 60, 35]
SEEDS = [0, 1, 2, 3, 42]
EPOCHS = 20000
LR = 5e-3


def rec_mse(X, rec):
    """MSE de reconstruccion por elemento (comparable entre AE lineal y PCA)."""
    return float(np.mean((np.asarray(X) - np.asarray(rec)) ** 2))


def train_linear(X, seed, epochs=EPOCHS, lr=LR):
    """AE totalmente lineal (identity en todas las capas) + MSE, full-batch."""
    rng = np.random.default_rng(seed)
    ae = Autoencoder(
        architecture=ARCH,
        hidden_activation="identity",
        output_activation="identity",
        bottleneck_activation="identity",
        initializer="glorot",
        seed=seed,
        loss_name="mse",
    )
    opt = Adam(lr=lr)
    losses = []
    for _ in range(epochs):
        ml, _ = ae.train_epoch(X, X, opt, batch_size=0, shuffle=False, rng=rng)
        losses.append(ml)
    rec = ae.reconstruct(X)
    s = pixel_error_summary(X, rec)
    return ae, losses, s, rec_mse(X, rec), rec


def pca_baseline(X):
    """PCA 2D analitica (SVD sobre X centrada). Devuelve latente, recon y metricas."""
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V2 = Vt[:2]                       # (2, 35)
    pcs = Xc @ V2.T                   # (N, 2) proyeccion a PC1, PC2
    rec = pcs @ V2 + mean            # reconstruccion de rango 2
    var_ratio = (S[:2] ** 2) / np.sum(S ** 2)
    s = pixel_error_summary(X, rec)
    return pcs, var_ratio, rec, s, rec_mse(X, rec)


# --------------------------------------------------------------- figura unica
def fig_results(seed_max, path):
    """Max pixel-error del AE lineal por seed (panel unico)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(SEEDS))
    bars = ax.bar(x, seed_max, color=PALETTE["negative"], edgecolor="k", alpha=0.85)
    ax.axhline(1, color=PALETTE["highlight"], ls="--", lw=1.8, label="objetivo max$\\leq$1")
    for b, v in zip(bars, seed_max):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in SEEDS])
    ax.set_ylabel("max pixel-error")
    ax.set_title("AE lineal: max pixel-error por seed")
    ax.set_ylim(0, max(seed_max) * 1.15)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, path)


def fig_worst_letters(X, labels, rec, k, path):
    """Las k letras peor reconstruidas por el AE lineal (== PCA), error en rojo."""
    rec_bin = (rec >= 0.5).astype(int)
    errs = pixel_errors(X, rec)
    order = np.argsort(-errs)[:k]
    fig, axes = plt.subplots(2, k, figsize=(k * 1.5, 3.6))
    for c, i in enumerate(order):
        orig = X[i].reshape(7, 5)
        rb = rec_bin[i].reshape(7, 5)
        axes[0, c].imshow(orig, cmap="Greys", vmin=0, vmax=1)
        axes[0, c].set_title(f"'{labels[i]}'", fontsize=11)
        axes[1, c].imshow(rb, cmap="Greys", vmin=0, vmax=1)
        yy, xx = np.where(rb != orig)
        axes[1, c].scatter(xx, yy, marker="s", s=45, facecolors="none",
                           edgecolors=PALETTE["highlight"], linewidths=1.6)
        axes[1, c].set_title(f"err={errs[i]}", fontsize=11, color=PALETTE["negative"])
        for ax in (axes[0, c], axes[1, c]):
            ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_ylabel("original", fontsize=10)
    axes[1, 0].set_ylabel("AE lineal", fontsize=10)
    fig.suptitle("AE lineal: peores reconstrucciones (rojo = pixel erroneo)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, path)


def _annotated_scatter(ax, Z, labels, color, title, xlabel="z1", ylabel="z2"):
    ax.scatter(Z[:, 0], Z[:, 1], c=color, s=60, alpha=0.65,
               edgecolors="k", linewidths=0.5, zorder=2)
    for i, lab in enumerate(labels):
        ax.annotate(lab, (Z[i, 0], Z[i, 1]), fontsize=10, fontweight="bold",
                    ha="center", va="center", zorder=3)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def fig_linear_latent(Z_lin, labels, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    _annotated_scatter(ax, Z_lin, labels, PALETTE["negative"],
                       "Espacio latente 2D del AE lineal")
    fig.tight_layout()
    save_fig(fig, path)


def fig_linear_vs_pca_latent(Z_lin, pcs, var_ratio, labels, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.2))
    _annotated_scatter(ax1, Z_lin, labels, PALETTE["negative"], "AE lineal (seed 0)")
    _annotated_scatter(ax2, pcs, labels, PALETTE["accent"], "PCA 2D",
                       xlabel=f"PC1 ({var_ratio[0]*100:.1f}% var)",
                       ylabel=f"PC2 ({var_ratio[1]*100:.1f}% var)")
    fig.suptitle("AE lineal vs PCA: subespacio equivalente, coordenadas no identicas",
                 fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, path)


def main():
    t0 = time.time()
    X, labels = load_font()

    # PCA de referencia
    pcs, var_ratio, pca_rec, pca_s, pca_mse = pca_baseline(X)
    print(f"[PCA] max={pca_s['max']} mean={pca_s['mean']:.3f} exactas={pca_s['n_exact']}/32 "
          f"rec_MSE={pca_mse:.4f} var2D={(var_ratio[0]+var_ratio[1])*100:.1f}%")

    # AE lineal por seed
    rows, seed_max, seed_mse = [], [], []
    ae_seed0 = None
    for seed in SEEDS:
        t = time.time()
        ae, _, s, mse, rec = train_linear(X, seed)
        seed_max.append(s["max"])
        seed_mse.append(mse)
        rows.append((seed, s["max"], s["mean"], s["n_exact"], mse))
        if seed == SEEDS[0]:
            ae_seed0 = ae
        print(f"[AE lineal seed={seed}] max={s['max']} mean={s['mean']:.3f} "
              f"exactas={s['n_exact']}/32 rec_MSE={mse:.4f} ({time.time()-t:.0f}s)",
              flush=True)

    lin_mean = float(np.mean(seed_max))
    lin_std = float(np.std(seed_max))
    lin_mse_mean = float(np.mean(seed_mse))
    print(f"[AE lineal] max pixel-error = {lin_mean:.2f} +- {lin_std:.2f} (5 seeds); "
          f"rec_MSE medio {lin_mse_mean:.4f} vs PCA {pca_mse:.4f} "
          f"(coinciden => converge a PCA)")

    fig_results(seed_max, os.path.join(OUT, "exp_linear_ae.png"))
    fig_worst_letters(X, labels, pca_rec, 6, os.path.join(OUT, "exp_linear_ae_worst.png"))
    if ae_seed0 is not None:
        Z_lin = ae_seed0.encode(X)
        fig_linear_latent(Z_lin, labels, os.path.join(OUT, "exp_linear_ae_latent.png"))
        fig_linear_vs_pca_latent(
            Z_lin, pcs, var_ratio, labels,
            os.path.join(OUT, "exp_linear_ae_vs_pca_latent.png")
        )

    with open(os.path.join(OUT, "exp_linear_ae.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "max_pixel", "mean_pixel", "n_exact", "rec_mse"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.4f}", r[3], f"{r[4]:.6f}"])
        w.writerow([])
        w.writerow(["pca", pca_s["max"], f"{pca_s['mean']:.4f}", pca_s["n_exact"],
                    f"{pca_mse:.6f}"])

    print(f"[ok] linear_ae_vs_pca completo en {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
