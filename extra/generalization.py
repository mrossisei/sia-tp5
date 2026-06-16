"""EXTRA-D — ¿El VAE de emojis memoriza o generaliza?

El EJ2 entrena con TODAS las variantes augmentadas y nunca mide en datos no
vistos (a diferencia del EJ3, que sí tiene test_rec). Acá traemos la disciplina
del TP3: split estratificado 80/20 por clase (cada emoji aporta ~20% de sus ~60
variantes al test held-out), entrenamos SOLO con el train y medimos rec en train
y en test a lo largo del entrenamiento.

  - Si rec_train << rec_test y rec_test sube -> memoriza (overfitting).
  - Si rec_train ~ rec_test -> generaliza (aprende a reconstruir emojis, no
    variantes puntuales).

Barremos la dimensión latente {2,8,16,32} para ver si más capacidad => más
overfit. HALLAZGO (ver generalization.csv): el gap NO crece con la dim latente
(8/16/32 dan gap ~+5; latente 2 underfitea con +13). Conecta con EXTRA-B: el
posterior collapse limita la capacidad EFECTIVA (~5-7 dims activas), así que dar
más latente no agrega sobreajuste.

FIGURAS (extra/results/):
  - generalization_curves.png : rec train vs test (latent 16) + gap final vs latente
  - generalization_heldout.png: reconstrucciones de variantes NO vistas (held-out)
"""

import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (RESULTS, ensure_dirs, load_emojis, make_vae,  # noqa: E402
                     fit_tracking, rec_bce, rec_mse, stratified_split)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from shared.plotting import save_fig, PALETTE  # noqa: E402

LATENTS = [2, 8, 16, 32]
DETAIL_LATENT = 16          # cuál se muestra en detalle (train vs test por época)
EPOCHS = 1200
SEED = 42


def plot_curves(runs, path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # Panel 1: train vs test para el latente "detalle"
    r = runs[DETAIL_LATENT]
    ax = axes[0]
    ax.plot(r["ev"], r["rec_tr"], color=PALETTE["primary"], lw=1.8,
            label="train")
    ax.plot(r["ev"], r["rec_te"], color=PALETTE["negative"], lw=1.8, ls="--",
            label="test (held-out)")
    ax.set_title(f"Reconstrucción train vs held-out (latent={DETAIL_LATENT})")
    ax.set_xlabel("Época"); ax.set_ylabel("BCE de reconstrucción")
    ax.grid(alpha=0.3); ax.legend()

    # Panel 2: rec final train/test y GAP vs dimensión latente
    ax = axes[1]
    lats = LATENTS
    tr = [runs[d]["rec_tr_final"] for d in lats]
    te = [runs[d]["rec_te_final"] for d in lats]
    ax.plot(lats, tr, "o-", color=PALETTE["primary"], label="train")
    ax.plot(lats, te, "s--", color=PALETTE["negative"], label="test (held-out)")
    for d in lats:
        gap = runs[d]["rec_te_final"] - runs[d]["rec_tr_final"]
        ax.annotate(f"+{gap:.1f}", (d, runs[d]["rec_te_final"]),
                    textcoords="offset points", xytext=(4, 6), fontsize=8,
                    color=PALETTE["negative"])
    ax.set_xscale("log", base=2); ax.set_xticks(lats)
    ax.set_xticklabels([str(d) for d in lats])
    ax.set_xlabel("dimensión latente")
    ax.set_ylabel("BCE final"); ax.set_title("Gap train↔test vs capacidad latente")
    ax.grid(alpha=0.3); ax.legend()

    fig.suptitle("EJ2 — Generalización del VAE (split 80/20 por clase)", y=1.02)
    fig.tight_layout()
    save_fig(fig, path)


def plot_heldout(vae, X_te, y_te, labels, image_shape, path, n=8):
    """Reconstrucciones de variantes NO vistas en entrenamiento."""
    rng = np.random.default_rng(0)
    pick = rng.permutation(len(X_te))[:n]
    X_hat = vae.reconstruct(X_te[pick])
    fig, axes = plt.subplots(2, n, figsize=(n * 1.2, 2.7))
    for j, i in enumerate(pick):
        axes[0, j].imshow(X_te[i].reshape(image_shape), cmap="gray",
                          vmin=0, vmax=1, interpolation="nearest")
        axes[1, j].imshow(X_hat[j].reshape(image_shape), cmap="gray",
                          vmin=0, vmax=1, interpolation="nearest")
        axes[0, j].set_title(labels[int(y_te[i])], fontsize=7)
        for r in (0, 1):
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
    axes[0, 0].set_ylabel("no visto", fontsize=9)
    axes[1, 0].set_ylabel("recon", fontsize=9)
    fig.suptitle(f"Reconstrucción de variantes HELD-OUT (latent={DETAIL_LATENT})",
                 y=1.04)
    fig.tight_layout()
    save_fig(fig, path)


def main():
    ensure_dirs()
    print("=== EXTRA-D: generalización (split train/test) ===")
    X, y, labels, image_shape = load_emojis()
    tr_idx, te_idx = stratified_split(y, test_frac=0.2, seed=0)
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_te = y[te_idx]
    print(f"train={len(tr_idx)}  test(held-out)={len(te_idx)}  "
          f"({len(np.unique(y))} clases)")

    runs = {}
    detail_vae = None
    for lat in LATENTS:
        vae = make_vae(X.shape[1], lat, [256, 64], [64, 256], seed=SEED)
        h = fit_tracking(vae, X_tr, X_te, epochs=EPOCHS, batch_size=64, lr=1e-3,
                         seed=SEED, eval_every=10)
        runs[lat] = h
        if lat == DETAIL_LATENT:
            detail_vae = vae
        print(f"[latent {lat:<2}] rec_train={h['rec_tr_final']:.2f}  "
              f"rec_test={h['rec_te_final']:.2f}  "
              f"gap={h['rec_te_final'] - h['rec_tr_final']:+.2f}")

    plot_curves(runs, os.path.join(RESULTS, "generalization_curves.png"))
    plot_heldout(detail_vae, X_te, y_te, labels, image_shape,
                 os.path.join(RESULTS, "generalization_heldout.png"))

    with open(os.path.join(RESULTS, "generalization.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["latent", "rec_train", "rec_test", "gap"])
        for lat in LATENTS:
            h = runs[lat]
            w.writerow([lat, f"{h['rec_tr_final']:.4f}", f"{h['rec_te_final']:.4f}",
                        f"{h['rec_te_final'] - h['rec_tr_final']:.4f}"])
    print(f"[ok] EXTRA-D -> {RESULTS}")


if __name__ == "__main__":
    main()
