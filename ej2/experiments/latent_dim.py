"""Estudio 1 — dim latente {2, 8, 16} sobre reconstrucción.

2 seeds × 3 dims = 6 corridas.
CSV:        exp_latent_dim.csv             (métricas por seed × dim)
PNG:        exp_latent_dim_reconstructions.png  (reconstrucciones comparativas)
PNG:        exp_latent_dim_mse.png         (barra de MSE ± std)
"""

import os, sys, csv, time
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.plotting import save_fig, PALETTE
from ej2.experiments._common import load_emojis, train_vae, img, RESULTS_DIR

DIMS   = [2, 8, 16]
SEEDS  = [0, 1, 2, 3, 42]
EPOCHS = 1000

SHOW_CLASSES = [
    "cara_feliz", "cara_triste", "fuego", "luna", "estrella",
    "corazon", "sol", "flor", "pulgar_arriba", "nube",
]


def main():
    X, y, labels, image_shape = load_emojis()
    label_list = list(labels)
    results = {d: [] for d in DIMS}

    t0 = time.time()
    for d in DIMS:
        for seed in SEEDS:
            t = time.time()
            vae, hist = train_vae(X, latent_dim=d, epochs=EPOCHS, seed=seed)
            X_hat = vae.reconstruct(X)
            rec_mse = float(np.mean(np.sum((X - X_hat) ** 2, axis=1)))
            results[d].append({
                "seed": seed, "vae": vae,
                "rec_final": hist["rec"][-1],
                "kl_final":  hist["kl"][-1],
                "rec_mse":   rec_mse,
            })
            print(f"[dim={d:2d} seed={seed:2d}] {time.time()-t:.1f}s  "
                  f"rec={hist['rec'][-1]:.3f}  KL={hist['kl'][-1]:.3f}  "
                  f"MSE={rec_mse:.4f}", flush=True)

    # CSV
    csv_path = os.path.join(RESULTS_DIR, "exp_latent_dim.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dim", "seed", "epochs", "rec_final", "kl_final", "rec_mse"])
        for d in DIMS:
            for r in results[d]:
                w.writerow([d, r["seed"], EPOCHS,
                            f"{r['rec_final']:.4f}",
                            f"{r['kl_final']:.4f}",
                            f"{r['rec_mse']:.4f}"])
    print(f"[csv] {csv_path}")

    # Métricas agregadas por dim
    means = {d: float(np.mean([r["rec_mse"] for r in results[d]])) for d in DIMS}
    stds  = {d: float(np.std( [r["rec_mse"] for r in results[d]])) for d in DIMS}

    # ================================================================
    # Figura 1: reconstrucciones comparativas (seed=42)
    # ================================================================

    # Un VAE por dim, el de seed=42
    def _by_seed(rows, s):
        return [r for r in rows if r["seed"] == s][0]

    rep_vaes = {d: _by_seed(results[d], 42)["vae"] for d in DIMS}

    # Un ejemplo por clase
    orig_vecs = []
    class_names = []
    for name in SHOW_CLASSES:
        if name in label_list:
            ci = label_list.index(name)
            idx = int(np.where(y == ci)[0][0])
            orig_vecs.append(X[idx])
            class_names.append(name)

    orig_vecs = np.array(orig_vecs)
    ncols = len(class_names)

    # Reconstrucción determinística
    rec_vecs = {}
    for d in DIMS:
        rec_vecs[d] = rep_vaes[d].reconstruct(orig_vecs)

    nrows = 1 + len(DIMS)
    fig1, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 2.0, nrows * 2.5))

    row_labels = ["Original"] + [f"dim={d}" for d in DIMS]

    for col in range(ncols):
        ax = axes[0, col]
        ax.imshow(img(orig_vecs[col], image_shape), cmap="gray",
                  interpolation="nearest", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(class_names[col].replace("_", " "), fontsize=10)

    for row, d in enumerate(DIMS, start=1):
        for col in range(ncols):
            ax = axes[row, col]
            ax.imshow(img(rec_vecs[d][col], image_shape), cmap="gray",
                      interpolation="nearest", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=11, fontweight="bold")

    subtitle_parts = [f"dim={d}: MSE = {means[d]:.2f} ± {stds[d]:.2f}" for d in DIMS]
    subtitle = "  |  ".join(subtitle_parts)

    fig1.suptitle("Reconstrucción por dimensión latente (z = μ(x), seed=42)",
                  fontsize=13, y=1.02)
    fig1.text(0.5, 0.0, subtitle, ha="center", fontsize=9,
              transform=fig1.transFigure)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    path1 = os.path.join(RESULTS_DIR, "exp_latent_dim_reconstructions.png")
    save_fig(fig1, path1)
    print(f"[ok] reconstrucciones -> {path1}")

    # ================================================================
    # Figura 2: barras de MSE ± std
    # ================================================================

    fig2, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(DIMS))
    colors = plt.get_cmap("viridis")(np.linspace(0.25, 0.75, len(DIMS)))

    bars = ax.bar(x, [means[d] for d in DIMS],
                  yerr=[stds[d] for d in DIMS],
                  color=colors, capsize=6, width=0.5,
                  error_kw=dict(elinewidth=1.6, capthick=1.6))

    ax.set_xticks(x)
    ax.set_xticklabels([f"dim={d}" for d in DIMS])
    ax.set_ylabel("rec MSE (media ± std)")
    ax.set_title(f"MSE de reconstrucción por dim latente ({len(SEEDS)} seeds)",
                 fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    for rect, d in zip(bars, DIMS):
        ax.text(rect.get_x() + rect.get_width() / 2,
                rect.get_height() + stds[d] + 0.05,
                f"{means[d]:.2f}", ha="center", va="bottom", fontsize=9)

    fig2.tight_layout()

    path2 = os.path.join(RESULTS_DIR, "exp_latent_dim_mse.png")
    save_fig(fig2, path2)
    print(f"[ok] mse bars -> {path2}")

    print(f"FIN_LATENT en {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
