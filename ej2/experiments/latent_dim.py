"""Estudio 1 (multi-seed) — dim latente {2, 8, 16}.

5 seeds × 3 dims = 15 corridas.
CSV: exp_latent_dim.csv  (una fila por seed × dim, metricas finales)
PNG: exp_latent_dim.png  (grilla de muestras de seed representativa
                          + barchart rec_MSE ± std)
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
EPOCHS = 400
N_QUAL = 8   # muestras cualitativas por dim


def rep_idx(rows, key="rec_final"):
    """Indice de la seed cuyo 'key' esta mas cerca de la mediana."""
    vals = np.array([r[key] for r in rows])
    return int(np.argmin(np.abs(vals - np.median(vals))))


def main():
    X, y, labels, image_shape = load_emojis()
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
            print(f"[dim={d:2d} seed={seed:3d}] {time.time()-t:.1f}s  "
                  f"rec={hist['rec'][-1]:.3f}  KL={hist['kl'][-1]:.3f}  "
                  f"MSE={rec_mse:.4f}", flush=True)

    # CSV completo
    csv_path = os.path.join(RESULTS_DIR, "exp_latent_dim.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dim", "seed", "rec_final", "kl_final", "rec_mse"])
        for d in DIMS:
            for r in results[d]:
                w.writerow([d, r["seed"],
                             f"{r['rec_final']:.4f}",
                             f"{r['kl_final']:.4f}",
                             f"{r['rec_mse']:.4f}"])
    print(f"[csv] {csv_path}")

    # Estadisticas por dim
    means = {d: np.mean([r["rec_mse"] for r in results[d]]) for d in DIMS}
    stds  = {d: np.std( [r["rec_mse"] for r in results[d]]) for d in DIMS}

    # Figura: grilla de muestras (seed representativa) + barchart
    n_img_rows = len(DIMS)
    fig = plt.figure(figsize=(N_QUAL * 1.25, n_img_rows * 1.3 + 2.8))
    gs = fig.add_gridspec(n_img_rows + 1, N_QUAL,
                          height_ratios=[1.0] * n_img_rows + [2.5],
                          hspace=0.35, wspace=0.05)

    cmap_dim = plt.get_cmap("viridis")(np.linspace(0.2, 0.8, len(DIMS)))

    for i, d in enumerate(DIMS):
        ri = rep_idx(results[d])
        vae_rep = results[d][ri]["vae"]
        samples, _ = vae_rep.generate(N_QUAL, seed=7)
        for j in range(N_QUAL):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(img(samples[j], image_shape), cmap="gray",
                      interpolation="nearest", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
        fig.add_subplot(gs[i, 0]).set_ylabel(
            f"dim={d}", fontsize=9, rotation=0, labelpad=28, va="center")
        row_title = (f"dim={d}  rec={means[d]:.1f}±{stds[d]:.2f}  "
                     f"(seed rep.={results[d][ri]['seed']})")
        fig.add_subplot(gs[i, N_QUAL // 2]).set_title(row_title, fontsize=8)

    # Barchart rec_MSE ± std
    ax_bar = fig.add_subplot(gs[n_img_rows, 2:N_QUAL - 2])
    x = np.arange(len(DIMS))
    bars = ax_bar.bar(x, [means[d] for d in DIMS],
                      yerr=[stds[d] for d in DIMS],
                      color=cmap_dim, capsize=6, width=0.5,
                      error_kw=dict(elinewidth=1.6, capthick=1.6))
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"dim={d}" for d in DIMS])
    ax_bar.set_ylabel("rec MSE (media ± std)")
    ax_bar.set_title(f"Reconstruccion por dim latente ({len(SEEDS)} seeds)", fontsize=9)
    ax_bar.grid(axis="y", alpha=0.3)
    for rect, d in zip(bars, DIMS):
        ax_bar.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + stds[d] + 0.05,
                    f"{means[d]:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"Estudio 1 — dim latente: muestras z~N(0,I) (seed representativa) "
                 f"y rec MSE (media±std, {len(SEEDS)} seeds)", y=1.01, fontsize=10)

    path = os.path.join(RESULTS_DIR, "exp_latent_dim.png")
    save_fig(fig, path)
    print(f"[ok] latent_dim completo en {time.time()-t0:.0f}s -> {path}")
    print("FIN_LATENT")


if __name__ == "__main__":
    main()
