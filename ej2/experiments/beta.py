"""Estudio 2 (multi-seed) — β {0.1, 1.0, 4.0}.

5 seeds × 3 betas = 15 corridas.
CSV: exp_beta.csv  (una fila por seed × beta, metricas finales)
PNG: exp_beta.png  (fila superior: scatter latente de seed representativa;
                    fila inferior: barchart rec_final y KL_final ± std)
"""

import os, sys, csv, time
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.plotting import save_fig, PALETTE
from ej2.experiments._common import load_emojis, train_vae, RESULTS_DIR

BETAS  = [0.1, 1.0, 4.0]
SEEDS  = [0, 1, 2, 3, 42]
EPOCHS = 400


def rep_idx(rows, key="rec_final"):
    vals = np.array([r[key] for r in rows])
    return int(np.argmin(np.abs(vals - np.median(vals))))


def main():
    X, y, labels, image_shape = load_emojis()
    n_classes = len(labels)
    results = {b: [] for b in BETAS}

    t0 = time.time()
    for b in BETAS:
        for seed in SEEDS:
            t = time.time()
            vae, hist = train_vae(X, latent_dim=2, beta=b, epochs=EPOCHS, seed=seed)
            X_hat = vae.reconstruct(X)
            rec_mse = float(np.mean(np.sum((X - X_hat) ** 2, axis=1)))
            results[b].append({
                "seed": seed, "vae": vae,
                "rec_final": hist["rec"][-1],
                "kl_final":  hist["kl"][-1],
                "rec_mse":   rec_mse,
            })
            print(f"[beta={b:.1f} seed={seed:3d}] {time.time()-t:.1f}s  "
                  f"rec={hist['rec'][-1]:.3f}  KL={hist['kl'][-1]:.3f}  "
                  f"MSE={rec_mse:.4f}", flush=True)

    # CSV completo
    csv_path = os.path.join(RESULTS_DIR, "exp_beta.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["beta", "seed", "rec_final", "kl_final", "rec_mse"])
        for b in BETAS:
            for r in results[b]:
                w.writerow([b, r["seed"],
                             f"{r['rec_final']:.4f}",
                             f"{r['kl_final']:.4f}",
                             f"{r['rec_mse']:.4f}"])
    print(f"[csv] {csv_path}")

    # Estadisticas por beta
    rec_mean = {b: np.mean([r["rec_final"] for r in results[b]]) for b in BETAS}
    rec_std  = {b: np.std( [r["rec_final"] for r in results[b]]) for b in BETAS}
    kl_mean  = {b: np.mean([r["kl_final"]  for r in results[b]]) for b in BETAS}
    kl_std   = {b: np.std( [r["kl_final"]  for r in results[b]]) for b in BETAS}

    # Figura: 2 filas — scatter (top) + barchart (bottom)
    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(2, len(BETAS) + 1,
                          height_ratios=[1.7, 1.0],
                          width_ratios=[1.0] * len(BETAS) + [0.05],
                          hspace=0.45, wspace=0.35)

    cmap = plt.get_cmap("tab20", n_classes)

    for i, b in enumerate(BETAS):
        ri = rep_idx(results[b])
        vae_rep = results[b][ri]["vae"]
        mu = vae_rep.encode(X)

        ax = fig.add_subplot(gs[0, i])
        for c in range(n_classes):
            m = (y == c)
            ax.scatter(mu[m, 0], mu[m, 1], s=10, color=cmap(c), alpha=0.6,
                       edgecolors="none")
        ax.set_title(
            f"β={b}\nrec={rec_mean[b]:.1f}±{rec_std[b]:.1f}  "
            f"KL={kl_mean[b]:.2f}±{kl_std[b]:.2f}",
            fontsize=9)
        ax.set_xlabel("z[0]"); ax.set_ylabel("z[1]")
        ax.grid(alpha=0.3)

    # Barchart rec ± std
    x = np.arange(len(BETAS))
    labels_x = [f"β={b}" for b in BETAS]
    bar_colors = [PALETTE["positive"], PALETTE["highlight"]]

    ax_rec = fig.add_subplot(gs[1, 0])
    ax_rec.bar(x, [rec_mean[b] for b in BETAS],
               yerr=[rec_std[b] for b in BETAS],
               color=PALETTE["positive"], capsize=6, width=0.5,
               error_kw=dict(elinewidth=1.6, capthick=1.6))
    ax_rec.set_xticks(x); ax_rec.set_xticklabels(labels_x)
    ax_rec.set_ylabel("rec loss (BCE)")
    ax_rec.set_title("Reconstruccion (media ± std)", fontsize=9)
    ax_rec.grid(axis="y", alpha=0.3)

    ax_kl = fig.add_subplot(gs[1, 1])
    ax_kl.bar(x, [kl_mean[b] for b in BETAS],
              yerr=[kl_std[b] for b in BETAS],
              color=PALETTE["highlight"], capsize=6, width=0.5,
              error_kw=dict(elinewidth=1.6, capthick=1.6))
    ax_kl.set_xticks(x); ax_kl.set_xticklabels(labels_x)
    ax_kl.set_ylabel("KL (nats)")
    ax_kl.set_title("Regularizacion KL (media ± std)", fontsize=9)
    ax_kl.grid(axis="y", alpha=0.3)

    # Tercer panel vacio o con nota
    ax_note = fig.add_subplot(gs[1, 2])
    ax_note.axis("off")
    ax_note.text(0.5, 0.5,
                 f"Scatter: seed representativa\n(rec mas cercana a la mediana).\n"
                 f"Barras: media ± std, {len(SEEDS)} seeds.",
                 ha="center", va="center", fontsize=8, wrap=True,
                 transform=ax_note.transAxes)

    fig.suptitle(f"Estudio 2 — efecto de β en el latente ({len(SEEDS)} seeds, {EPOCHS} epocas)",
                 y=1.01, fontsize=11)

    path = os.path.join(RESULTS_DIR, "exp_beta.png")
    save_fig(fig, path)
    print(f"[ok] beta completo en {time.time()-t0:.0f}s -> {path}")
    print("FIN_BETA")


if __name__ == "__main__":
    main()
