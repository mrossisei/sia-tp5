"""Estudio 4 (multi-seed) — KL warmup {0, 200} epocas.

5 seeds × 2 warmups = 10 corridas.
CSV: exp_kl_warmup.csv  (una fila por seed × epoch × warmup — historiales completos
                          + exp_kl_warmup_summary.csv para metricas finales)
PNG: exp_kl_warmup.png  (curvas rec y KL con banda media ± std por warmup)
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

WARMUPS = [0, 200]
SEEDS   = [0, 1, 2, 3, 42]
EPOCHS  = 400


def make_schedule(warmup, beta_target=1.0):
    if warmup <= 0:
        return None
    return lambda ep: beta_target * min(1.0, (ep + 1) / warmup)


def main():
    X, y, labels, image_shape = load_emojis()
    # results[w] = lista de dicts por seed (incluye hist completo)
    results = {w: [] for w in WARMUPS}

    t0 = time.time()
    for w in WARMUPS:
        sched = make_schedule(w)
        for seed in SEEDS:
            t = time.time()
            vae, hist = train_vae(X, latent_dim=2, beta=1.0,
                                  beta_schedule=sched,
                                  epochs=EPOCHS, seed=seed)
            X_hat = vae.reconstruct(X)
            rec_mse = float(np.mean(np.sum((X - X_hat) ** 2, axis=1)))
            results[w].append({
                "seed": seed,
                "rec_final": hist["rec"][-1],
                "kl_final":  hist["kl"][-1],
                "rec_mse":   rec_mse,
                "hist_rec":  hist["rec"],
                "hist_kl":   hist["kl"],
            })
            print(f"[warmup={w:3d} seed={seed:3d}] {time.time()-t:.1f}s  "
                  f"rec={hist['rec'][-1]:.3f}  KL={hist['kl'][-1]:.3f}  "
                  f"MSE={rec_mse:.4f}", flush=True)

    # CSV historiales completos (warmup, seed, epoch, rec, kl)
    csv_hist = os.path.join(RESULTS_DIR, "exp_kl_warmup.csv")
    with open(csv_hist, "w", newline="") as f:
        w_csv = csv.writer(f)
        w_csv.writerow(["warmup", "seed", "epoch", "rec", "kl"])
        for w in WARMUPS:
            for r in results[w]:
                for ep in range(EPOCHS):
                    w_csv.writerow([w, r["seed"], ep + 1,
                                    f"{r['hist_rec'][ep]:.4f}",
                                    f"{r['hist_kl'][ep]:.4f}"])
    print(f"[csv] {csv_hist}")

    # CSV resumen metricas finales
    csv_summ = os.path.join(RESULTS_DIR, "exp_kl_warmup_summary.csv")
    with open(csv_summ, "w", newline="") as f:
        w_csv = csv.writer(f)
        w_csv.writerow(["warmup", "seed", "rec_final", "kl_final", "rec_mse"])
        for w in WARMUPS:
            for r in results[w]:
                w_csv.writerow([w, r["seed"],
                                 f"{r['rec_final']:.4f}",
                                 f"{r['kl_final']:.4f}",
                                 f"{r['rec_mse']:.4f}"])
    print(f"[csv] {csv_summ}")

    # Figura: 2 paneles (rec y KL) con banda media ± std
    epochs_ax = np.arange(1, EPOCHS + 1)
    colors = {0: PALETTE["primary"], 200: PALETTE["secondary"]}
    labels_w = {0: "sin warmup (β=1 desde ep. 0)",
                200: "warmup=200 (β sube en 200 ep.)"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for w in WARMUPS:
        rec_mat = np.array([r["hist_rec"] for r in results[w]])
        kl_mat  = np.array([r["hist_kl"]  for r in results[w]])
        rec_m, rec_s = rec_mat.mean(axis=0), rec_mat.std(axis=0)
        kl_m,  kl_s  = kl_mat.mean(axis=0),  kl_mat.std(axis=0)
        col = colors[w]; lab = labels_w[w]

        axes[0].plot(epochs_ax, rec_m, color=col, lw=1.6, label=lab)
        axes[0].fill_between(epochs_ax, rec_m - rec_s, rec_m + rec_s,
                             color=col, alpha=0.20)
        axes[1].plot(epochs_ax, kl_m, color=col, lw=1.6, label=lab)
        axes[1].fill_between(epochs_ax, kl_m - kl_s, kl_m + kl_s,
                             color=col, alpha=0.20)

    axes[0].set_title("Reconstruccion por epoca")
    axes[0].set_xlabel("Epoca"); axes[0].set_ylabel("rec loss")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].set_title("KL por epoca")
    axes[1].set_xlabel("Epoca"); axes[1].set_ylabel("KL (nats)")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    fig.suptitle(f"Estudio 4 — KL warmup: media ± std, {len(SEEDS)} seeds, {EPOCHS} epocas",
                 y=1.02, fontsize=11)
    fig.tight_layout()

    path = os.path.join(RESULTS_DIR, "exp_kl_warmup.png")
    save_fig(fig, path)
    print(f"[ok] kl_warmup completo en {time.time()-t0:.0f}s -> {path}")
    print("FIN_WARMUP")


if __name__ == "__main__":
    main()
