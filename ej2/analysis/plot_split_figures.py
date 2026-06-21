"""Genera figuras separadas para Estudios 1 y 2 del VAE.

Lee los CSV ya guardados, encuentra la seed representativa (rec_mse mas
cercana a la mediana) y reentrena SOLO esas seeds para obtener los modelos.

Salidas en ej2/results/:
  exp_latent_dim_samples.png  grilla limpia de muestras por dim
  exp_latent_dim_bars.png     barchart rec_MSE ± std por dim
  exp_beta_scatter.png        scatter latente por β (seed representativa)
  exp_beta_bars.png           barchart rec y KL ± std por β
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

EPOCHS = 400


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_csv(path, key_col, val_cols):
    """Lee CSV y devuelve dict key -> {col: [values]}."""
    data = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            k = r[key_col]
            data.setdefault(k, {c: [] for c in val_cols})
            for c in val_cols:
                data[k][c].append(float(r[c]))
    return data


def rep_seed(seeds, values):
    """Seed cuyo value esta mas cerca de la mediana."""
    arr = np.array(values)
    idx = int(np.argmin(np.abs(arr - np.median(arr))))
    return seeds[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Estudio 1 — dim latente
# ──────────────────────────────────────────────────────────────────────────────

def make_latent_dim_figures(X, image_shape):
    csv_path = os.path.join(RESULTS_DIR, "exp_latent_dim.csv")
    raw = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            d = int(r["dim"])
            raw.setdefault(d, {"seeds": [], "rec_mse": []})
            raw[d]["seeds"].append(int(r["seed"]))
            raw[d]["rec_mse"].append(float(r["rec_mse"]))

    DIMS = sorted(raw.keys())
    means = {d: np.mean(raw[d]["rec_mse"]) for d in DIMS}
    stds  = {d: np.std( raw[d]["rec_mse"]) for d in DIMS}

    # Entrenar seed representativa por dim
    print("Entrenando seeds representativas para latent_dim ...")
    vaes = {}
    for d in DIMS:
        s = rep_seed(raw[d]["seeds"], raw[d]["rec_mse"])
        t = time.time()
        vae, _ = train_vae(X, latent_dim=d, epochs=EPOCHS, seed=s)
        vaes[d] = (vae, s)
        print(f"  dim={d} seed={s}  {time.time()-t:.1f}s", flush=True)

    # ── Figura 1: grilla de muestras (limpia) ─────────────────────────────────
    N_QUAL = 8
    fig, axes = plt.subplots(len(DIMS), N_QUAL,
                             figsize=(N_QUAL * 1.3, len(DIMS) * 1.4))
    axes = np.atleast_2d(axes)
    for i, d in enumerate(DIMS):
        vae_rep, s = vaes[d]
        samples, _ = vae_rep.generate(N_QUAL, seed=7)
        for j in range(N_QUAL):
            axes[i, j].imshow(img(samples[j], image_shape),
                              cmap="gray", interpolation="nearest",
                              vmin=0, vmax=1)
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
        axes[i, 0].set_ylabel(f"dim={d}", fontsize=10, rotation=0,
                               labelpad=30, va="center")
    fig.suptitle("Estudio 1 — muestras $z\\sim\\mathcal{N}(0,I)$ por dim latente"
                 " (seed representativa)", y=1.01)
    fig.tight_layout()
    p1 = os.path.join(RESULTS_DIR, "exp_latent_dim_samples.png")
    save_fig(fig, p1)
    print(f"[ok] {p1}")

    # ── Figura 2: barchart rec_MSE ± std ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(DIMS))
    cols = plt.get_cmap("viridis")(np.linspace(0.2, 0.8, len(DIMS)))
    bars = ax.bar(x, [means[d] for d in DIMS],
                  yerr=[stds[d] for d in DIMS],
                  color=cols, capsize=7, width=0.5,
                  error_kw=dict(elinewidth=1.8, capthick=1.8))
    ax.set_xticks(x)
    ax.set_xticklabels([f"dim={d}" for d in DIMS], fontsize=11)
    ax.set_ylabel("rec MSE  (media ± std)")
    ax.set_title(f"Reconstruccion por dim latente  ({len(raw[DIMS[0]]['seeds'])} seeds)",
                 fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    for rect, d in zip(bars, DIMS):
        ax.text(rect.get_x() + rect.get_width() / 2,
                rect.get_height() + stds[d] + 0.05,
                f"{means[d]:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    p2 = os.path.join(RESULTS_DIR, "exp_latent_dim_bars.png")
    save_fig(fig, p2)
    print(f"[ok] {p2}")


# ──────────────────────────────────────────────────────────────────────────────
# Estudio 2 — beta
# ──────────────────────────────────────────────────────────────────────────────

def make_beta_figures(X, y, image_shape):
    csv_path = os.path.join(RESULTS_DIR, "exp_beta.csv")
    raw = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            b = float(r["beta"])
            raw.setdefault(b, {"seeds": [], "rec_final": [], "kl_final": [], "rec_mse": []})
            raw[b]["seeds"].append(int(r["seed"]))
            raw[b]["rec_final"].append(float(r["rec_final"]))
            raw[b]["kl_final"].append(float(r["kl_final"]))
            raw[b]["rec_mse"].append(float(r["rec_mse"]))

    BETAS = sorted(raw.keys())
    rec_mean = {b: np.mean(raw[b]["rec_final"]) for b in BETAS}
    rec_std  = {b: np.std( raw[b]["rec_final"]) for b in BETAS}
    kl_mean  = {b: np.mean(raw[b]["kl_final"])  for b in BETAS}
    kl_std   = {b: np.std( raw[b]["kl_final"])  for b in BETAS}

    n_classes = len(np.unique(y))

    print("Entrenando seeds representativas para beta ...")
    vaes = {}
    for b in BETAS:
        s = rep_seed(raw[b]["seeds"], raw[b]["rec_final"])
        t = time.time()
        vae, _ = train_vae(X, latent_dim=2, beta=b, epochs=EPOCHS, seed=s)
        vaes[b] = (vae, s)
        print(f"  beta={b} seed={s}  {time.time()-t:.1f}s", flush=True)

    # ── Figura 3: scatter latente (seed representativa) ───────────────────────
    cmap = plt.get_cmap("tab20", n_classes)
    fig, axes = plt.subplots(1, len(BETAS), figsize=(len(BETAS) * 4.5, 4.2))
    axes = np.atleast_1d(axes)
    for ax, b in zip(axes, BETAS):
        vae_rep, s = vaes[b]
        mu = vae_rep.encode(X)
        for c in range(n_classes):
            m = (y == c)
            ax.scatter(mu[m, 0], mu[m, 1], s=12, color=cmap(c),
                       alpha=0.65, edgecolors="none")
        ax.set_title(f"$\\beta={b}$", fontsize=12)
        ax.set_xlabel("$z[0]$"); ax.set_ylabel("$z[1]$")
        ax.grid(alpha=0.3)
    fig.suptitle("Estudio 2 — espacio latente por $\\beta$ (seed representativa)",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    p3 = os.path.join(RESULTS_DIR, "exp_beta_scatter.png")
    save_fig(fig, p3)
    print(f"[ok] {p3}")

    # ── Figura 4: barchart rec y KL ± std ─────────────────────────────────────
    x = np.arange(len(BETAS))
    xlabels = [f"β={b}" for b in BETAS]
    n_seeds = len(raw[BETAS[0]]["seeds"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.2))
    ax1.bar(x, [rec_mean[b] for b in BETAS],
            yerr=[rec_std[b] for b in BETAS],
            color=PALETTE["positive"], capsize=7, width=0.5,
            error_kw=dict(elinewidth=1.8, capthick=1.8))
    ax1.set_xticks(x); ax1.set_xticklabels(xlabels, fontsize=11)
    ax1.set_ylabel("rec loss (BCE)")
    ax1.set_title("Reconstruccion (media ± std)")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, [kl_mean[b] for b in BETAS],
            yerr=[kl_std[b] for b in BETAS],
            color=PALETTE["highlight"], capsize=7, width=0.5,
            error_kw=dict(elinewidth=1.8, capthick=1.8))
    ax2.set_xticks(x); ax2.set_xticklabels(xlabels, fontsize=11)
    ax2.set_ylabel("KL  (nats)")
    ax2.set_title("Regularizacion KL (media ± std)")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Estudio 2 — tradeoff rec/KL por $\\beta$  ({n_seeds} seeds)",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    p4 = os.path.join(RESULTS_DIR, "exp_beta_bars.png")
    save_fig(fig, p4)
    print(f"[ok] {p4}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    X, y, labels, image_shape = load_emojis()
    t0 = time.time()
    make_latent_dim_figures(X, image_shape)
    make_beta_figures(X, y, image_shape)
    print(f"[ok] plot_split_figures completo en {time.time()-t0:.0f}s")
    print("FIN_SPLIT")


if __name__ == "__main__":
    main()
