"""EJ1.b — Estudio conjunto ARQUITECTURA x CUELLO del DAE.

Barre todas las combinaciones de:
  - arquitectura: poco_profunda / media / profunda_ancha
  - cuello:       2D / 4D / 8D

Protocolo fijo: salt&pepper online p=0.10, 12000 epocas, 5 seeds, Adam lr=1e-3.
Metrica: pixel-error medio a p=0.10 (20 reps) y exactas/32 en limpio.

Genera (en ej1/results/denoising/):
  - dae_arch_bottleneck_heatmap.png : heatmap arquitectura x cuello
                                      (pixel-error medio, exactas en limpio)
  - dae_arch_bottleneck.csv         : tabla completa
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
from shared.optimizers import Adam
from shared.metrics import pixel_errors, pixel_error_summary
from shared.plotting import save_fig
from ej1.models.autoencoder import Autoencoder
from ej1.models.denoising import apply_noise

OUT = os.path.join(REPO_ROOT, "ej1", "results", "denoising")

NOISE_TYPE   = "salt_pepper"
TRAIN_LEVEL  = 0.10
EVAL_LEVEL   = 0.10
SEEDS        = [0, 1, 2, 3, 42]
N_REPS       = 20
EPOCHS       = 12000

BOTTLENECKS = [2, 4, 8]

ARCHITECTURES = {
    "poco profunda\n[35,20,k,20,35]":          lambda k: [35, 20, k, 20, 35],
    "media\n[35,30,15,k,15,30,35]":            lambda k: [35, 30, 15, k, 15, 30, 35],
    "profunda-ancha\n[35,60,40,20,k,...]":     lambda k: [35, 60, 40, 20, k, 20, 40, 60, 35],
}


def eval_mean_error(model, X, p, rng_seed=11):
    rng = np.random.default_rng(rng_seed)
    errs = [
        pixel_errors(X, model.reconstruct(apply_noise(X, NOISE_TYPE, p, rng))).mean()
        for _ in range(N_REPS)
    ]
    return float(np.mean(errs))


def train_dae(X, arch, bottleneck, seed):
    rng = np.random.default_rng(seed)
    ae = Autoencoder(
        architecture=arch,
        hidden_activation="tanh",
        output_activation="logistic",
        bottleneck_activation="identity",
        bottleneck_size=bottleneck,
        initializer="glorot",
        seed=seed,
        loss_name="bce",
    )
    opt = Adam(lr=1e-3)
    for _ in range(EPOCHS):
        Xn = apply_noise(X, NOISE_TYPE, TRAIN_LEVEL, rng)
        ae.train_epoch(Xn, X, opt, batch_size=0, shuffle=False, rng=rng)
    return ae


def plot_heatmap(mean_err, std_err, mean_exact, std_exact, arch_labels, path):
    n_arch = len(arch_labels)
    n_bn   = len(BOTTLENECKS)

    fig, ax = plt.subplots(figsize=(7, 4.2))

    im0 = ax.imshow(mean_err, cmap="viridis_r", aspect="auto",
                    vmin=0, vmax=np.max(mean_err))
    ax.set_xticks(range(n_bn))
    ax.set_xticklabels([f"{k}D" for k in BOTTLENECKS])
    ax.set_yticks(range(n_arch))
    ax.set_yticklabels(arch_labels, fontsize=10)
    ax.set_xlabel("dimension del cuello")
    ax.set_ylabel("arquitectura")
    ax.set_title(f"pixel-error medio a p={EVAL_LEVEL}  "
                 f"({len(SEEDS)} seeds × {N_REPS} realizaciones)")
    for i in range(n_arch):
        for j in range(n_bn):
            v = mean_err[i, j]
            color = "white" if v > np.max(mean_err) * 0.55 else "black"
            ax.text(j, i, f"{v:.2f}\n±{std_err[i,j]:.2f}",
                    ha="center", va="center", fontsize=9, color=color)
    fig.colorbar(im0, ax=ax, label="pixel-error medio")

    fig.tight_layout()
    save_fig(fig, path)


def main():
    X, _ = load_font()
    arch_labels = list(ARCHITECTURES.keys())
    arch_fns    = list(ARCHITECTURES.values())

    mean_err   = np.zeros((len(arch_labels), len(BOTTLENECKS)))
    std_err    = np.zeros_like(mean_err)
    mean_exact = np.zeros_like(mean_err)
    std_exact  = np.zeros_like(mean_err)

    rows_csv = []
    t0 = time.time()

    for i, (label, arch_fn) in enumerate(zip(arch_labels, arch_fns)):
        for j, k in enumerate(BOTTLENECKS):
            arch = arch_fn(k)
            per_seed_err, per_seed_exact = [], []
            for seed in SEEDS:
                dae = train_dae(X, arch, k, seed)
                per_seed_err.append(eval_mean_error(dae, X, EVAL_LEVEL))
                per_seed_exact.append(
                    pixel_error_summary(X, dae.reconstruct(X))["n_exact"]
                )

            m_err   = float(np.mean(per_seed_err))
            s_err   = float(np.std(per_seed_err))
            m_exact = float(np.mean(per_seed_exact))
            s_exact = float(np.std(per_seed_exact))
            mean_err[i, j]   = m_err
            std_err[i, j]    = s_err
            mean_exact[i, j] = m_exact
            std_exact[i, j]  = s_exact

            short = label.replace("\n", " ")
            print(f"[{short:16s} / cuello {k}D]  "
                  f"err={m_err:.3f}+-{s_err:.3f}  exact={m_exact:.1f}/32",
                  flush=True)
            rows_csv.append([short, k, f"{m_err:.4f}", f"{s_err:.4f}", f"{m_exact:.2f}", f"{s_exact:.2f}"])

    os.makedirs(OUT, exist_ok=True)

    plot_heatmap(mean_err, std_err, mean_exact, std_exact, arch_labels,
                 os.path.join(OUT, "dae_arch_bottleneck_heatmap.png"))

    csv_path = os.path.join(OUT, "dae_arch_bottleneck.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arquitectura", "cuello", "err_medio", "err_std", "exactas_limpio", "exactas_std"])
        w.writerows(rows_csv)

    print(f"\n[ok] dae_arch_bottleneck completo en {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
