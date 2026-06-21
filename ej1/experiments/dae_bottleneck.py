"""EJ1.b.1 — Barrido del TAMANO DE CUELLO del DAE (justifica la arquitectura).

La consigna 1b pide "plantear una arquitectura conveniente y explicar la
eleccion". El AE basico (1a) usa cuello 2D, pero el DAE pasa a 8D: este
experimento mide empiricamente por que. Entrenamos UN DAE POR (TAMANO DE CUELLO,
SEED) con K = 2, 4, 8 y 5 seeds (mismo protocolo de rigor que 1a), con el MISMO
entrenamiento que main_denoising (salt & pepper online, target limpio, p del
config) y cruzamos K x nivel de ruido de evaluacion. Reportamos media +- std
SOBRE LAS SEEDS (no es suerte de una seed).

Lectura esperada: con K=2 el cuello tiene que limpiar Y comprimir a 2 numeros,
asi que pierde capacidad; al agrandar el cuello el denoising mejora y satura.
Se elige el K mas chico que ya da buen denoising sin romper la reconstruccion
limpia.

Genera (en ej1/results/denoising/):
  - dae_bottleneck_curves.png  : pixel-error vs nivel de eval, una curva por K
                                 con barras de error (std sobre 5 seeds)
  - dae_bottleneck_heatmap.png : matriz K x nivel-eval (media) + baseline AE basico
  - dae_bottleneck.csv         : tabla completa (media y std por seed, exactas/32)
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
from shared.config_loader import load_yaml
from shared.plotting import save_fig, PALETTE
from ej1.models.autoencoder import Autoencoder
from ej1.models.denoising import apply_noise

OUT = os.path.join(REPO_ROOT, "ej1", "results", "denoising")

NOISE_TYPE = "salt_pepper"
BOTTLENECKS = [2, 4, 8]
SEEDS = [0, 1, 2, 3, 42]          # mismo protocolo de rigor que 1a
EVAL_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]
N_REPS = 20                       # realizaciones de ruido por celda (promediadas)


def arch_for(k):
    """Arquitectura simetrica con cuello K (misma forma que el DAE del config)."""
    return [35, 60, 40, 20, int(k), 20, 40, 60, 35]


def train_dae(X, cfg, k, seed):
    """Mismo entrenamiento que main_denoising pero con cuello K y seed dada."""
    rng = np.random.default_rng(seed)
    ae = Autoencoder(
        architecture=arch_for(k),
        hidden_activation=cfg["hidden_activation"],
        output_activation=cfg["output_activation"],
        bottleneck_activation=cfg["bottleneck_activation"],
        bottleneck_size=int(k),
        initializer=cfg.get("initializer", "glorot"),
        seed=seed,
        loss_name=cfg["loss"],
    )
    opt = Adam(lr=cfg["learning_rate"])
    level = cfg["noise"]["train_level"]
    for _ in range(cfg["epochs"]):
        Xn = apply_noise(X, NOISE_TYPE, level, rng)
        ae.train_epoch(Xn, X, opt, batch_size=cfg.get("batch_size", 0),
                       shuffle=False, rng=rng)
    return ae


def train_basic(X, bcfg, seed):
    """AE basico 2D: input = target = X (sin ruido), igual que main_autoencoder."""
    rng = np.random.default_rng(seed)
    ae = Autoencoder(
        architecture=bcfg["architecture"],
        hidden_activation=bcfg["hidden_activation"],
        output_activation=bcfg["output_activation"],
        bottleneck_activation=bcfg["bottleneck_activation"],
        initializer=bcfg.get("initializer", "glorot"),
        seed=seed,
        loss_name=bcfg["loss"],
    )
    opt = Adam(lr=bcfg["learning_rate"])
    for _ in range(bcfg["epochs"]):
        ae.train_epoch(X, X, opt, batch_size=bcfg.get("batch_size", 0),
                       shuffle=False, rng=rng)
    return ae


def evaluate(model, X, levels, seed=11):
    """pixel-error medio (32 patrones x N_REPS realizaciones) por nivel."""
    means = []
    for lvl in levels:
        rng = np.random.default_rng(seed + int(lvl * 1000))
        errs = [
            pixel_errors(X, model.reconstruct(apply_noise(X, NOISE_TYPE, lvl, rng))).mean()
            for _ in range(N_REPS)
        ]
        means.append(float(np.mean(errs)))
    return means


def plot_heatmap(mean_m, std_m, row_labels, path):
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    im = ax.imshow(mean_m, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(EVAL_LEVELS)))
    ax.set_xticklabels([f"{l:.2f}" for l in EVAL_LEVELS])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("nivel de ruido en EVALUACION (salt & pepper)")
    ax.set_ylabel("modelo (tamano de cuello)")
    for i in range(mean_m.shape[0]):
        for j in range(mean_m.shape[1]):
            v = mean_m[i, j]
            color = "white" if v > mean_m.max() * 0.5 else "black"
            ax.text(j, i, f"{v:.2f}±{std_m[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color=color)
    fig.colorbar(im, ax=ax, label="pixel-error medio")
    ax.set_title("DAE: tamano de cuello x nivel de evaluacion "
                 f"(media ± std sobre {len(SEEDS)} seeds x {N_REPS} realizaciones)")
    fig.tight_layout()
    save_fig(fig, path)


def plot_curves(mean_dae, std_dae, base_mean, base_std, path):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(BOTTLENECKS)))
    for i, k in enumerate(BOTTLENECKS):
        ax.errorbar(EVAL_LEVELS, mean_dae[i], yerr=std_dae[i], fmt="-o",
                    color=colors[i], capsize=5, capthick=1.6, elinewidth=1.6,
                    markersize=5, label=f"DAE cuello {k}D")
    ax.errorbar(EVAL_LEVELS, base_mean, yerr=base_std, fmt="--s",
                color=PALETTE["negative"], capsize=5, capthick=1.6, elinewidth=1.6,
                markersize=5, label="AE basico 2D (sin denoising)")
    ax.set_xlabel("nivel de ruido en evaluacion (salt & pepper)")
    ax.set_ylabel("pixel-error medio")
    ax.set_title(f"Capacidad de eliminar ruido segun el cuello "
                 f"(media $\\pm$ std, {len(SEEDS)} seeds)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    save_fig(fig, path)


def main():
    full = load_yaml(os.path.join(REPO_ROOT, "ej1", "config.yaml"))
    cfg, bcfg = full["denoising"], full["basic"]
    X, _ = load_font()

    mean_rows, std_rows = [], []
    clean_exact = {}  # k -> (media, std) de exactas/32 en limpio sobre seeds
    t0 = time.time()
    for k in BOTTLENECKS:
        t = time.time()
        per_seed = []        # [n_seeds, n_levels]
        exacts = []
        for seed in SEEDS:
            dae = train_dae(X, cfg, k, seed=seed)
            per_seed.append(evaluate(dae, X, EVAL_LEVELS))
            exacts.append(pixel_error_summary(X, dae.reconstruct(X))["n_exact"])
        per_seed = np.array(per_seed)
        mean_rows.append(per_seed.mean(axis=0))
        std_rows.append(per_seed.std(axis=0))
        clean_exact[k] = (float(np.mean(exacts)), float(np.std(exacts)))
        print(f"[cuello {k}D] {time.time()-t:5.1f}s  "
              f"eval(media): {['%.2f' % m for m in per_seed.mean(axis=0)]}  "
              f"limpio exactas: {clean_exact[k][0]:.1f}+-{clean_exact[k][1]:.1f}/32",
              flush=True)

    mean_dae = np.array(mean_rows)
    std_dae = np.array(std_rows)

    # Baseline: AE basico 2D entrenado SIN ruido, mismas 5 seeds (media ± std).
    base_per_seed = np.array([evaluate(train_basic(X, bcfg, s), X, EVAL_LEVELS)
                              for s in SEEDS])
    base_mean = base_per_seed.mean(axis=0)
    base_std = base_per_seed.std(axis=0)
    print(f"[baseline AE basico 2D] eval(media): {['%.2f' % m for m in base_mean]}")

    mean_m = np.vstack([mean_dae, base_mean])
    std_m = np.vstack([std_dae, base_std])
    plot_curves(mean_dae, std_dae, base_mean, base_std,
                os.path.join(OUT, "dae_bottleneck_curves.png"))
    plot_heatmap(mean_m, std_m, [f"DAE cuello {k}D" for k in BOTTLENECKS] + ["AE basico 2D"],
                 os.path.join(OUT, "dae_bottleneck_heatmap.png"))

    csv_path = os.path.join(OUT, "dae_bottleneck.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        head = ["modelo", "exactas_limpio_media/32", "exactas_limpio_std"]
        for l in EVAL_LEVELS:
            head += [f"eval_{l:.2f}_media", f"eval_{l:.2f}_std"]
        w.writerow(head)
        for i, k in enumerate(BOTTLENECKS):
            row = [f"DAE cuello {k}D", f"{clean_exact[k][0]:.2f}", f"{clean_exact[k][1]:.2f}"]
            for j in range(len(EVAL_LEVELS)):
                row += [f"{mean_dae[i, j]:.4f}", f"{std_dae[i, j]:.4f}"]
            w.writerow(row)
        base_row = ["AE basico 2D", "", ""]
        for j in range(len(EVAL_LEVELS)):
            base_row += [f"{base_mean[j]:.4f}", f"{base_std[j]:.4f}"]
        w.writerow(base_row)

    print(f"[ok] dae_bottleneck completo en {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
