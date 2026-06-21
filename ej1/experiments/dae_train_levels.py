"""EJ1.b.2 — Barrido del NIVEL DE RUIDO DE ENTRENAMIENTO del DAE.

La consigna pide "distorsionen las entradas en diferentes niveles y estudien
la capacidad del Autoencoder de eliminar el ruido". Lectura fuerte: entrenamos
UN DAE POR NIVEL de ruido de train (salt & pepper; procedimiento de
Autoencoders.pdf, slide 22: entrada X~ ruidosa regenerada cada época, target X
limpio) y cruzamos nivel de train x nivel de evaluación.

Genera (en ej1/results/denoising/):
  - exp_train_levels_heatmap.png : matriz train-level x eval-level de
    pixel-error medio (+ fila baseline: AE básico sin denoising)
  - exp_train_levels_curves.png  : pixel-error vs nivel de eval, una curva
    por DAE entrenado
  - exp_train_levels.csv         : tabla completa
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
from shared.metrics import pixel_errors
from shared.config_loader import load_yaml
from shared.plotting import save_fig, PALETTE
from ej1.models.autoencoder import Autoencoder
from ej1.models.denoising import apply_noise

OUT = os.path.join(REPO_ROOT, "ej1", "results", "denoising")
BASIC_MODEL = os.path.join(REPO_ROOT, "ej1", "results", "basic", "model.npz")

NOISE_TYPE = "salt_pepper"
TRAIN_LEVELS = [0.05, 0.10, 0.20, 0.30]
SEEDS = [0, 1, 2, 3, 42]          # mismo protocolo de rigor que 1a
EVAL_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]
N_REPS = 20  # realizaciones de ruido por celda (promediadas)


def train_dae(X, cfg, level, seed):
    """Mismo entrenamiento que main_denoising: ruido online, target limpio."""
    rng = np.random.default_rng(seed)
    ae = Autoencoder(
        architecture=cfg["architecture"],
        hidden_activation=cfg["hidden_activation"],
        output_activation=cfg["output_activation"],
        bottleneck_activation=cfg["bottleneck_activation"],
        bottleneck_size=cfg.get("bottleneck_size", 2),
        initializer=cfg.get("initializer", "glorot"),
        seed=seed,
        loss_name=cfg["loss"],
    )
    opt = Adam(lr=cfg["learning_rate"])
    for _ in range(cfg["epochs"]):
        Xn = apply_noise(X, NOISE_TYPE, level, rng)
        ae.train_epoch(Xn, X, opt, batch_size=cfg.get("batch_size", 0),
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
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    im = ax.imshow(mean_m, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(EVAL_LEVELS)))
    ax.set_xticklabels([f"{l:.2f}" for l in EVAL_LEVELS])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("nivel de ruido en EVALUACIÓN (salt & pepper)")
    ax.set_ylabel("modelo (nivel de ruido en TRAIN)")
    for i in range(mean_m.shape[0]):
        for j in range(mean_m.shape[1]):
            v = mean_m[i, j]
            color = "white" if v > mean_m.max() * 0.5 else "black"
            # Baseline (ultima fila): sin std (modelo unico).
            txt = f"{v:.2f}" if std_m[i, j] < 0 else f"{v:.2f}±{std_m[i, j]:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(im, ax=ax, label="pixel-error medio")
    ax.set_title("DAE: nivel de train × nivel de evaluación "
                 f"(media ± std sobre {len(SEEDS)} seeds × {N_REPS} realizaciones)")
    fig.tight_layout()
    save_fig(fig, path)


def plot_curves(mean_m, std_m, row_labels, path):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(row_labels) - 1))
    for i, lab in enumerate(row_labels):
        if i < len(row_labels) - 1:
            ax.errorbar(EVAL_LEVELS, mean_m[i], yerr=std_m[i], fmt="-o",
                        color=colors[i], capsize=4, capthick=1.4, elinewidth=1.4,
                        markersize=5, label=lab)
        else:  # baseline AE básico (modelo unico, sin barras)
            ax.plot(EVAL_LEVELS, mean_m[i], "--s", color=PALETTE["negative"],
                    label=lab)
    ax.set_xlabel("nivel de ruido en evaluación (salt & pepper)")
    ax.set_ylabel("pixel-error medio")
    ax.set_title(f"Capacidad de eliminar ruido según el nivel usado en train "
                 f"(media ± std, {len(SEEDS)} seeds)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    save_fig(fig, path)


def main():
    cfg = load_yaml(os.path.join(REPO_ROOT, "ej1", "config.yaml"))["denoising"]
    X, _ = load_font()

    mean_rows, std_rows = [], []
    row_labels = []
    t0 = time.time()
    for lvl in TRAIN_LEVELS:
        t = time.time()
        per_seed = []  # [n_seeds, n_levels]
        for seed in SEEDS:
            dae = train_dae(X, cfg, lvl, seed=seed)
            per_seed.append(evaluate(dae, X, EVAL_LEVELS))
        per_seed = np.array(per_seed)
        mean_rows.append(per_seed.mean(axis=0))
        std_rows.append(per_seed.std(axis=0))
        row_labels.append(f"DAE train p={lvl:.2f}")
        print(f"[train p={lvl:.2f}] {time.time()-t:5.1f}s  "
              f"eval(media): {['%.2f' % m for m in per_seed.mean(axis=0)]}  "
              f"(limpio={per_seed.mean(axis=0)[EVAL_LEVELS.index(0.0)]:.2f})",
              flush=True)

    # Baseline: AE básico (entrenado SIN ruido) del run principal. Modelo unico
    # (no swept) -> std marcado como negativo para no anotarlo.
    ae_basic = Autoencoder.load(BASIC_MODEL)
    base_means = evaluate(ae_basic, X, EVAL_LEVELS)
    mean_rows.append(np.array(base_means))
    std_rows.append(np.full(len(EVAL_LEVELS), -1.0))
    row_labels.append("AE básico (sin denoising)")
    print(f"[baseline AE básico] eval: {['%.2f' % m for m in base_means]}")

    mean_m = np.array(mean_rows)
    std_m = np.array(std_rows)
    plot_heatmap(mean_m, std_m, row_labels, os.path.join(OUT, "exp_train_levels_heatmap.png"))
    plot_curves(mean_m, std_m, row_labels, os.path.join(OUT, "exp_train_levels_curves.png"))

    csv_path = os.path.join(OUT, "exp_train_levels.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        head = ["modelo"]
        for l in EVAL_LEVELS:
            head += [f"eval_{l:.2f}_media", f"eval_{l:.2f}_std"]
        w.writerow(head)
        for lab, m, s in zip(row_labels, mean_rows, std_rows):
            row = [lab]
            for j in range(len(EVAL_LEVELS)):
                row += [f"{m[j]:.4f}", ("" if s[j] < 0 else f"{s[j]:.4f}")]
            w.writerow(row)

    print(f"[ok] dae_train_levels completo en {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
