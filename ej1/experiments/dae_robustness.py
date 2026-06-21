"""EJ1.b.2 — Robustez al ruido: DAE vs AE basico, con 5 seeds y barras de error.

La slide de robustez comparaba un DAE y un AE basico UNICOS (promediando solo
realizaciones de ruido). Aca lo llevamos al mismo protocolo de rigor que el resto
de 1b: entrenamos 5 DAE y 5 AE basico (seeds [0,1,2,3,42]) y reportamos
media +- std SOBRE LAS SEEDS para AMBOS modelos, en los 3 tipos de ruido.

  - DAE: arquitectura del config 'denoising' (cuello 8D lineal), entrenado con
    salt&pepper online p=0.10 (target limpio).
  - AE basico: arquitectura del config 'basic' (cuello 2D lineal), entrenado SIN
    ruido (input = target = X). Es el baseline 'no entrenado para denoising'.

Genera (en ej1/results/denoising/):
  - dae_robustness_sweep.png : 3 paneles (salt&pepper / gaussian / masking),
    curvas DAE vs AE basico con barras de error (std sobre 5 seeds)
  - dae_robustness.csv        : tabla completa (media y std por modelo y tipo)
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

SEEDS = [0, 1, 2, 3, 42]
N_REPS = 20  # realizaciones de ruido por celda (promediadas)


def train_dae(X, cfg, seed):
    """DAE: salt&pepper online, target limpio (igual que main_denoising)."""
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
    ntype, lvl = cfg["noise"]["train_type"], cfg["noise"]["train_level"]
    for _ in range(cfg["epochs"]):
        Xn = apply_noise(X, ntype, lvl, rng)
        ae.train_epoch(Xn, X, opt, batch_size=cfg.get("batch_size", 0),
                       shuffle=False, rng=rng)
    return ae


def train_basic(X, cfg, seed):
    """AE basico: input = target = X (sin ruido), igual que main_autoencoder."""
    rng = np.random.default_rng(seed)
    ae = Autoencoder(
        architecture=cfg["architecture"],
        hidden_activation=cfg["hidden_activation"],
        output_activation=cfg["output_activation"],
        bottleneck_activation=cfg["bottleneck_activation"],
        initializer=cfg.get("initializer", "glorot"),
        seed=seed,
        loss_name=cfg["loss"],
    )
    opt = Adam(lr=cfg["learning_rate"])
    for _ in range(cfg["epochs"]):
        ae.train_epoch(X, X, opt, batch_size=cfg.get("batch_size", 0),
                       shuffle=False, rng=rng)
    return ae


def eval_model(model, X, ntype, levels, seed=11):
    """pixel-error medio (32 patrones x N_REPS realizaciones) por nivel."""
    means = []
    for lvl in levels:
        rng = np.random.default_rng(seed + int(lvl * 1000))
        errs = [
            pixel_errors(X, model.reconstruct(apply_noise(X, ntype, lvl, rng))).mean()
            for _ in range(N_REPS)
        ]
        means.append(float(np.mean(errs)))
    return means


def main():
    cfg = load_yaml(os.path.join(REPO_ROOT, "ej1", "config.yaml"))
    dcfg, bcfg = cfg["denoising"], cfg["basic"]
    noise_types = dcfg["noise"]["sweep_types"]
    levels = dcfg["noise"]["sweep_levels"]
    X, _ = load_font()

    # Entrenar 5 DAE y 5 AE basico.
    daes, basics = [], []
    t0 = time.time()
    for seed in SEEDS:
        t = time.time()
        daes.append(train_dae(X, dcfg, seed))
        basics.append(train_basic(X, bcfg, seed))
        print(f"[seed {seed}] entrenados DAE+basico en {time.time()-t:.1f}s", flush=True)

    # Evaluar: por tipo de ruido, media +- std sobre seeds para cada modelo.
    results = {}  # (grupo, tipo) -> (mean[levels], std[levels])
    for ntype in noise_types:
        for name, models in [("DAE", daes), ("AE basico", basics)]:
            per_seed = np.array([eval_model(m, X, ntype, levels) for m in models])
            results[(name, ntype)] = (per_seed.mean(axis=0), per_seed.std(axis=0))

    # Figura: 3 paneles con barras de error en ambos modelos.
    fig, axes = plt.subplots(1, len(noise_types), figsize=(5 * len(noise_types), 4.5),
                             squeeze=False)
    for ti, ntype in enumerate(noise_types):
        ax = axes[0, ti]
        for name, color in [("DAE", PALETTE["primary"]), ("AE basico", PALETTE["secondary"])]:
            mean, std = results[(name, ntype)]
            ax.errorbar(levels, mean, yerr=std, fmt="-o", color=color, capsize=4,
                        capthick=1.4, elinewidth=1.4, markersize=5, label=name)
        ax.set_xlabel("nivel de ruido")
        ax.set_ylabel("pixel-error medio (vs limpio)")
        ax.set_title(f"ruido: {ntype}")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Robustez al ruido: DAE vs AE basico "
                 f"(media $\\pm$ std sobre {len(SEEDS)} seeds, {N_REPS} realizaciones)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, os.path.join(OUT, "dae_robustness_sweep.png"))

    # CSV.
    with open(os.path.join(OUT, "dae_robustness.csv"), "w", newline="") as f:
        w = csv.writer(f)
        head = ["modelo", "ruido"]
        for l in levels:
            head += [f"{l:.2f}_media", f"{l:.2f}_std"]
        w.writerow(head)
        for (name, ntype), (mean, std) in results.items():
            row = [name, ntype]
            for j in range(len(levels)):
                row += [f"{mean[j]:.4f}", f"{std[j]:.4f}"]
            w.writerow(row)

    print(f"[ok] dae_robustness completo en {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
