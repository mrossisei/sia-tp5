"""EJ1.b.1 (ablacion) — Activacion del CUELLO del DAE: identidad vs tanh.

Cuello 8D fijo (el elegido en el Estudio 1), entrenamiento de denoising
identico (salt&pepper online p del config, target limpio). Variamos SOLO la
activacion del cuello y comparamos a 5 seeds:
  - reconstruccion limpia (exactas/32)
  - pixel-error vs nivel de ruido de evaluacion (salt&pepper), media +- std.

Objetivo: ver si hay ganador real o si empatan (decidir si vale una slide).

Genera (en ej1/results/denoising/):
  - dae_bnactivation_curves.png : curvas identidad vs tanh con barras de error
  - dae_bnactivation.csv        : tabla completa
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
ACTIVATIONS = ["identity", "tanh"]
SEEDS = [0, 1, 2, 3, 42]
EVAL_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]
N_REPS = 20


def train_dae(X, cfg, activation, seed):
    """DAE con cuello 8D y la activacion de cuello indicada."""
    rng = np.random.default_rng(seed)
    ae = Autoencoder(
        architecture=cfg["architecture"],
        hidden_activation=cfg["hidden_activation"],
        output_activation=cfg["output_activation"],
        bottleneck_activation=activation,
        bottleneck_size=cfg.get("bottleneck_size", 8),
        initializer=cfg.get("initializer", "glorot"),
        seed=seed,
        loss_name=cfg["loss"],
    )
    opt = Adam(lr=cfg["learning_rate"])
    lvl = cfg["noise"]["train_level"]
    for _ in range(cfg["epochs"]):
        Xn = apply_noise(X, NOISE_TYPE, lvl, rng)
        ae.train_epoch(Xn, X, opt, batch_size=cfg.get("batch_size", 0),
                       shuffle=False, rng=rng)
    return ae


def evaluate(model, X, levels, seed=11):
    means = []
    for lvl in levels:
        rng = np.random.default_rng(seed + int(lvl * 1000))
        errs = [
            pixel_errors(X, model.reconstruct(apply_noise(X, NOISE_TYPE, lvl, rng))).mean()
            for _ in range(N_REPS)
        ]
        means.append(float(np.mean(errs)))
    return means


def main():
    cfg = load_yaml(os.path.join(REPO_ROOT, "ej1", "config.yaml"))["denoising"]
    X, _ = load_font()

    res = {}  # act -> (mean[levels], std[levels], exact_mean, exact_std)
    t0 = time.time()
    for act in ACTIVATIONS:
        t = time.time()
        per_seed, exacts = [], []
        for seed in SEEDS:
            dae = train_dae(X, cfg, act, seed)
            per_seed.append(evaluate(dae, X, EVAL_LEVELS))
            exacts.append(pixel_error_summary(X, dae.reconstruct(X))["n_exact"])
        per_seed = np.array(per_seed)
        res[act] = (per_seed.mean(axis=0), per_seed.std(axis=0),
                    float(np.mean(exacts)), float(np.std(exacts)))
        print(f"[cuello 8D / {act:8s}] {time.time()-t:5.1f}s  "
              f"limpio exactas={res[act][2]:.1f}±{res[act][3]:.1f}/32  "
              f"eval(media)={['%.3f' % m for m in res[act][0]]}", flush=True)

    # Figura comparativa.
    fig, ax = plt.subplots(figsize=(8.5, 5))
    colors = {"identity": PALETTE["primary"], "tanh": PALETTE["secondary"]}
    for act in ACTIVATIONS:
        mean, std, _, _ = res[act]
        ax.errorbar(EVAL_LEVELS, mean, yerr=std, fmt="-o", color=colors[act],
                    capsize=5, capthick=1.6, elinewidth=1.6, markersize=5,
                    label=f"cuello {act}")
    ax.set_xlabel("nivel de ruido en evaluacion (salt & pepper)")
    ax.set_ylabel("pixel-error medio")
    ax.set_title(f"DAE cuello 8D: activacion del cuello identidad vs tanh "
                 f"(media $\\pm$ std, {len(SEEDS)} seeds)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    save_fig(fig, os.path.join(OUT, "dae_bnactivation_curves.png"))

    with open(os.path.join(OUT, "dae_bnactivation.csv"), "w", newline="") as f:
        w = csv.writer(f)
        head = ["activacion_cuello", "exactas_limpio_media/32", "exactas_limpio_std"]
        for l in EVAL_LEVELS:
            head += [f"eval_{l:.2f}_media", f"eval_{l:.2f}_std"]
        w.writerow(head)
        for act in ACTIVATIONS:
            mean, std, em, es = res[act]
            row = [act, f"{em:.2f}", f"{es:.2f}"]
            for j in range(len(EVAL_LEVELS)):
                row += [f"{mean[j]:.4f}", f"{std[j]:.4f}"]
            w.writerow(row)

    print(f"[ok] dae_bottleneck_activation completo en {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
