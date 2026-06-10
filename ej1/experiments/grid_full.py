"""EJ1.a.2 — Grilla COMPLETA optimizador × lr × seeds a presupuesto largo.

Versión "sin censura" del estudio rápido (opt_lr_heatmap corre 6000 épocas con
una seed): acá cada celda corre 30000 épocas × 5 seeds (75 runs, estilo
run_grid de TP3), para separar "no converge" de "necesitaba más presupuesto"
y reportar media ± std en lugar de una corrida suelta.

Genera (en ej1/results/basic/):
  - exp_grid_full_heatmap.png : max-pixel medio y época de convergencia media
                                (anotado con n_seeds que convergen y ±std)
  - exp_grid_full.csv         : todas las corridas (config × seed)

Pensado para run_overnight.sh (~30-60 min con la implementación vectorizada).
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
from shared.plotting import save_fig
from ej1.experiments.optimization_study import train  # runner compartido

OUT = os.path.join(REPO_ROOT, "ej1", "results", "basic")
ARCH = [35, 60, 40, 20, 2, 20, 40, 60, 35]
DEFAULT_OPTS = ["adam", "momentum", "gd"]
DEFAULT_LRS = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
DEFAULT_SEEDS = [0, 1, 2, 3, 42]
DEFAULT_EPOCHS = 30000


def _heatmap(ax, M, ann, title, cbar_label, fig, xlabels, ylabels):
    masked = np.ma.masked_invalid(M)
    im = ax.imshow(masked, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(xlabels)
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("learning rate")
    ax.set_ylabel("optimizador")
    ax.set_title(title, fontsize=10)
    vmax = np.nanmax(M) if np.isfinite(M).any() else 1.0
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            txt = ann[i][j]
            if np.isnan(M[i, j]):
                ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                        color="red", fontweight="bold")
            else:
                color = "white" if M[i, j] > vmax * 0.5 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                        color=color)
    fig.colorbar(im, ax=ax, label=cbar_label)


def main(epochs=DEFAULT_EPOCHS, seeds=None, opts=None, lrs=None, suffix=""):
    seeds = DEFAULT_SEEDS if seeds is None else seeds
    opts = DEFAULT_OPTS if opts is None else opts
    lrs = DEFAULT_LRS if lrs is None else lrs

    X, _ = load_font()
    n_runs = len(opts) * len(lrs) * len(seeds)
    print(f"[grid_full] {len(opts)}x{len(lrs)} celdas x {len(seeds)} seeds = "
          f"{n_runs} corridas de {epochs} épocas", flush=True)

    # por celda: listas de max_pixel y conv_epoch (nan si no converge)
    cell_max = [[[] for _ in lrs] for _ in opts]
    cell_conv = [[[] for _ in lrs] for _ in opts]
    rows = []
    t0 = time.time()
    k = 0
    for i, opt_name in enumerate(opts):
        for j, lr in enumerate(lrs):
            for seed in seeds:
                k += 1
                t = time.time()
                _, _, s, ce = train(X, ARCH, "tanh", "identity", "bce",
                                    opt_name, lr, epochs=epochs, seed=seed)
                cell_max[i][j].append(float(s["max"]))
                cell_conv[i][j].append(float(ce) if ce is not None else np.nan)
                rows.append([opt_name, lr, seed, s["max"], s["n_exact"],
                             "" if ce is None else int(ce)])
                el = time.time() - t0
                eta = el / k * (n_runs - k) / 60.0
                print(f"  [{k:3d}/{n_runs}] {opt_name} lr={lr:g} seed={seed}: "
                      f"max={s['max']} conv={ce} ({time.time()-t:.0f}s, "
                      f"ETA {eta:.0f} min)", flush=True)

    # matrices media (nan-aware) + anotaciones "media±std\nn/5"
    M_max = np.full((len(opts), len(lrs)), np.nan)
    M_conv = np.full((len(opts), len(lrs)), np.nan)
    ann_max = [["" for _ in lrs] for _ in opts]
    ann_conv = [["" for _ in lrs] for _ in opts]
    for i in range(len(opts)):
        for j in range(len(lrs)):
            mx = np.array(cell_max[i][j])
            cv = np.array(cell_conv[i][j])
            M_max[i, j] = float(np.mean(mx))
            ann_max[i][j] = f"{np.mean(mx):.1f}±{np.std(mx):.1f}"
            n_ok = int(np.sum(~np.isnan(cv)))
            if n_ok > 0:
                ok = cv[~np.isnan(cv)]
                M_conv[i, j] = float(np.mean(ok))
                ann_conv[i][j] = f"{np.mean(ok):.0f}±{np.std(ok):.0f}\n{n_ok}/{len(seeds)}"
            else:
                ann_conv[i][j] = f"✗ 0/{len(seeds)}"

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    xlabels = [f"{lr:g}" for lr in lrs]
    _heatmap(axes[0], M_max, ann_max,
             "max pixel-error final (media ± std sobre seeds)",
             "max pixel-error", fig, xlabels, opts)
    _heatmap(axes[1], M_conv, ann_conv,
             "época de convergencia (media ± std; n/seeds que convergen)",
             "época", fig, xlabels, opts)
    fig.suptitle(f"Grilla completa optimizador × lr — {epochs} épocas × "
                 f"{len(seeds)} seeds", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, os.path.join(OUT, f"exp_grid_full_heatmap{suffix}.png"))

    with open(os.path.join(OUT, f"exp_grid_full{suffix}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["optimizador", "lr", "seed", "max_pixel", "n_exact", "conv_epoch"])
        w.writerows(rows)

    print(f"[ok] grid_full completo en {(time.time()-t0)/60.0:.1f} min -> {OUT}",
          flush=True)


if __name__ == "__main__":
    main()
