"""Figuras del Denoising AE (EJ1.b). Consume el DAE y el AE basico entrenados.

Genera (en ej1/results/denoising/):
  - noise_examples.png : filas limpio / ruidoso / denoised para varias letras y niveles
  - noise_sweep.png    : nivel de ruido vs pixel-error medio, DAE vs AE basico
                         (una curva por tipo de ruido)
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.fonts import ROWS, COLS
from shared.metrics import pixel_errors
from shared.plotting import save_fig, PALETTE
from ej1.models.denoising import apply_noise


def _grid(vec):
    return np.asarray(vec).reshape(ROWS, COLS)


def plot_noise_examples(dae, X, labels, path, noise_type="salt_pepper",
                        levels=(0.1, 0.2, 0.3), char_idxs=(1, 5, 15, 19),
                        seed=123):
    """Para cada letra y nivel: limpio / ruidoso / denoised."""
    rng = np.random.default_rng(seed)
    n_chars = len(char_idxs)
    n_levels = len(levels)
    # Filas: por cada (char): 1 limpio + (ruidoso,denoised) por nivel
    ncols = 1 + 2 * n_levels
    fig, axes = plt.subplots(n_chars, ncols, figsize=(ncols * 1.3, n_chars * 1.5))
    if n_chars == 1:
        axes = axes[None, :]
    for r, ci in enumerate(char_idxs):
        x = X[ci:ci + 1]
        axes[r, 0].imshow(_grid(x[0]), cmap="Greys", vmin=0, vmax=1)
        axes[r, 0].set_ylabel(f"'{labels[ci]}'", fontsize=10, rotation=0,
                              labelpad=15, va="center")
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        if r == 0:
            axes[r, 0].set_title("limpio", fontsize=9)
        for li, lvl in enumerate(levels):
            xn = apply_noise(x, noise_type, lvl, rng)
            xd = dae.reconstruct(xn)
            xd_bin = (xd >= 0.5).astype(int)
            c_noisy = 1 + 2 * li
            c_den = c_noisy + 1
            axes[r, c_noisy].imshow(_grid(xn[0]), cmap="Greys", vmin=0, vmax=1)
            axes[r, c_noisy].set_xticks([]); axes[r, c_noisy].set_yticks([])
            err = int(pixel_errors(x, xd)[0])
            axes[r, c_den].imshow(_grid(xd_bin[0]), cmap="Greys", vmin=0, vmax=1)
            axes[r, c_den].set_xticks([]); axes[r, c_den].set_yticks([])
            col = PALETTE["positive"] if err <= 1 else PALETTE["negative"]
            # err siempre como xlabel (consistente en todas las filas)
            axes[r, c_den].set_xlabel(f"err={err}", fontsize=8, color=col)
            if r == 0:
                axes[r, c_noisy].set_title(f"ruido {lvl}", fontsize=9)
                axes[r, c_den].set_title("denoised", fontsize=9)
    fig.suptitle(f"Denoising AE — ruido '{noise_type}': limpio / ruidoso / reconstruido",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, path)


def plot_noise_sweep(dae, ae_basic, X, path, noise_types, levels, n_reps=10,
                     seed=7):
    """Curva nivel de ruido vs pixel-error medio, DAE vs AE basico, por tipo.

    Promedia sobre n_reps realizaciones de ruido y los 32 patrones.
    Devuelve un dict {(modelo, tipo): [errores por nivel]} con los numeros.
    """
    results = {}
    fig, axes = plt.subplots(1, len(noise_types), figsize=(5 * len(noise_types), 4.5),
                             squeeze=False)
    for ti, ntype in enumerate(noise_types):
        ax = axes[0, ti]
        for model, name, color in [(dae, "DAE", PALETTE["primary"]),
                                   (ae_basic, "AE basico", PALETTE["secondary"])]:
            means = []
            for lvl in levels:
                rng = np.random.default_rng(seed + int(lvl * 1000))
                errs = []
                for _ in range(n_reps):
                    Xn = apply_noise(X, ntype, lvl, rng)
                    rec = model.reconstruct(Xn)
                    errs.append(pixel_errors(X, rec).mean())
                means.append(float(np.mean(errs)))
            results[(name, ntype)] = means
            ax.plot(levels, means, "-o", color=color, label=name)
        ax.set_xlabel("nivel de ruido")
        ax.set_ylabel("pixel-error medio (vs limpio)")
        ax.set_title(f"ruido: {ntype}")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Robustez al ruido: DAE vs AE basico (32 letras, "
                 f"{n_reps} realizaciones)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, path)
    return results


def generate_all(dae, ae_basic, X, labels, out_dir, cfg_noise):
    os.makedirs(out_dir, exist_ok=True)
    p = lambda name: os.path.join(out_dir, name)
    train_type = cfg_noise["train_type"]
    plot_noise_examples(dae, X, labels, p("noise_examples.png"),
                        noise_type=train_type, levels=(0.1, 0.2, 0.3))
    results = plot_noise_sweep(dae, ae_basic, X, p("noise_sweep.png"),
                               cfg_noise["sweep_types"], cfg_noise["sweep_levels"])
    return results
