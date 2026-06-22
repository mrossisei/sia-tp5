"""EJ1.b — Barrido de ARQUITECTURA del DAE (cuello fijo en 8D, p fijo en 0.10).

Pregunta: la arquitectura profunda-ancha que usamos (ganadora en el AE basico)
es tambien la mejor para denoising? O con una red mas chica alcanza?

Las mismas 3 familias del Eje 3 del AE basico, con cuello 8D y protocolo DAE:
  - poco_profunda : [35, 20, 8, 20, 35]
  - media         : [35, 30, 15, 8, 15, 30, 35]
  - profunda_ancha: [35, 60, 40, 20, 8, 20, 40, 60, 35]  <- config actual

Protocolo identico al resto de los estudios DAE:
  salt&pepper online, train_level=0.10, 12000 epocas, 5 seeds, Adam lr=1e-3.
  Evaluacion con p FIJO = 0.10 (nivel de entrenamiento), 20 reps.

Metricas (mismo formato que exp_architecture_ms.png del AE):
  - pixel-error medio final a p=0.10 (media +- std sobre 5 seeds)
  - epoca de convergencia: primera epoca con pixel-error medio <= CONV_THRESHOLD

Genera (en ej1/results/denoising/):
  - dae_architecture.png  : figura dos paneles (error final / velocidad)
  - dae_architecture.csv  : tabla completa por seed
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
from shared.plotting import save_fig, PALETTE
from ej1.models.autoencoder import Autoencoder
from ej1.models.denoising import apply_noise

OUT = os.path.join(REPO_ROOT, "ej1", "results", "denoising")

BOTTLENECK   = 8
NOISE_TYPE   = "salt_pepper"
TRAIN_LEVEL  = 0.10
EVAL_LEVEL   = 0.10        # p fijo para la comparacion de arquitecturas
SEEDS        = [0, 1, 2, 3, 42]
N_REPS       = 20          # realizaciones de ruido por evaluacion
EPOCHS       = 12000
EVAL_EVERY   = 300         # cada cuantas epocas evaluar para detectar convergencia
CONV_THRESHOLD = 1.0       # pixel-error medio <= este valor => convergio

ARCHITECTURES = {
    "poco profunda\n[35,20,8,20,35]":       [35, 20, 8, 20, 35],
    "media\n[35,30,15,8,15,30,35]":         [35, 30, 15, 8, 15, 30, 35],
    "profunda-ancha\n[35,60,40,20,8,...]":  [35, 60, 40, 20, 8, 20, 40, 60, 35],
}


def eval_mean_error(model, X, p, rng_seed=11):
    """pixel-error medio sobre 32 letras x N_REPS realizaciones de ruido."""
    rng = np.random.default_rng(rng_seed)
    errs = [
        pixel_errors(X, model.reconstruct(apply_noise(X, NOISE_TYPE, p, rng))).mean()
        for _ in range(N_REPS)
    ]
    return float(np.mean(errs))


def train_dae(X, arch, seed):
    """Entrena el DAE y registra el pixel-error medio cada EVAL_EVERY epocas."""
    rng = np.random.default_rng(seed)
    ae = Autoencoder(
        architecture=arch,
        hidden_activation="tanh",
        output_activation="logistic",
        bottleneck_activation="identity",
        bottleneck_size=BOTTLENECK,
        initializer="glorot",
        seed=seed,
        loss_name="bce",
    )
    opt = Adam(lr=1e-3)
    conv_epoch = None

    for ep in range(EPOCHS):
        Xn = apply_noise(X, NOISE_TYPE, TRAIN_LEVEL, rng)
        ae.train_epoch(Xn, X, opt, batch_size=0, shuffle=False, rng=rng)

        if ep % EVAL_EVERY == 0 and conv_epoch is None:
            err = eval_mean_error(ae, X, EVAL_LEVEL, rng_seed=ep)
            if err <= CONV_THRESHOLD:
                conv_epoch = ep

    final_err = eval_mean_error(ae, X, EVAL_LEVEL)
    if final_err <= CONV_THRESHOLD and conv_epoch is None:
        conv_epoch = EPOCHS

    return final_err, conv_epoch


def plot_figure(labels, means, stds, cmeans, cstds, nconv, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    y = np.arange(len(labels))

    # Panel izquierdo: pixel-error medio final a p=EVAL_LEVEL
    colors = [PALETTE["positive"] if m <= CONV_THRESHOLD else PALETTE["negative"]
              for m in means]
    axes[0].barh(y, means, xerr=stds, color=colors, edgecolor="k", alpha=0.85, capsize=4)
    axes[0].axvline(CONV_THRESHOLD, color=PALETTE["highlight"], ls="--", lw=1.5,
                    label=f"umbral $\\leq${CONV_THRESHOLD:.0f}")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([l.replace("\n", " ") for l in labels], fontsize=9)
    axes[0].set_xlabel(f"pixel-error medio a p={EVAL_LEVEL} (media $\\pm$ std)")
    axes[0].set_title("Error final")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, axis="x", alpha=0.3)
    axes[0].invert_yaxis()

    # Panel derecho: epoca de convergencia + n/seeds
    cm_plot = [0.0 if np.isnan(m) else m for m in cmeans]
    cs_plot = [0.0 if np.isnan(s) else s for s in cstds]
    axes[1].barh(y, cm_plot, xerr=cs_plot, color=PALETTE["primary"],
                 edgecolor="k", alpha=0.85, capsize=4)
    extents = [cm_plot[i] + cs_plot[i] for i in range(len(labels))]
    xmax = max(extents + [1]) * 1.35
    for yi, (m, n) in enumerate(zip(cmeans, nconv)):
        if np.isnan(m):
            txt = f"no converge ({n}/{len(SEEDS)})"
            xpos = xmax * 0.02
        else:
            txt = f"{n}/{len(SEEDS)}"
            xpos = cm_plot[yi] + cs_plot[yi] + xmax * 0.02
        axes[1].text(xpos, yi, txt, va="center", fontsize=8)
    axes[1].set_xlim(0, xmax)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([l.replace("\n", " ") for l in labels], fontsize=9)
    axes[1].set_xlabel(f"epoca de convergencia (pixel-error medio $\\leq${CONV_THRESHOLD:.0f})")
    axes[1].set_title("Velocidad (n/seeds que convergen)")
    axes[1].grid(True, axis="x", alpha=0.3)
    axes[1].invert_yaxis()

    fig.suptitle(
        f"DAE cuello {BOTTLENECK}D: arquitectura  ({len(SEEDS)} seeds, {EPOCHS} epocas)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, path)


def main():
    X, _ = load_font()
    labels = list(ARCHITECTURES.keys())

    all_means, all_stds, all_cmeans, all_cstds, all_nconv = [], [], [], [], []

    t0 = time.time()
    for label, arch in ARCHITECTURES.items():
        t = time.time()
        final_errs, conv_epochs = [], []
        for seed in SEEDS:
            err, ce = train_dae(X, arch, seed)
            final_errs.append(err)
            conv_epochs.append(float(ce) if ce is not None else np.nan)

        m_err = float(np.mean(final_errs))
        s_err = float(np.std(final_errs))
        conv_ok = [c for c in conv_epochs if not np.isnan(c)]
        m_conv = float(np.mean(conv_ok)) if conv_ok else np.nan
        s_conv = float(np.std(conv_ok)) if conv_ok else np.nan
        n_conv = len(conv_ok)

        all_means.append(m_err)
        all_stds.append(s_err)
        all_cmeans.append(m_conv)
        all_cstds.append(s_conv if not np.isnan(s_conv) else 0.0)
        all_nconv.append(n_conv)

        short = label.replace("\n", " ")
        print(
            f"[{short}] {time.time()-t:5.1f}s  "
            f"err_medio={m_err:.3f}+-{s_err:.3f}  "
            f"conv={m_conv:.0f}+-{s_conv:.0f} ({n_conv}/{len(SEEDS)} seeds)",
            flush=True,
        )

    os.makedirs(OUT, exist_ok=True)
    plot_figure(labels, all_means, all_stds, all_cmeans, all_cstds, all_nconv,
                os.path.join(OUT, "dae_architecture.png"))

    csv_path = os.path.join(OUT, "dae_architecture.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arquitectura", "err_medio_media", "err_medio_std",
                    "conv_epoch_media", "conv_epoch_std", "n_conv"])
        for i, label in enumerate(labels):
            w.writerow([
                label.replace("\n", " "),
                f"{all_means[i]:.4f}", f"{all_stds[i]:.4f}",
                f"{all_cmeans[i]:.1f}" if not np.isnan(all_cmeans[i]) else "nan",
                f"{all_cstds[i]:.1f}",
                all_nconv[i],
            ])

    print(f"\n[ok] dae_architecture completo en {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
