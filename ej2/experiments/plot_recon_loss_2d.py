"""Grafica el Estudio 3 (2D) BCE vs MSE a partir de los datos en disco.

NO entrena: lee
  exp_reconloss_2d_summary.csv   (metricas) -> barras rec-MSE y KL
  exp_reconloss_2d_arrays.npz    (imagenes de generacion) -> panel z~N(0,I)
Genera exp_vae_reconloss.png. Editalo y volve a correr cuantas veces quieras;
para regenerar los datos corre antes recon_loss_2d.py (una sola vez).

Nota: el .npz tambien guarda mu(x) y las clases por si se quiere un scatter del
latente, pero en 2D las medias de BCE y MSE casi no se distinguen (el colapso
de MSE esta en la varianza sigma, no en mu) -> la decision se ve mejor en la
generacion y en las barras rec-MSE/KL.
"""

import os, sys, csv

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from shared.plotting import save_fig, PALETTE
from ej2.experiments._common import RESULTS_DIR

LOSSES = ["bce", "mse"]
CSV_PATH = os.path.join(RESULTS_DIR, "exp_reconloss_2d_summary.csv")
NPZ_PATH = os.path.join(RESULTS_DIR, "exp_reconloss_2d_arrays.npz")
OUT_PATH = os.path.join(RESULTS_DIR, "exp_vae_reconloss.png")


def load_metrics():
    """Devuelve {loss: {metric: [valores por seed]}} desde el CSV."""
    m = {l: {"rec_mse": [], "kl": [], "sharpness": []} for l in LOSSES}
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            l = row["recon_loss"]
            for k in ("rec_mse", "kl", "sharpness"):
                m[l][k].append(float(row[k]))
    return m


def bar_panel(ax, metrics, key, title, colors):
    x = np.arange(len(LOSSES))
    means = [np.mean(metrics[l][key]) for l in LOSSES]
    stds = [np.std(metrics[l][key]) for l in LOSSES]
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors)
    ax.bar_label(bars, labels=[f"{m:.2f}" for m in means],
                 padding=6, fontsize=9, fontweight="bold")
    ax.set_ylim(0, max(means) * 1.28)
    ax.set_xticks(x); ax.set_xticklabels([l.upper() for l in LOSSES])
    ax.set_title(title, fontsize=9)


def main():
    if not (os.path.exists(CSV_PATH) and os.path.exists(NPZ_PATH)):
        sys.exit("Faltan datos. Corre primero: python3 "
                 "ej2/experiments/recon_loss_2d.py")

    metrics = load_metrics()
    arr = np.load(NPZ_PATH, allow_pickle=True)
    gen = {"bce": arr["gen_bce"], "mse": arr["gen_mse"]}
    image_shape = tuple(int(v) for v in arr["image_shape"])
    n_samples = gen["bce"].shape[0]

    fig = plt.figure(figsize=(11, 4.6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[2.2, 1.0], height_ratios=[1, 1],
                           hspace=0.45, wspace=0.28)

    # panel cualitativo (izq): generacion BCE arriba, MSE abajo (mismos z)
    gs_q = gridspec.GridSpecFromSubplotSpec(2, n_samples, subplot_spec=gs[:, 0],
                                            hspace=0.15, wspace=0.08)
    for r, l in enumerate(LOSSES):
        for j in range(n_samples):
            ax = fig.add_subplot(gs_q[r, j])
            ax.imshow(gen[l][j].reshape(image_shape), cmap="gray",
                      vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(l.upper(), fontsize=11, rotation=0,
                              labelpad=18, va="center")
    fig.text(0.34, 0.93, "Generacion z~N(0,I) (latente 2D, mismos z)",
             ha="center", fontsize=11)

    colors = [PALETTE["positive"], PALETTE["negative"]]
    bar_panel(fig.add_subplot(gs[0, 1]), metrics, "rec_mse",
              "rec-MSE (comun, menor mejor)", colors)
    bar_panel(fig.add_subplot(gs[1, 1]), metrics, "kl",
              "KL (nats): MSE colapsa el latente", colors)

    fig.suptitle("Estudio 3 (2D) --- BCE vs MSE: con MSE el latente colapsa y la "
                 "generacion se va a gris", y=1.0, fontsize=11)
    save_fig(fig, OUT_PATH)
    print(f"[ok] figura -> {OUT_PATH}")


if __name__ == "__main__":
    main()
