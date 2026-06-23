"""Grafica el Estudio 4 (2D) a partir de exp_arch_2d_summary.csv. NO entrena.

Dos paneles, sin la redundancia del scatter:
  izq: LOSS COMPLETA (rec_BCE + beta*KL) apilada -> se ve el total que de verdad
       minimizamos Y cuanto aporta el KL (si una arq mas grande sube el KL y
       compensa la ganancia en reconstruccion).
  der: rec-MSE (metrica comun entre estudios).
En ambos, el nº de parametros va como etiqueta -> no hace falta scatter aparte.
"""

import os, sys, csv
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.plotting import save_fig, PALETTE
from ej2.experiments._common import RESULTS_DIR

BETA = 1.0
ORDER = ["angosta", "base", "ancha", "profunda"]
CSV_PATH = os.path.join(RESULTS_DIR, "exp_arch_2d_summary.csv")
OUT_PATH = os.path.join(RESULTS_DIR, "exp_vae_arch.png")


def load():
    """Devuelve {arch: {metric: media}} + n_params por arch."""
    acc = defaultdict(lambda: defaultdict(list))
    npar = {}
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            a = row["arch"]
            for k in ("rec_mse", "rec_bce", "kl"):
                acc[a][k].append(float(row[k]))
            npar[a] = int(row["n_params"])
    # loss total por seed (para su media y desvio correctos: std(suma) != suma(std))
    for a in acc:
        acc[a]["loss"] = [rb + BETA * k
                          for rb, k in zip(acc[a]["rec_bce"], acc[a]["kl"])]
    stats = {a: {k: float(np.mean(v)) for k, v in d.items()} for a, d in acc.items()}
    stds = {a: {k: float(np.std(v)) for k, v in d.items()} for a, d in acc.items()}
    return stats, stds, npar


def main():
    if not os.path.exists(CSV_PATH):
        sys.exit("Falta el CSV. Corre primero: python3 ej2/experiments/arch_2d.py")
    stats, stds, npar = load()
    archs = [a for a in ORDER if a in stats]
    x = np.arange(len(archs))
    pl = [f"{npar[a]/1000:.0f}k" for a in archs]  # etiqueta de params

    fig, axL = plt.subplots(1, 1, figsize=(6.5, 4.4))

    # --- izq: loss total apilada = rec_BCE + beta*KL ----------------------
    rec_bce = np.array([stats[a]["rec_bce"] for a in archs])
    kl = np.array([stats[a]["kl"] for a in archs])
    klw = BETA * kl
    total = rec_bce + klw
    loss_std = np.array([stds[a]["loss"] for a in archs])
    b1 = axL.bar(x, rec_bce, color=PALETTE["primary"], label="rec (BCE)")
    b2 = axL.bar(x, klw, bottom=rec_bce, color=PALETTE["highlight"],
                 label=r"$\beta\cdot$KL")
    axL.errorbar(x, total, yerr=loss_std, fmt="none", ecolor="black", capsize=4)
    # KL adentro de la banda roja (su tendencia es el mecanismo: sube con capacidad)
    for xi, rb, k in zip(x, rec_bce, kl):
        axL.text(xi, rb + klw.max() / 2, f"KL\n{k:.1f}", ha="center",
                 va="center", fontsize=7, color="white", fontweight="bold")
    # total arriba de cada barra (por encima de la barra de error)
    for xi, t, s in zip(x, total, loss_std):
        axL.text(xi, t + s, f"{t:.1f} $\\pm$ {s:.1f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")
    axL.set_ylim(0, total.max() * 1.18)
    axL.set_xticks(x); axL.set_xticklabels(archs)
    axL.set_title("Loss completa = rec(BCE) + " r"$\beta\cdot$KL  (menor mejor):"
                  "\ncasi no cambia --- el KL sube y compensa la mejora en rec",
                  fontsize=9)
    axL.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    save_fig(fig, OUT_PATH)
    print(f"[ok] figura -> {OUT_PATH}")


if __name__ == "__main__":
    main()
