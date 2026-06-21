"""EJ1.a.2 — Estudio de optimizacion MULTI-SEED (drill-downs + resumen).

Reemplaza el barrido de 1 seed de optimization_study.py por uno con 5 seeds
(media +- std), para que cada eje sea un experimento reproducible y el resumen
quede al FINAL (consolidando lo visto). El eje optimizador x lr vive aparte
(grid_full.py / opt_lr_heatmap.py).

Cada eje se barre variando UNA pieza sobre la config base
([35,60,40,20,2,...], tanh, cuello identity, BCE, Adam 1e-3), 6000 epocas,
seeds [0,1,2,3,42]. Metrica: max pixel-error final y epoca de convergencia
(1a epoca con max<=1).

Genera (en ej1/results/basic/):
  - exp_loss_ms.png         : BCE vs MSE         (max-pixel y conv-epoch, media+-std)
  - exp_activation_ms.png   : tanh/relu/logistica
  - exp_architecture_ms.png : poco profunda / media / profunda-ancha
  - exp_summary_ms.png      : resumen consolidado (max-pixel media+-std, todos los ejes)
  - exp_optimization_ms.csv : tabla eje/config/seed -> max, conv (resumen legible)
  - exp_optimization_ms.npz : DATOS CRUDOS de cada corrida para re-graficar SIN
                              re-entrenar: curvas de loss por epoca, pixel-error por
                              patron (32), max y conv de cada (eje, config, seed).
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
from shared.plotting import save_fig, PALETTE
from ej1.experiments.optimization_study import train

OUT = os.path.join(REPO_ROOT, "ej1", "results", "basic")
BASE = [35, 60, 40, 20, 2, 20, 40, 60, 35]
SEEDS = [0, 1, 2, 3, 42]
EPOCHS = 6000

# Cada eje: lista de (label, dict de overrides sobre la config base).
AXES = {
    "loss": [
        ("BCE", dict(loss="bce")),
        ("MSE", dict(loss="mse")),
    ],
    "activacion": [
        ("tanh", dict(hidden="tanh")),
        ("relu", dict(hidden="relu")),
        ("logistica", dict(hidden="logistic")),
    ],
    "arquitectura": [
        ("poco profunda\n[35,20,2,20,35]", dict(arch=[35, 20, 2, 20, 35])),
        ("media\n[35,30,15,2,15,30,35]", dict(arch=[35, 30, 15, 2, 15, 30, 35])),
        ("profunda-ancha\n[35,60,40,20,2,...]", dict(arch=BASE)),
    ],
}

DEFAULTS = dict(arch=BASE, hidden="tanh", bact="identity", loss="bce",
                opt="adam", lr=1e-3)


def run_config(X, ov):
    """Corre una config (5 seeds). Devuelve lista de dicts por seed con TODO lo
    necesario para re-graficar: max, conv, curva de loss y pixel-error por patron."""
    cfg = {**DEFAULTS, **ov}
    out = []
    for seed in SEEDS:
        _, losses, s, ce = train(X, cfg["arch"], cfg["hidden"], cfg["bact"],
                                cfg["loss"], cfg["opt"], cfg["lr"],
                                epochs=EPOCHS, seed=seed)
        out.append({
            "max": s["max"],
            "conv": float(ce) if ce is not None else np.nan,
            "losses": np.asarray(losses, dtype=np.float32),
            "per_pattern": np.asarray(s["per_pattern"], dtype=np.int16),
            "n_exact": s["n_exact"],
            "cfg": cfg,
        })
    return out


def _stats(results, label, key):
    """(media, std, n_conv) de la metrica `key` ('max' o 'conv') sobre las seeds."""
    vals = np.array([r[key] for r in results[label]], dtype=float)
    ok = vals[~np.isnan(vals)]
    if key == "conv":  # solo seeds que convergieron
        n_conv = int(np.sum(~np.isnan(vals)))
        m = float(np.mean(ok)) if len(ok) else np.nan
        sd = float(np.std(ok)) if len(ok) else 0.0
    else:              # max-pixel: todas las seeds
        n_conv = len(vals)
        m, sd = float(np.mean(vals)), float(np.std(vals))
    return m, sd, n_conv


def axis_figure(title, labels, results, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    y = np.arange(len(labels))

    # Panel A: max pixel-error final (media +- std), color pass/fail
    means = [_stats(results, l, "max")[0] for l in labels]
    stds = [_stats(results, l, "max")[1] for l in labels]
    colors = [PALETTE["positive"] if m <= 1 else PALETTE["negative"] for m in means]
    axes[0].barh(y, means, xerr=stds, color=colors, edgecolor="k", alpha=0.85, capsize=4)
    axes[0].axvline(1, color=PALETTE["highlight"], ls="--", lw=1.5, label="objetivo max$\\leq$1")
    axes[0].set_yticks(y); axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].set_xlabel("max pixel-error (media $\\pm$ std)")
    axes[0].set_title("Error final")
    axes[0].legend(fontsize=8); axes[0].grid(True, axis="x", alpha=0.3)
    axes[0].invert_yaxis()

    # Panel B: epoca de convergencia (media +- std) + n/seeds que convergen
    cmeans = [_stats(results, l, "conv")[0] for l in labels]
    cstds = [_stats(results, l, "conv")[1] for l in labels]
    nconv = [_stats(results, l, "conv")[2] for l in labels]
    cm_plot = [0 if np.isnan(m) else m for m in cmeans]
    axes[1].barh(y, cm_plot, xerr=cstds, color=PALETTE["primary"],
                 edgecolor="k", alpha=0.85, capsize=4)
    # xmax debe contemplar barra + error (std) + espacio para la etiqueta a la derecha.
    extents = [cm_plot[i] + cstds[i] for i in range(len(labels))]
    xmax = max(extents + [1]) * 1.30
    for yi, (m, n) in enumerate(zip(cmeans, nconv)):
        txt = f"{n}/{len(SEEDS)}" if not np.isnan(m) else f"no converge ({n}/{len(SEEDS)})"
        xpos = (cm_plot[yi] + cstds[yi] + xmax * 0.02) if not np.isnan(m) else xmax * 0.02
        axes[1].text(xpos, yi, txt, va="center", fontsize=8)
    axes[1].set_xlim(0, xmax)
    axes[1].set_yticks(y); axes[1].set_yticklabels(labels, fontsize=9)
    axes[1].set_xlabel("epoca de convergencia (max$\\leq$1)")
    axes[1].set_title("Velocidad (n/seeds que convergen)")
    axes[1].grid(True, axis="x", alpha=0.3); axes[1].invert_yaxis()

    fig.suptitle(f"{title}  ({len(SEEDS)} seeds, {EPOCHS} epocas)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, path)


def summary_figure(summary, path):
    """summary: lista de (label, mean_max, std_max). Resumen consolidado."""
    fig, ax = plt.subplots(figsize=(11, 6))
    labels = [s[0] for s in summary]
    means = [s[1] for s in summary]
    stds = [s[2] for s in summary]
    colors = [PALETTE["positive"] if m <= 1 else PALETTE["negative"] for m in means]
    y = np.arange(len(labels))
    ax.barh(y, means, xerr=stds, color=colors, edgecolor="k", alpha=0.85, capsize=3)
    ax.axvline(1, color=PALETTE["highlight"], ls="--", lw=1.3, label="objetivo max$\\leq$1")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("max pixel-error (media $\\pm$ std sobre 5 seeds)")
    ax.set_title(f"Resumen: config $\\to$ max pixel-error ({EPOCHS} epocas, {len(SEEDS)} seeds)")
    ax.legend(); ax.grid(True, axis="x", alpha=0.3); ax.invert_yaxis()
    fig.tight_layout()
    save_fig(fig, path)


def main():
    t0 = time.time()
    X, _ = load_font()

    csv_rows = []      # resumen legible
    summary = []       # (label, mean_max, std_max) consolidado
    # Acumuladores de datos crudos para el .npz (1 fila por corrida).
    raw = {"eje": [], "config": [], "seed": [], "max": [], "conv": [],
           "n_exact": [], "losses": [], "per_pattern": [],
           "hidden": [], "loss": [], "lr": [], "arch": []}

    fig_names = {"loss": "exp_loss_ms.png", "activacion": "exp_activation_ms.png",
                 "arquitectura": "exp_architecture_ms.png"}
    titles = {"loss": "Funcion de perdida: BCE vs MSE",
              "activacion": "Activacion oculta: tanh / relu / logistica",
              "arquitectura": "Arquitectura: profundidad / ancho"}

    print(f"=== Estudio multi-seed: {len(SEEDS)} seeds {SEEDS}, {EPOCHS} epocas ===")
    for axis, configs in AXES.items():
        results, labels = {}, []
        for label, ov in configs:
            t = time.time()
            results[label] = run_config(X, ov)
            labels.append(label)
            clean = label.replace("\n", " ")

            mmax, smax, _ = _stats(results, label, "max")
            mconv, sconv, nconv = _stats(results, label, "conv")
            summary.append((f"{axis}: {label.splitlines()[0]}", mmax, smax))

            # impresion legible por config
            conv_txt = (f"{mconv:.0f}+-{sconv:.0f} ({nconv}/{len(SEEDS)})"
                        if not np.isnan(mconv) else f"no converge (0/{len(SEEDS)})")
            per_seed = ", ".join(f"s{sd}:max={r['max']}"
                                 f"{'' if np.isnan(r['conv']) else '@'+str(int(r['conv']))}"
                                 for sd, r in zip(SEEDS, results[label]))
            print(f"[{axis}: {clean}] max={mmax:.2f}+-{smax:.2f} | conv={conv_txt} "
                  f"| {per_seed} ({time.time()-t:.0f}s)", flush=True)

            for sd, r in zip(SEEDS, results[label]):
                csv_rows.append([axis, clean, sd, r["max"],
                                 "" if np.isnan(r["conv"]) else int(r["conv"])])
                raw["eje"].append(axis); raw["config"].append(clean)
                raw["seed"].append(sd); raw["max"].append(r["max"])
                raw["conv"].append(r["conv"]); raw["n_exact"].append(r["n_exact"])
                raw["losses"].append(r["losses"])
                raw["per_pattern"].append(r["per_pattern"])
                raw["hidden"].append(r["cfg"]["hidden"]); raw["loss"].append(r["cfg"]["loss"])
                raw["lr"].append(r["cfg"]["lr"]); raw["arch"].append(str(r["cfg"]["arch"]))
        axis_figure(titles[axis], labels, results, os.path.join(OUT, fig_names[axis]))

    summary_figure(summary, os.path.join(OUT, "exp_summary_ms.png"))

    # ---- CSV resumen legible ----
    csv_path = os.path.join(OUT, "exp_optimization_ms.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eje", "config", "seed", "max_pixel", "conv_epoch"])
        w.writerows(csv_rows)

    # ---- NPZ datos crudos (re-graficar sin re-entrenar) ----
    npz_path = os.path.join(OUT, "exp_optimization_ms.npz")
    np.savez_compressed(
        npz_path,
        eje=np.array(raw["eje"]), config=np.array(raw["config"]),
        seed=np.array(raw["seed"], dtype=np.int32),
        max_pixel=np.array(raw["max"], dtype=np.int16),
        conv_epoch=np.array(raw["conv"], dtype=np.float32),   # nan = no converge
        n_exact=np.array(raw["n_exact"], dtype=np.int16),
        losses=np.stack(raw["losses"]),                       # (n_runs, EPOCHS) float32
        per_pattern=np.stack(raw["per_pattern"]),             # (n_runs, 32) int16
        hidden=np.array(raw["hidden"]), loss=np.array(raw["loss"]),
        lr=np.array(raw["lr"], dtype=np.float32), arch=np.array(raw["arch"]),
        seeds=np.array(SEEDS, dtype=np.int32), epochs=EPOCHS,
    )

    # ---- resumen consolidado impreso ----
    print("\n=== RESUMEN consolidado (max pixel-error, media +- std sobre 5 seeds) ===")
    for label, m, sd in summary:
        flag = "OK <=1" if m <= 1 else "FALLA"
        print(f"  {label:<32} {m:5.2f} +- {sd:4.2f}   [{flag}]")
    print(f"\n[csv] {csv_path}")
    print(f"[npz] {npz_path}  (claves: eje, config, seed, max_pixel, conv_epoch, "
          f"n_exact, losses[{len(csv_rows)}x{EPOCHS}], per_pattern[{len(csv_rows)}x32], "
          f"hidden, loss, lr, arch)")
    print(f"[ok] optimization_multiseed completo en {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
