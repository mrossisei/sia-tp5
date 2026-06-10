"""EJ1.a.2 — Estudio de arquitecturas y tecnicas de optimizacion.

Barre, ejecuta DE VERDAD y compara (figuras + tabla CSV en results/basic/):
  - Arquitectura (profundidad/ancho) vs max-pixel-error
  - Optimizador (GD vs Momentum vs Adam): curvas de loss
  - Learning rate vs convergencia
  - Activacion oculta (tanh / relu / logistic) vs max-pixel-error
  - Loss MSE vs BCE: curvas y error final

No es exhaustivo, si representativo. Cada corrida mide pixel_error_summary.
"""

import os
import sys
import csv

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.fonts import load_font
from shared.optimizers import Adam, Momentum, GradientDescent
from shared.metrics import pixel_error_summary
from shared.plotting import save_fig, PALETTE
from ej1.models.autoencoder import Autoencoder

OUT = os.path.join(REPO_ROOT, "ej1", "results", "basic")
EPOCHS = 6000          # presupuesto comun para comparar de forma justa
SEED = 0


def make_opt(name, lr):
    if name == "adam":
        return Adam(lr=lr)
    if name == "momentum":
        return Momentum(lr=lr, momentum=0.9)
    if name == "gd":
        return GradientDescent(lr=lr)
    raise ValueError(name)


def train(X, arch, hidden, bact, loss, opt_name, lr, epochs=EPOCHS, seed=SEED,
          record_every=50):
    rng = np.random.default_rng(seed)
    ae = Autoencoder(architecture=arch, hidden_activation=hidden,
                     output_activation="logistic", bottleneck_activation=bact,
                     initializer="glorot", seed=seed, loss_name=loss)
    opt = make_opt(opt_name, lr)
    losses = []
    conv_epoch = None  # primera epoca con max<=1
    for ep in range(epochs):
        ml, _ = ae.train_epoch(X, X, opt, batch_size=0, shuffle=False, rng=rng)
        losses.append(ml)
        if ep % record_every == 0 and conv_epoch is None:
            if pixel_error_summary(X, ae.reconstruct(X))["max"] <= 1:
                conv_epoch = ep
    s = pixel_error_summary(X, ae.reconstruct(X))
    if s["max"] <= 1 and conv_epoch is None:
        conv_epoch = epochs
    return ae, losses, s, conv_epoch


def exp_architecture(X, rows):
    archs = {
        "poco profunda [35,20,2,20,35]": [35, 20, 2, 20, 35],
        "media [35,30,15,2,15,30,35]": [35, 30, 15, 2, 15, 30, 35],
        "profunda/ancha [35,60,40,20,2,...]": [35, 60, 40, 20, 2, 20, 40, 60, 35],
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, arch in archs.items():
        _, losses, s, ce = train(X, arch, "tanh", "identity", "bce", "adam", 1e-3)
        ax.plot(losses, label=f"{name} (max={s['max']}, exact={s['n_exact']})")
        rows.append({"grupo": "arquitectura", "config": name, "max_pixel": s["max"],
                     "n_exact": s["n_exact"], "n_le1": s["n_le1"],
                     "loss_final": losses[-1], "epoca_converge": ce})
        print(f"[arch] {name}: max={s['max']} exact={s['n_exact']} conv={ce}")
    ax.set_yscale("log"); ax.set_xlabel("Epoca"); ax.set_ylabel("Loss (log)")
    ax.set_title("Arquitectura vs convergencia (BCE, tanh, Adam)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    save_fig(fig, os.path.join(OUT, "exp_architecture.png"))


def exp_optimizer(X, rows):
    arch = [35, 60, 40, 20, 2, 20, 40, 60, 35]
    fig, ax = plt.subplots(figsize=(9, 5))
    for opt_name, lr in [("adam", 1e-3), ("momentum", 1e-2), ("gd", 1e-2)]:
        _, losses, s, ce = train(X, arch, "tanh", "identity", "bce", opt_name, lr)
        ax.plot(losses, label=f"{opt_name} lr={lr} (max={s['max']})")
        rows.append({"grupo": "optimizador", "config": f"{opt_name} lr={lr}",
                     "max_pixel": s["max"], "n_exact": s["n_exact"],
                     "n_le1": s["n_le1"], "loss_final": losses[-1],
                     "epoca_converge": ce})
        print(f"[opt] {opt_name} lr={lr}: max={s['max']} exact={s['n_exact']} conv={ce}")
    ax.set_yscale("log"); ax.set_xlabel("Epoca"); ax.set_ylabel("Loss (log)")
    ax.set_title("Optimizador vs convergencia (BCE, tanh, cuello identity)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    save_fig(fig, os.path.join(OUT, "exp_optimizer.png"))


def exp_learning_rate(X, rows):
    arch = [35, 60, 40, 20, 2, 20, 40, 60, 35]
    fig, ax = plt.subplots(figsize=(9, 5))
    for lr in [1e-4, 5e-4, 1e-3, 5e-3]:
        _, losses, s, ce = train(X, arch, "tanh", "identity", "bce", "adam", lr)
        ax.plot(losses, label=f"lr={lr} (max={s['max']}, conv={ce})")
        rows.append({"grupo": "learning_rate", "config": f"adam lr={lr}",
                     "max_pixel": s["max"], "n_exact": s["n_exact"],
                     "n_le1": s["n_le1"], "loss_final": losses[-1],
                     "epoca_converge": ce})
        print(f"[lr] lr={lr}: max={s['max']} exact={s['n_exact']} conv={ce}")
    ax.set_yscale("log"); ax.set_xlabel("Epoca"); ax.set_ylabel("Loss (log)")
    ax.set_title("Learning rate vs convergencia (Adam, BCE, tanh)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    save_fig(fig, os.path.join(OUT, "exp_learning_rate.png"))


def exp_activation(X, rows):
    arch = [35, 60, 40, 20, 2, 20, 40, 60, 35]
    fig, ax = plt.subplots(figsize=(9, 5))
    for hidden in ["tanh", "relu", "logistic"]:
        _, losses, s, ce = train(X, arch, hidden, "identity", "bce", "adam", 1e-3)
        ax.plot(losses, label=f"{hidden} (max={s['max']}, exact={s['n_exact']})")
        rows.append({"grupo": "activacion", "config": f"hidden={hidden}",
                     "max_pixel": s["max"], "n_exact": s["n_exact"],
                     "n_le1": s["n_le1"], "loss_final": losses[-1],
                     "epoca_converge": ce})
        print(f"[act] {hidden}: max={s['max']} exact={s['n_exact']} conv={ce}")
    ax.set_yscale("log"); ax.set_xlabel("Epoca"); ax.set_ylabel("Loss (log)")
    ax.set_title("Activacion oculta vs convergencia (Adam, BCE)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    save_fig(fig, os.path.join(OUT, "exp_activation.png"))


def exp_loss(X, rows):
    arch = [35, 60, 40, 20, 2, 20, 40, 60, 35]
    fig, ax = plt.subplots(figsize=(9, 5))
    for loss in ["mse", "bce"]:
        _, losses, s, ce = train(X, arch, "tanh", "identity", loss, "adam", 1e-3)
        # normalizar para comparar escalas distintas (MSE vs BCE) en una sola fig
        l = np.array(losses); l = l / l[0]
        ax.plot(l, label=f"{loss.upper()} (max={s['max']}, exact={s['n_exact']})")
        rows.append({"grupo": "loss", "config": loss.upper(),
                     "max_pixel": s["max"], "n_exact": s["n_exact"],
                     "n_le1": s["n_le1"], "loss_final": losses[-1],
                     "epoca_converge": ce})
        print(f"[loss] {loss}: max={s['max']} exact={s['n_exact']} conv={ce}")
    ax.set_yscale("log"); ax.set_xlabel("Epoca")
    ax.set_ylabel("Loss normalizada (log)")
    ax.set_title("MSE vs BCE (Adam, tanh, cuello identity)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    save_fig(fig, os.path.join(OUT, "exp_loss.png"))


def summary_bar(rows):
    """Figura resumen: max-pixel-error por config (todos los grupos)."""
    fig, ax = plt.subplots(figsize=(11, 6))
    names = [f"{r['grupo']}: {r['config']}" for r in rows]
    maxs = [r["max_pixel"] for r in rows]
    colors = [PALETTE["positive"] if m <= 1 else PALETTE["negative"] for m in maxs]
    y = np.arange(len(names))
    ax.barh(y, maxs, color=colors, edgecolor="k")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7)
    ax.axvline(1, color=PALETTE["highlight"], ls="--", lw=1.2, label="objetivo max<=1")
    ax.set_xlabel("max pixel-error")
    ax.set_title("Resumen: config -> max pixel-error (6000 epocas, seed=0)")
    ax.legend(); ax.invert_yaxis()
    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, "exp_summary_maxpixel.png"))


def main():
    X, _ = load_font()
    rows = []
    exp_architecture(X, rows)
    exp_optimizer(X, rows)
    exp_learning_rate(X, rows)
    exp_activation(X, rows)
    exp_loss(X, rows)
    summary_bar(rows)

    csv_path = os.path.join(OUT, "optimization_table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["grupo", "config", "max_pixel",
                                          "n_exact", "n_le1", "loss_final",
                                          "epoca_converge"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[tabla] {csv_path}")
    print("[ok] estudio de optimizacion completo")


if __name__ == "__main__":
    main()
