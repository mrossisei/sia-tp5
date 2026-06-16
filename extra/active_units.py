"""EXTRA-B — Unidades activas del latente (posterior collapse).

PREGUNTA: el barrido de dimensión latente (EJ2 emojis y EJ3 MNIST) muestra que
la calidad SATURA entre 16 y 32 dimensiones. ¿Por qué? Hipótesis estándar:
aunque el latente tenga 64 dimensiones, el VAE solo USA unas pocas; el resto
"colapsa" al prior (posterior collapse) y no transporta información.

MÉTRICA: KL por dimensión (en nats), promediada sobre los datos:
    KL_j = E_x[ -1/2 (1 + logvar_j - mu_j^2 - exp(logvar_j)) ]
Una dimensión está "activa" si KL_j > umbral (0.01, criterio habitual). Si la
red colapsa la dim j, q(z_j|x) ~ N(0,1) = prior  =>  KL_j ~ 0.

DATOS:
  - EJ3/MNIST: NO re-entrena nada; lee los modelos d2_L{2,8,16,32,64} ya
    guardados en ej3/results/ y mide KL por dimensión sobre el test de MNIST.
  - EJ2/emojis: entrena un barrido chico latent ∈ {2,8,16,32} (cacheado en
    extra/results/models/) y mide lo mismo sobre los emojis.

FIGURAS (extra/results/):
  - active_units_kl_spectrum.png : KL por dimensión (ordenada) por modelo MNIST
  - active_units_vs_latent.png   : unidades activas vs dim latente (ambos datasets)
                                   con la recta y=x (todas activas) de referencia
"""

import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (REPO_ROOT, RESULTS, MODELS, ensure_dirs, load_emojis,  # noqa: E402
                     load_mnist_test, make_vae, fit_simple, kl_per_dim)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from shared.plotting import save_fig, PALETTE  # noqa: E402
from ej2.models.vae import VAE  # noqa: E402

THRESHOLD = 0.01
EJ3_RESULTS = os.path.join(REPO_ROOT, "ej3", "results")
EJ3_LATENTS = [2, 8, 16, 32, 64]
EMOJI_LATENTS = [2, 8, 16, 32]


def _ej3_model_path(latent):
    return os.path.join(EJ3_RESULTS, f"d2_L{latent}_s42_model.npz")


def _ej3_hist_epochs(latent):
    """Épocas realmente entrenadas (len del historial), para anotar honestamente
    los modelos que quedaron a medio entrenar (p.ej. L64)."""
    p = os.path.join(EJ3_RESULTS, f"d2_L{latent}_s42_hist.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    return int(len(d["total"]))


def collect_ej3():
    """Lee modelos MNIST guardados y mide KL por dim sobre el test. No entrena."""
    Xte, _ = load_mnist_test(n=5000)
    out = []
    for lat in EJ3_LATENTS:
        path = _ej3_model_path(lat)
        if not os.path.exists(path):
            print(f"[ej3] FALTA {path} (saltado)")
            continue
        vae = VAE.load(path)
        kl = kl_per_dim(vae, Xte)
        n_act = int(np.sum(kl > THRESHOLD))
        eps = _ej3_hist_epochs(lat)
        out.append({"latent": lat, "kl": kl, "active": n_act, "epochs": eps})
        print(f"[ej3/MNIST] L{lat:<2}: activas={n_act}/{lat}  "
              f"KL_total={kl.sum():.2f}  (entrenado {eps} ép)")
    return out


def collect_emojis():
    """Entrena (o carga cache) el barrido de latente sobre emojis y mide KL."""
    X, y, labels, _ = load_emojis()
    out = []
    for lat in EMOJI_LATENTS:
        cache = os.path.join(MODELS, f"emoji_L{lat}.npz")
        if os.path.exists(cache):
            vae = VAE.load(cache)
        else:
            print(f"[emoji] entrenando L{lat} (600 ép)...", flush=True)
            vae = make_vae(X.shape[1], lat, [256, 64], [64, 256], seed=42)
            fit_simple(vae, X, epochs=600, batch_size=64, lr=1e-3, seed=42)
            vae.save(cache)
        kl = kl_per_dim(vae, X)
        n_act = int(np.sum(kl > THRESHOLD))
        out.append({"latent": lat, "kl": kl, "active": n_act})
        print(f"[ej2/emoji] L{lat:<2}: activas={n_act}/{lat}  KL_total={kl.sum():.2f}")
    return out


def plot_kl_spectrum(ej3, path):
    """KL por dimensión (ordenada de mayor a menor) para cada modelo MNIST."""
    n = len(ej3)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.2), squeeze=False)
    for ax, r in zip(axes[0], ej3):
        kl_sorted = np.sort(r["kl"])[::-1]
        idx = np.arange(1, len(kl_sorted) + 1)
        colors = [PALETTE["primary"] if v > THRESHOLD else PALETTE["negative"]
                  for v in kl_sorted]
        ax.bar(idx, kl_sorted, color=colors, width=0.9)
        ax.axhline(THRESHOLD, color=PALETTE["highlight"], ls="--", lw=1.0)
        ax.set_title(f"L{r['latent']}: {r['active']} activas", fontsize=10)
        ax.set_xlabel("dim (ordenada)")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, axis="y")
    axes[0][0].set_ylabel("KL de la dim (nats, log)")
    fig.suptitle("MNIST (EJ3): KL por dimensión latente — la mayoría colapsa al "
                 f"prior (KL<{THRESHOLD}, en rojo)", y=1.02)
    fig.tight_layout()
    save_fig(fig, path)


def plot_active_vs_latent(ej3, emoji, path):
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    lat_max = max(EJ3_LATENTS)
    # recta y=x: "todas las dimensiones activas"
    ax.plot([1, lat_max], [1, lat_max], color="gray", ls=":", lw=1.4,
            label="y = x (todas activas)")

    for data, name, color, marker in [
        (ej3, "MNIST (EJ3, d2)", PALETTE["primary"], "o"),
        (emoji, "emojis (EJ2)", PALETTE["positive"], "s"),
    ]:
        lats = [r["latent"] for r in data]
        acts = [r["active"] for r in data]
        ax.plot(lats, acts, marker=marker, color=color, lw=1.8, label=name)
        for r in data:
            ax.annotate(f"{r['active']}", (r["latent"], r["active"]),
                        textcoords="offset points", xytext=(6, 5), fontsize=8,
                        color=color)

    ax.set_xscale("log", base=2)
    ax.set_xticks(EJ3_LATENTS)
    ax.set_xticklabels([str(d) for d in EJ3_LATENTS])
    ax.set_xlabel("dimensión latente (total)")
    ax.set_ylabel(f"unidades activas (KL > {THRESHOLD})")
    ax.set_title("Unidades activas vs dimensión latente:\nel VAE usa un nº "
                 "limitado de dimensiones aunque le des más", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, path)


def main():
    ensure_dirs()
    print("=== EXTRA-B: unidades activas / posterior collapse ===")
    ej3 = collect_ej3()
    emoji = collect_emojis()

    if ej3:
        plot_kl_spectrum(ej3, os.path.join(RESULTS, "active_units_kl_spectrum.png"))
    plot_active_vs_latent(ej3, emoji,
                          os.path.join(RESULTS, "active_units_vs_latent.png"))

    # CSV crudo
    with open(os.path.join(RESULTS, "active_units.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "latent", "active_units", "kl_total", "epochs"])
        for r in ej3:
            w.writerow(["mnist", r["latent"], r["active"],
                        f"{r['kl'].sum():.4f}", r["epochs"]])
        for r in emoji:
            w.writerow(["emoji", r["latent"], r["active"],
                        f"{r['kl'].sum():.4f}", ""])
    # arrays crudos de KL por dim (para regraficar)
    np.savez(os.path.join(RESULTS, "active_units_kl.npz"),
             **{f"mnist_L{r['latent']}": r["kl"] for r in ej3},
             **{f"emoji_L{r['latent']}": r["kl"] for r in emoji})
    print(f"[ok] EXTRA-B -> {RESULTS}")


if __name__ == "__main__":
    main()
