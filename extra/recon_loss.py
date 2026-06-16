"""EXTRA-E — MSE vs BCE como pérdida de reconstrucción del VAE.

El VAE soporta ambas pérdidas; el EJ2 usa BCE sin comparar. Experimento clásico:
con imágenes en [0,1], la BCE penaliza fuerte los píxeles mal clasificados y
empuja la salida hacia 0/1 (reconstrucción NÍTIDA); la MSE penaliza el error
cuadrático y tiende a la media -> reconstrucción más BORROSA/gris.

Entrena dos VAE idénticos (misma arquitectura, latente, épocas, semilla, salida
logística) cambiando SOLO recon_loss, y compara:
  - visual: original / recon BCE / recon MSE (una variante por clase)
  - métricas comunes: rec_MSE (en ambos) y nitidez = fracción de píxeles
    "confiados" (|p-0.5|>0.4); más alto = más binario/nítido.

FIGURA (extra/results/): recon_loss_mse_vs_bce.png
"""

import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (RESULTS, ensure_dirs, load_emojis, make_vae,  # noqa: E402
                     fit_simple, rec_mse)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from shared.plotting import save_fig  # noqa: E402

LATENT = 16
EPOCHS = 600
SEED = 42


def sharpness(vae, X):
    """Fracción media de píxeles 'confiados' (|p-0.5|>0.4): más alto = más nítido."""
    p = vae.reconstruct(X)
    return float(np.mean(np.abs(p - 0.5) > 0.4))


def main():
    ensure_dirs()
    print("=== EXTRA-E: MSE vs BCE ===")
    X, y, labels, image_shape = load_emojis()

    models = {}
    for recon in ("bce", "mse"):
        vae = make_vae(X.shape[1], LATENT, [256, 64], [64, 256],
                       recon_loss=recon, output_activation="logistic", seed=SEED)
        fit_simple(vae, X, epochs=EPOCHS, batch_size=64, lr=1e-3, seed=SEED)
        models[recon] = vae
        print(f"[{recon.upper()}] rec_MSE={rec_mse(vae, X):.2f}  "
              f"nitidez={sharpness(vae, X):.3f}")

    # una variante (la primera) por clase, hasta 10 clases
    classes = list(np.unique(y))[:10]
    cols = [int(np.where(y == c)[0][0]) for c in classes]
    rec_bce_img = models["bce"].reconstruct(X[cols])
    rec_mse_img = models["mse"].reconstruct(X[cols])

    rows = [("original", X[cols]),
            (f"BCE\n(nitidez {sharpness(models['bce'], X):.2f})", rec_bce_img),
            (f"MSE\n(nitidez {sharpness(models['mse'], X):.2f})", rec_mse_img)]

    n = len(cols)
    fig, axes = plt.subplots(3, n, figsize=(n * 1.15, 3.8))
    for r, (name, imgs) in enumerate(rows):
        for j in range(n):
            axes[r, j].imshow(imgs[j].reshape(image_shape), cmap="gray",
                              vmin=0, vmax=1, interpolation="nearest")
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
            if r == 0:
                axes[r, j].set_title(labels[int(y[cols[j]])], fontsize=7)
        axes[r, 0].set_ylabel(name, fontsize=8, rotation=0, labelpad=26,
                              va="center")
    fig.suptitle(f"VAE emojis (latent={LATENT}): BCE más nítida, MSE más borrosa",
                 y=1.03)
    fig.tight_layout()
    save_fig(fig, os.path.join(RESULTS, "recon_loss_mse_vs_bce.png"))

    with open(os.path.join(RESULTS, "recon_loss.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["recon_loss", "rec_mse", "sharpness"])
        for recon in ("bce", "mse"):
            w.writerow([recon, f"{rec_mse(models[recon], X):.4f}",
                        f"{sharpness(models[recon], X):.4f}"])
    print(f"[ok] EXTRA-E -> {RESULTS}")


if __name__ == "__main__":
    main()
