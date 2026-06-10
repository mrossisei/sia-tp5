"""Entrypoint EJ2 — VAE sobre emojis.

Orquesta:
  1) GRADIENT-CHECK (obligatorio, falla ruidosamente si no pasa).
  2) Carga el dataset de emojis.
  3) Entrena el VAE principal (latent_dim=2 para el manifold), Adam, mini-batch.
  4) Sanity checks (shapes, rango [0,1], NaN, reconstrucción).
  5) Guarda el modelo y delega TODAS las figuras a analysis/vae.py.

Uso:
    python3 ej2/main_vae.py
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np

from shared.config_loader import load_yaml
from shared.optimizers import build_optimizer
from ej2.models.vae import VAE, gradcheck
from ej2.analysis import vae as analysis


def load_emojis(path):
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype(np.float64)
    y = d["y"]
    labels = [str(s) for s in d["labels"]]
    image_shape = tuple(int(v) for v in d["image_shape"])
    return X, y, labels, image_shape


def main():
    cfg = load_yaml(os.path.join(REPO_ROOT, "ej2", "config.yaml"))
    vcfg = cfg["vae"]
    results_dir = os.path.join(REPO_ROOT, "ej2", "results")
    os.makedirs(results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) GRADIENT-CHECK (antes de entrenar; falla ruidosamente).
    # ------------------------------------------------------------------
    print("=" * 70)
    print("GRADIENT-CHECK (analítico vs diferencias finitas centrales)")
    print("=" * 70)
    rel_err = gradcheck(seed=0, eps_fd=1e-5, tol=1e-4, verbose=True)
    print(f"gradient-check PASÓ. error relativo máx = {rel_err:.3e}\n")

    # ------------------------------------------------------------------
    # 2) Dataset
    # ------------------------------------------------------------------
    data_path = os.path.join(REPO_ROOT, cfg["data"]["path"])
    X, y, labels, image_shape = load_emojis(data_path)
    print(f"Dataset: X={X.shape}  clases={len(labels)}  img={image_shape}")
    assert X.ndim == 2 and X.shape[1] == vcfg["input_dim"], "shape de X inesperada"
    assert X.min() >= 0.0 and X.max() <= 1.0, "X fuera de [0,1]"
    assert not np.isnan(X).any(), "NaN en X"

    # ------------------------------------------------------------------
    # 3) Entrenamiento (VAE principal, latent_dim=2)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"ENTRENAMIENTO VAE  latent_dim={vcfg['latent_dim']}  "
          f"recon={vcfg['recon_loss']}  beta={vcfg['beta']}")
    print("=" * 70)
    vae = VAE(
        input_dim=vcfg["input_dim"],
        encoder_hidden=vcfg["encoder_hidden"],
        latent_dim=vcfg["latent_dim"],
        decoder_hidden=vcfg["decoder_hidden"],
        hidden_activation=vcfg["hidden_activation"],
        output_activation=vcfg["output_activation"],
        recon_loss=vcfg["recon_loss"],
        beta=vcfg["beta"],
        seed=vcfg["seed"],
    )
    optimizer = build_optimizer(vcfg)

    # KL warmup opcional: beta sube linealmente de 0 a beta en warmup épocas.
    warmup = int(vcfg.get("kl_warmup_epochs", 0))
    beta_target = float(vcfg["beta"])
    if warmup > 0:
        def beta_schedule(ep):
            return beta_target * min(1.0, (ep + 1) / warmup)
    else:
        beta_schedule = None

    hist = vae.fit(
        X, optimizer,
        epochs=vcfg["epochs"], batch_size=vcfg["batch_size"],
        beta_schedule=beta_schedule, seed=vcfg["seed"],
        verbose=True, log_every=max(1, vcfg["epochs"] // 12),
    )

    print(f"\nLosses finales:  total={hist['total'][-1]:.4f}  "
          f"rec={hist['rec'][-1]:.4f}  KL={hist['kl'][-1]:.4f}")

    # ------------------------------------------------------------------
    # 4) Sanity checks de salida
    # ------------------------------------------------------------------
    X_hat = vae.reconstruct(X)
    assert X_hat.shape == X.shape, "shape de reconstrucción inesperada"
    assert X_hat.min() >= 0.0 and X_hat.max() <= 1.0, "x_hat fuera de [0,1]"
    assert not np.isnan(X_hat).any(), "NaN en reconstrucción"
    mu = vae.encode(X)
    assert mu.shape == (X.shape[0], vcfg["latent_dim"]), "shape de mu inesperada"
    recon_mse = float(np.mean(np.sum((X - X_hat) ** 2, axis=1)))
    print(f"Sanity: x_hat en [{X_hat.min():.3f}, {X_hat.max():.3f}]  "
          f"MSE recon (z=mu) = {recon_mse:.4f}")

    # ------------------------------------------------------------------
    # 5) Guardar modelo y generar figuras
    # ------------------------------------------------------------------
    model_path = os.path.join(results_dir, "vae_model.npz")
    vae.save(model_path)
    print(f"\nModelo guardado en {model_path}")

    print("\nGenerando figuras de analysis/...")
    paths = analysis.make_all_figures(vae, X, y, labels, image_shape, hist, results_dir)
    for p in paths:
        print(f"  -> {p}")

    print("\nLISTO.")


if __name__ == "__main__":
    main()
