"""EJ1.b — Denoising Autoencoder sobre font.h.

Entrena un DAE reusando la clase Autoencoder (misma arquitectura simetrica, con
el cuello algo mas ancho porque el denoising no exige 2D). Augmentation ONLINE:
cada epoca se regenera el ruido y se entrena con train_epoch(X_noisy, X_clean).
Evalua la capacidad de quitar ruido a distintos niveles y compara con el AE
basico (que no fue entrenado para denoising). Delega figuras a analysis/.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np

from shared.fonts import load_font, INPUT_DIM
from shared.optimizers import Adam
from shared.metrics import pixel_errors, pixel_error_summary
from shared.config_loader import load_yaml
from ej1.models.autoencoder import Autoencoder
from ej1.models.denoising import apply_noise
from ej1.analysis.denoising import generate_all

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "denoising")
BASIC_MODEL = os.path.join(HERE, "results", "basic", "model.npz")


def train_dae(X, cfg):
    """Entrena el DAE con augmentation online (ruido regenerado cada epoca)."""
    seed = cfg["seed"]
    rng = np.random.default_rng(seed)
    ae = Autoencoder(
        architecture=cfg["architecture"],
        hidden_activation=cfg["hidden_activation"],
        output_activation=cfg["output_activation"],
        bottleneck_activation=cfg["bottleneck_activation"],
        bottleneck_size=cfg.get("bottleneck_size", 2),
        initializer=cfg.get("initializer", "glorot"),
        seed=seed,
        weight_decay=cfg.get("weight_decay", 0.0),
        loss_name=cfg["loss"],
    )
    opt = Adam(lr=cfg["learning_rate"])
    ntype = cfg["noise"]["train_type"]
    nlevel = cfg["noise"]["train_level"]

    losses = []
    for ep in range(cfg["epochs"]):
        # Augmentation online: ruido nuevo en cada epoca, target = limpio.
        X_noisy = apply_noise(X, ntype, nlevel, rng)
        ml, _ = ae.train_epoch(X_noisy, X, opt, batch_size=cfg.get("batch_size", 0),
                               shuffle=False, rng=rng)
        losses.append(ml)
    return ae, losses


def evaluate(model, X, ntype, levels, n_reps=20, seed=11):
    """pixel-error medio del modelo frente a ruido (promedio sobre reps)."""
    out = {}
    for lvl in levels:
        rng = np.random.default_rng(seed + int(lvl * 1000))
        errs = []
        for _ in range(n_reps):
            Xn = apply_noise(X, ntype, lvl, rng)
            errs.append(pixel_errors(X, model.reconstruct(Xn)).mean())
        out[lvl] = float(np.mean(errs))
    return out


def main():
    cfg = load_yaml(os.path.join(HERE, "config.yaml"))["denoising"]
    X, labels = load_font()

    assert X.shape == (32, INPUT_DIM)
    print(f"[sanity] X={X.shape} rango=[{X.min()},{X.max()}]")
    print(f"[config] arch={cfg['architecture']} cuello={cfg.get('bottleneck_size',2)} "
          f"ruido_train={cfg['noise']['train_type']}@{cfg['noise']['train_level']}")

    # --- Entrenar DAE ---
    print(f"\n[train] DAE ({cfg['epochs']} epocas, augmentation online)")
    dae, losses = train_dae(X, cfg)

    # Sanity de salida
    rec = dae.reconstruct(X)
    assert not np.any(np.isnan(rec)) and 0 <= rec.min() and rec.max() <= 1
    s_clean = pixel_error_summary(X, rec)
    print(f"  loss final          : {losses[-1]:.4e}")
    print(f"  (entrada limpia) max : {s_clean['max']}  mean: {s_clean['mean']:.3f}  "
          f"exact: {s_clean['n_exact']}/32")

    # --- AE basico (baseline sin denoising) ---
    ae_basic = Autoencoder.load(BASIC_MODEL)

    # --- Evaluacion: capacidad de quitar ruido a distintos niveles ---
    ntype = cfg["noise"]["train_type"]
    levels = cfg["noise"]["sweep_levels"]
    dae_eval = evaluate(dae, X, ntype, levels)
    base_eval = evaluate(ae_basic, X, ntype, levels)
    print(f"\n=== pixel-error medio vs nivel de ruido ('{ntype}') ===")
    print(f"  {'nivel':>6} | {'DAE':>8} | {'AE basico':>10}")
    for lvl in levels:
        print(f"  {lvl:>6.2f} | {dae_eval[lvl]:>8.3f} | {base_eval[lvl]:>10.3f}")

    # Resumen extra por tipo de ruido (DAE)
    print("\n=== DAE: pixel-error medio por tipo de ruido ===")
    for nt in cfg["noise"]["sweep_types"]:
        ev = evaluate(dae, X, nt, levels)
        line = "  ".join(f"{lvl}:{ev[lvl]:.2f}" for lvl in levels)
        print(f"  {nt:>12}: {line}")

    # --- Guardar modelo + figuras ---
    model_path = os.path.join(RESULTS, "model.npz")
    dae.save(model_path)
    print(f"\n[guardado] DAE -> {model_path}")

    generate_all(dae, ae_basic, X, labels, RESULTS, cfg["noise"])
    print(f"[figuras] generadas en {RESULTS}")


if __name__ == "__main__":
    main()
