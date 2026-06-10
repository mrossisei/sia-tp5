"""EJ1.a — Autoencoder basico sobre font.h (objetivo: max <= 1 pixel de error).

Orquesta: carga font, prueba varias seeds (el optimo tiene minimos locales),
entrena la mejor (Adam, full-batch, EarlyStopping monitor=loss), mide
pixel_error_summary, guarda el modelo y delega TODAS las figuras a analysis/.
Incluye sanity checks (shapes, rango de pixeles, NaN, pixel-error).
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np

from shared.fonts import load_font, INPUT_DIM
from shared.optimizers import Adam
from shared.regularization import EarlyStopping
from shared.metrics import pixel_error_summary
from shared.config_loader import load_yaml
from ej1.models.autoencoder import Autoencoder
from ej1.analysis.autoencoder import generate_all

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "basic")


def build_ae(cfg, seed):
    return Autoencoder(
        architecture=cfg["architecture"],
        hidden_activation=cfg["hidden_activation"],
        output_activation=cfg["output_activation"],
        bottleneck_activation=cfg["bottleneck_activation"],
        initializer=cfg.get("initializer", "glorot"),
        seed=seed,
        weight_decay=cfg.get("weight_decay", 0.0),
        loss_name=cfg["loss"],
    )


def train_one(X, cfg, seed, epochs, collect_losses=False):
    """Entrena un AE con una seed. Devuelve (ae, losses, best_loss, summary)."""
    rng = np.random.default_rng(seed)
    ae = build_ae(cfg, seed)
    opt = Adam(lr=cfg["learning_rate"])

    es_cfg = cfg.get("early_stopping", {})
    es = None
    if es_cfg.get("enabled", False):
        es = EarlyStopping(patience=es_cfg.get("patience", 4000),
                           min_delta=es_cfg.get("min_delta", 1e-7))

    losses = []
    for ep in range(epochs):
        mean_loss, _ = ae.train_epoch(X, X, opt, batch_size=cfg.get("batch_size", 0),
                                      shuffle=False, rng=rng)
        if collect_losses:
            losses.append(mean_loss)
        if es is not None and es(mean_loss, ae.get_params(), epoch=ep):
            if es.best_params is not None:
                ae.set_params(es.best_params)
            break
    else:
        if es is not None and es.best_params is not None:
            ae.set_params(es.best_params)

    summary = pixel_error_summary(X, ae.reconstruct(X))
    best_loss = es._best if es is not None else (losses[-1] if losses else mean_loss)
    best_epoch = es.best_epoch if es is not None else (len(losses) - 1)
    return ae, losses, float(best_loss), summary, best_epoch


def main():
    cfg = load_yaml(os.path.join(HERE, "config.yaml"))["basic"]
    X, labels = load_font()

    # --- Sanity checks de entrada ---
    assert X.shape == (32, INPUT_DIM), f"X shape inesperado: {X.shape}"
    assert set(np.unique(X)).issubset({0.0, 1.0}), "X debe ser binario {0,1}"
    print(f"[sanity] X={X.shape} rango=[{X.min()},{X.max()}] labels={len(labels)}")

    epochs = cfg["epochs"]
    seeds = cfg.get("seeds_to_try", [cfg["seed"]])

    # --- Fase 1: probar varias seeds con un presupuesto corto y elegir la mejor ---
    short = min(8000, epochs)
    print(f"\n[fase 1] probando seeds {seeds} ({short} epocas c/u)")
    results = []
    for seed in seeds:
        _, _, bl, s, _ = train_one(X, cfg, seed, short, collect_losses=False)
        results.append((seed, s["max"], s["n_exact"], bl))
        print(f"  seed={seed}: max={s['max']} n_exact={s['n_exact']}/32 "
              f"n_le1={s['n_le1']}/32 loss={bl:.3e}")

    # Mejor = menor max, luego mayor n_exact, luego menor loss.
    results.sort(key=lambda r: (r[1], -r[2], r[3]))
    best_seed = results[0][0]
    print(f"\n[fase 1] mejor seed = {best_seed} (max={results[0][1]}, "
          f"n_exact={results[0][2]})")

    # --- Fase 2: entrenar la mejor seed a fondo, recolectando la curva de loss ---
    print(f"\n[fase 2] entrenamiento final seed={best_seed} ({epochs} epocas)")
    ae, losses, best_loss, summary, best_epoch = train_one(
        X, cfg, best_seed, epochs, collect_losses=True)

    # --- Sanity checks de salida ---
    rec = ae.reconstruct(X)
    assert rec.shape == X.shape, f"rec shape {rec.shape}"
    assert not np.any(np.isnan(rec)), "NaN en la reconstruccion"
    assert 0.0 <= rec.min() and rec.max() <= 1.0, "salida fuera de [0,1]"
    Z = ae.encode(X)
    assert Z.shape == (32, 2), f"latente shape {Z.shape}"

    print("\n=== RESULTADO AE BASICO ===")
    print(f"  epocas entrenadas : {len(losses)} (mejor en epoca {best_epoch})")
    print(f"  loss (mejor)      : {best_loss:.4e}")
    print(f"  max pixel-error   : {summary['max']}")
    print(f"  mean pixel-error  : {summary['mean']:.4f}")
    print(f"  letras exactas    : {summary['n_exact']}/32")
    print(f"  letras con <=1    : {summary['n_le1']}/32")
    print(f"  OBJETIVO (max<=1) : {'CUMPLIDO' if summary['success'] else 'NO cumplido'}")
    if not summary["success"]:
        fail = [labels[i] for i in np.where(summary["per_pattern"] > 1)[0]]
        print(f"  letras que fallan : {fail}")

    # --- Guardar modelo ---
    model_path = os.path.join(RESULTS, "model.npz")
    ae.save(model_path)
    print(f"\n[guardado] modelo -> {model_path}")

    # --- Delegar figuras a analysis ---
    generate_all(ae, X, labels, RESULTS, losses=losses, best_epoch=best_epoch)
    print(f"[figuras] generadas en {RESULTS}")


if __name__ == "__main__":
    main()
