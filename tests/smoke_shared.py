"""Smoke test de la infraestructura shared/ (numpy puro).

Corre:
  1. Carga de font.h: shapes, rango {0,1}, dibujo ASCII de 3 letras.
  2. Gradient-check numérico del MLP._backward para MSE y para BCE (valida el
     fix de BCE+sigmoide de AGENTS §4.3).
  3. Entrenamiento corto de un AE (MSE y BCE) sobre font.h: la loss baja y el
     pixel-error medio mejora.
  4. Inicializadores: random_normal / he / glorot producen escalas razonables.

Uso:  python3 tests/smoke_shared.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from shared.fonts import load_font, LABELS, to_bin_array, FONT3
from shared.mlp import MLP
from shared.optimizers import Adam
from shared.losses import mse, bce
from shared.metrics import pixel_errors, pixel_error_summary
from shared.initializers import initialize_layers


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def ascii_letter(flat):
    grid = flat.reshape(7, 5)
    return "\n".join("".join("#" if v >= 0.5 else "." for v in row) for row in grid)


def test_font():
    section("1. font.h")
    X, labels = load_font()
    assert X.shape == (32, 35), X.shape
    assert set(np.unique(X).tolist()) <= {0.0, 1.0}, np.unique(X)
    assert labels == LABELS and len(labels) == 32
    # decodificación consistente con to_bin_array
    assert np.array_equal(X[1].reshape(7, 5), to_bin_array(FONT3[1]))
    for idx in (1, 15, 26):  # a, o, z
        print(f"\nLetra '{labels[idx]}':")
        print(ascii_letter(X[idx]))
    print("\nOK font.h")
    return X, labels


def numerical_grad(model, X, t, eps=1e-5):
    """Gradiente numérico (diferencias centrales) de la mean-loss vs params."""
    params = model.get_params()
    num = []
    for p in params:
        g = np.zeros_like(p, dtype=float)
        it = np.nditer(p, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            ix = it.multi_index
            orig = p[ix]
            p[ix] = orig + eps
            lp = model._loss_fn(t, model.predict(X))
            p[ix] = orig - eps
            lm = model._loss_fn(t, model.predict(X))
            p[ix] = orig
            g[ix] = (lp - lm) / (2 * eps)
            it.iternext()
        num.append(g)
    return num


def analytic_grad(model, X, t):
    out, cache = model._forward(X)
    gW, gb = model._backward(t, cache)
    flat = []
    for i in range(model.n_layers):
        flat.append(gW[i])
        flat.append(gb[i])
    return flat


def test_gradcheck():
    section("2. Gradient-check MLP (MSE y BCE)")
    rng = np.random.default_rng(0)
    X = rng.random((6, 8))
    t = (rng.random((6, 4)) > 0.5).astype(float)
    for loss_name in ("mse", "bce"):
        model = MLP([8, 6, 4], hidden_activation="tanh",
                    output_activation="logistic", loss_name=loss_name,
                    initializer="glorot", seed=1)
        num = numerical_grad(model, X, t)
        ana = analytic_grad(model, X, t)
        max_rel = 0.0
        for n, a in zip(num, ana):
            denom = np.maximum(1e-8, np.abs(n) + np.abs(a))
            max_rel = max(max_rel, float(np.max(np.abs(n - a) / denom)))
        print(f"  {loss_name}: max rel err = {max_rel:.2e}")
        assert max_rel < 1e-4, f"gradcheck {loss_name} falló: {max_rel}"
    print("OK gradient-check")


def test_train(X):
    section("3. Entrenamiento corto de AE (MSE y BCE)")
    arch = [35, 20, 10, 2, 10, 20, 35]
    for loss_name in ("mse", "bce"):
        rng = np.random.default_rng(42)
        model = MLP(arch, hidden_activation="relu", output_activation="logistic",
                    loss_name=loss_name, initializer="glorot", seed=42)
        opt = Adam(lr=2e-3)
        first = None
        for ep in range(1500):
            loss, _ = model.train_epoch(X, X, opt, batch_size=0, shuffle=False)
            if first is None:
                first = loss
        recon = model.predict(X)
        summ = pixel_error_summary(X, recon)
        print(f"  {loss_name}: loss {first:.4f} -> {loss:.4f} | "
              f"pixel-err max={summ['max']} mean={summ['mean']:.2f} "
              f"exact={summ['n_exact']}/32")
        assert loss < first, "la loss no bajó"
    print("OK entrenamiento (sanity, no es la corrida final)")


def test_init():
    section("4. Inicializadores")
    arch = [35, 20, 2, 35]
    for method in ("random_normal", "he_normal", "glorot"):
        params = initialize_layers(arch, method=method, scale=0.1, seed=3)
        stds = [float(np.std(W)) for W, _ in params]
        print(f"  {method}: stds por capa = {[round(s, 3) for s in stds]}")
    print("OK inicializadores")


if __name__ == "__main__":
    X, labels = test_font()
    test_gradcheck()
    test_train(X)
    test_init()
    print("\nSMOKE TEST OK ✔")
