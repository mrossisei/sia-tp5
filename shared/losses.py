import numpy as np

# Estabilidad numérica para BCE: evita log(0) clipeando las probabilidades.
_BCE_EPS = 1e-12


def mse(y_true, y_pred):
    """ECM 0.5·||t - y||² por muestra, promediado sobre el batch.

    Es la loss "natural" del autoencoder: L(X, X') = ||X - X'||²
    (Autoencoders.pdf, slides 3-4). El 0.5 simplifica la derivada.
    """
    diff = y_true - y_pred
    if diff.ndim == 1:
        return 0.5 * float(np.mean(diff ** 2))
    sum_axes = tuple(range(1, diff.ndim))
    return 0.5 * float(np.mean(np.sum(diff ** 2, axis=sum_axes)))


def mse_deriv(y_true, y_pred):
    return y_pred - y_true


def bce(y_true, y_pred):
    """Binary cross-entropy promediada por muestra (suma sobre features).

    Fórmula de la teoría (Autoencoders.pdf, slide 51):
        Hp(q) = -(1/N) Σ_i { y_i·log p(y_i) + (1-y_i)·log(1-p(y_i)) }
    interpretando cada píxel como una Bernoulli con probabilidad p = salida
    sigmoide. Es la loss natural para píxeles binarios en {0,1}; un
    reconstructor perfecto da BCE = 0 (misma slide).
    Clipea ``y_pred`` a [eps, 1-eps] para evitar ``log(0)`` (ver AGENTS §4.3).
    """
    p = np.clip(y_pred, _BCE_EPS, 1.0 - _BCE_EPS)
    terms = y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)
    if terms.ndim == 1:
        return -float(np.mean(terms))
    sum_axes = tuple(range(1, terms.ndim))
    return -float(np.mean(np.sum(terms, axis=sum_axes)))


def bce_deriv(y_true, y_pred):
    """dBCE/dy_pred. Cuidado: diverge cerca de 0/1.

    En la práctica el ``MLP`` fusiona sigmoide+BCE y usa el gradiente estable
    ``(y_pred - y_true)`` (el factor ``y(1-y)`` se cancela). Esta función existe
    por completitud / gradient-checking con salida ``identity``.
    """
    p = np.clip(y_pred, _BCE_EPS, 1.0 - _BCE_EPS)
    return (p - y_true) / (p * (1.0 - p))


def build_loss(name):
    """Devuelve (loss_fn, loss_deriv_fn) para el nombre dado.

    Soporta "mse" y "bce". Para agregar una nueva loss:
      1. Definir `nombre_loss(y_true, y_pred) -> escalar`
      2. Definir `nombre_loss_deriv(y_true, y_pred) -> ndarray per-sample`
      3. Agregar una rama acá.
    """
    if name == "mse":
        return mse, mse_deriv
    if name == "bce":
        return bce, bce_deriv
    raise ValueError(f"Unknown loss: {name}")
