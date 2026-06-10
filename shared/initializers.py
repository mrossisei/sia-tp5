import numpy as np


def _scale_for(method, n_in, n_out, scale):
    """Devuelve el desvío estándar a usar según el método de inicialización.

    - random_normal: usa `scale` tal cual.
    - he_normal:      sqrt(2 / fan_in)            (recomendado para ReLU)
    - xavier/glorot:  sqrt(2 / (fan_in + fan_out)) (recomendado para tanh/logistic)

    Por qué: en redes profundas, pesos iniciales mal escalados hacen que los
    gradientes se desvanezcan o exploten al retropropagar por muchas capas
    ("Fundamental Deep Learning Problem": DeepLearning.pdf, slide 86). La
    inicialización de Xavier aparece en la teoría justamente como uno de los
    remedios (DeepLearning.pdf, slide 88). La idea es elegir sigma para que la
    varianza de activaciones y gradientes se mantenga ~constante capa a capa.
    Papers: Glorot & Bengio 2010 (xavier); He et al. 2015 (he, para ReLU).
    """
    if method == "random_normal":
        return float(scale)
    if method == "he_normal":
        return float(np.sqrt(2.0 / n_in))
    if method in ("xavier", "glorot", "xavier_normal", "glorot_normal"):
        return float(np.sqrt(2.0 / (n_in + n_out)))
    raise ValueError(f"Unknown initializer: {method}")


def initialize_layer(n_in, n_out=1, method="random_normal", scale=0.1, seed=42):
    """Return (W, b) for a single layer.

    Layout:
      n_out == 1:  W shape (n_in,)        b is a Python float (scalar)
      n_out > 1 :  W shape (n_out, n_in)  b shape (n_out,)

    The random draws are arranged so that, for n_out=1, the resulting (W, b)
    is identical (modulo reshaping) to the legacy single-vector initializer
    which produced ``rng.normal(0, scale, size=n_in+1)`` with index 0 = bias.
    """
    rng = np.random.default_rng(seed)
    scale = _scale_for(method, n_in, n_out, scale)

    if n_out == 1:
        arr = rng.normal(0.0, scale, size=n_in + 1)
        return arr[1:].copy(), float(arr[0])
    arr = rng.normal(0.0, scale, size=n_out * (n_in + 1))
    b = arr[:n_out].copy()
    W = arr[n_out:].reshape(n_out, n_in).copy()
    return W, b


def initialize_layers(architecture, method="random_normal", scale=0.1, seed=42):
    """Initialize weights and biases for all layers of an MLP.

    architecture: [n_in, h1, h2, ..., n_out]
    Returns: list of (W, b) tuples, one per layer.
    Each layer gets a different seed derived from the base seed.
    """
    n_layers = len(architecture) - 1
    seed_seq = np.random.SeedSequence(seed)
    layer_seeds = seed_seq.spawn(n_layers)

    params = []
    for i in range(n_layers):
        W, b = initialize_layer(
            architecture[i], n_out=architecture[i + 1],
            method=method, scale=scale, seed=layer_seeds[i],
        )
        params.append((W, b))
    return params
