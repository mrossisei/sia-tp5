"""Funciones de activación y sus derivadas (expresadas en función de la salida).

Notas de teoría:
  - logistic/tanh: sigmoides clásicas de TP3; `logistic` da salidas en (0,1),
    lo que permite leer cada píxel reconstruido como una probabilidad
    (necesario para BCE, Autoencoders.pdf slide 51).
  - relu: "Rectified Linear Neurons" (DeepLearning.pdf, slide 100; CNN.pdf,
    slide 27). Al no saturar para h>0, su derivada no se achica y mitiga el
    desvanecimiento del gradiente en redes profundas (DeepLearning.pdf,
    slide 86). Usada en el VAE de EJ2.
  - identity: capa lineal; se usa en el cuello del AE (código latente sin
    saturar) y en las cabezas mu/logvar del VAE (la capa z del VAE actúa como
    un perceptrón lineal con activación identidad: Autoencoders.pdf, slide 83).
"""

import numpy as np


def activate(h, name, beta=1.0):
    if name == "identity":
        return h.copy()
    elif name == "logistic":
        return 1.0 / (1.0 + np.exp(-np.clip(beta * h, -500, 500)))
    elif name == "tanh":
        return np.tanh(beta * h)
    elif name == "step":
        return np.where(h > 0, 1.0, -1.0)
    elif name == "relu":
        return np.maximum(0.0, h)
    raise ValueError(f"Unknown activation: {name}")


def activate_deriv(O, name, beta=1.0):
    """Derivative of activation w.r.t. pre-activation h, expressed via output O.

    For 'step', returns a pseudo-derivative (identity) since the true
    derivative is 0 almost everywhere and undefined at 0.
    For 'relu', O is the post-activation output so relu'(h) = (O > 0).
    """
    if name == "identity":
        return np.ones_like(O)
    elif name == "logistic":
        # sigma'(h) = beta * sigma(h) * (1 - sigma(h)) = beta * O * (1-O)
        return beta * O * (1.0 - O)
    elif name == "tanh":
        # tanh'(h) = beta * (1 - tanh(h)^2) = beta * (1 - O^2)
        return beta * (1.0 - O ** 2)
    elif name == "step":
        return np.ones_like(O)
    elif name == "relu":
        # relu'(h) = 1 si h>0, 0 si no (subgradiente 0 en h=0)
        return np.where(O > 0, 1.0, 0.0)
    raise ValueError(f"Unknown activation: {name}")
