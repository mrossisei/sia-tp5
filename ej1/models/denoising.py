"""Funciones de ruido para el Denoising Autoencoder (numpy puro).

Procedimiento DAE (Autoencoders.pdf, slides 20-22, en especial slide 22):
  "se modela el ruido, es decir, se lo caracteriza numéricamente
   (e.g. salt-and-pepper) o en general con una función de distribución de
   probabilidad (e.g. Gaussiano) y se agrega a los datos X generando una
   nueva instancia X̃. Sin embargo, la salida que se pone en la red
   corresponde al dato X sin alterar."
Es decir: entrada = X̃ ruidosa, target = X limpio. El Encoder/Decoder preserva
el contenido de información relevante y "filtra" el ruido (slide 21). Las dos
familias de ruido de la slide están implementadas acá (salt_pepper, gaussian)
más una tercera (masking/oclusión) para evaluar generalización.

Todas reciben un rng (np.random.Generator) para ser deterministas y aptas para
augmentation online (regenerar el ruido en cada época). Devuelven una copia
ruidosa de X en el rango [0,1].

El DAE reusa la MISMA clase Autoencoder (arquitectura simétrica con cuello de
botella): si la red comprime y reconstruye bien las letras limpias, sólo cambia
el OBJETIVO de entrenamiento (entrada ruidosa -> salida limpia). El cuello puede
ser 2D (como el AE básico) o algo más ancho, porque el denoising no exige una
representación estrictamente bidimensional.
"""

import numpy as np


def salt_pepper(X, p, rng):
    """Flip de cada pixel con probabilidad p (sal y pimienta sobre {0,1}).

    Ruido "caracterizado numéricamente" (Autoencoders.pdf, slide 22). Con prob
    p, el pixel se invierte (0->1, 1->0). Sobre datos binarios esto es el
    equivalente clásico de sal-y-pimienta.
    """
    X = np.asarray(X, dtype=np.float64)
    flip = rng.random(X.shape) < p
    out = X.copy()
    out[flip] = 1.0 - out[flip]
    return np.clip(out, 0.0, 1.0)


def gaussian(X, sigma, rng):
    """Ruido gaussiano aditivo: x~ = x + N(0, sigma^2), clipeado a [0,1].

    Ruido modelado "con una función de distribución de probabilidad"
    (Autoencoders.pdf, slide 22).
    """
    X = np.asarray(X, dtype=np.float64)
    noise = rng.normal(0.0, sigma, size=X.shape)
    return np.clip(X + noise, 0.0, 1.0)


def masking(X, frac, rng):
    """Pone a 0 una fracción `frac` de píxeles elegidos al azar (oclusión)."""
    X = np.asarray(X, dtype=np.float64)
    mask = rng.random(X.shape) < frac
    out = X.copy()
    out[mask] = 0.0
    return np.clip(out, 0.0, 1.0)


# Registro de tipos de ruido por nombre (para barridos parametrizados).
NOISE_FNS = {
    "salt_pepper": salt_pepper,
    "gaussian": gaussian,
    "masking": masking,
}


def apply_noise(X, noise_type, level, rng):
    """Aplica el ruido `noise_type` con intensidad `level` (p o sigma o frac)."""
    if noise_type not in NOISE_FNS:
        raise ValueError(f"Tipo de ruido desconocido: {noise_type}")
    return NOISE_FNS[noise_type](X, level, rng)
