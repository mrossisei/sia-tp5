"""Utilidades compartidas por los experimentos del VAE.

Cada experimento entrena uno o varios VAE variando un hiperparámetro y guarda
una figura en ej2/results/. No es analysis/ "oficial" pero respeta la regla de
no mezclar (matplotlib se usa SOLO acá, no en models/).
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

from shared.optimizers import Adam
from ej2.models.vae import VAE


RESULTS_DIR = os.path.join(REPO_ROOT, "ej2", "results")
DATA_PATH = os.path.join(REPO_ROOT, "ej2", "data", "emojis.npz")


def load_emojis():
    d = np.load(DATA_PATH, allow_pickle=True)
    X = d["X"].astype(np.float64)
    y = d["y"]
    labels = [str(s) for s in d["labels"]]
    image_shape = tuple(int(v) for v in d["image_shape"])
    return X, y, labels, image_shape


def train_vae(X, latent_dim=2, beta=1.0, beta_schedule=None, epochs=400,
              batch_size=64, lr=1e-3, seed=42,
              encoder_hidden=(256, 64), decoder_hidden=(64, 256)):
    vae = VAE(
        input_dim=X.shape[1],
        encoder_hidden=list(encoder_hidden),
        latent_dim=latent_dim,
        decoder_hidden=list(decoder_hidden),
        hidden_activation="relu",
        output_activation="logistic",
        recon_loss="bce",
        beta=beta,
        seed=seed,
    )
    opt = Adam(lr=lr)
    hist = vae.fit(X, opt, epochs=epochs, batch_size=batch_size,
                   beta=beta, beta_schedule=beta_schedule, seed=seed,
                   verbose=False)
    return vae, hist


def img(vec, image_shape):
    return np.asarray(vec, dtype=np.float64).reshape(image_shape)
