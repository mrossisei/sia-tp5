"""Utilidades compartidas por los experimentos EXTRA.

Reutiliza el VAE de ej2/models/vae.py (numpy puro, ya gradchequeado) y las
piezas de shared/. matplotlib NO se importa acá (solo en los scripts que grafican).
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

from shared.optimizers import Adam
from ej2.models.vae import VAE

RESULTS = os.path.join(REPO_ROOT, "extra", "results")
MODELS = os.path.join(RESULTS, "models")
EMOJIS = os.path.join(REPO_ROOT, "ej2", "data", "emojis.npz")
MNIST = os.path.join(REPO_ROOT, "ej3", "data", "mnist.npz")

_BCE_EPS = 1e-12


def ensure_dirs():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)


# ----------------------------------------------------------------- datasets
def load_emojis(path=EMOJIS):
    """Devuelve (X float64 en [0,1], y clases, labels, image_shape)."""
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype(np.float64)
    y = d["y"]
    labels = [str(s) for s in d["labels"]]
    image_shape = tuple(int(v) for v in d["image_shape"])
    return X, y, labels, image_shape


def load_mnist_test(n=5000, path=MNIST, seed=0):
    """Subconjunto determinista del test de MNIST, normalizado a [0,1]."""
    d = np.load(path, allow_pickle=True)
    Xte = d["X_test"].astype(np.float64) / 255.0
    yte = d["y_test"]
    if 0 < n < len(Xte):
        idx = np.random.default_rng(seed).permutation(len(Xte))[:n]
        Xte, yte = Xte[idx], yte[idx]
    return Xte, yte


# ----------------------------------------------------------------- métricas
def rec_bce(vae, X):
    """BCE de reconstrucción (z = mu, determinista), sumada en píxeles y
    promediada en muestras — la misma métrica que el rec del entrenamiento."""
    x_hat = vae.reconstruct(X)
    p = np.clip(x_hat, _BCE_EPS, 1.0 - _BCE_EPS)
    return float(np.mean(np.sum(-(X * np.log(p) + (1.0 - X) * np.log(1.0 - p)), axis=1)))


def rec_mse(vae, X):
    """MSE de reconstrucción (z = mu): métrica COMÚN para comparar modelos
    entrenados con distinta loss."""
    x_hat = vae.reconstruct(X)
    return float(np.mean(np.sum((X - x_hat) ** 2, axis=1)))


def kl_per_dim(vae, X):
    """KL por dimensión latente, promediado sobre los datos (en nats).

    KL_j = E_x[ -1/2 (1 + logvar_j - mu_j^2 - exp(logvar_j)) ]
    Un VAE con "posterior collapse" en la dim j da KL_j ~ 0 (la dim no se usa:
    mu_j ~ 0 y logvar_j ~ 0 -> q(z_j|x) = prior). Usa encode_params (mismo clip
    de logvar que el forward), así el número coincide con lo que la red usa.
    """
    mu, logvar = vae.encode_params(X)
    kl = -0.5 * (1.0 + logvar - mu ** 2 - np.exp(logvar))  # (N, D)
    return kl.mean(axis=0)  # (D,)


def active_units(vae, X, threshold=0.01):
    """Cantidad de dimensiones latentes 'activas' (KL_j > threshold)."""
    return int(np.sum(kl_per_dim(vae, X) > threshold))


# ----------------------------------------------------------------- modelos
def make_vae(input_dim, latent_dim, encoder_hidden, decoder_hidden,
             recon_loss="bce", output_activation="logistic", beta=1.0, seed=42):
    return VAE(
        input_dim=input_dim,
        encoder_hidden=list(encoder_hidden),
        latent_dim=latent_dim,
        decoder_hidden=list(decoder_hidden),
        hidden_activation="relu",
        output_activation=output_activation,
        recon_loss=recon_loss,
        beta=beta,
        seed=seed,
    )


def fit_simple(vae, X, epochs=600, batch_size=64, lr=1e-3, beta=1.0, seed=42):
    """Entrena con Adam (sin tracking). Devuelve hist de vae.fit."""
    opt = Adam(lr=lr)
    return vae.fit(X, opt, epochs=epochs, batch_size=batch_size, beta=beta,
                   seed=seed, verbose=False)


def fit_tracking(vae, X_tr, X_te, epochs=1500, batch_size=64, lr=1e-3,
                 beta=1.0, seed=42, eval_every=10):
    """Igual que vae.fit pero evalúa rec en train Y en test (held-out) cada
    eval_every épocas. Devuelve dict con curvas (ev, rec_tr, rec_te) + finales.

    Mismo bucle que VAE.fit: por batch muestrea eps fresco, forward(sample=True),
    backward, opt.step(get_flat_params, grads), set_flat_params.
    """
    opt = Adam(lr=lr)
    rng = np.random.default_rng(seed)
    N = X_tr.shape[0]
    ev, rec_tr, rec_te = [], [], []
    for ep in range(epochs):
        idx = rng.permutation(N)
        for s in range(0, N, batch_size):
            Xb = X_tr[idx[s:s + batch_size]]
            eps = rng.standard_normal((Xb.shape[0], vae.latent_dim))
            _, cache = vae.forward(Xb, eps=eps, sample=True)
            grads = vae.backward(cache, beta=beta)
            vae.set_flat_params(opt.step(vae.get_flat_params(), grads))
        if ep % eval_every == 0 or ep == epochs - 1:
            ev.append(ep)
            rec_tr.append(rec_bce(vae, X_tr))
            rec_te.append(rec_bce(vae, X_te))
    return {
        "ev": np.array(ev),
        "rec_tr": np.array(rec_tr),
        "rec_te": np.array(rec_te),
        "rec_tr_final": rec_tr[-1],
        "rec_te_final": rec_te[-1],
    }


def stratified_split(y, test_frac=0.2, seed=0):
    """Índices train/test estratificados por clase (cada clase aporta ~test_frac
    de sus variantes al test). Determinista."""
    rng = np.random.default_rng(seed)
    tr_idx, te_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_te = max(1, int(round(len(idx) * test_frac)))
        te_idx.extend(idx[:n_te].tolist())
        tr_idx.extend(idx[n_te:].tolist())
    return np.array(sorted(tr_idx)), np.array(sorted(te_idx))
