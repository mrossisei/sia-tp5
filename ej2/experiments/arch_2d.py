"""Estudio 4 (2D) — capacidad de la arquitectura, juzgada con la LOSS COMPLETA.
[SOLO ENTRENA Y VUELCA A DISCO — no grafica]

Pregunta: ¿cuanta capacidad fuera del cuello 2D conviene? Pero NO la juzgamos
solo con reconstruccion: logueamos tambien KL y la loss real que minimizamos
(rec_BCE + beta*KL), porque una arquitectura mas grande puede reconstruir mejor
metiendo mas info en el latente -> sube el KL -> el penalty compensa parte de la
ganancia. Asi vemos si el ranking aguanta bajo la loss completa.

4 arquitecturas x 3 seeds = 12 corridas (latente 2D, BCE, beta=1).
Corre UNA vez; despues graficas con plot_arch_2d.py (lee de disco, no entrena).

Salida (ej2/results/):
  exp_arch_2d_summary.csv   arch, seed, n_params, rec_mse, rec_bce, kl
"""

import os, sys, csv, time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import numpy as np

from shared.optimizers import Adam
from ej2.models.vae import VAE
from ej2.experiments._common import load_emojis, RESULTS_DIR

LATENT = 2
BETA = 1.0
EPOCHS = 400
SEEDS = [42, 0, 1]

# nombre -> (encoder_hidden, decoder_hidden)
ARCHS = {
    "angosta":  ([128],          [128]),
    "base":     ([256, 64],      [64, 256]),
    "ancha":    ([512, 128],     [128, 512]),
    "profunda": ([256, 128, 64], [64, 128, 256]),
}

CSV_PATH = os.path.join(RESULTS_DIR, "exp_arch_2d_summary.csv")
_EPS = 1e-12


def train(X, enc, dec, seed):
    vae = VAE(
        input_dim=X.shape[1],
        encoder_hidden=list(enc),
        latent_dim=LATENT,
        decoder_hidden=list(dec),
        hidden_activation="relu",
        output_activation="logistic",
        recon_loss="bce",
        beta=BETA,
        seed=seed,
    )
    vae.fit(X, Adam(lr=1e-3), epochs=EPOCHS, batch_size=64, beta=BETA,
            seed=seed, verbose=False)
    return vae


def n_params(vae):
    return int(sum(np.asarray(p).size for p in vae.get_flat_params()))


def rec_mse(vae, X):
    """MSE de reconstruccion (z=mu): metrica COMUN entre estudios."""
    x_hat = vae.reconstruct(X)
    return float(np.mean(np.sum((X - x_hat) ** 2, axis=1)))


def rec_bce(vae, X):
    """BCE de reconstruccion (z=mu): el TERMINO de reconstruccion de la loss."""
    x_hat = np.clip(vae.reconstruct(X), _EPS, 1.0 - _EPS)
    return float(np.mean(np.sum(-(X * np.log(x_hat) + (1 - X) * np.log(1 - x_hat)),
                                axis=1)))


def kl_total(vae, X):
    """KL(q(z|x)||N(0,I)) sumado en dims, promediado en datos (nats)."""
    mu, logvar = vae.encode_params(X)
    kl = -0.5 * (1.0 + logvar - mu ** 2 - np.exp(logvar))
    return float(kl.sum(axis=1).mean())


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()
    X, y, labels, image_shape = load_emojis()

    rows = []
    for name, (enc, dec) in ARCHS.items():
        for seed in SEEDS:
            t = time.time()
            vae = train(X, enc, dec, seed)
            r = {
                "arch": name, "seed": seed, "n_params": n_params(vae),
                "rec_mse": rec_mse(vae, X), "rec_bce": rec_bce(vae, X),
                "kl": kl_total(vae, X),
            }
            rows.append(r)
            print(f"[{name:8s} seed={seed:2d}] {time.time()-t:.1f}s  "
                  f"params={r['n_params']:>6d}  rec_mse={r['rec_mse']:.2f}  "
                  f"rec_bce={r['rec_bce']:.1f}  kl={r['kl']:.2f}  "
                  f"loss={r['rec_bce']+BETA*r['kl']:.1f}")

    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arch", "seed", "n_params",
                                          "rec_mse", "rec_bce", "kl"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items()})

    print("\n=== RESUMEN (media sobre seeds) ===")
    for name in ARCHS:
        sub = [r for r in rows if r["arch"] == name]
        rb = np.mean([r["rec_bce"] for r in sub])
        kl = np.mean([r["kl"] for r in sub])
        print(f"{name:8s}: params={sub[0]['n_params']:>6d}  "
              f"rec-MSE={np.mean([r['rec_mse'] for r in sub]):.2f}  "
              f"rec_BCE={rb:.1f}  KL={kl:.2f}  "
              f"LOSS={rb + BETA*kl:.1f}")
    print(f"[ok] datos -> {CSV_PATH}")
    print(f"[ok] arch_2d (train) en {time.time()-t0:.0f}s")
    print("FIN_ARCH_2D")


if __name__ == "__main__":
    main()
