"""Estudio 3 (2D) — BCE vs MSE como loss de reconstruccion, en el latente 2D
que usa la corrida final.  [SOLO ENTRENA Y VUELCA A DISCO — no grafica]

Motivacion: el VAE final es 2D y generativo. No tiene sentido elegir la loss
mirando reconstruccion en 16D (una dim que no usamos): la elegimos en 2D y
mirando la CONSECUENCIA GENERATIVA. El mecanismo es independiente de la dim
(BCE da gradiente limpio (x_hat - x) con salida logistica; MSE+logistica se
desvanece en pixeles saturados -> tiende a la media gris), pero con MSE el
termino de reconstruccion es mas chico, el KL pesa relativamente mas y el
latente COLAPSA -> menos info al decoder -> generacion gris.

3 seeds x 2 loss = 6 corridas (latente 2D). Corre UNA vez; despues graficas
todas las veces que quieras con plot_recon_loss_2d.py (lee de disco, no entrena).

Salidas (ej2/results/):
  exp_reconloss_2d_summary.csv   metricas por recon_loss x seed (rec_mse, kl, sharpness)
  exp_reconloss_2d_arrays.npz    generacion z~N(0,I) (BCE/MSE) + medias mu(x) del
                                 latente + clases y (para el scatter de colapso)
"""

import os, sys, csv, time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import numpy as np

from shared.optimizers import Adam
from ej2.models.vae import VAE
from ej2.experiments._common import load_emojis, RESULTS_DIR

LATENT = 2                 # <- la dim de la corrida final (no 16D)
LOSSES = ["bce", "mse"]
SEEDS = [42, 0, 1]
EPOCHS = 600
N_SAMPLES = 8              # muestras z~N(0,I) para el panel cualitativo
REP_SEED = 42             # seed representativa del panel cualitativo

CSV_PATH = os.path.join(RESULTS_DIR, "exp_reconloss_2d_summary.csv")
NPZ_PATH = os.path.join(RESULTS_DIR, "exp_reconloss_2d_arrays.npz")


def train(X, recon_loss, seed):
    vae = VAE(
        input_dim=X.shape[1],
        encoder_hidden=[256, 64],
        latent_dim=LATENT,
        decoder_hidden=[64, 256],
        hidden_activation="relu",
        output_activation="logistic",
        recon_loss=recon_loss,
        beta=1.0,
        seed=seed,
    )
    vae.fit(X, Adam(lr=1e-3), epochs=EPOCHS, batch_size=64, beta=1.0,
            seed=seed, verbose=False)
    return vae


def rec_mse(vae, X):
    """MSE de reconstruccion (z=mu): metrica COMUN para comparar ambas loss."""
    x_hat = vae.reconstruct(X)
    return float(np.mean(np.sum((X - x_hat) ** 2, axis=1)))


def kl_total(vae, X):
    """KL(q(z|x)||N(0,I)) sumado en las dims latentes, promediado en datos (nats)."""
    mu, logvar = vae.encode_params(X)
    kl = -0.5 * (1.0 + logvar - mu ** 2 - np.exp(logvar))  # (N, D)
    return float(kl.sum(axis=1).mean())


def sharpness(vae, X):
    """Fraccion de pixeles 'confiados' (|p-0.5|>0.4): mas alto = mas nitido."""
    p = vae.reconstruct(X)
    return float(np.mean(np.abs(p - 0.5) > 0.4))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()
    X, y, labels, image_shape = load_emojis()

    metrics = {l: {"rec_mse": [], "kl": [], "sharp": []} for l in LOSSES}
    rep_models = {}
    for l in LOSSES:
        for seed in SEEDS:
            t = time.time()
            vae = train(X, l, seed)
            metrics[l]["rec_mse"].append(rec_mse(vae, X))
            metrics[l]["kl"].append(kl_total(vae, X))
            metrics[l]["sharp"].append(sharpness(vae, X))
            if seed == REP_SEED:
                rep_models[l] = vae
            print(f"[{l.upper()} seed={seed:2d}] {time.time()-t:.1f}s  "
                  f"rec_mse={metrics[l]['rec_mse'][-1]:.2f}  "
                  f"kl={metrics[l]['kl'][-1]:.2f}  "
                  f"sharp={metrics[l]['sharp'][-1]:.3f}")

    # --- muestras generativas: MISMO z~N(0,I) decodificado con cada loss ------
    rng = np.random.default_rng(0)
    Z = rng.standard_normal(size=(N_SAMPLES, LATENT))
    gen = {l: rep_models[l].decode(Z) for l in LOSSES}

    # --- medias mu(x) del latente (para el scatter colapso BCE vs MSE) --------
    mu = {l: rep_models[l].encode_params(X)[0] for l in LOSSES}  # (N, 2)

    # =================================================================== CSV
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["recon_loss", "seed", "rec_mse", "kl", "sharpness"])
        for l in LOSSES:
            for i, seed in enumerate(SEEDS):
                w.writerow([l, seed,
                            f"{metrics[l]['rec_mse'][i]:.4f}",
                            f"{metrics[l]['kl'][i]:.4f}",
                            f"{metrics[l]['sharp'][i]:.4f}"])

    # =================================================================== NPZ
    # Imagenes de generacion (lo unico que NO se puede reconstruir desde el CSV
    # porque depende de los modelos). El plot script las lee de aca.
    np.savez(NPZ_PATH,
             gen_bce=gen["bce"], gen_mse=gen["mse"], Z=Z,
             mu_bce=mu["bce"], mu_mse=mu["mse"], y=y,
             labels=np.array(labels, dtype=object),
             image_shape=np.array(image_shape), rep_seed=REP_SEED)

    print("\n=== RESUMEN (media sobre seeds) ===")
    for l in LOSSES:
        print(f"{l.upper()}: rec-MSE={np.mean(metrics[l]['rec_mse']):.2f}  "
              f"KL={np.mean(metrics[l]['kl']):.2f}  "
              f"nitidez={np.mean(metrics[l]['sharp']):.3f}")
    print(f"[ok] datos -> {CSV_PATH}")
    print(f"[ok] arrays -> {NPZ_PATH}")
    print(f"[ok] reconloss_2d (train) en {time.time()-t0:.0f}s")
    print("FIN_RECONLOSS_2D")


if __name__ == "__main__":
    main()
