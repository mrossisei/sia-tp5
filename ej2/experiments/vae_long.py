"""EJ2 — Corrida LARGA del VAE (overnight) con checkpoints y resume.

Es la única pieza genuinamente compute-bound del TP5: el VAE principal corta
en 600 épocas con la reconstrucción todavía bajando. Acá entrenamos por horas
con checkpoints periódicos (modelo + estado de Adam + historial), de modo que:
  - se puede cortar en cualquier momento y el último checkpoint queda usable;
  - se puede reanudar con --resume;
  - al final (o al cortar) quedan las curvas y la suite completa de figuras.

Uso típico (ver run_overnight.sh):
  python3 ej2/experiments/vae_long.py --epochs 50000 --ckpt-every 1000
  python3 ej2/experiments/vae_long.py --data ej2/data/emojis_24.npz --tag 24px \
          --encoder 512 128 --decoder 128 512 --epochs 12000 --ckpt-every 250
  python3 ej2/experiments/vae_long.py ... --resume   # continúa del último ckpt

Salidas en ej2/results/long_{tag}/:
  ckpt_model.npz / ckpt_state.npz  (checkpoint rodante, escritura atómica)
  loss_history.csv, loss_curve.png (actualizados en cada checkpoint)
  vae_model.npz + suite de figuras (al terminar)
"""

import argparse
import csv
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import numpy as np

from shared.optimizers import Adam
from ej2.models.vae import VAE


# ----------------------------------------------------------------- utilidades
def load_dataset(path):
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype(np.float64)
    y = d["y"]
    labels = [str(s) for s in d["labels"]]
    image_shape = tuple(int(v) for v in d["image_shape"])
    return X, y, labels, image_shape


def atomic_savez(path, **arrays):
    """Escribe el .npz a un tmp y renombra: un corte a mitad de write no
    corrompe el checkpoint anterior."""
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def save_checkpoint(out_dir, vae, opt, hist, epoch):
    model_path = os.path.join(out_dir, "ckpt_model.npz")
    # VAE.save no es atómico: guardamos a tmp y renombramos.
    tmp_model = model_path + ".tmp.npz"
    vae.save(tmp_model)
    os.replace(tmp_model, model_path)

    state = {"epoch": np.array(epoch),
             "hist_total": np.array(hist["total"]),
             "hist_rec": np.array(hist["rec"]),
             "hist_kl": np.array(hist["kl"]),
             "hist_beta": np.array(hist["beta"])}
    adam = opt.state_dict()
    state["adam_t"] = np.array(adam["t"])
    state["adam_lr"] = np.array(adam["lr"])
    if adam["m"] is not None:
        for i, (m, v) in enumerate(zip(adam["m"], adam["v"])):
            state[f"adam_m{i}"] = m
            state[f"adam_v{i}"] = v
        state["adam_n"] = np.array(len(adam["m"]))
    else:
        state["adam_n"] = np.array(0)
    atomic_savez(os.path.join(out_dir, "ckpt_state.npz"), **state)


def load_checkpoint(out_dir, opt):
    """Devuelve (vae, hist, epoch) y deja `opt` con su estado cargado."""
    vae = VAE.load(os.path.join(out_dir, "ckpt_model.npz"))
    d = np.load(os.path.join(out_dir, "ckpt_state.npz"), allow_pickle=True)
    hist = {"total": list(d["hist_total"]), "rec": list(d["hist_rec"]),
            "kl": list(d["hist_kl"]), "beta": list(d["hist_beta"])}
    n = int(d["adam_n"])
    state = {"lr": float(d["adam_lr"]), "t": int(d["adam_t"]),
             "m": [d[f"adam_m{i}"] for i in range(n)] if n else None,
             "v": [d[f"adam_v{i}"] for i in range(n)] if n else None}
    opt.load_state_dict(state)
    return vae, hist, int(d["epoch"])


def dump_history(out_dir, hist):
    with open(os.path.join(out_dir, "loss_history.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "total", "rec", "kl", "beta"])
        for i in range(len(hist["total"])):
            w.writerow([i + 1, f"{hist['total'][i]:.6f}", f"{hist['rec'][i]:.6f}",
                        f"{hist['kl'][i]:.6f}", f"{hist['beta'][i]:.4f}"])


def plot_quick_curve(out_dir, hist):
    """Curva rápida (se actualiza en cada checkpoint, para espiar el progreso)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from shared.plotting import save_fig

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, key, title in zip(axes, ["total", "rec", "kl"],
                              ["Loss total", "Reconstrucción", "KL"]):
        ax.plot(np.arange(1, len(hist[key]) + 1), hist[key], lw=0.9)
        ax.set_title(title)
        ax.set_xlabel("Época")
        ax.grid(alpha=0.3)
        if key != "kl":
            ax.set_yscale("log")
    fig.suptitle(f"VAE largo — {len(hist['total'])} épocas", y=1.02)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "loss_curve.png"))


# ----------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description="Corrida larga del VAE con checkpoints.")
    p.add_argument("--data", default=os.path.join(REPO_ROOT, "ej2", "data", "emojis.npz"))
    p.add_argument("--tag", default="16px", help="sufijo del dir de resultados")
    p.add_argument("--latent", type=int, default=2)
    p.add_argument("--encoder", type=int, nargs="+", default=[256, 64])
    p.add_argument("--decoder", type=int, nargs="+", default=[64, 256])
    p.add_argument("--epochs", type=int, default=50000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--kl-warmup", type=int, default=0,
                   help="épocas de warmup lineal del beta (0 = sin warmup)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--resume", action="store_true",
                   help="continúa desde el último checkpoint del mismo tag")
    args = p.parse_args()

    out_dir = os.path.join(REPO_ROOT, "ej2", "results", f"long_{args.tag}")
    os.makedirs(out_dir, exist_ok=True)

    X, y, labels, image_shape = load_dataset(args.data)
    N, D = X.shape
    print(f"[setup] dataset={args.data}  X={X.shape}  img={image_shape}  "
          f"out={out_dir}", flush=True)

    opt = Adam(lr=args.lr)
    ckpt_exists = os.path.exists(os.path.join(out_dir, "ckpt_state.npz"))
    if args.resume and ckpt_exists:
        vae, hist, start_epoch = load_checkpoint(out_dir, opt)
        # rng re-sembrado con la época: no repite el stream ya consumido.
        rng = np.random.default_rng(args.seed + start_epoch)
        print(f"[resume] retomando en la época {start_epoch} "
              f"(hist={len(hist['total'])} épocas, Adam t={opt._t})", flush=True)
    else:
        if args.resume:
            print("[resume] no hay checkpoint previo: arranco de cero.", flush=True)
        vae = VAE(input_dim=D, encoder_hidden=args.encoder, latent_dim=args.latent,
                  decoder_hidden=args.decoder, hidden_activation="relu",
                  output_activation="logistic", recon_loss="bce",
                  beta=args.beta, seed=args.seed)
        hist = {"total": [], "rec": [], "kl": [], "beta": []}
        start_epoch = 0
        rng = np.random.default_rng(args.seed)

    n_params = sum(int(np.asarray(p_).size) for p_ in vae.get_flat_params())
    print(f"[setup] params={n_params:,}  latent={args.latent}  "
          f"enc={args.encoder} dec={args.decoder}  epochs objetivo={args.epochs}",
          flush=True)

    bs = N if args.batch <= 0 else int(args.batch)
    t0 = time.time()
    done = start_epoch
    try:
        # Mismo loop que VAE.fit, pero con rng persistente y checkpoints.
        for ep in range(start_epoch, args.epochs):
            if args.kl_warmup > 0:
                cur_beta = args.beta * min(1.0, (ep + 1) / args.kl_warmup)
            else:
                cur_beta = args.beta
            idx = rng.permutation(N)
            ep_t = ep_r = ep_k = 0.0
            n_b = 0
            for start in range(0, N, bs):
                Xb = X[idx[start:start + bs]]
                eps = rng.standard_normal(size=(Xb.shape[0], vae.latent_dim))
                _, cache = vae.forward(Xb, eps=eps, sample=True)
                Lt, Lr, Lk = vae.loss(cache, beta=cur_beta)
                grads = vae.backward(cache, beta=cur_beta)
                vae.set_flat_params(opt.step(vae.get_flat_params(), grads))
                ep_t += Lt; ep_r += Lr; ep_k += Lk
                n_b += 1
            hist["total"].append(ep_t / n_b)
            hist["rec"].append(ep_r / n_b)
            hist["kl"].append(ep_k / n_b)
            hist["beta"].append(cur_beta)
            done = ep + 1

            if done % args.log_every == 0 or done == args.epochs:
                rate = (done - start_epoch) / max(1e-9, time.time() - t0)
                eta_h = (args.epochs - done) / max(1e-9, rate) / 3600.0
                print(f"  ép {done:6d}/{args.epochs}  L={hist['total'][-1]:9.4f}  "
                      f"rec={hist['rec'][-1]:9.4f}  KL={hist['kl'][-1]:7.4f}  "
                      f"({rate:5.1f} ép/s, ETA {eta_h:5.2f} h)", flush=True)

            if done % args.ckpt_every == 0:
                save_checkpoint(out_dir, vae, opt, hist, done)
                dump_history(out_dir, hist)
                plot_quick_curve(out_dir, hist)
    except KeyboardInterrupt:
        print(f"\n[corte] interrumpido en la época {done}; guardo checkpoint...",
              flush=True)

    # ------------------------------------------------------------- cierre
    save_checkpoint(out_dir, vae, opt, hist, done)
    dump_history(out_dir, hist)
    plot_quick_curve(out_dir, hist)
    vae.save(os.path.join(out_dir, "vae_model.npz"))

    X_hat = vae.reconstruct(X)
    rec_mse = float(np.mean(np.sum((X - X_hat) ** 2, axis=1)))
    print(f"\n[fin] épocas={done}  L={hist['total'][-1]:.4f}  "
          f"rec={hist['rec'][-1]:.4f}  KL={hist['kl'][-1]:.4f}  "
          f"rec_MSE={rec_mse:.4f}  ({(time.time()-t0)/3600.0:.2f} h)", flush=True)

    print("[figuras] generando suite completa...", flush=True)
    from ej2.analysis import vae as analysis
    paths = analysis.make_all_figures(vae, X, y, labels, image_shape, hist, out_dir)
    for pth in paths:
        print(f"  -> {pth}", flush=True)
    print("[ok] vae_long listo.", flush=True)


if __name__ == "__main__":
    main()
