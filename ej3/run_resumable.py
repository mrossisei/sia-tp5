"""Runner PAUSABLE Y CONTINUABLE para los experimentos del EJ3 (VAE sobre MNIST).

Idea: hay una cola de "jobs" (cada job = una arquitectura del barrido de
profundidad). El runner los entrena uno por uno guardando CHECKPOINTS atómicos
cada N épocas. Podés:

    python3 ej3/run_resumable.py              # corre / reanuda TODOS los jobs
    python3 ej3/run_resumable.py --status     # muestra el progreso y sale
    python3 ej3/run_resumable.py --only d2_L16 # corre/reanuda un solo job
    python3 ej3/run_resumable.py --reset all   # borra checkpoints (¡empieza de cero!)

PAUSAR: Ctrl-C (SIGINT) o SIGTERM. El runner termina la época en curso, guarda
checkpoint y sale limpio. Volvé a correrlo y CONTINÚA EXACTO donde quedó
(reanuda modelo + optimizador Adam + estado del RNG + historial + época).
Aunque se corte la luz, perdés a lo sumo `checkpoint_every` épocas.

Qué se guarda en cada checkpoint (ej3/results/checkpoints/<job>.npz):
  - época completada, tiempo acumulado
  - parámetros del VAE (lista plana)
  - estado del optimizador Adam (m, v, t)
  - estado del generador aleatorio (para shuffles/eps reproducibles tras reanudar)
  - historial de losses (train total/rec/KL y test rec)
  - args del constructor del VAE (para reinstanciarlo solo)

NO importa matplotlib (es entrenamiento). Las figuras las hace
ej3/analysis/mnist_vae.py a partir de los .npz que deja este runner.
"""

import argparse
import json
import os
import signal
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

from shared.config_loader import load_yaml
from shared.optimizers import Adam
from ej2.models.vae import VAE, gradcheck          # reutilizamos EL MISMO VAE del EJ2
from ej3.experiments.depth_study import make_jobs, describe

EJ3_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EJ3_DIR, "results")
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
MANIFEST = os.path.join(CKPT_DIR, "manifest.json")
_BCE_EPS = 1e-12

# --------------------------------------------------------------------- señales
_STOP = {"flag": False}


def _install_signal_handlers():
    def handler(signum, frame):
        if _STOP["flag"]:
            print("\n[segunda señal] salida inmediata (sin guardar de nuevo).")
            sys.exit(130)
        _STOP["flag"] = True
        print(f"\n[señal {signum}] PAUSANDO: guardo checkpoint al terminar la "
              f"época actual y salgo. (Ctrl-C otra vez = salir ya)")
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


# ------------------------------------------------------------------ checkpoint
def ckpt_path(job_id):
    return os.path.join(CKPT_DIR, f"{job_id}.npz")


def save_checkpoint(path, *, epoch, vae, opt, rng, hist, elapsed, ctor):
    """Escritura ATÓMICA: escribe a un tmp y hace os.replace (no se corrompe)."""
    params = vae.get_flat_params()
    tmp = path + ".writing"
    with open(tmp, "wb") as f:
        np.savez(
            f,
            epoch=np.int64(epoch),
            elapsed_sec=np.float64(elapsed),
            n_params=np.int64(len(params)),
            vae_ctor=np.array(ctor, dtype=object),
            opt_state=np.array(opt.state_dict(), dtype=object),
            rng_state=np.array(rng.bit_generator.state, dtype=object),
            hist=np.array(hist, dtype=object),
            **{f"p{i}": np.asarray(p) for i, p in enumerate(params)},
        )
    os.replace(tmp, path)


def load_checkpoint(path):
    d = np.load(path, allow_pickle=True)
    n = int(d["n_params"])
    return {
        "epoch": int(d["epoch"]),
        "elapsed_sec": float(d["elapsed_sec"]),
        "vae_params": [d[f"p{i}"] for i in range(n)],
        "vae_ctor": d["vae_ctor"].item(),
        "opt_state": d["opt_state"].item(),
        "rng_state": d["rng_state"].item(),
        "hist": d["hist"].item(),
    }


def job_status(job):
    """('done'|'in_progress'|'pending', epoch_completadas)."""
    p = ckpt_path(job["id"])
    if not os.path.exists(p):
        return "pending", 0
    try:
        d = np.load(p, allow_pickle=True)
        ep = int(d["epoch"])
    except Exception:
        return "pending", 0
    if ep >= job["epochs"]:
        return "done", ep
    return "in_progress", ep


# ----------------------------------------------------------------------- datos
def load_mnist(cfg):
    path = os.path.join(REPO_ROOT, cfg["data"]["path"])
    if not os.path.exists(path):
        print(f"ERROR: no existe {path}. Corré primero:\n  python3 ej3/data/build_mnist.py")
        sys.exit(1)
    d = np.load(path)
    Xtr = d["X_train"].astype(np.float64) / 255.0
    ytr = d["y_train"]
    Xte = d["X_test"].astype(np.float64) / 255.0
    yte = d["y_test"]

    rng = np.random.default_rng(int(cfg["data"].get("data_seed", 0)))
    ntr = int(cfg["data"].get("train_size", 0))
    nte = int(cfg["data"].get("test_size", 0))
    if 0 < ntr < len(Xtr):
        idx = rng.permutation(len(Xtr))[:ntr]
        Xtr, ytr = Xtr[idx], ytr[idx]
    if 0 < nte < len(Xte):
        idx = rng.permutation(len(Xte))[:nte]
        Xte, yte = Xte[idx], yte[idx]
    return Xtr, ytr, Xte, yte


def job_to_ctor(job, input_dim):
    return {
        "input_dim": int(input_dim),
        "encoder_hidden": list(job["encoder_hidden"]),
        "latent_dim": int(job["latent_dim"]),
        "decoder_hidden": list(job["decoder_hidden"]),
        "hidden_activation": job.get("hidden_activation", "relu"),
        "output_activation": job.get("output_activation", "logistic"),
        "recon_loss": job.get("recon_loss", "bce"),
        "beta": float(job["beta"]),
        "seed": int(job["seed"]),
    }


def count_params(ctor):
    """Cuenta parámetros sin entrenar (para el reporte de tamaño)."""
    vae = VAE(**ctor)
    return sum(int(np.asarray(p).size) for p in vae.get_flat_params())


def eval_test_rec(vae, X_test, batch=2000):
    """BCE de reconstrucción en test (z=mu, determinístico). Sum sobre features,
    promedio sobre muestras — misma métrica que el rec de entrenamiento."""
    total, n = 0.0, 0
    for s in range(0, len(X_test), batch):
        Xb = X_test[s:s + batch]
        x_hat = vae.reconstruct(Xb)
        p = np.clip(x_hat, _BCE_EPS, 1.0 - _BCE_EPS)
        rec = -np.sum(Xb * np.log(p) + (1.0 - Xb) * np.log(1.0 - p), axis=1)
        total += float(rec.sum())
        n += len(Xb)
    return total / max(1, n)


# --------------------------------------------------------------- entrenamiento
def run_job(job, Xtr, Xte, cfg):
    """Entrena (o reanuda) un job hasta job['epochs'] o hasta que pidan pausa.
    Devuelve 'done' o 'paused'."""
    rcfg = cfg["runner"]
    bs = int(cfg["vae"]["batch_size"])
    lr = float(cfg["vae"]["learning_rate"])
    beta = float(job["beta"])
    epochs = int(job["epochs"])
    ck_every = int(rcfg.get("checkpoint_every", 5))
    eval_every = int(rcfg.get("eval_every", 5))
    log_every = int(rcfg.get("log_every", 1))
    path = ckpt_path(job["id"])
    N = Xtr.shape[0]

    ctor = job_to_ctor(job, Xtr.shape[1])

    # ---- reanudar o empezar de cero ----
    if os.path.exists(path):
        st = load_checkpoint(path)
        ctor = st["vae_ctor"]
        vae = VAE(**ctor)
        vae.set_flat_params(st["vae_params"])
        opt = Adam(lr=lr)
        opt.load_state_dict(st["opt_state"])
        rng = np.random.default_rng()
        rng.bit_generator.state = st["rng_state"]
        start_epoch = st["epoch"]
        hist = st["hist"]
        elapsed = st["elapsed_sec"]
        print(f"  ↻ REANUDANDO desde época {start_epoch}/{epochs}")
    else:
        vae = VAE(**ctor)
        opt = Adam(lr=lr)
        rng = np.random.default_rng(int(job["seed"]))
        start_epoch = 0
        hist = {"total": [], "rec": [], "kl": [], "beta": [], "test_rec": []}
        elapsed = 0.0
        nparams = sum(int(np.asarray(p).size) for p in vae.get_flat_params())
        print(f"  ▷ EMPEZANDO de cero ({nparams:,} parámetros)")

    if start_epoch >= epochs:
        finalize(job, vae, hist, elapsed)
        return "done"

    # ---- loop de épocas ----
    for ep in range(start_epoch, epochs):
        t0 = time.time()
        idx = rng.permutation(N)
        et = er = ek = 0.0
        nb = 0
        for s in range(0, N, bs):
            bi = idx[s:s + bs]
            Xb = Xtr[bi]
            eps = rng.standard_normal(size=(Xb.shape[0], vae.latent_dim))
            x_hat, cache = vae.forward(Xb, eps=eps, sample=True)
            Lt, Lr, Lk = vae.loss(cache, beta=beta)
            grads = vae.backward(cache, beta=beta)
            new_params = opt.step(vae.get_flat_params(), grads)
            vae.set_flat_params(new_params)
            et += Lt; er += Lr; ek += Lk; nb += 1
        dt = time.time() - t0
        elapsed += dt

        hist["total"].append(et / nb)
        hist["rec"].append(er / nb)
        hist["kl"].append(ek / nb)
        hist["beta"].append(beta)
        if (ep + 1) % eval_every == 0 or ep == epochs - 1:
            tr = eval_test_rec(vae, Xte)
        else:
            tr = hist["test_rec"][-1] if hist["test_rec"] else float("nan")
        hist["test_rec"].append(tr)

        if (ep + 1) % log_every == 0 or ep == epochs - 1:
            done_ep = ep + 1
            eta = dt * (epochs - done_ep)
            print(f"    época {done_ep:4d}/{epochs}  L={hist['total'][-1]:8.3f}  "
                  f"rec={hist['rec'][-1]:8.3f}  KL={hist['kl'][-1]:7.3f}  "
                  f"test_rec={tr:8.3f}  ({dt:4.1f}s/ép, ETA {eta/60:5.1f} min)",
                  flush=True)

        must_ckpt = (ep + 1) % ck_every == 0 or ep == epochs - 1 or _STOP["flag"]
        if must_ckpt:
            save_checkpoint(path, epoch=ep + 1, vae=vae, opt=opt, rng=rng,
                            hist=hist, elapsed=elapsed, ctor=ctor)

        if _STOP["flag"]:
            print(f"  ⏸ PAUSADO en época {ep + 1}/{epochs} (checkpoint guardado).")
            return "paused"

    finalize(job, vae, hist, elapsed)
    return "done"


def finalize(job, vae, hist, elapsed):
    """Guarda el modelo final y el historial crudo (para re-graficar sin re-entrenar)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_path = os.path.join(RESULTS_DIR, f"{job['id']}_model.npz")
    vae.save(model_path)
    hist_path = os.path.join(RESULTS_DIR, f"{job['id']}_hist.npz")
    np.savez(
        hist_path,
        total=np.array(hist["total"]), rec=np.array(hist["rec"]),
        kl=np.array(hist["kl"]), beta=np.array(hist["beta"]),
        test_rec=np.array(hist["test_rec"]),
        elapsed_sec=np.float64(elapsed),
        n_hidden=np.int64(job["n_hidden"]),
        encoder_hidden=np.array(job["encoder_hidden"]),
        latent_dim=np.int64(job["latent_dim"]),
    )
    print(f"  ✓ TERMINADO '{job['id']}'  (entrenó {elapsed/60:.1f} min)  "
          f"-> {os.path.basename(model_path)}, {os.path.basename(hist_path)}")


# ------------------------------------------------------------------- manifest
def write_manifest(jobs):
    os.makedirs(CKPT_DIR, exist_ok=True)
    data = {"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "jobs": []}
    for j in jobs:
        status, ep = job_status(j)
        data["jobs"].append({
            "id": j["id"], "n_hidden": j["n_hidden"],
            "encoder_hidden": j["encoder_hidden"],
            "epochs": j["epochs"], "completed_epochs": ep, "status": status,
        })
    tmp = MANIFEST + ".writing"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, MANIFEST)


def print_status(jobs, Xtr_shape=None):
    print("=" * 78)
    print("ESTADO DE LOS EXPERIMENTOS (EJ3 — VAE sobre MNIST)")
    print("=" * 78)
    for j in jobs:
        status, ep = job_status(j)
        bar = {"done": "✓ done   ", "in_progress": "… parcial", "pending": "· pend.  "}[status]
        extra = ""
        hp = os.path.join(RESULTS_DIR, f"{j['id']}_hist.npz")
        if os.path.exists(hp):
            h = np.load(hp)
            extra = f"  test_rec_final={float(h['test_rec'][-1]):.2f}  ({float(h['elapsed_sec'])/60:.1f} min)"
        print(f"  {bar}  {j['id']:<10} {j['n_hidden']} ocultas  "
              f"{ep:4d}/{j['epochs']} ép{extra}")
    print("=" * 78)


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Runner pausable/continuable del EJ3")
    ap.add_argument("--status", action="store_true", help="muestra progreso y sale")
    ap.add_argument("--only", type=str, default=None, help="corre un solo job por id")
    ap.add_argument("--reset", type=str, default=None,
                    help="borra checkpoint(s): un id, o 'all' (¡empieza de cero!)")
    ap.add_argument("--no-gradcheck", action="store_true", help="saltea el gradient-check inicial")
    ap.add_argument("--config", type=str, default=os.path.join(EJ3_DIR, "config.yaml"),
                    help="ruta al config.yaml (por defecto ej3/config.yaml)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    jobs = make_jobs(cfg)
    os.makedirs(CKPT_DIR, exist_ok=True)

    if args.reset:
        targets = jobs if args.reset == "all" else [j for j in jobs if j["id"] == args.reset]
        if not targets:
            print(f"no hay job con id '{args.reset}'")
            return
        for j in targets:
            p = ckpt_path(j["id"])
            if os.path.exists(p):
                os.remove(p)
                print(f"borrado checkpoint de {j['id']}")
        write_manifest(jobs)
        return

    if args.status:
        print_status(jobs)
        return

    if args.only:
        jobs_run = [j for j in jobs if j["id"] == args.only]
        if not jobs_run:
            print(f"no hay job con id '{args.only}'. Disponibles: {[j['id'] for j in jobs]}")
            return
    else:
        jobs_run = jobs

    _install_signal_handlers()

    # Banner + plan
    print("=" * 78)
    print("EJ3 — VAE sobre MNIST: barrido de PROFUNDIDAD (runner pausable)")
    print("=" * 78)
    print(describe(jobs))
    print("-" * 78)
    print("Pausar: Ctrl-C (guarda checkpoint y sale). Reanudar: volvé a correr esto.")
    print("=" * 78)

    # Gradient-check (convención del repo: falla ruidosamente si el VAE está mal)
    if not args.no_gradcheck:
        print("\nGRADIENT-CHECK del VAE (analítico vs diferencias finitas)...")
        rel = gradcheck(seed=0, verbose=True)
        print(f"gradient-check PASÓ (err rel máx = {rel:.2e})\n")

    Xtr, ytr, Xte, yte = load_mnist(cfg)
    print(f"MNIST: train={Xtr.shape}  test={Xte.shape}  "
          f"(subset de config: train_size={cfg['data']['train_size']}, "
          f"test_size={cfg['data']['test_size']})\n")

    write_manifest(jobs)
    for j in jobs_run:
        status, ep = job_status(j)
        if status == "done":
            print(f"[{j['id']}] ya está terminado ({ep}/{j['epochs']}), salteando.")
            continue
        print(f"\n[{j['id']}]  {j['n_hidden']} capas ocultas  "
              f"enc={j['encoder_hidden']} -> z{j['latent_dim']} -> dec={j['decoder_hidden']}")
        result = run_job(j, Xtr, Xte, cfg)
        write_manifest(jobs)
        if result == "paused":
            print("\n⏸  Runner pausado. Volvé a correr `python3 ej3/run_resumable.py` "
                  "para continuar donde quedó.")
            return

    print("\n" + "=" * 78)
    print("TODOS LOS JOBS TERMINADOS.")
    print_status(jobs)
    print("Generá las figuras con:  python3 ej3/analysis/mnist_vae.py")


if __name__ == "__main__":
    main()
