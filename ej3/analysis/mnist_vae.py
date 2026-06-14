"""Figuras del EJ3 (VAE sobre MNIST) — análisis del barrido de PROFUNDIDAD.

Lee los .npz que deja ej3/run_resumable.py (modelos + historiales) y produce
las figuras en ej3/results/. NO re-entrena: re-grafica a partir de datos crudos
(regla del repo). matplotlib se usa SÓLO acá.

Agrupa las REALIZACIONES (semillas) de cada arquitectura: promedia sobre
semillas para las curvas/resumen y usa la MEJOR réplica para las imágenes.

Uso:
    python3 ej3/analysis/mnist_vae.py [config.yaml]
"""

import os
import sys
import glob

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.config_loader import load_yaml
from ej2.models.vae import VAE
from ej3.experiments.depth_study import make_jobs
import ej3.run_resumable as R

EJ3_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(EJ3_DIR, "results")
IMG = (28, 28)


def _paths(job_id):
    return (os.path.join(RESULTS_DIR, f"{job_id}_model.npz"),
            os.path.join(RESULTS_DIR, f"{job_id}_hist.npz"))


def _available(jobs):
    out = []
    for j in jobs:
        mp, hp = _paths(j["id"])
        if os.path.exists(mp) and os.path.exists(hp):
            out.append(j)
    return out


def _group_by_arch(jobs):
    """name -> {'n_hidden', 'jobs': [...]} ordenado por profundidad."""
    groups = {}
    for j in jobs:
        g = groups.setdefault(j["name"], {"n_hidden": j["n_hidden"], "jobs": []})
        g["jobs"].append(j)
    return dict(sorted(groups.items(), key=lambda kv: kv[1]["n_hidden"]))


def _hist(job):
    return np.load(_paths(job["id"])[1])


def _best_job(group):
    """La réplica (semilla) con menor test_rec final = mejor reconstrucción."""
    return min(group["jobs"], key=lambda j: float(_hist(j)["test_rec"][-1]))


def _mean_curve(group, key):
    """Media por época sobre semillas (recorta a la longitud común; ignora NaN)."""
    arrs = [np.asarray(_hist(j)[key], dtype=float) for j in group["jobs"]]
    L = min(len(a) for a in arrs)
    return np.nanmean(np.stack([a[:L] for a in arrs]), axis=0)


def _grid(ax, images, n_cols, title=None):
    n = len(images)
    n_rows = int(np.ceil(n / n_cols))
    canvas = np.ones((n_rows * IMG[0], n_cols * IMG[1]))
    for i, vec in enumerate(images):
        r, c = divmod(i, n_cols)
        canvas[r*IMG[0]:(r+1)*IMG[0], c*IMG[1]:(c+1)*IMG[1]] = np.asarray(vec).reshape(IMG)
    ax.imshow(canvas, cmap="gray_r", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)


def fig_reconstructions(group, Xte, yte, n=10):
    job = _best_job(group)
    vae = VAE.load(_paths(job["id"])[0])
    picks = [np.where(yte == d)[0][0] for d in range(10) if len(np.where(yte == d)[0])]
    Xs = Xte[picks]
    Xr = vae.reconstruct(Xs)
    fig, axes = plt.subplots(2, 1, figsize=(n, 2.4))
    _grid(axes[0], Xs, n, "originales (test)")
    _grid(axes[1], Xr, n, f"reconstruidas — {group['n_hidden']} capas ocultas "
                          f"(mejor réplica: {job['id']})")
    fig.tight_layout()
    p = os.path.join(RESULTS_DIR, f"recon_{job['name']}_L{job['latent_dim']}.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    return p


def fig_samples(group, n=40, seed=0):
    job = _best_job(group)
    vae = VAE.load(_paths(job["id"])[0])
    samples, _ = vae.generate(n, seed=seed)
    fig, ax = plt.subplots(figsize=(8, 0.8 * (n / 10)))
    _grid(ax, samples, 10, f"muestras generadas z~N(0,I) — {group['n_hidden']} capas "
                           f"ocultas (mejor réplica: {job['id']})")
    fig.tight_layout()
    p = os.path.join(RESULTS_DIR, f"samples_{job['name']}_L{job['latent_dim']}.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    return p


def fig_curves(groups):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, g in groups.items():
        lab = f"{name} ({g['n_hidden']} oc.)"
        axes[0].plot(_mean_curve(g, "rec"), label=lab)
        axes[1].plot(_mean_curve(g, "test_rec"), label=lab)
    axes[0].set_title("Reconstrucción en TRAIN (BCE, media de semillas)")
    axes[1].set_title("Reconstrucción en TEST / held-out (BCE, media de semillas)")
    for ax in axes:
        ax.set_xlabel("época"); ax.set_ylabel("BCE por muestra")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("EJ3 — efecto de la PROFUNDIDAD sobre la reconstrucción", fontsize=12)
    fig.tight_layout()
    lat = next(iter(groups.values()))["jobs"][0]["latent_dim"]
    p = os.path.join(RESULTS_DIR, f"depth_curves_L{lat}.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    return p


def fig_depth_summary(groups):
    """Resultado central: métrica final vs cantidad de capas (con réplicas)."""
    nh, mean_test, mean_train, mean_min = [], [], [], []
    pts_x, pts_y = [], []
    for name, g in groups.items():
        finals_te = [float(_hist(j)["test_rec"][-1]) for j in g["jobs"]]
        finals_tr = [float(_hist(j)["rec"][-1]) for j in g["jobs"]]
        mins = [float(_hist(j)["elapsed_sec"]) / 60.0 for j in g["jobs"]]
        nh.append(g["n_hidden"])
        mean_test.append(np.mean(finals_te)); mean_train.append(np.mean(finals_tr))
        mean_min.append(np.mean(mins))
        pts_x += [g["n_hidden"]] * len(finals_te); pts_y += finals_te

    order = np.argsort(nh)
    nh = np.array(nh)[order]
    mean_test = np.array(mean_test)[order]; mean_train = np.array(mean_train)[order]
    mean_min = np.array(mean_min)[order]

    fig, ax1 = plt.subplots(figsize=(7.5, 4.8))
    ax1.plot(nh, mean_test, "o-", color="C0", label="test rec (media de semillas)")
    ax1.scatter(pts_x, pts_y, color="C0", alpha=0.4, s=25, label="cada realización")
    ax1.plot(nh, mean_train, "s--", color="C2", label="train rec (media)")
    ax1.set_xlabel("cantidad de capas ocultas del encoder")
    ax1.set_ylabel("BCE de reconstrucción (menor = mejor)")
    ax1.set_xticks(nh); ax1.grid(alpha=0.3); ax1.legend(loc="upper right", fontsize=8)
    ax2 = ax1.twinx()
    ax2.bar(nh, mean_min, width=0.25, alpha=0.2, color="C3")
    ax2.set_ylabel("tiempo de entrenamiento [min]  (barras)")
    best = nh[int(np.argmin(mean_test))]
    ax1.set_title(f"EJ3 — ¿ayuda agregar capas?  (mejor test rec con {best} capas ocultas)")
    fig.tight_layout()
    lat = next(iter(groups.values()))["jobs"][0]["latent_dim"]
    p = os.path.join(RESULTS_DIR, f"depth_summary_L{lat}.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    return p


def fig_manifold(job, n=20, span=3.0):
    """Manifold 2D clásico del VAE: decodifica una grilla de z sobre
    [-span, span]^2 y la tilea. Sólo tiene sentido con latente 2."""
    vae = VAE.load(_paths(job["id"])[0])
    grid = np.linspace(-span, span, n)
    imgs = []
    for yi in grid[::-1]:          # filas: z2 de mayor a menor
        for xi in grid:            # columnas: z1 de menor a mayor
            imgs.append(vae.decode(np.array([[xi, yi]]))[0])
    fig, ax = plt.subplots(figsize=(8, 8))
    _grid(ax, imgs, n, f"manifold latente 2D — {job['id']} (grilla z1×z2 en [-3,3])")
    p = os.path.join(RESULTS_DIR, f"manifold_{job['name']}_L{job['latent_dim']}.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    return p


def fig_latent_scatter(job, Xte, yte):
    """Scatter del espacio latente (z=mu) coloreado por dígito. Sólo latente 2."""
    vae = VAE.load(_paths(job["id"])[0])
    mu = vae.encode(Xte)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sc = ax.scatter(mu[:, 0], mu[:, 1], c=yte, cmap="tab10", s=6, alpha=0.5)
    ax.set_xlabel("z1"); ax.set_ylabel("z2")
    ax.set_title(f"espacio latente 2D (test) — {job['id']}")
    fig.colorbar(sc, ax=ax, ticks=range(10), label="dígito")
    p = os.path.join(RESULTS_DIR, f"scatter_{job['name']}_L{job['latent_dim']}.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    return p


def fig_interpolation(job, Xte, steps=10, pairs=6, seed=0):
    """Interpola linealmente en el latente entre pares de dígitos y decodifica
    (muestra que el espacio es continuo/sin huecos)."""
    vae = VAE.load(_paths(job["id"])[0])
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(pairs):
        i, j = int(rng.integers(len(Xte))), int(rng.integers(len(Xte)))
        za, zb = vae.encode(Xte[i:i+1])[0], vae.encode(Xte[j:j+1])[0]
        for t in np.linspace(0.0, 1.0, steps):
            rows.append(vae.decode(((1.0 - t) * za + t * zb)[None])[0])
    fig, ax = plt.subplots(figsize=(steps, pairs))
    _grid(ax, rows, steps, f"interpolación en el latente — {job['id']}")
    p = os.path.join(RESULTS_DIR, f"interp_{job['name']}_L{job['latent_dim']}.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    return p


def fig_latent_summary(arch="d2"):
    """Figura clave del experimento de LATENTE: test_rec final vs dimensión
    latente (escanea {arch}_L*_hist.npz). Independiente de la config."""
    pts = []
    for f in sorted(glob.glob(os.path.join(RESULTS_DIR, f"{arch}_L*_hist.npz"))):
        h = np.load(f)
        pts.append((int(h["latent_dim"]), float(h["test_rec"][-1]), float(h["rec"][-1])))
    if len(pts) < 2:
        return None
    pts.sort()
    lats = [p[0] for p in pts]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(lats, [p[1] for p in pts], "o-", label="test rec")
    ax.plot(lats, [p[2] for p in pts], "s--", color="C2", label="train rec")
    ax.set_xscale("log", base=2); ax.set_xticks(lats)
    ax.set_xticklabels([str(l) for l in lats])
    ax.set_xlabel("dimensión latente"); ax.set_ylabel("BCE de reconstrucción (menor = mejor)")
    ax.set_title(f"EJ3 — efecto del LATENTE en la reconstrucción ({arch})")
    ax.grid(alpha=0.3); ax.legend()
    p = os.path.join(RESULTS_DIR, f"latent_summary_{arch}.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    return p


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(EJ3_DIR, "config.yaml")
    cfg = load_yaml(cfg_path)
    jobs = _available(make_jobs(cfg))
    if not jobs:
        print("No hay jobs terminados todavía. Corré primero: python3 ej3/run_resumable.py")
        return
    groups = _group_by_arch(jobs)
    print(f"Arquitecturas con resultados: "
          f"{[(n, len(g['jobs'])) for n, g in groups.items()]} (nombre, #realizaciones)")

    _, _, Xte, yte = R.load_mnist(cfg)

    paths = []
    for name, g in groups.items():
        paths.append(fig_reconstructions(g, Xte, yte))
        paths.append(fig_samples(g))
    paths.append(fig_curves(groups))
    if len(groups) >= 2:
        paths.append(fig_depth_summary(groups))

    # visualizaciones de presentación (lucen el "que se vea bueno")
    lat = next(iter(groups.values()))["jobs"][0]["latent_dim"]
    bestjob = min((j for g in groups.values() for j in g["jobs"]),
                  key=lambda j: float(_hist(j)["test_rec"][-1]))
    paths.append(fig_interpolation(bestjob, Xte))
    if lat == 2:
        bj = _best_job(next(iter(groups.values())))
        paths.append(fig_manifold(bj))
        paths.append(fig_latent_scatter(bj, Xte, yte))
    lp = fig_latent_summary("d2")
    if lp:
        paths.append(lp)

    print("\nfiguras generadas:")
    for p in paths:
        print(f"  -> {p}")


if __name__ == "__main__":
    main()
