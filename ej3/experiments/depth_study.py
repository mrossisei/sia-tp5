"""Definición del experimento de PROFUNDIDAD del EJ3.

Construye la lista de "jobs" (una arquitectura por job) a partir del config.
Cada job es un dict autocontenido con todo lo que el runner necesita para
instanciar el VAE y entrenarlo. El runner (ej3/run_resumable.py) los procesa
uno por uno con checkpointing.

NO importa matplotlib (es definición de experimento, no análisis).
"""


def make_jobs(cfg):
    """Devuelve la lista de jobs del barrido de profundidad.

    Todos comparten latente, beta, épocas, lr, seed y datos: lo ÚNICO que
    cambia es la cantidad/ancho de capas ocultas. Así el experimento aísla el
    efecto de la profundidad sobre reconstrucción/generación.
    """
    vcfg = cfg["vae"]
    archs = cfg["experiments"]["depth"]["architectures"]

    jobs = []
    for a in archs:
        enc = [int(n) for n in a["encoder_hidden"]]
        dec = [int(n) for n in a["decoder_hidden"]]
        job = {
            "id": f"{a['name']}_L{int(vcfg['latent_dim'])}",
            "name": a["name"],
            "encoder_hidden": enc,
            "decoder_hidden": dec,
            "n_hidden": len(enc),                 # cuántas capas ocultas tiene el encoder
            "latent_dim": int(vcfg["latent_dim"]),
            "beta": float(vcfg["beta"]),
            "epochs": int(vcfg["epochs"]),
            "seed": int(vcfg["seed"]),
            "hidden_activation": "relu",
            "output_activation": "logistic",
            "recon_loss": "bce",
        }
        jobs.append(job)
    return jobs


def describe(jobs):
    """Texto legible del plan de experimentos (para banner del runner)."""
    lines = []
    for j in jobs:
        lines.append(
            f"  {j['id']:<10}  {j['n_hidden']} capas ocultas  "
            f"enc={j['encoder_hidden']} -> z{j['latent_dim']} -> dec={j['decoder_hidden']}"
        )
    return "\n".join(lines)
