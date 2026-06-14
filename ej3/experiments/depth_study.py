"""Definición del experimento de PROFUNDIDAD del EJ3.

Construye la lista de "jobs" a partir del config. Cada job = (una arquitectura,
una semilla). El runner (ej3/run_resumable.py) los procesa uno por uno con
checkpointing.

ORDEN de los jobs: semilla EXTERNA, arquitectura interna. Es decir, primero
corre las 4 profundidades con la 1ª semilla (barrido completo de profundidad),
y recién después repite con la 2ª (la réplica). Así, si se corta temprano, ya
queda el experimento entero con una semilla.

NO importa matplotlib (es definición de experimento, no análisis).
"""


def make_jobs(cfg):
    """Devuelve la lista de jobs del barrido de profundidad (arquitectura × semilla).

    Todos comparten latente, beta, épocas, lr y datos: lo ÚNICO que cambia entre
    arquitecturas es la cantidad/ancho de capas ocultas (para aislar el efecto
    de la profundidad), y entre semillas, la inicialización + el orden de datos.
    """
    vcfg = cfg["vae"]
    dcfg = cfg["experiments"]["depth"]
    archs = dcfg["architectures"]
    seeds = [int(s) for s in dcfg.get("seeds", [vcfg.get("seed", 42)])]
    latent = int(vcfg["latent_dim"])

    jobs = []
    for sd in seeds:                       # semilla externa: barrido completo por semilla
        for a in archs:
            enc = [int(n) for n in a["encoder_hidden"]]
            dec = [int(n) for n in a["decoder_hidden"]]
            jobs.append({
                "id": f"{a['name']}_L{latent}_s{sd}",
                "name": a["name"],
                "encoder_hidden": enc,
                "decoder_hidden": dec,
                "n_hidden": len(enc),          # capas ocultas del encoder
                "latent_dim": latent,
                "beta": float(vcfg["beta"]),
                "epochs": int(vcfg["epochs"]),
                "seed": int(sd),
                "hidden_activation": "relu",
                "output_activation": "logistic",
                "recon_loss": "bce",
            })
    return jobs


def describe(jobs):
    """Texto legible del plan (agrupado por arquitectura) para el banner."""
    by_arch = {}
    for j in jobs:
        by_arch.setdefault(j["name"], {"job": j, "seeds": []})
        by_arch[j["name"]]["seeds"].append(j["seed"])
    lines = []
    for name, info in by_arch.items():
        j = info["job"]
        lines.append(
            f"  {name:<4} {j['n_hidden']} capas ocultas  "
            f"enc={j['encoder_hidden']} -> z{j['latent_dim']} -> dec={j['decoder_hidden']}"
            f"   semillas={info['seeds']}"
        )
    return "\n".join(lines)
