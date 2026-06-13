"""Verifica que el runner del EJ3 REANUDA EXACTO desde un checkpoint.

Propiedad clave: entrenar 4 épocas de corrido debe dar EXACTAMENTE el mismo
modelo que entrenar 2, "pausar" (checkpoint) y reanudar 2 más. Si el checkpoint
no guardara TODO el estado (pesos + Adam + estado del RNG), las corridas
divergirían. Es la prueba de que la pausa/continuación no altera el resultado.

Uso:  python3 tests/smoke_ej3_resume.py
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

import ej3.run_resumable as R


def _tiny_cfg():
    return {
        "vae": {"batch_size": 32, "learning_rate": 1e-3},
        "runner": {"checkpoint_every": 2, "eval_every": 1, "log_every": 99},
    }


def _tiny_job(job_id, epochs):
    return {
        "id": job_id, "name": job_id, "n_hidden": 1,
        "encoder_hidden": [32], "decoder_hidden": [32],
        "latent_dim": 4, "beta": 1.0, "epochs": epochs, "seed": 7,
        "hidden_activation": "relu", "output_activation": "logistic",
        "recon_loss": "bce",
    }


def _final_params(job_id):
    d = R.load_checkpoint(R.ckpt_path(job_id))
    return d["vae_params"], d["hist"], d["epoch"]


def _cleanup(*job_ids):
    for jid in job_ids:
        for p in (R.ckpt_path(jid),
                  os.path.join(R.RESULTS_DIR, f"{jid}_model.npz"),
                  os.path.join(R.RESULTS_DIR, f"{jid}_hist.npz")):
            if os.path.exists(p):
                os.remove(p)


def main():
    cfg = _tiny_cfg()
    rng = np.random.default_rng(0)
    Xtr = (rng.random((160, 784)) > 0.5).astype(np.float64)
    Xte = (rng.random((40, 784)) > 0.5).astype(np.float64)

    ID_A, ID_B = "_smoke_straight", "_smoke_resumed"
    _cleanup(ID_A, ID_B)
    R._STOP["flag"] = False

    # A) 4 épocas de corrido
    R.run_job(_tiny_job(ID_A, 4), Xtr, Xte, cfg)
    pA, hA, eA = _final_params(ID_A)

    # B) 2 épocas -> (checkpoint) -> reanudar 2 más
    R.run_job(_tiny_job(ID_B, 2), Xtr, Xte, cfg)      # corta en época 2, deja checkpoint
    _, _, eMid = _final_params(ID_B)
    assert eMid == 2, f"checkpoint intermedio mal: época {eMid}"
    R.run_job(_tiny_job(ID_B, 4), Xtr, Xte, cfg)      # reanuda 2 -> 4
    pB, hB, eB = _final_params(ID_B)

    assert eA == eB == 4, f"épocas finales {eA} vs {eB}"
    assert len(pA) == len(pB), "distinta cantidad de parámetros"
    max_diff = max(float(np.max(np.abs(a - b))) for a, b in zip(pA, pB))
    hist_diff = max(abs(x - y) for x, y in zip(hA["total"], hB["total"]))

    print(f"épocas: corrido={eA}  reanudado={eB}")
    print(f"diferencia MÁXIMA en pesos (corrido vs reanudado): {max_diff:.3e}")
    print(f"diferencia MÁXIMA en loss por época:               {hist_diff:.3e}")

    assert max_diff == 0.0, f"los pesos DIFIEREN tras reanudar (max {max_diff:.3e})"
    assert hist_diff == 0.0, f"el historial DIFIERE tras reanudar (max {hist_diff:.3e})"
    print("OK: reanudar (por límite de épocas) da resultado BIT-IDÉNTICO. ✓")

    # C) PAUSA POR SEÑAL: el flag _STOP debe cortar tras la época en curso,
    #    dejar checkpoint, y reanudar hasta el final igual que el corrido.
    ID_C = "_smoke_signal"
    _cleanup(ID_C)
    R._STOP["flag"] = True                                   # simula Ctrl-C ya pedido
    res = R.run_job(_tiny_job(ID_C, 4), Xtr, Xte, cfg)       # hace 1 época y pausa
    _, _, eC1 = _final_params(ID_C)
    assert res == "paused" and eC1 == 1, f"pausa por señal mal: res={res} ép={eC1}"
    R._STOP["flag"] = False                                  # "vuelvo a correr el script"
    res2 = R.run_job(_tiny_job(ID_C, 4), Xtr, Xte, cfg)      # reanuda 1 -> 4
    pC, hC, eC = _final_params(ID_C)
    assert res2 == "done" and eC == 4
    sig_diff = max(float(np.max(np.abs(a - b))) for a, b in zip(pA, pC))
    print(f"pausa por señal en época {eC1}, reanudado a {eC}; "
          f"diferencia máx vs corrido: {sig_diff:.3e}")
    _cleanup(ID_A, ID_B, ID_C)
    assert sig_diff == 0.0, f"pausa por señal alteró los pesos (max {sig_diff:.3e})"
    print("OK: pausar por señal y reanudar también da BIT-IDÉNTICO. ✓")


if __name__ == "__main__":
    main()
