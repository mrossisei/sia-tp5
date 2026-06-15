"""Smoke test del VAE (EJ2): automatiza el gradient-check OBLIGATORIO.

AGENTS.md (§9.2 y §15) marca el gradient-check numérico del VAE (backprop
analítico vs diferencias finitas centrales) como sanity check obligatorio antes
de confiar en el entrenamiento. Hasta ahora sólo corría invocando el módulo a
mano (`python3 ej2/models/vae.py`) o como gate del runner del EJ3; este test lo
mete en la suite para que falle ruidosamente si alguien rompe el backward.

Corre el gradcheck sobre varias seeds. Debe terminar con `SMOKE VAE OK`.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from ej2.models.vae import gradcheck

TOL = 1e-4


def main():
    worst = 0.0
    for seed in (0, 1, 2):
        rel = gradcheck(seed=seed, tol=TOL, verbose=False)
        print(f"[gradcheck VAE] seed={seed}: err rel máx = {rel:.2e}")
        assert rel < TOL, f"gradient-check FALLÓ (seed={seed}): {rel:.2e} >= {TOL}"
        worst = max(worst, rel)
    print(f"\nSMOKE VAE OK (peor err rel = {worst:.2e} < tol {TOL})")


if __name__ == "__main__":
    main()
