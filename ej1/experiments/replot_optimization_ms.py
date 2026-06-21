"""Re-grafica las figuras del estudio multi-seed DESDE exp_optimization_ms.npz,
sin reentrenar. Reconstruye la estructura `results[label] -> [{max, conv}, ...]`
que esperan axis_figure/summary_figure y vuelve a guardar los PNG.

Uso: python -m ej1.experiments.replot_optimization_ms
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import numpy as np

from ej1.experiments.optimization_multiseed import OUT, axis_figure, summary_figure

# titles/fig_names viven dentro de main(); los redefinimos aca (mismos valores).
FIG_NAMES = {"loss": "exp_loss_ms.png", "activacion": "exp_activation_ms.png",
             "arquitectura": "exp_architecture_ms.png"}
TITLES = {"loss": "Funcion de perdida: BCE vs MSE",
          "activacion": "Activacion oculta: tanh / relu / logistica",
          "arquitectura": "Arquitectura: profundidad / ancho"}


def main():
    npz_path = os.path.join(OUT, "exp_optimization_ms.npz")
    d = np.load(npz_path, allow_pickle=True)
    eje = d["eje"]; config = d["config"]
    max_pixel = d["max_pixel"]; conv = d["conv_epoch"]

    summary = []
    for axis in ("loss", "activacion", "arquitectura"):
        mask_axis = eje == axis
        # preservar orden de aparicion de las configs
        labels = list(dict.fromkeys(config[mask_axis].tolist()))
        results = {}
        for lab in labels:
            m = mask_axis & (config == lab)
            results[lab] = [{"max": int(mx), "conv": float(cv)}
                            for mx, cv in zip(max_pixel[m], conv[m])]
            vals = np.array([r["max"] for r in results[lab]], dtype=float)
            summary.append((f"{axis}: {lab.splitlines()[0]}",
                            float(vals.mean()), float(vals.std())))
        axis_figure(TITLES[axis], labels, results, os.path.join(OUT, FIG_NAMES[axis]))
        print(f"[ok] {FIG_NAMES[axis]}")

    summary_figure(summary, os.path.join(OUT, "exp_summary_ms.png"))
    print("[ok] exp_summary_ms.png")
    print(f"[done] figuras regeneradas desde {npz_path}")


if __name__ == "__main__":
    main()
