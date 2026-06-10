#!/usr/bin/env bash
# =============================================================================
# TP5 — Experimentos pesados (dejar corriendo de noche).
#
# Lanzar con:
#     nohup ./run_overnight.sh > overnight.log 2>&1 &
# Seguir el progreso con:
#     tail -f overnight.log
#
# Etapas (secuenciales; cada una deja resultados aunque se corte la siguiente):
#   1. EJ1: grilla completa optimizador x lr (3x5) x 5 seeds x 30000 épocas
#      -> ej1/results/basic/exp_grid_full_heatmap.png (+ CSV)        [~30-60 min]
#   2. EJ2: VAE largo 16x16, latent=2, 50000 épocas, checkpoint c/1000
#      -> ej2/results/long_16px/                                     [~2-4 h]
#   3. EJ2: dataset 24x24 (120 variantes/clase) + VAE grande
#      [576,512,128]->2->[128,512,576], 30000 épocas, checkpoint c/250
#      -> ej2/data/emojis_24.npz + ej2/results/long_24px/            [~4-6 h]
#
# Total estimado: ~6-7 h (medido a ~11 ép/s la etapa 2 y ~1.8 ép/s la etapa 3).
# Los VAE largos guardan checkpoints atómicos: si a la mañana la etapa 3 sigue
# corriendo, se puede cortar (Ctrl+C o kill) y el último checkpoint + figuras
# parciales quedan usables; o continuarla luego con:
#     python3 ej2/experiments/vae_long.py --data ej2/data/emojis_24.npz \
#         --tag 24px --encoder 512 128 --decoder 128 512 \
#         --epochs 30000 --ckpt-every 250 --resume
# =============================================================================
set -u
cd "$(dirname "$0")"

echo "============================================================"
echo "[overnight] inicio: $(date)"
echo "============================================================"

echo; echo "[etapa 1/3] EJ1 grilla completa (30000 ep x 5 seeds)"; echo
python3 ej1/experiments/grid_full.py
echo "[etapa 1/3] terminada: $(date)"

echo; echo "[etapa 2/3] VAE largo 16x16 (50000 épocas)"; echo
python3 ej2/experiments/vae_long.py --epochs 50000 --ckpt-every 1000
echo "[etapa 2/3] terminada: $(date)"

echo; echo "[etapa 3/3] dataset 24x24 + VAE grande (12000 épocas)"; echo
if [ ! -f ej2/data/emojis_24.npz ]; then
    python3 ej2/data/build_emojis.py --size 24 --per-class 120 \
        --out ej2/data/emojis_24.npz
fi
python3 ej2/experiments/vae_long.py --data ej2/data/emojis_24.npz --tag 24px \
    --encoder 512 128 --decoder 128 512 --epochs 30000 --ckpt-every 250

echo
echo "============================================================"
echo "[overnight] TODO terminado: $(date)"
echo "============================================================"
