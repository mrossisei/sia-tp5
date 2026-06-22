# Presentacion EJ2 (Beamer)

Presentacion standalone del **EJ2 - Variational Autoencoder sobre emojis**.

## Requisitos

- Una distribucion LaTeX con `pdflatex`.
- El tema **Metropolis** (`beamertheme-metropolis`).
- Las figuras ya generadas en:
  - `ej2/results/`
  - `extra/results/`

La presentacion las lee con:

```tex
\graphicspath{{../ej2/results/}{../extra/results/}}
```

Si faltara alguna figura principal de EJ2, regenerarla con:

```bash
python3 ej2/main_vae.py
python3 ej2/experiments/prior_extrapolation.py
```

Si faltaran las figuras extra usadas en la version extendida:

```bash
python3 extra/recon_loss.py
python3 extra/generalization.py
python3 extra/active_units.py
python3 extra/vae_letters.py
```

## Compilar

Desde este directorio:

```bash
pdflatex main.tex
pdflatex main.tex
```

Esto produce `main.pdf`.

## Figuras usadas

De `ej2/results/`:

- `dataset_sample.png`
- `exp_beta.png`
- `samples.png`
- `exp_prior_extrapolation.png`
- `exp_latent_dim.png`
- `reconstructions.png`
- `latent_scatter.png`
- `manifold_grid.png`
- `interpolation.png`
- `comparison_600_vs_50k.png`

De `extra/results/`:

- `letters_ae_vs_vae_scatter.png`
- `recon_loss_mse_vs_bce.png`
- `generalization_curves.png`
- `active_units_vs_latent.png`
