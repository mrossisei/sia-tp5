# Presentación TP5 (Beamer)

Presentación en LaTeX (Beamer, tema **Metropolis**, en español), consistente con
TP3/TP4.

## Requisitos

- Una distribución LaTeX (TeX Live / MiKTeX) con `pdflatex`.
- El tema **Metropolis** (`beamertheme-metropolis`). En TeX Live se instala con:
  ```bash
  tlmgr install beamertheme-metropolis
  ```
  En Debian/Ubuntu suele venir en `texlive-latex-extra`.
- Las **figuras ya generadas** en `ej1/results/` y `ej2/results/`. La presentación
  las lee mediante `\graphicspath`:
  ```
  \graphicspath{{../ej1/results/basic/}{../ej1/results/denoising/}{../ej2/results/}}
  ```
  Si faltara alguna figura, regenerala corriendo los entrypoints (ver el README
  raíz del repositorio).

## Compilar

Desde este directorio (`presentacion/`), compilar **dos veces** para resolver el
índice y las referencias:

```bash
pdflatex main.tex
pdflatex main.tex
```

Esto produce `main.pdf`.

> Nota: Metropolis recomienda compilar con `xelatex` o `lualatex` para mejores
> fuentes, pero `pdflatex` funciona. Si usás `xelatex`:
> ```bash
> xelatex main.tex
> xelatex main.tex
> ```

## Figuras incluidas

Todas las rutas de `\includegraphics` apuntan a archivos que existen en
`ej1/results/{basic,denoising}/` y `ej2/results/`:

- **EJ1.a** (`basic/`): `learning_curve`, `reconstruction_grid`, `latent_scatter`,
  `latent_scatter_pca`, `new_letter_generation`, `exp_architecture`,
  `exp_optimizer`, `exp_learning_rate`, `exp_activation`, `exp_loss`.
- **EJ1.b** (`denoising/`): `noise_examples`, `noise_sweep`.
- **EJ2** (`ej2/results/`): `dataset_sample`, `loss_curves`, `reconstructions`,
  `latent_scatter`, `manifold_grid`, `samples`, `interpolation`,
  `exp_latent_dim`, `exp_beta`, `exp_kl_warmup`.
