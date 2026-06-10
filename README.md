# TP5 — Deep Learning: Autoencoders y Variational Autoencoders

Trabajo Práctico 5 de Sistemas de Inteligencia Artificial (ITBA). Implementación
de **Autoencoders (AE)**, **Denoising Autoencoders (DAE)** y **Variational
Autoencoders (VAE)** en **numpy puro**, sin frameworks de Deep Learning.

Se reutiliza la maquinaria de MLP con backpropagation de TP3 (forward/backward,
optimizadores, activaciones, inicializadores) y se extiende con lo propio del
aprendizaje profundo: BCE numéricamente estable, inicialización según la
activación (Glorot/He) y la derivación completa del VAE (ELBO, KL, reparametrización).

---

## 1. Ejercicios

- **EJ1.a — Autoencoder básico.** AE con cuello de botella 2D sobre el dataset
  `font.h` (32 letras de 7×5 = 35 píxeles). Objetivo: reconstruir cada letra con
  un error **≤ 1 píxel**. Incluye un estudio de arquitecturas, optimizadores,
  learning rate, activación y función de pérdida, además de un scatter del espacio
  latente y la generación de una "letra nueva".
- **EJ1.b — Denoising Autoencoder.** DAE entrenado con ruido sal-y-pimienta
  (augmentation online) que aprende a limpiar entradas corruptas. Se barren
  distintos tipos y niveles de ruido y se compara contra el AE básico.
- **EJ2 — Variational Autoencoder.** VAE sobre un dataset de **emojis** 16×16 en
  escala de grises. Incluye gradient-check del backprop, espacio latente 2D,
  grilla generativa (manifold), muestreo, interpolación y experimentos de
  `latent_dim`, `beta` (β-VAE) y `kl_warmup`.

---

## 2. Estructura del repositorio

```
sia-tp5/
├── shared/                     # Librería compartida (numpy puro, portada de TP3)
│   ├── mlp.py                  # MLP con backprop (forward/backward, train_epoch)
│   ├── optimizers.py           # Adam, Momentum, GradientDescent, build_optimizer
│   ├── activations.py          # identity/logistic/tanh/relu/step
│   ├── initializers.py         # random_normal / he_normal / glorot
│   ├── losses.py               # mse, bce (BCE numéricamente estable)
│   ├── metrics.py              # pixel_errors, pixel_error_summary
│   ├── regularization.py       # EarlyStopping
│   ├── fonts.py                # carga de font.h (X 32×35 en {0,1})
│   ├── config_loader.py        # load_yaml
│   └── plotting.py             # save_fig, paleta, curvas
├── ej1/
│   ├── config.yaml             # config de AE básico y denoising
│   ├── models/                 # Autoencoder, ruido (denoising) — NO importa matplotlib
│   ├── analysis/               # todo el ploteo (usa shared.plotting.save_fig)
│   ├── experiments/            # estudio de optimización (arch/opt/lr/act/loss)
│   ├── main_autoencoder.py     # entrypoint EJ1.a
│   ├── main_denoising.py       # entrypoint EJ1.b
│   └── results/                # figuras + modelos (.npz) + tabla CSV
│       ├── basic/
│       └── denoising/
├── ej2/
│   ├── config.yaml             # config del VAE y de sus experimentos
│   ├── data/
│   │   ├── build_emojis.py     # genera emojis.npz desde NotoColorEmoji
│   │   ├── emojis.npz          # dataset (960 muestras, 16 clases)
│   │   └── README.md           # documentación del dataset
│   ├── models/vae.py           # VAE (encoder/decoder, reparam, ELBO) + gradcheck
│   ├── analysis/vae.py         # figuras del VAE
│   ├── experiments/            # latent_dim, beta, kl_warmup
│   ├── main_vae.py             # entrypoint EJ2
│   └── results/                # figuras + modelo (vae_model.npz)
├── tests/
│   └── smoke_shared.py         # smoke test de la librería compartida
├── presentacion/               # presentación Beamer (LaTeX, tema Metropolis)
├── requirements.txt
└── README.md
```

Convención (heredada de TP3/TP4): `models/` **no** importa matplotlib; todo el
ploteo vive en `analysis/` (vía `shared.plotting.save_fig`); cada `main_*.py`
orquesta y hace sanity checks (shapes, rango de píxeles, NaN, pixel-error). La
configuración se lee de `ejN/config.yaml` y el determinismo se controla con seeds.

---

## 3. Instalación

Requiere Python 3.10+ y las dependencias de `requirements.txt` (numpy, matplotlib,
pyyaml y pillow — esta última sólo para regenerar el dataset de emojis del EJ2).

```bash
pip install -r requirements.txt
```

---

## 4. Cómo correr

Todos los entrypoints se ejecutan desde la raíz del repositorio (cada `main_*.py`
agrega el repo al `sys.path` automáticamente).

### Smoke test de la librería compartida

```bash
python3 tests/smoke_shared.py
```

Verifica la carga de `font.h`, un gradient-check del MLP (MSE y BCE), un
entrenamiento corto de AE y los inicializadores. Debe terminar con `SMOKE TEST OK`.

### EJ1.a — Autoencoder básico

```bash
python3 ej1/main_autoencoder.py
```

Prueba varias seeds, entrena la mejor a fondo (Adam, full-batch, EarlyStopping),
reporta el pixel-error, guarda el modelo en `ej1/results/basic/model.npz` y genera
las figuras en `ej1/results/basic/`.

Estudio de optimización (arquitecturas / optimizadores / learning rate / activación
/ loss) que produce `exp_*.png` y `optimization_table.csv`:

```bash
python3 ej1/experiments/optimization_study.py
```

### EJ1.b — Denoising Autoencoder

```bash
python3 ej1/main_denoising.py
```

Entrena el DAE con augmentation online (ruido sal-y-pimienta regenerado cada
época), lo evalúa frente a distintos tipos/niveles de ruido, compara con el AE
básico, guarda el modelo en `ej1/results/denoising/model.npz` y genera las figuras.
Requiere haber corrido antes `main_autoencoder.py` (usa su modelo como baseline).

### EJ2 — dataset de emojis (opcional, ya está generado)

```bash
python3 ej2/data/build_emojis.py
```

Regenera `ej2/data/emojis.npz` (960 muestras, 16 clases, 16×16 en grises) a partir
de NotoColorEmoji. Requiere la fuente instalada en
`/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf` y `pillow`. El repo ya incluye
`emojis.npz`, por lo que este paso **no** es necesario para correr el VAE.

### EJ2 — VAE

```bash
python3 ej2/main_vae.py
```

Corre el gradient-check (obligatorio, falla ruidosamente si no pasa), carga los
emojis, entrena el VAE principal (latent_dim=2, BCE, β=1, Adam), guarda el modelo
en `ej2/results/vae_model.npz` y genera las figuras en `ej2/results/`.

Experimentos del VAE:

```bash
python3 ej2/experiments/latent_dim.py   # -> ej2/results/exp_latent_dim.png
python3 ej2/experiments/beta.py         # -> ej2/results/exp_beta.png
python3 ej2/experiments/kl_warmup.py    # -> ej2/results/exp_kl_warmup.png
```

---

## 5. Resultados

### EJ1.a — Autoencoder básico

Arquitectura `[35, 60, 40, 20, 2, 20, 40, 60, 35]`, tanh en las ocultas, latente
lineal, salida logística, pérdida BCE, Adam (lr=1e-3), full-batch.

| Métrica | Valor |
|---|---|
| max pixel-error (sobre las 32 letras, umbral 0.5) | **0** |
| mean pixel-error | 0.0000 |
| letras exactas | **32/32** |
| letras con ≤1 píxel | **32/32** |
| BCE final (mejor época) | 1.74e-06 |

El objetivo de **≤ 1 píxel se cumple con margen** (máximo error = 0). Las 5 seeds
probadas `[0, 1, 2, 3, 42]` alcanzaron max=0 y 32/32 letras exactas, y el modelo
recargado desde `model.npz` reproduce el resultado.

Como baseline lineal, **PCA 2D** captura sólo 27.1% de la varianza (PC1 15.2% +
PC2 11.9%), muy por debajo de la reconstrucción perfecta del AE no lineal con el
mismo cuello 2D.

**Estudio de optimización** (6000 épocas, seed=0; "época de convergencia" = primera
con max≤1):

- **Arquitectura:** shallow @3650 ép · media @2800 ép · profunda-ancha @600 ép
  (todas llegan a max=0).
- **Optimizador:** Adam @600 ép · Momentum @500 ép · GD @4300 ép (todos max=0).
- **Learning rate (Adam):** 1e-4 @5400 · 5e-4 @1250 · 1e-3 @600 · 5e-3 @200 (todos max=0).
- **Activación oculta:** tanh @600 (max=0) · relu @700 (max=0) · logística **no
  converge** (max=7, 27/32 exactas).
- **Pérdida:** BCE @600 (max=0) vs MSE max=3 (30/32) en 6000 épocas.

### EJ1.b — Denoising Autoencoder

DAE entrenado con sal-y-pimienta p=0.10 (augmentation online, 12000 épocas).
pixel-error **medio** sobre las 32 letras (promedio de repeticiones):

| Nivel sal-y-pimienta | DAE | AE básico |
|---|---|---|
| limpio | 0.00 (32/32 exactas) | — |
| p=0.05 | 0.027 | 6.86 |
| p=0.10 | 0.267 | 9.88 |
| p=0.20 | 2.65 | 12.32 |
| p=0.30 | 6.77 | 14.82 |
| p=0.40 | 11.27 | 16.94 |

El DAE reduce el error **~25-250×** respecto del AE básico en niveles bajos/medios.
Además **generaliza a ruido no visto**: gaussiano (σ=0.1: 0.00 · σ=0.3: 0.02 ·
σ=0.4: 0.15) y masking (frac=0.2: 0.33 · 0.3: 0.88 · 0.4: 2.12).

### EJ2 — VAE

- **Gradient-check:** error relativo máximo **1.25e-10** (tolerancia 1e-4) → **PASA**.
  Otras ramas verificadas: mse+identity 6.4e-11, mse+logistic 4.1e-11, relu+bce
  1.9e-10, sin cuerpo de encoder 1.3e-10.
- **VAE principal** (latent_dim=2, BCE, β=1, 600 épocas, Adam): loss total final
  117.995 (rec 112.219 + KL 5.776). MSE de reconstrucción (z=μ) = 4.576. Salidas
  en [0.000, 0.973], sin NaN.
- **latent_dim** (400 ép): dim=2 rec=114.06 / KL=5.52 / rec_MSE=5.27 · dim=8
  rec=106.06 / KL=8.86 / rec_MSE=1.61 · dim=16 rec=105.89 / KL=8.77 / rec_MSE=1.63
  (más dimensiones → mucha mejor reconstrucción).
- **beta** (400 ép, latente 2): β=0.1 rec=111.30 / KL=10.00 · β=1.0 rec=114.06 /
  KL=5.52 · β=4.0 rec=120.57 / KL=3.25 (trade-off rec/KL clásico del β-VAE).
- **kl_warmup** (400 ép): sin warmup rec=114.06 / KL=5.52 · warmup=100 rec=113.64 /
  KL=5.58 (el warmup deja crecer el KL temprano y termina con reconstrucción algo mejor).

---

## 6. Figuras generadas

- **EJ1.a** (`ej1/results/basic/`): `learning_curve.png`, `reconstruction_grid.png`,
  `latent_scatter.png`, `latent_scatter_pca.png`, `pixel_error_hist.png`,
  `new_letter_generation.png`, `exp_architecture.png`, `exp_optimizer.png`,
  `exp_learning_rate.png`, `exp_activation.png`, `exp_loss.png`,
  `exp_summary_maxpixel.png`.
- **EJ1.b** (`ej1/results/denoising/`): `noise_examples.png`, `noise_sweep.png`.
- **EJ2** (`ej2/results/`): `dataset_sample.png`, `loss_curves.png`,
  `latent_scatter.png`, `manifold_grid.png`, `samples.png`, `interpolation.png`,
  `reconstructions.png`, `exp_latent_dim.png`, `exp_beta.png`, `exp_kl_warmup.png`.

---

## 7. Presentación

La presentación (Beamer, tema Metropolis, en español) está en `presentacion/`.
Ver `presentacion/README.md` para instrucciones de compilación.

---

## 8. Créditos del dataset de emojis

Los glifos provienen de **NotoColorEmoji** (Google Fonts, licencia SIL Open Font
License 1.1). Ver `ej2/data/README.md` para el detalle de generación.
