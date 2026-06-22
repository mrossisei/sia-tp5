# Experimentos de EJ2

## Contexto general

EJ2 implementa un **Variational Autoencoder (VAE)** en numpy puro sobre un dataset de **emojis 16x16 en escala de grises**.

### Configuración base común

- Dataset: `ej2/data/emojis.npz`
- Tamaño de imagen: `16x16`
- Cantidad de muestras: `960`
- Cantidad de clases: `16`
- Encoder hidden: `[256, 64]`
- Decoder hidden: `[64, 256]`
- Activación oculta: `relu`
- Activación de salida: `logistic`
- Loss de reconstrucción: `bce`
- Optimizador: `Adam`

Archivo helper compartido:
- `ej2/experiments/_common.py`

---

## 1. Corrida principal del VAE

**Script:** `ej2/main_vae.py`

### Objetivo

Entrenar el modelo principal del EJ2 y generar todas las figuras estándar del ejercicio.

### Configuración

- `latent_dim = 2`
- `beta = 1.0`
- `epochs = 600`
- `batch_size = 64`
- `seed = 42`

### Particularidades

- Corre un **gradient-check obligatorio** antes de entrenar.
- Entrena el VAE principal sobre el dataset de emojis.
- Genera visualizaciones del espacio latente y de la capacidad generativa.

### Salidas

- `ej2/results/vae_model.npz`
- `ej2/results/loss_curves.png`
- `ej2/results/latent_scatter.png`
- `ej2/results/manifold_grid.png`
- `ej2/results/samples.png`
- `ej2/results/interpolation.png`
- `ej2/results/reconstructions.png`

---

## 2. Barrido de dimensión latente

**Script:** `ej2/experiments/latent_dim.py`

### Objetivo

Comparar cómo cambia la reconstrucción y la generación al variar la dimensión del espacio latente.

### Variable barrida

- `latent_dim in {2, 8, 16}`

### Configuración

- `epochs = 400`
- `seed = 42`

### Métricas reportadas

- `rec_final`
- `kl_final`
- `rec_mse`

### Visualización

- Se generan muestras decodificadas desde `z ~ N(0, I)` para cada dimensión latente.

### Salida

- `ej2/results/exp_latent_dim.png`

---

## 3. Barrido de beta (β-VAE)

**Script:** `ej2/experiments/beta.py`

### Objetivo

Estudiar el trade-off entre reconstrucción y regularización del latente al variar el peso del término KL.

### Variable barrida

- `beta in {0.1, 1.0, 4.0}`

### Configuración

- `latent_dim = 2`
- `epochs = 400`
- `seed = 42`

### Métricas reportadas

- `rec`
- `kl`
- `rec_mse`

### Visualización

- Scatter latente 2D para cada valor de `beta`.

### Salida

- `ej2/results/exp_beta.png`

---

## 4. KL warmup / annealing

**Script:** `ej2/experiments/kl_warmup.py`

### Objetivo

Evaluar si conviene introducir el término KL de forma gradual para estabilizar el entrenamiento.

### Variable barrida

- `warmup in {0, 100}`

### Configuración

- `latent_dim = 2`
- `beta_target = 1.0`
- `epochs = 400`
- `seed = 42`

### Métricas reportadas

- `rec` final
- `kl` final

### Visualización

- Curvas por época de:
- reconstrucción
- KL

### Salida

- `ej2/results/exp_kl_warmup.png`

---

## 5. Variabilidad entre seeds

**Script:** `ej2/experiments/seeds.py`

### Objetivo

Medir la estabilidad del VAE principal frente a distintas semillas aleatorias.

### Variable barrida

- `seed in {42, 7, 123}`

### Configuración

- `latent_dim = 2`
- `epochs = 600`

### Métricas reportadas

- `total`
- `rec`
- `kl`
- `rec_mse`

### Visualización

- Curvas individuales por seed
- Media
- Banda `± std`

### Salidas

- `ej2/results/exp_seeds.png`
- `ej2/results/exp_seeds.csv`

---

## 6. Corrida larga del VAE principal

**Script:** `ej2/experiments/vae_long.py`

### Objetivo

Entrenar el VAE principal durante muchas más épocas que la corrida estándar, ya que a 600 épocas la reconstrucción seguía mejorando.

### Configuración típica

- Dataset: `ej2/data/emojis.npz`
- `tag = 16px`
- `latent = 2`
- Encoder: `[256, 64]`
- Decoder: `[64, 256]`
- `epochs = 50000`
- `batch = 64`
- `lr = 1e-3`
- `beta = 1.0`

### Particularidades

- Checkpoints periódicos
- Reanudación con `--resume`
- Snapshots intermedios del modelo

### Salidas

Directorio:
- `ej2/results/long_16px/`

Archivos típicos:
- `vae_model.npz`
- `loss_history.csv`
- `loss_curve.png`
- `loss_curves.png`
- `latent_scatter.png`
- `manifold_grid.png`
- `samples.png`
- `interpolation.png`
- `reconstructions.png`
- `snapshots/model_epNNNNNN.npz`

---

## 7. Corrida larga en dataset 24x24

**Script:** `ej2/experiments/vae_long.py`

### Objetivo

Repetir la corrida larga en un dataset de emojis de mayor resolución.

### Configuración típica

- Dataset: `ej2/data/emojis_24.npz`
- `tag = 24px`
- Encoder: `[512, 128]`
- Decoder: `[128, 512]`
- `epochs = 30000`
- `ckpt_every = 250`

### Salidas

Directorio:
- `ej2/results/long_24px/`

Incluye el mismo tipo de archivos que la corrida larga 16px.

---

## 8. Análisis posterior de corridas largas

**Script:** `ej2/analysis/long_run.py`

### Objetivo

Analizar corridas largas ya entrenadas sin reentrenar nada.

### Insumos

- Snapshots de `long_16px`
- Snapshots de `long_24px`
- Historiales de loss guardados

### Qué genera

- `ej2/results/evolution_16px.png`
- `ej2/results/evolution_24px.png`
- `ej2/results/comparison_600_vs_50k.png`

### Qué muestra

- Evolución del decoder a lo largo del entrenamiento
- Comparación entre la corrida normal de `600` épocas y la corrida larga de `50000`

---

## Resumen rápido

EJ2 tiene:

1. Una corrida principal del VAE
2. Un barrido de dimensión latente
3. Un barrido de `beta`
4. Un experimento de KL warmup
5. Un experimento de robustez entre seeds
6. Una corrida larga del modelo principal
7. Una corrida larga sobre emojis 24x24
8. Un análisis posterior de las corridas largas
