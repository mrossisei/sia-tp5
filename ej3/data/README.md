# Dataset EJ3 — MNIST 28×28

`mnist.npz` se genera con `build_mnist.py` (no se versiona: es regenerable).

```bash
python3 ej3/data/build_mnist.py
```

## Qué hace

1. Descarga los 4 archivos IDX clásicos de MNIST desde mirrors públicos
   (S3 de PyTorch / CVDF de Google; el dominio original `yann.lecun.com` suele
   estar caído). Quedan en `ej3/data/raw/` (tampoco se versionan).
2. Los parsea con **numpy puro** (`gzip` + `struct` de la stdlib, sin
   torchvision/keras/sklearn — consistente con todo el TP).
3. Guarda `ej3/data/mnist.npz` comprimido.

## Contenido de `mnist.npz`

| clave | shape | tipo | qué es |
|-------|-------|------|--------|
| `X_train` | (60000, 784) | uint8 | imágenes de train, 0..255 (28×28 aplanado) |
| `y_train` | (60000,) | uint8 | dígito 0..9 |
| `X_test` | (10000, 784) | uint8 | imágenes de test |
| `y_test` | (10000,) | uint8 | dígito 0..9 |
| `image_shape` | (2,) | int | `[28, 28]` |

El loader del runner normaliza a `float64` en `[0,1]` (divide por 255) y toma el
subconjunto determinista que indica `config.yaml` (`train_size`/`test_size`).

## Crédito

MNIST — Yann LeCun, Corinna Cortes, Christopher J.C. Burges. Dominio público /
uso académico habitual.
