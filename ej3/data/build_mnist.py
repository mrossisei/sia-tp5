"""
build_mnist.py — Genera el dataset MNIST para el VAE alternativo (ejercicio 3).

EJ3 es una VARIANTE del EJ2: el mismo VAE (ej2/models/vae.py) pero sobre MNIST
28x28 en vez de emojis 16x16, para (a) probar un dataset más complejo/estándar
y (b) barrer la CANTIDAD DE CAPAS OCULTAS como experimento.

Fuente: archivos IDX clásicos de MNIST (LeCun et al.). Se bajan de mirrors
públicos (el dominio yann.lecun.com suele estar caído / con 403):
    - https://ossci-datasets.s3.amazonaws.com/mnist/   (mirror de PyTorch)
    - https://storage.googleapis.com/cvdf-datasets/mnist/  (mirror de CVDF)

numpy PURO: el parser usa SÓLO la librería estándar (gzip, struct, urllib) +
numpy. No se usa torchvision/keras/sklearn ni ninguna lib de ML, consistente
con el resto del TP.

Salida: ej3/data/mnist.npz con claves
    X_train  uint8 (60000, 784)   — imágenes aplanadas, 0..255
    y_train  uint8 (60000,)       — dígito 0..9
    X_test   uint8 (10000, 784)
    y_test   uint8 (10000,)
    image_shape  array [28, 28]

Se guardan como uint8 (compacto); el loader normaliza a float64 en [0,1].

Uso:
    python3 ej3/data/build_mnist.py            # baja (si falta) y construye
    python3 ej3/data/build_mnist.py --force    # rebaja y reconstruye
"""

import argparse
import gzip
import os
import struct
import sys
import urllib.request

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "raw")
OUT_NPZ = os.path.join(SCRIPT_DIR, "mnist.npz")

# Mirrors a probar en orden (el primero que responda gana).
MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download(fname, dest, timeout=60):
    """Descarga fname desde el primer mirror que funcione. Devuelve True/False."""
    last_err = None
    for base in MIRRORS:
        url = base + fname
        try:
            print(f"  bajando {url} ...", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "sia-tp5/ej3"})
            with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
                f.write(r.read())
            size = os.path.getsize(dest)
            if size < 1000:  # respuesta vacía / error encubierto
                raise IOError(f"archivo demasiado chico ({size} bytes)")
            print(f"    OK ({size/1e6:.1f} MB)")
            return True
        except Exception as e:  # noqa: BLE001 — queremos probar el próximo mirror
            last_err = e
            print(f"    falló: {e}")
            if os.path.exists(dest):
                os.remove(dest)
    print(f"  ERROR: no se pudo bajar {fname} de ningún mirror ({last_err})")
    return False


def ensure_raw(force=False):
    """Garantiza que los 4 .gz estén en raw/. Devuelve True si están todos."""
    os.makedirs(RAW_DIR, exist_ok=True)
    ok = True
    for fname in FILES.values():
        dest = os.path.join(RAW_DIR, fname)
        if force and os.path.exists(dest):
            os.remove(dest)
        if not os.path.exists(dest):
            ok = ensure_one(fname, dest) and ok
        else:
            print(f"  ya existe {fname} ({os.path.getsize(dest)/1e6:.1f} MB)")
    return ok


def ensure_one(fname, dest):
    return _download(fname, dest)


def read_idx_images(path):
    """Parser IDX de imágenes (magic 2051): big-endian, uint8."""
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"magic inesperado en {path}: {magic} (esperaba 2051)")
        buf = f.read(n * rows * cols)
    data = np.frombuffer(buf, dtype=np.uint8).reshape(n, rows * cols)
    return data, (rows, cols)


def read_idx_labels(path):
    """Parser IDX de labels (magic 2049): big-endian, uint8."""
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"magic inesperado en {path}: {magic} (esperaba 2049)")
        buf = f.read(n)
    return np.frombuffer(buf, dtype=np.uint8)


def build(force=False):
    print("=" * 70)
    print("MNIST -> ej3/data/mnist.npz  (numpy puro)")
    print("=" * 70)

    if not ensure_raw(force=force):
        print("\nNo se pudieron obtener los archivos crudos de MNIST.")
        print("Si no hay internet, descargá manualmente estos 4 archivos a")
        print(f"  {RAW_DIR}/")
        for f in FILES.values():
            print(f"    - {f}")
        print("y volvé a correr este script.")
        return 1

    print("\nparseando IDX ...")
    X_train, shape_tr = read_idx_images(os.path.join(RAW_DIR, FILES["train_images"]))
    y_train = read_idx_labels(os.path.join(RAW_DIR, FILES["train_labels"]))
    X_test, shape_te = read_idx_images(os.path.join(RAW_DIR, FILES["test_images"]))
    y_test = read_idx_labels(os.path.join(RAW_DIR, FILES["test_labels"]))

    assert shape_tr == shape_te == (28, 28), f"shape inesperada: {shape_tr}/{shape_te}"
    assert X_train.shape == (60000, 784), f"X_train {X_train.shape}"
    assert X_test.shape == (10000, 784), f"X_test {X_test.shape}"
    assert y_train.shape == (60000,) and y_test.shape == (10000,)
    assert X_train.max() <= 255 and X_train.min() >= 0

    print(f"  X_train={X_train.shape}  y_train={y_train.shape}  "
          f"clases={sorted(np.unique(y_train).tolist())}")
    print(f"  X_test ={X_test.shape}  y_test ={y_test.shape}")

    np.savez_compressed(
        OUT_NPZ,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        image_shape=np.array([28, 28]),
    )
    print(f"\nguardado: {OUT_NPZ} ({os.path.getsize(OUT_NPZ)/1e6:.1f} MB)")
    print("LISTO.")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Construye ej3/data/mnist.npz")
    ap.add_argument("--force", action="store_true", help="rebaja y reconstruye")
    args = ap.parse_args()
    sys.exit(build(force=args.force))


if __name__ == "__main__":
    main()
