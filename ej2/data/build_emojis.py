"""
build_emojis.py — Genera el dataset de emojis para el VAE (ejercicio 2).

Fuente: NotoColorEmoji.ttf (Google Fonts, Apache License 2.0)
        /usr/share/fonts/truetype/noto/NotoColorEmoji.ttf

Salida: ej2/data/emojis.npz con claves:
    X           float32 (N, 256)  — imágenes aplanadas, rango [0,1]
    y           int     (N,)      — índice de clase
    labels      array de strings  — nombre por clase
    image_shape array [16, 16]

También guarda: ej2/results/dataset_sample.png (grilla de muestras).

Método:
    - Rasteriza cada emoji con PIL (NotoColorEmoji, tamaño 109)
    - Convierte a escala de grises 'L', redimensiona a 16x16 (LANCZOS)
    - Augmentation: ~60 variantes por clase con traslaciones ±2px,
      escala ±5%, rotación ±10° (determinista, seed fijo)
"""

import os
import sys

# Ruta al repo root para poder importar shared/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EJ2_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(EJ2_DIR)
sys.path.insert(0, REPO_ROOT)

import numpy as np
from PIL import Image, ImageFont, ImageDraw

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
FONT_PATH = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
FONT_SIZE = 109          # único tamaño reconocido por NotoColorEmoji
IMG_CANVAS = 128         # canvas para rasterizar antes de recortar
TARGET_SIZE = 16         # salida final 16x16
AUGMENT_PER_CLASS = 60   # variantes por clase (incluyendo el original)
SEED = 42
MIN_NONZERO = 1000       # mínimo de píxeles con tinta para aceptar el emoji

OUT_NPZ = os.path.join(EJ2_DIR, "data", "emojis.npz")
OUT_PNG = os.path.join(EJ2_DIR, "results", "dataset_sample.png")

# ---------------------------------------------------------------------------
# Emojis a incluir: (nombre_clase, carácter_unicode)
# ---------------------------------------------------------------------------
EMOJI_CANDIDATES = [
    ("cara_feliz",       "😀"),
    ("cara_encantada",   "😍"),
    ("cara_gafas",       "😎"),
    ("cara_triste",      "😢"),
    ("cara_enojada",     "😡"),
    ("cara_dormida",     "😴"),
    ("cara_asustada",    "😱"),
    ("cara_sorprendida", "😯"),
    ("corazon",          "❤"),
    ("estrella",         "⭐"),
    ("sol",              "☀"),
    ("nube",             "☁"),
    ("luna",             "🌙"),
    ("fuego",            "🔥"),
    ("flor",             "🌸"),
    ("pulgar_arriba",    "👍"),
]


# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------

def rasterizar_emoji(char: str, font: ImageFont.FreeTypeFont) -> np.ndarray | None:
    """
    Rasteriza un emoji a escala de grises 16x16 normalizado [0,1].
    Retorna None si el glifo queda vacío (< MIN_NONZERO píxeles con tinta).
    """
    img = Image.new("RGBA", (IMG_CANVAS, IMG_CANVAS), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((10, 10), char, font=font, embedded_color=True)

    gray = img.convert("L")
    arr = np.array(gray, dtype=np.float32)

    if np.count_nonzero(arr) < MIN_NONZERO:
        return None

    # Recortar bounding-box con margen de 2px
    rows = np.any(arr > 0, axis=1)
    cols = np.any(arr > 0, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    margin = 2
    r0 = max(0, r0 - margin)
    r1 = min(IMG_CANVAS - 1, r1 + margin)
    c0 = max(0, c0 - margin)
    c1 = min(IMG_CANVAS - 1, c1 + margin)

    cropped = Image.fromarray(arr[r0:r1+1, c0:c1+1])
    resized = cropped.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
    out = np.clip(np.array(resized, dtype=np.float32) / 255.0, 0.0, 1.0)
    return out  # shape (16, 16), rango [0,1]


def augmentar(base: np.ndarray, n: int, rng: np.random.Generator) -> list[np.ndarray]:
    """
    Genera n variantes augmentadas de la imagen base (16x16).
    Transformaciones: traslación ±2px, escala ±5%, rotación ±10°.
    La imagen original siempre se incluye como primera variante.
    """
    variantes = [base.copy()]

    # Aumentamos PIL para manejar las transformaciones
    pil_base = Image.fromarray((base * 255).astype(np.uint8), mode="L")

    for _ in range(n - 1):
        img = pil_base.copy()

        # Escala aleatoria: zoom en [0.90, 1.10]
        scale = rng.uniform(0.90, 1.10)
        new_w = max(1, int(TARGET_SIZE * scale))
        new_h = max(1, int(TARGET_SIZE * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)

        # Centrar sobre canvas 16x16
        canvas = Image.new("L", (TARGET_SIZE, TARGET_SIZE), 0)
        x_off = (TARGET_SIZE - new_w) // 2
        y_off = (TARGET_SIZE - new_h) // 2
        canvas.paste(img, (x_off, y_off))
        img = canvas

        # Rotación aleatoria ±10°
        angle = rng.uniform(-10, 10)
        img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)

        # Traslación aleatoria ±2px
        tx = int(rng.integers(-2, 3))
        ty = int(rng.integers(-2, 3))
        img = img.transform(
            (TARGET_SIZE, TARGET_SIZE),
            Image.AFFINE,
            (1, 0, -tx, 0, 1, -ty),
            resample=Image.BILINEAR,
            fillcolor=0,
        )

        arr = np.clip(np.array(img, dtype=np.float32) / 255.0, 0.0, 1.0)
        variantes.append(arr)

    return variantes


def guardar_hoja_contacto(X_por_clase: list[np.ndarray], labels: list[str], ruta: str):
    """
    Guarda una grilla PNG con una muestra por clase para verificación visual.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_clases = len(labels)
    cols = 8
    rows = (n_clases + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.8))
    axes = np.array(axes).reshape(rows, cols)

    for i, (img, label) in enumerate(zip(X_por_clase, labels)):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.imshow(img.reshape(TARGET_SIZE, TARGET_SIZE), cmap="gray", vmin=0, vmax=1)
        ax.set_title(label, fontsize=7, pad=2)
        ax.axis("off")

    # Ocultar ejes sobrantes
    for i in range(n_clases, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis("off")

    fig.suptitle("Dataset de emojis — una muestra por clase", fontsize=10, y=1.01)
    plt.tight_layout()

    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Hoja de contacto guardada en: {ruta}")


# ---------------------------------------------------------------------------
# Script principal
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Construyendo dataset de emojis (16x16, escala de grises)")
    print("=" * 60)

    rng = np.random.default_rng(SEED)

    # Cargar fuente
    print(f"\nCargando fuente: {FONT_PATH}")
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    print("Fuente cargada OK.")

    X_list = []       # imágenes aplanadas (N, 256)
    y_list = []       # etiquetas de clase (N,)
    labels = []       # nombre de cada clase
    muestras_repr = []  # una muestra por clase (para la grilla)

    print(f"\nRasterizando y augmentando {len(EMOJI_CANDIDATES)} emojis...")
    clase_idx = 0
    for nombre, char in EMOJI_CANDIDATES:
        base = rasterizar_emoji(char, font)
        if base is None:
            print(f"  DESCARTADO (vacío): {nombre} {char}")
            continue

        variantes = augmentar(base, AUGMENT_PER_CLASS, rng)
        for v in variantes:
            X_list.append(v.flatten())
            y_list.append(clase_idx)

        labels.append(nombre)
        muestras_repr.append(base)
        print(f"  OK  clase {clase_idx:2d}  {nombre:20s} {char}  "
              f"base_max={base.max():.3f}  variantes={len(variantes)}")
        clase_idx += 1

    n_clases = len(labels)
    X = np.array(X_list, dtype=np.float32)   # (N, 256)
    y = np.array(y_list, dtype=int)           # (N,)
    labels_arr = np.array(labels)
    image_shape = np.array([TARGET_SIZE, TARGET_SIZE])

    print(f"\nResumen del dataset:")
    print(f"  N total de muestras : {X.shape[0]}")
    print(f"  Dimensión (aplanada): {X.shape[1]}")
    print(f"  Número de clases    : {n_clases}")
    print(f"  Rango X             : [{X.min():.4f}, {X.max():.4f}]")
    print(f"  NaN en X            : {np.isnan(X).sum()}")
    print(f"  Clases              : {list(labels)}")

    # Verificaciones
    assert X.shape[1] == 256, f"Esperado 256, got {X.shape[1]}"
    assert X.min() >= 0.0 and X.max() <= 1.0, "Rango fuera de [0,1]"
    assert not np.isnan(X).any(), "Hay NaN en X"
    assert len(labels) == n_clases, "Mismatch labels"
    assert y.min() == 0 and y.max() == n_clases - 1, "Índices de clase inconsistentes"
    print("\nTodas las verificaciones pasaron.")

    # Guardar NPZ
    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez(OUT_NPZ, X=X, y=y, labels=labels_arr, image_shape=image_shape)
    print(f"\nDataset guardado en: {OUT_NPZ}")

    # Hoja de contacto
    guardar_hoja_contacto(muestras_repr, labels, OUT_PNG)

    print("\nListo.")
    return X, y, labels_arr, image_shape


if __name__ == "__main__":
    main()
