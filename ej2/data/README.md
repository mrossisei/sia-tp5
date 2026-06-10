# Dataset de emojis — ej2/data

## Descripción

Dataset de imágenes de emojis en escala de grises 16x16 para entrenamiento del VAE (ejercicio 2).

## Archivo generado: `emojis.npz`

| Clave         | Tipo      | Shape       | Descripción                              |
|---------------|-----------|-------------|------------------------------------------|
| `X`           | float32   | (N, 256)    | Imágenes aplanadas, rango [0,1]          |
| `y`           | int64     | (N,)        | Índice de clase (0 a n_classes-1)        |
| `labels`      | str array | (n_classes,)| Nombre de cada clase                     |
| `image_shape` | int array | (2,)        | [16, 16]                                 |

### Estadísticas del dataset generado

- **N total**: 960 muestras
- **Clases**: 16
- **Muestras por clase**: 60 (augmentation aplicado)
- **Rango de X**: [0.0, 1.0]

## Clases incluidas

| Índice | Nombre            | Emoji |
|--------|-------------------|-------|
| 0      | cara_feliz        | 😀    |
| 1      | cara_encantada    | 😍    |
| 2      | cara_gafas        | 😎    |
| 3      | cara_triste       | 😢    |
| 4      | cara_enojada      | 😡    |
| 5      | cara_dormida      | 😴    |
| 6      | cara_asustada     | 😱    |
| 7      | cara_sorprendida  | 😯    |
| 8      | corazon           | ❤    |
| 9      | estrella          | ⭐    |
| 10     | sol               | ☀    |
| 11     | nube              | ☁    |
| 12     | luna              | 🌙    |
| 13     | fuego             | 🔥    |
| 14     | flor              | 🌸    |
| 15     | pulgar_arriba     | 👍    |

## Fuente de los glifos

**NotoColorEmoji** — Google Fonts  
Ruta: `/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf`  
Licencia: **SIL Open Font License 1.1** (OFL-1.1)  
URL: https://fonts.google.com/noto/specimen/Noto+Color+Emoji

La licencia OFL-1.1 permite uso libre incluyendo aplicaciones académicas y de investigación.

## Método de generación

1. Rasterización con PIL a tamaño 109px (único tamaño soportado por NotoColorEmoji)
2. Conversión a escala de grises (modo 'L')
3. Recorte del bounding box con margen de 2px
4. Redimensionado a 16x16 con filtro LANCZOS
5. Normalización a [0,1] con clip
6. Augmentation: 60 variantes por clase con:
   - Escala aleatoria ±10%
   - Rotación aleatoria ±10°
   - Traslación aleatoria ±2px
   - Seed fijo = 42 (determinismo)

## Reproducción

```bash
python3 ej2/data/build_emojis.py
```

La hoja de contacto (una muestra por clase) se guarda en `ej2/results/dataset_sample.png`.
