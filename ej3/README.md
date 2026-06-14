# EJ3 — VAE sobre MNIST (variante alternativa del EJ2)

EJ3 **no es un ejercicio del enunciado**: es una **versión alternativa del EJ2**
para comparar contra los emojis y, sobre todo, para responder una duda concreta:

> En un TP de *Deep Learning*, ¿no estamos usando **muy pocas capas**?
> ¿Cambia algo si agregamos más capas ocultas? ¿Y si usamos un dataset más
> complejo y estándar como MNIST 28×28?

Por eso EJ3 hace dos cosas:

1. **Mismo VAE, dataset más complejo.** Reutiliza **exactamente** el VAE de
   `ej2/models/vae.py` (numpy puro, ya gradchequeado), pero sobre **MNIST 28×28**
   (`input_dim = 784`) en vez de emojis 16×16. No se reimplementa el modelo.
2. **Experimento de PROFUNDIDAD.** Entrena el mismo VAE con 1, 2, 3 y 4 capas
   ocultas (todo lo demás igual) y mide si agregar capas mejora la
   reconstrucción en *test* (held-out) y la generación, y a qué costo de tiempo.

Todo se corre con un **runner pausable y continuable**: lo dejás trabajando,
lo cortás cuando querés (Ctrl-C) y al volver continúa **exacto** donde quedó.

---

## Cómo correr

```bash
# 1) Construir el dataset (baja MNIST de mirrors públicos; numpy puro)
python3 ej3/data/build_mnist.py

# 2) Correr / reanudar TODOS los experimentos de profundidad
python3 ej3/run_resumable.py

# 3) Ver progreso cuando quieras
python3 ej3/run_resumable.py --status

# 4) Figuras (después de que haya jobs terminados)
python3 ej3/analysis/mnist_vae.py
```

### Pausar y continuar

- **Pausar:** `Ctrl-C` (SIGINT). El runner termina la época en curso, guarda un
  checkpoint y sale limpio.
- **Continuar:** volvé a correr `python3 ej3/run_resumable.py`. Reanuda desde el
  último checkpoint, restaurando **modelo + optimizador Adam + estado del
  generador aleatorio + historial + época**. La continuación es
  **bit-idéntica** a una corrida sin interrupciones (verificado en
  `tests/smoke_ej3_resume.py`).
- **Robustez:** además de la pausa elegante, guarda un checkpoint cada
  `checkpoint_every` épocas (config). Si se corta la luz o lo matás con `kill -9`,
  perdés a lo sumo esas pocas épocas.

### Extender en el futuro (aunque ya haya terminado)

El experimento es **continuable hacia adelante**: si terminó las 500 épocas y
querés más, no se re-entrena de cero, sigue desde donde quedó (restaura modelo +
Adam + RNG).

```bash
python3 ej3/run_resumable.py --extend 300    # 300 épocas MÁS por job
python3 ej3/run_resumable.py --epochs 1000   # llevar el target a 1000 (absoluto)
```

(También podés editar `epochs:` en el config y volver a correr: detecta que el
checkpoint tiene menos épocas que el nuevo target y entrena sólo lo que falta.)
Verificado: extender un job ya terminado **appendea** al historial, no reinicia.

### Otros comandos

```bash
python3 ej3/run_resumable.py --only d2_L16_s42  # un solo experimento (arch+semilla)
python3 ej3/run_resumable.py --reset all        # borra checkpoints (¡de cero!)
python3 ej3/run_resumable.py --no-gradcheck     # saltea el gradient-check inicial
```

---

## Qué se configura (`ej3/config.yaml`)

Viene preparado para una corrida **overnight (≈10 h)**: 60.000 imágenes, 500
épocas, 2 realizaciones (semillas) por arquitectura, latente 16, batch 256.

- `vae.latent_dim`: **16** (deja capacidad para que la profundidad pueda mostrar
  efecto; con 2D todo sale igual de borroso y enmascara el experimento). Poné
  `2` si querés el manifold 2D estilo EJ2.
- `data.train_size` / `data.test_size`: subconjunto determinista de MNIST.
  **`0` = todas** (60000 / 10000). Para una prueba rápida poné `20000`.
- `vae.epochs`, `batch_size`, `learning_rate`, `beta`: hiperparámetros comunes.
- `runner.checkpoint_every` / `eval_every` / `log_every`.
- `experiments.depth.seeds`: realizaciones por arquitectura (`[42, 123]` = 2).
  Bajalo a `[42]` para ir más rápido (~5 h) o agregá una 3ª para más robustez.
- `experiments.depth.architectures`: la lista de profundidades a barrer. Editala
  para agregar/quitar; el runner se adapta solo.

**Tiempos medidos** (60k, batch 256, latente 16): ~4 s/época la red de 1 capa,
~12 s/época la de 4 capas. 4 arquitecturas × 500 épocas × 2 semillas ≈ **10 h**.
Si lo cortás, reanuda; el orden de los jobs deja un barrido completo de
profundidad (1 semilla) listo en ~5 h.

---

## Salidas

- `ej3/results/checkpoints/<job>.npz` — checkpoints (transitorios, regenerables).
- `ej3/results/<job>_model.npz` — modelo final (cargable con `VAE.load`).
- `ej3/results/<job>_hist.npz` — historial crudo (loss train/test por época,
  tiempo). Permite re-graficar sin re-entrenar.
- `ej3/results/*.png` — figuras: reconstrucciones, muestras generadas, curvas y
  el resumen **profundidad vs reconstrucción/tiempo** (el resultado central).

---

## Archivos

```
ej3/
├── config.yaml                 # hiperparámetros + lista de arquitecturas
├── run_resumable.py            # ★ runner pausable/continuable (entrena)
├── data/
│   └── build_mnist.py          # descarga + parser IDX (numpy puro) -> mnist.npz
├── experiments/
│   └── depth_study.py          # define los jobs del barrido de profundidad
└── analysis/
    └── mnist_vae.py            # figuras (matplotlib SOLO acá)
```

El VAE vive en `ej2/models/vae.py` (se reutiliza, no se duplica). El dataset
`mnist.npz` y los checkpoints se regeneran y por eso no se versionan.
