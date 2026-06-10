# AGENTS.md — TP5: Deep Learning (Autoencoders / VAE)

> **Sistemas de Inteligencia Artificial — ITBA, 2026**
>
> Este es el documento maestro del TP5: codifica la consigna, las decisiones de
> diseño, las anclas teóricas (lo visto en clase) y el plan de implementación
> ejercicio por ejercicio. Está pensado para trabajar **en base a esto**.
> Mantiene la convención de TP3 y TP4 (numpy puro, `models/` + `analysis/` +
> `main_*.py`, `config.yaml`, presentación Beamer).

---

## Índice

1. [Resumen del TP](#1-resumen-del-tp)
2. [Consigna (enunciado transcripto)](#2-consigna-enunciado-transcripto)
3. [Decisiones de diseño](#3-decisiones-de-diseño)
4. [Dependencia con TPs anteriores (reuso de TP3)](#4-dependencia-con-tps-anteriores-reuso-de-tp3)
5. [Estructura del proyecto](#5-estructura-del-proyecto)
6. [El dataset `font.h`](#6-el-dataset-fonth)
7. [Anclas teóricas](#7-anclas-teóricas)
8. [Plan EJ1 — Autoencoder básico + Denoising](#8-plan-ej1--autoencoder-básico--denoising)
9. [Plan EJ2 — VAE sobre emojis](#9-plan-ej2--vae-sobre-emojis)
10. [Plan experimental (mapeo a la consigna)](#10-plan-experimental-mapeo-a-la-consigna)
11. [Criterios de éxito y métricas](#11-criterios-de-éxito-y-métricas)
12. [Presentación](#12-presentación)
13. [Entrega](#13-entrega)
14. [Roadmap de trabajo](#14-roadmap-de-trabajo)
15. [Notas para el agente](#15-notas-para-el-agente)

---

## 1. Resumen del TP

El TP5 cierra la cursada con **Deep Learning**. Se implementan tres redes,
todas en **numpy puro** (sin frameworks), sobre la infraestructura de MLP +
backprop que ya construimos en TP3:

| Bloque | Red | Dataset | Objetivo central |
|--------|-----|---------|------------------|
| **EJ1.a** | Autoencoder básico | `font.h` (32 patrones 7×5) | Reconstruir las 32 letras vía un cuello de botella **2D**, con **≤ 1 pixel de error** |
| **EJ1.b** | Denoising Autoencoder | `font.h` | Quitar ruido a entradas distorsionadas |
| **EJ2** | Variational Autoencoder (VAE) | **emojis** (baja resolución) | Espacio latente probabilístico + **generar** muestras nuevas |

La columna vertebral teórica está en `material/clases-teoricas/Autoencoders.pdf`
(95 diapositivas, autocontenidas, con la derivación completa del VAE). El resto
de los PDF (`DeepLearning.pdf`, `CNN.pdf`, `Capsule.pdf`) son conceptuales y
sólo aportan contexto.

---

## 2. Consigna (enunciado transcripto)

Fuente: `material/Enunciado.pdf`.

### 1. Autoencoders

**a)** Implementar un Autoencoder básico para las imágenes binarias de la lista
de caracteres del archivo `font.h`.

1. Plantear una arquitectura de red para el Codificador y Decodificador que
   permita representar los datos de entrada en un **espacio latente de dos
   dimensiones**.
2. Describir y estudiar las diferentes **arquitecturas y técnicas de
   optimización** que se fueron aplicando para permitir que la red aprenda todo
   el set de datos o un subconjunto. **El objetivo es aprender los 32 patrones
   de 5×7 en un espacio latente de 2 dimensiones con un error máximo de 1 pixel
   incorrecto.** Si es un subconjunto, mostrar por qué no fue posible aprender
   el dataset completo.
3. Realizar el gráfico en **dos dimensiones** que muestre los datos de entrada
   en el espacio latente.
4. Mostrar cómo la red puede **generar una nueva letra** que no pertenece al
   conjunto de entrenamiento.

**b)** Sobre el mismo dataset, implementar una variante **Denoising Autoencoder**.

1. Plantear una arquitectura conveniente para esta tarea. Explicar la elección.
2. **Distorsionar las entradas en diferentes niveles** y estudiar la capacidad
   del autoencoder de eliminar el ruido.

### 2. VAE (Variational Autoencoder)

Extender el autoencoder anterior para que opere como un VAE.

- **a)** Elegir (o construir) un conjunto de datos nuevo (por ejemplo, emojis).
- **b)** Modificar el autoencoder planteando un esquema variacional para
  solucionar el problema de la representación en el espacio latente.
- **c)** Usar el autoencoder anterior para **generar una nueva muestra** que
  juzguemos que pertenece al conjunto de datos presentado (Generative
  Autoencoder).

---

## 3. Decisiones de diseño

Decisiones tomadas al planificar el TP (confirmar/registrar cambios acá):

1. **Implementación: numpy puro, reutilizando TP3.** Portamos la infraestructura
   de `sia-tp3/shared/` (MLP con backprop, optimizers, activaciones,
   inicializadores, early stopping, métricas) a `shared/` de este repo y
   construimos AE / DAE / VAE encima. Sin PyTorch/TensorFlow, consistente con
   TP3 y TP4. El VAE se implementa con la derivación de backprop de las clases
   teóricas (ver [§7.4](#74-vae)).
2. **Dataset del VAE: emojis** (confirmado) — **16×16, escala de grises,
   ~10–20 emojis**. Reconocibles para juzgar la generación a ojo.
3. **Presentación: Beamer LaTeX** (tema Metropolis), consistente con TP3/TP4.
   Figuras leídas vía `\graphicspath` desde `ejN/results/`.
4. **Formato de configuración: YAML** (`ejN/config.yaml`), una config por
   ejercicio con secciones por bloque (`basic`, `denoising`, `vae`).
5. **Layout `models/` vs `analysis/`** (idéntico a TP4): `models/` implementa
   los algoritmos y **no importa matplotlib**; `analysis/` consume sus salidas y
   produce figuras/métricas; los `main_*.py` orquestan.
6. **Un entrypoint por bloque** (`main_autoencoder.py`, `main_denoising.py`,
   `main_vae.py`). Permite correr y entregar bloques de forma independiente.
7. **`results/` subdividido por bloque**. Cada `main_*.py` escribe sólo a su
   subcarpeta.
8. **Loss del AE básico: MSE** (sale gratis con el `MLP` de TP3). BCE se estudia
   como "técnica de optimización" y requiere un pequeño ajuste en `mlp.py`
   (ver [§4](#4-dependencia-con-tps-anteriores-reuso-de-tp3)).
9. **Sin regularización en el AE básico**: queremos que la red **memorice** los
   32 patrones (overfitting deseado). El término KL del VAE es la única
   regularización "buscada".
10. **EJ1.a es memorización, sin split** (confirmado): las 32 letras son
    train = test; el criterio ≤1 pixel se mide sobre el mismo set (no se puede
    reconstruir una letra que el código 2D nunca vio).
11. **Criterio ≤1 pixel: "por patrón, máx ≤ 1"** (confirmado): cada una de las
    32 letras con ≤1 pixel incorrecto. Ver [§11](#11-criterios-de-éxito-y-métricas).
12. **Latente del VAE: 2D principal + corrida >2D** (confirmado): 2D para
    graficar el manifold; una corrida con 8–16 dims para mejor generación.

---

## 4. Dependencia con TPs anteriores (reuso de TP3)

> El `AGENTS.md` de TP4 fija la convención de **consultar antes de reusar código
> de TPs anteriores**. **Confirmado por el usuario:** se reusa TP3 **siempre que
> sea adecuado para este TP y se incorpore lo nuevo/importante de deep learning**
> (ver [§4.3](#43-lo-nuevo-de-deep-learning-a-incorporar-más-allá-de-tp3)).
> También **confirmado: numpy puro es requisito** (como en todos los TPs previos).

Se **portan** estos módulos de `sia-tp3/shared/` a `sia-tp5/shared/` (copiar,
no importar cruzado entre repos, para mantener TP5 self-contained):

| Módulo TP3 | Qué aporta | Cambios para TP5 |
|------------|-----------|------------------|
| `mlp.py` | `MLP`: `_forward`→`(out,(h_list,V_list))`, `_backward`, `train_epoch(X,t,opt,batch,shuffle,rng)`, `get/set_params`, `save/load`, `predict` | Ver caveat ⚠️ abajo |
| `optimizers.py` | `GradientDescent`, `Momentum`, `Adam`, `build_optimizer(cfg)`; interfaz `step(params, grads)` sobre lista plana | Reusar tal cual (sirve para el VAE) |
| `activations.py` | `activate(h,name,beta)` y `activate_deriv(O,name,beta)` para identity/logistic/tanh/step/relu | Reusar tal cual |
| `initializers.py` | `initialize_layers(arch, method, scale, seed)` con `random_normal`/`he_normal` | **Agregar `xavier`/`glorot`** (importante en DL; ver §4.3) |
| `regularization.py` | `EarlyStopping(patience, min_delta)`; `l2_penalty`, `l2_gradient` | Reusar tal cual |
| `losses.py` | `mse`, `mse_deriv`, `build_loss(name)` | **Agregar `bce`** (loss natural para píxeles binarios; ver §4.3 y caveat) |
| `metrics.py` | matrices de confusión, accuracy, etc. | Agregar `pixel_error` específico del TP |
| `config_loader.py` | `load_yaml(path)` | Reusar tal cual |
| `utils.py` (TP3) / `plotting.py` (TP4) | `save_fig(fig, path, dpi=150)`, paleta, helpers de curvas | Unificar en `shared/plotting.py` |

### ⚠️ Caveat clave: el `_backward` del `MLP` asume MSE

En `mlp.py` el delta de salida está **hardcodeado** como:

```python
delta = (output - t) * activate_deriv(output, self._activations[-1], self.beta)
```

Esto es exactamente `mse_deriv(t, output) · σ'(h)`, **correcto para MSE**. El
autoencoder básico (target = input, loss MSE, salida `logistic`) funciona con el
`MLP` **tal cual**, sin tocar nada.

Si queremos estudiar **BCE** (binary cross-entropy, natural para píxeles en
{0,1}): con salida sigmoide, el delta correcto es `(output - t)` **sin** el
factor `σ'(h)` (se cancela en la derivación BCE+sigmoide). Por lo tanto, para
soportar BCE hay que generalizar `_backward` para usar `self._loss_deriv_fn` y
fusionar el caso sigmoide+BCE. Implementamos **ambas**: MSE (baseline, sale
gratis) y BCE (la loss apropiada para datos binarios). Ver [§4.3](#43-lo-nuevo-de-deep-learning-a-incorporar-más-allá-de-tp3).

### El `MLP` no alcanza para el VAE

El VAE necesita: dos cabezas de salida en el encoder (μ y logσ²), el truco de
reparametrización (capa estocástica) y una loss doble (reconstrucción + KL). El
`train_epoch`/`_backward` del `MLP` no modelan eso. **El VAE se implementa como
clase propia** en `ej2/models/vae.py`, reutilizando `activate`, `activate_deriv`,
`initialize_layers` y los **optimizers** (que operan sobre cualquier lista plana
de parámetros). Ver [§9](#9-plan-ej2--vae-sobre-emojis).

### 4.3 Lo nuevo de Deep Learning a incorporar (más allá de TP3)

El MLP de TP3 resuelve clasificación supervisada; reusarlo es válido, pero hay
cosas **propias de deep learning** que no trae y que sí importan acá. Se
incorporan explícitamente (no se copia TP3 por inercia):

- **Inicialización según la activación.** El `random_normal(scale=0.1)` de TP3
  puede frenar la convergencia. Agregar y usar **Xavier/Glorot** (para
  `tanh`/`logistic`) y **He** (para `relu`, ya está). Es una de las "técnicas de
  optimización" que pide la consigna 1.a.2.
- **BCE como loss para datos binarios.** Los píxeles están en {0,1}: la binary
  cross-entropy con salida sigmoide es la elección natural (gradiente limpio
  `(output - t)`), además del MSE. Implementar ambas y compararlas.
  **Estabilidad numérica:** clipear el argumento del `log` (`eps≈1e-12`) para
  evitar `log(0)`.
- **Tied weights (pesos atados)** — técnica clásica de autoencoders: el decoder
  reusa la traspuesta de los pesos del encoder (`V = Wᵀ`). Menos parámetros,
  regulariza, y es un experimento interesante (atados vs no atados).
- **Estabilidad numérica en general:** sigmoide ya clipeada en TP3; en el VAE,
  usar **log-varianza** (previsto en §7.4) y cuidar el `exp`.
- **Las "técnicas de optimización" de 1.a.2** se barren como ejes DL:
  **inicialización × activación × optimizador × learning rate × loss × batch
  size**. Esa matriz es la respuesta concreta a esa parte de la consigna.
- **(Opcional) Scheduling:** decaimiento del learning rate y **KL warmup**
  (annealing del término KL) en el VAE para estabilizar el entrenamiento.

> En resumen: reusamos la *maquinaria* de TP3 (forward/backward/optimizers),
> pero las *decisiones de diseño* (init, loss, regularización, scheduling) se
> toman con criterio de deep learning.

---

## 5. Estructura del proyecto

```
sia-tp5/
├── AGENTS.md                      # este archivo
├── README.md                      # README público (cómo correr)
├── requirements.txt               # numpy, matplotlib, pyyaml, pillow (+ opc. sklearn/pandas)
├── material/                      # provisto por la cátedra (NO tocar)
│   ├── Enunciado.pdf
│   ├── font.h                     # 32 patrones 7×5 (Font3)
│   ├── fonts.ipynb                # notebook de referencia para decodificar font.h
│   └── clases-teoricas/*.pdf
├── context/                       # teoría destilada (anclas, no desviarse)
│   ├── Autoencoders.md            # AE básico, DAE, regularización, generación
│   └── VAE.md                     # ELBO, KL, reparam, backprop del VAE
├── shared/                        # PORTADO de TP3 + extensiones (ver §4)
│   ├── __init__.py
│   ├── activations.py
│   ├── initializers.py
│   ├── losses.py                  # + bce
│   ├── optimizers.py
│   ├── regularization.py
│   ├── mlp.py
│   ├── metrics.py                 # + pixel_error
│   ├── config_loader.py
│   ├── plotting.py                # save_fig, paleta, helpers
│   └── fonts.py                   # loader de font.h → X (32, 35)
├── ej1/                           # Autoencoders: básico + denoising
│   ├── config.yaml                # secciones: basic, denoising
│   ├── models/
│   │   ├── __init__.py
│   │   └── autoencoder.py         # wrapper sobre MLP: encode() / decode()
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── autoencoder.py         # scatter latente, grilla reconstrucción, letra nueva
│   │   └── denoising.py           # curvas de denoising vs nivel de ruido
│   ├── experiments/               # barridos (ver §10)
│   │   ├── arch_search/           # arquitecturas → ≤1 pixel
│   │   ├── optimizer/             # GD / Momentum / Adam
│   │   ├── learning_rate/
│   │   ├── activation/            # tanh / relu / logistic
│   │   └── noise_levels/          # DAE: barrido de ruido
│   ├── main_autoencoder.py        # entrypoint 1.a
│   ├── main_denoising.py          # entrypoint 1.b
│   └── results/{basic,denoising}/
├── ej2/                           # VAE sobre emojis
│   ├── config.yaml
│   ├── data/
│   │   ├── build_emojis.py        # rasteriza/descarga emojis → .npz
│   │   └── emojis.npz             # dataset generado (gitignore si pesa)
│   ├── models/
│   │   ├── __init__.py
│   │   └── vae.py                 # encoder(μ,logσ²)+reparam+decoder+loss+backward
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── vae.py                 # espacio latente, grilla generativa, interpolación
│   ├── experiments/
│   │   ├── latent_dim/            # 2D vs >2D
│   │   ├── beta/                  # peso del KL (β-VAE)
│   │   └── kl_warmup/             # annealing del KL
│   ├── main_vae.py
│   └── results/
└── presentacion/
    ├── main.tex                   # Beamer Metropolis
    └── README.md                  # cómo compilar
```

---

## 6. El dataset `font.h`

- **32 patrones**, cada uno **7 filas × 5 columnas = 35 píxeles binarios** {0,1}.
- En `font.h` el array relevante es `Font3[32][7]`: cada patrón son **7 bytes**
  (uno por fila); de cada byte interesan los **5 bits menos significativos** (las
  5 columnas). Los 3 bits altos se descartan.
- Caracteres: rango ASCII **0x60–0x7f** →
  `` ` `` `a b c d e f g h i j k l m n o p q r s t u v w x y z { | } ~` y `DEL`.
  Sirven como **labels** para anotar el scatter latente.

### Decodificación (de `material/fonts.ipynb`)

```python
def to_bin_array(encoded_char):           # encoded_char: 7 enteros (filas)
    bin_array = np.zeros((7, 5), dtype=int)
    for row in range(7):
        current = encoded_char[row]
        for col in range(5):
            bin_array[row][4 - col] = current & 1
            current >>= 1
    return bin_array
```

### `shared/fonts.py` (a implementar)

```python
def load_font(flatten=True):
    """Devuelve (X, labels).
    X: (32, 35) float en {0,1} si flatten, o (32, 7, 5) si no.
    labels: lista de 32 nombres de caracter ('`','a',...,'DEL').
    """
```

Notas:
- Embeber `Font3` (copiado de `font.h`) y `to_bin_array` acá; no parsear C.
- El dataset es **chiquito (32×35)**: entra entero en memoria, se entrena a
  **full-batch**, y el "test" para el criterio de ≤1 pixel es el **mismo set de
  entrenamiento** (es un problema de memorización/compresión, no de
  generalización).

---

## 7. Anclas teóricas

Resumen operativo de `material/clases-teoricas/Autoencoders.pdf`. Las fórmulas
están como en las slides; se citan páginas para volver a la fuente. (Detalle
extendido → `context/Autoencoders.md` y `context/VAE.md`.)

### 7.1 Autoencoder básico (pp. 2–6)

Dos MLP acopladas: el **Encoder** comprime y el **Decoder** reconstruye, con la
salida de igual dimensión que la entrada.

$$Z = f(X) = h(XW + b) \qquad X' = g(Z) = h(ZV + b)$$

Se entrena como un MLP normal pero con **target = input**, minimizando

$$L(X, X') = \lVert X - X' \rVert^2 \quad (\text{MSE})$$

o, para píxeles binarios, **binary cross-entropy** (p. 51):

$$H_p(q) = -\tfrac{1}{N}\sum_i \big[ y_i \log p(y_i) + (1-y_i)\log(1-p(y_i)) \big]$$

El **cuello de botella** (capa central angosta, aquí de **2 neuronas**) fuerza
una codificación eficiente. Es la única "restricción" del AE básico.

### 7.2 Espacio latente y relación con PCA (pp. 7–16)

Un **autoencoder lineal equivale a PCA**: el código $Z$ son las proyecciones en
las componentes principales ($T_{PCA}(X) = XE = Z$). Con activaciones **no
lineales**, el AE es una **extensión no lineal de PCA** (p. 16). Útil para el
informe: comparar el scatter latente del AE con una proyección PCA 2D.

### 7.3 Denoising Autoencoder (pp. 20–22)

Se **modela el ruido** y se agrega a la entrada generando $\tilde{X}$, pero **el
target sigue siendo el $X$ limpio**:

> entrada = $\tilde{X}$ (ruidoso), salida esperada = $X$ (limpio).

Tipos de ruido nombrados: **salt-and-pepper**, **Gaussiano**, **Rayleigh**. La
estructura encoder/decoder preserva la información relevante y descarta el ruido.

### 7.4 VAE

Esta es la parte densa. El AE básico tiene un espacio latente "con agujeros":
no todo $z$ decodifica a algo válido (pp. 35–37). El VAE le da **estructura
estadística** al latente para poder samplear.

**Planteo probabilístico (pp. 55–61).** Encoder modela $q_\phi(z\mid x)$,
decoder modela $p_\theta(x\mid z)$. Se asume prior $p(z)=\mathcal{N}(0, I)$ y
posterior aproximado gaussiano $q_\phi(z\mid x)=\mathcal{N}(\mu(x),\Sigma(x))$
con $\Sigma$ diagonal.

**ELBO / Variational Lower Bound (pp. 62–69).**

$$\log p(x) = \underbrace{KL\!\big(q(z)\,\Vert\,p(z\mid x)\big)}_{\ge 0} + \mathcal{L}, \qquad \mathcal{L} \le \log p(x)$$

$$\boxed{\;\mathcal{L} = \mathbb{E}_{q(z)}\!\big[\log p(x\mid z)\big] - KL\!\big(q(z)\,\Vert\,p(z)\big)\;}$$

**Función de costo a minimizar (pp. 73, 78)** = reconstrucción + regularizador KL:

$$-\mathcal{L} = \underbrace{\lVert X - X' \rVert^2}_{\text{reconstrucción}} + \underbrace{KL\big(q(z)\Vert p(z)\big)}_{\text{regularizador}}$$

**KL entre gaussianas, con log-varianza por estabilidad numérica (pp. 76–77).**
La red predice $\log\sigma^2$ (no $\sigma^2$). Con $\Sigma \equiv \log\sigma^2$:

$$\boxed{\;KL = -\tfrac{1}{2}\sum_k \Big(1 + \log\sigma_k^2 - \mu_k^2 - e^{\log\sigma_k^2}\Big)\;}$$

**Truco de reparametrización (pp. 80–83).** Para poder retropropagar a través del
muestreo, se saca el azar a una entrada externa $\epsilon$:

$$\boxed{\;z = \mu(x) + \sigma(x)\odot\epsilon, \qquad \sigma = e^{\frac{1}{2}\log\sigma^2}, \qquad \epsilon\sim\mathcal{N}(0,I)\;}$$

La capa $z$ se comporta como un **perceptrón lineal** (activación identidad).

**Backprop del VAE (pp. 84–89) — receta exacta a implementar:**

1. Hay **dos gradientes**: el de reconstrucción y el de regularización (KL).
2. Los pesos del **DECODER** se actualizan **sólo** con el gradiente de
   reconstrucción (igual que un MLP normal).
3. Los pesos del **ENCODER** se actualizan con **ambos** gradientes.
4. El gradiente de reconstrucción que llega al encoder se obtiene multiplicando
   el gradiente que baja del decoder ($\partial L/\partial z$) por la derivada
   de la reparametrización respecto de $\mu$ y de $\log\sigma^2$:
   - $\dfrac{\partial z}{\partial \mu} = 1$
   - $\dfrac{\partial z}{\partial \log\sigma^2} = \tfrac{1}{2}\,\sigma\odot\epsilon = \tfrac{1}{2}(z-\mu)$
5. El gradiente del KL respecto de las cabezas del encoder (analítico):
   - $\dfrac{\partial KL}{\partial \mu} = \mu$
   - $\dfrac{\partial KL}{\partial \log\sigma^2} = \tfrac{1}{2}\big(e^{\log\sigma^2} - 1\big)$
6. Se **suman** las contribuciones en cada cabeza del encoder:
   - $\delta_\mu = \dfrac{\partial L_{rec}}{\partial z} + \mu$
   - $\delta_{\log\sigma^2} = \dfrac{\partial L_{rec}}{\partial z}\cdot\tfrac{1}{2}(z-\mu) + \tfrac{1}{2}\big(e^{\log\sigma^2}-1\big)$

   y se sigue el backprop estándar hacia atrás por el cuerpo del encoder.

> El término KL **no depende de la salida del decoder**; depende sólo de $\mu$ y
> $\log\sigma^2$ (variables internas). Se calcula analíticamente (p. 82).

**Generación (pp. 80–81).** Se samplea $z\sim\mathcal{N}(0,I)$ y se pasa por el
**Decoder**. El KL es lo que fuerza al latente a ser continuo y muestreable.

### 7.5 Optimización (transversal)

Las slides de AE sólo nombran descenso por gradiente
$\omega^{t+1} = \omega^t - \eta\,\partial J/\partial\omega$ (p. 84). Los
optimizadores concretos (**Adam, Momentum, GD**) los traemos de TP3. Para el AE
de 32 patrones, **Adam + full-batch + muchas épocas** es el caballito de batalla.

---

## 8. Plan EJ1 — Autoencoder básico + Denoising

### 8.1 `ej1/models/autoencoder.py`

Wrapper fino sobre el `MLP` de TP3. La arquitectura es **simétrica** con cuello
de botella 2D, p. ej. `[35, 30, 20, 10, 2, 10, 20, 30, 35]`.

- Entrena como **un solo MLP** con `train_epoch(X, X, optimizer, batch_size=0)`
  (target = input, full-batch).
- Métodos a agregar:
  - `encode(X)` → corre el forward por las capas del encoder y devuelve la
    activación del cuello de botella (índice de la capa de tamaño 2 en
    `V_list`). Se obtiene de `_forward` y se corta `V_list[idx_bottleneck]`.
  - `decode(Z)` → corre el forward por las capas del decoder partiendo de `Z`
    (usa `self.weights[idx:]`, `self.biases[idx:]`, `activate`).
  - `reconstruct(X) = decode(encode(X))`.
- Salida `logistic` (píxeles en (0,1)); binarizar a 0.5 para el conteo de error.

> El bottleneck puede usar activación **lineal/identity** (espacio latente sin
> saturar) o `tanh` (acotado, lindo para graficar). Probar ambas.

### 8.2 `main_autoencoder.py` (consigna 1.a.1–1.a.4)

1. `X, labels = load_font()`.
2. Construir AE con la mejor arquitectura encontrada (ver `experiments/`).
3. Entrenar con Adam + EarlyStopping (monitor = loss de reconstrucción), guardar
   curva de loss.
4. **Métrica ≤1 pixel** (ver [§11](#11-criterios-de-éxito-y-métricas)): reportar
   `max_i` de píxeles incorrectos y cuántas letras quedan exactas / con 1 error.
5. Generar las figuras (delegar a `analysis/autoencoder.py`).

### 8.3 `ej1/analysis/autoencoder.py`

- **Scatter latente 2D** (1.a.3): `encode(X)` → scatter con cada punto anotado
  con su carácter. Comparar opcionalmente con PCA 2D (ancla teórica §7.2).
- **Grilla de reconstrucción**: original vs reconstruido (7×5) para las 32
  letras, resaltando píxeles erróneos.
- **Generación de letra nueva (1.a.4)**: elegir un punto del latente **que no
  corresponde a ninguna letra** (p. ej. el **punto medio entre dos códigos**,
  o un punto interpolado en una recta entre dos letras) y `decode()`-arlo.
  Mostrar el barrido/interpolación. Como el AE básico no regulariza el latente,
  elegir puntos **dentro de la nube** de códigos existentes.

### 8.4 Denoising — `ej1/models/denoising.py` + `main_denoising.py` (1.b)

- La arquitectura puede ser la misma del AE básico (justificar: si comprime y
  reconstruye bien, sólo cambia el objetivo de entrenamiento). Probar también un
  cuello **un poco más ancho** (el denoising no exige 2D).
- **Ruido** (helpers en `models/denoising.py`):
  - *salt-and-pepper*: flip de cada pixel con probabilidad `p`.
  - *gaussiano*: $\tilde{x} = x + \mathcal{N}(0,\sigma^2)$ (clip a [0,1]).
  - *masking*: poner a 0 una fracción de píxeles.
- **Entrenamiento**: cada época, `train_epoch(X_noisy, X_clean, ...)` regenerando
  el ruido (data augmentation online → más robustez).
- **Estudio (1.b.2)**: barrer niveles de ruido `p ∈ {0.05, 0.1, 0.2, 0.3, ...}`,
  medir error de reconstrucción sobre entradas ruidosas; comparar **DAE vs AE
  básico** frente a ruido. Figuras: entrada ruidosa → salida limpia, y curva
  *nivel de ruido vs pixel-error*.

### 8.5 `ej1/config.yaml` (esqueleto)

```yaml
basic:
  architecture: [35, 30, 20, 10, 2, 10, 20, 30, 35]
  hidden_activation: "relu"        # probar tanh / relu / logistic
  output_activation: "logistic"
  bottleneck_activation: "identity"
  loss: "mse"                       # bce = experimento opcional
  optimizer: "adam"
  learning_rate: 0.001
  epochs: 20000
  batch_size: 0                     # full-batch
  seed: 42
  early_stopping: { enabled: true, patience: 2000, min_delta: 1.0e-6 }

denoising:
  architecture: [35, 30, 20, 10, 2, 10, 20, 30, 35]
  noise: { type: "salt_pepper", levels: [0.05, 0.1, 0.2, 0.3] }
  optimizer: "adam"
  learning_rate: 0.001
  epochs: 20000
  seed: 42
```

---

## 9. Plan EJ2 — VAE sobre emojis

### 9.1 Dataset de emojis — `ej2/data/build_emojis.py`

- **Confirmado:** **16×16, escala de grises, ~10–20 emojis** (input_dim = 256).
- **Generar** los bitmaps con **Pillow** rasterizando glifos de una fuente de
  emojis (p. ej. Noto Emoji / Twemoji) a **16×16**, escala de grises,
  normalizados a [0,1]. Guardar `emojis.npz` (X, labels).
- Elegir un set acotado y coherente (~10–20 caras/símbolos). Opcional:
  **augmentation** (pequeños shifts/rotaciones) para tener más muestras.
- Documentar la fuente y la licencia de los emojis en `ej2/data/README.md`.
- *Fallback* si no hay fuente de emojis a mano: dataset de **formas geométricas**
  generadas (círculos/cuadrados/triángulos con variaciones) — mismo pipeline.

### 9.2 `ej2/models/vae.py` (núcleo del TP)

Clase autocontenida (numpy). Reusa `activate`/`activate_deriv`,
`initialize_layers` y un `Adam` de `shared/`. Parámetros como **lista plana**
para usar `optimizer.step(params, grads)`.

**Componentes** (notación de [§7.4](#74-vae)):

- **Encoder body**: MLP `x → h_enc` (1–2 capas ocultas, relu/tanh).
- **Cabezas**: `μ = h_enc·Wμ + bμ`, `logσ² = h_enc·Wlv + blv` (lineales).
- **Reparam**: `z = μ + exp(0.5·logσ²) ⊙ ε`, `ε ~ N(0,I)`.
- **Decoder**: `z → h_dec → x̂` (salida `logistic` o `identity` según se use BCE
  o MSE de reconstrucción).
- **Loss**: `L = L_rec + β·L_KL`, con `L_KL = -½ Σ(1 + logσ² − μ² − exp(logσ²))`.
  `β` controla el peso del KL (β-VAE / annealing).
- **`backward`**: implementar la receta de los 6 pasos de [§7.4](#74-vae):
  decoder sólo con grad de reconstrucción; encoder con `δ_μ` y `δ_logσ²`
  (reconstrucción + KL), y backprop estándar por el cuerpo del encoder.
- **`fit(X, ...)`**: loop de épocas (mini-batch), `optimizer.step` sobre la lista
  de params; trackear `loss`, `loss_rec`, `loss_kl`.
- **`generate(n)` / `decode(z)`**: samplear `z~N(0,I)` → decoder.
- **Sanity check obligatorio**: **gradient check numérico** (diferencias finitas)
  contra el `backward` analítico en una red chica, antes de confiar en el
  entrenamiento. Es el punto más propenso a bugs.

### 9.3 `ej2/analysis/vae.py`

- **Curvas de loss**: total, reconstrucción y KL por separado.
- **Espacio latente** (si latent_dim=2): scatter de `μ(x)` coloreado por clase.
- **Grilla generativa** (2.c): barrer una grilla en el latente 2D (vía la
  inversa de la CDF normal para cubrir la densidad del prior) y decodificar cada
  punto → "mapa" de emojis generados (el plot clásico del manifold del VAE).
- **Muestra nueva juzgable** (2.c): `generate()` y mostrar varias muestras
  sampleadas de $\mathcal{N}(0,I)$.
- **Interpolación**: recorrer una recta entre dos códigos y decodificar
  (morphing de un emoji a otro) — muestra la continuidad del latente.

### 9.4 `ej2/config.yaml` (esqueleto)

```yaml
vae:
  input_dim: 256                    # 16x16 (confirmado)
  encoder_hidden: [256, 64]
  latent_dim: 2                     # corrida principal 2D (manifold); + corrida >2D (8–16) en experiments/latent_dim
  decoder_hidden: [64, 256]
  hidden_activation: "relu"
  output_activation: "logistic"
  recon_loss: "bce"                 # o "mse"
  beta: 1.0                         # peso del KL (β-VAE)
  kl_warmup_epochs: 0               # annealing opcional
  optimizer: "adam"
  learning_rate: 0.001
  epochs: 500
  batch_size: 32
  seed: 42
data:
  path: "ej2/data/emojis.npz"
```

---

## 10. Plan experimental (mapeo a la consigna)

Cada ítem del enunciado se contesta con figuras/tablas concretas. Patrón
`run.py` (computa, pickle de resultados) + `plot.py` (figura), como TP3.

### EJ1.a — Autoencoder básico

| Consigna | Experimento / entregable |
|----------|--------------------------|
| 1.a.1 arquitectura 2D | Diagrama de la arquitectura encoder/decoder elegida |
| 1.a.2 arquitecturas y optimización | `experiments/arch_search/` (profundidad/ancho), `optimizer/` (GD vs Momentum vs Adam), `learning_rate/`, `activation/` (tanh/relu/logistic), MSE vs BCE. Tabla: arquitectura → `max pixel-error`, épocas a converger |
| 1.a.2 (subconjunto) | Si no se logran las 32 con ≤1 pixel: mostrar cuántas sí, qué letras fallan, y argumentar el cuello de botella 2D como límite de capacidad |
| 1.a.3 scatter latente | `analysis`: scatter 2D anotado (+ comparación PCA opcional) |
| 1.a.4 letra nueva | `analysis`: interpolación/punto nuevo en el latente decodificado |

### EJ1.b — Denoising

| Consigna | Experimento / entregable |
|----------|--------------------------|
| 1.b.1 arquitectura | Justificación de la arquitectura (misma del AE o cuello más ancho) |
| 1.b.2 niveles de ruido | `experiments/noise_levels/`: barrido de `p`/`σ`, curva *ruido vs pixel-error*, grilla ruidoso→limpio, comparación DAE vs AE básico |

### EJ2 — VAE

| Consigna | Experimento / entregable |
|----------|--------------------------|
| 2.a dataset | `build_emojis.py` + muestra del dataset |
| 2.b esquema variacional | Curvas loss (rec/KL), scatter latente 2D, gradient-check OK |
| 2.c generación | Grilla generativa (manifold), muestras de `N(0,I)`, interpolación |
| extra | `experiments/`: `latent_dim` (2 vs >2), `beta` (β-VAE), `kl_warmup` |

---

## 11. Criterios de éxito y métricas

`shared/metrics.py` → agregar:

```python
def pixel_errors(X_true, X_pred, threshold=0.5):
    """Píxeles incorrectos por patrón. Devuelve array (N,) de enteros."""
    pred_bin = (X_pred >= threshold).astype(int)
    return np.sum(pred_bin != X_true.astype(int), axis=1)
```

- **Objetivo EJ1.a (interpretación confirmada — "por patrón: máx ≤ 1"):** cada
  uno de los **32 patrones** se reconstruye con **≤ 1 pixel incorrecto** ⇒ éxito
  si `max(pixel_errors(X, reconstruct(X))) ≤ 1`. Reportar también el histograma
  de errores y el promedio.
- **Realismo:** comprimir 32 patrones de 35 dims a **2 dims** con ≤1 pixel es
  **difícil** (es justo lo que el enunciado anticipa). Estrategia: maximizar
  capacidad **fuera** del cuello (capas encoder/decoder anchas), Adam, miles de
  épocas, sin regularización (memorización deseada), probar `bottleneck` lineal
  vs `tanh`. Si no se llega a las 32, entregar el **mejor subconjunto** y la
  explicación cuantitativa.
- **EJ1.b:** pixel-error medio sobre entradas ruidosas, por nivel de ruido.
- **EJ2:** loss de reconstrucción + KL; evaluación **cualitativa** de la
  generación (las muestras "parecen" emojis del set).

---

## 12. Presentación

Beamer LaTeX (tema **Metropolis**), en `presentacion/main.tex`, consistente con
TP3/TP4. Figuras vía `\graphicspath` desde `ej1/results/` y `ej2/results/`.
Compilar con `pdflatex main.tex` (×2). Estructura sugerida:

1. Problema y dataset (`font.h`, decodificación).
2. AE básico: arquitectura, búsqueda (arquitecturas/optimizadores), curva de
   loss, **scatter latente 2D**, criterio ≤1 pixel, generación de letra nueva.
3. Denoising: ruido, barrido de niveles, ejemplos ruidoso→limpio.
4. VAE: motivación (los "agujeros" del latente), ELBO/KL/reparam (1 slide de
   teoría), dataset de emojis, espacio latente, **grilla generativa**,
   interpolación.
5. Conclusiones y limitaciones (cuello 2D, capacidad, etc.).

---

## 13. Entrega

Por Campus (igual que TP3/TP4):
- La **presentación**.
- El **repositorio** con `README.md` y archivo(s) de configuración.
- El **hash del commit** final: `git log --oneline -1`.

---

## 14. Roadmap de trabajo

> **ESTADO: COMPLETADO Y VERIFICADO** (2026-06). Todo implementado en numpy puro,
> ejecutado de verdad y chequeado por verificación adversaria independiente (4/4 OK).
> Resultados reales:
> - **EJ1.a:** `max pixel-error = 0` → **las 32 letras exactas** (objetivo era ≤1).
>   Config: `[35,60,40,20,2,20,40,60,35]`, hidden `tanh`, cuello `identity` 2D, loss
>   `BCE`, Adam. Las 5 seeds probadas logran 32/32. PCA lineal 2D sólo capta 27,1%
>   de la varianza → justifica el AE no lineal. Estudio de optimización
>   (arquitectura/optimizador/lr/activación/MSE-vs-BCE) ejecutado, tabla + figuras.
> - **EJ1.b:** DAE (cuello 8D, ruido online) reduce el pixel-error ~25–250× vs el AE
>   básico frente a salt&pepper/gaussiano/masking; barrido de niveles graficado.
> - **EJ2:** VAE propio (numpy) sobre 960 emojis (16 clases, 16×16 gris desde
>   NotoColorEmoji). **Gradient-check numérico = 1,25e-10** (con "dientes":
>   corromper el backward lo hace fallar). Genera caras/estrellas/lunas reconocibles;
>   manifold 2D suave; interpolación. Experimentos `latent_dim`/`beta`/`kl_warmup`.
> - **Entregables:** `README.md`, `context/{Autoencoders,VAE}.md`, presentación Beamer
>   `presentacion/main.tex` (compila a PDF de 30 páginas con pdflatex), 30 figuras.
> - **Base `shared/`** portada de TP3 + extensiones DL (Glorot, BCE estable, fix del
>   backward MSE/BCE, `pixel_errors`); `tests/smoke_shared.py` con gradient-check.
> - **Batería extendida** (2026-06, densidad estilo TP3/TP4): DAE entrenado a 4
>   niveles de ruido (heatmap train×eval: sweet spot p∈[0.10,0.20], trade-off
>   limpio-vs-robustez con p=0.30 → err limpio 0.56); grilla optimizador×lr
>   (Adam converge max=0 en los 5 lr, Momentum sólo lr≥1e-3, GD sólo 1e-2);
>   multi-seed 5×6 configs con media±std (adam@1e-3 conv 750±184 en 5/5 seeds,
>   gd 4/5; adam@1e-4 2/5); VAE 3 seeds → loss final 118.02±0.14 (<0.3%),
>   seed=42 reproduce exactamente la corrida principal. Todo el código clave
>   comentado con referencias (PDF, slide) a las clases teóricas.

Orden propuesto (cada paso es chico y verificable):

1. **Scaffolding**: crear estructura de carpetas, `requirements.txt`,
   `README.md`, `shared/__init__.py`.
2. **Portar `shared/`** desde TP3 (mlp, optimizers, activations, initializers,
   regularization, losses, metrics, config_loader, plotting). Smoke test.
3. **`shared/fonts.py`**: cargar font.h, verificar visualmente 2–3 letras.
4. **EJ1.a**: `models/autoencoder.py` (encode/decode) → `main_autoencoder.py` →
   primera corrida, medir pixel-error. Iterar arquitectura hasta acercarse a las
   32. Figuras de `analysis/`.
5. **EJ1.a experimentos**: `arch_search`, `optimizer`, `learning_rate`,
   `activation` (responden 1.a.2).
6. **EJ1.b**: ruido + `main_denoising.py` + barrido de niveles.
7. **EJ2 dataset**: `build_emojis.py`.
8. **EJ2 VAE**: `models/vae.py` con **gradient-check** antes de entrenar.
   Entrenar, curvas, scatter latente.
9. **EJ2 generación**: grilla generativa, muestras, interpolación.
10. **EJ2 experimentos**: `latent_dim`, `beta`, `kl_warmup`.
11. **Presentación** Beamer + README final + commit + hash.

---

## 15. Notas para el agente

- **No tocar `material/`** (provisto por la cátedra).
- Priorizar la teoría de `context/` y de `material/clases-teoricas/Autoencoders.pdf`
  sobre implementaciones genéricas de internet. Si se desvía de la teoría,
  **avisar y justificar**.
- Mantener `models/` **sin matplotlib**; toda figura va en `analysis/`.
- **Sanity checks inline** (asserts) en cada `main_*.py`: shapes, rango de
  píxeles, NaN, pixel-error. Para el VAE, **gradient-check numérico** es
  obligatorio antes de confiar en resultados.
- Documentar decisiones e hiperparámetros acá (sección §3) a medida que se toman.
- Determinismo: fijar `seed` y usar `np.random.default_rng(seed)`.
- El dataset de `font.h` es de memorización: el "test" del criterio ≤1 pixel es
  el propio set de entrenamiento (no hay split).
```
