# Autoencoders — Ancla Teórica

> Fuente principal: `material/clases-teoricas/Autoencoders.pdf` (95 diapositivas, 2026).
> Las citas al pie indican número de diapositiva (p. N).

---

## 1. Definición y arquitectura básica (pp. 2–5)

Un **Autoencoder** es una arquitectura de red neuronal **no supervisada** compuesta
por dos MLP acopladas:

- El **Encoder** comprime la entrada $X$ en una representación interna compacta $Z$
  (el *latent code* o código latente).
- El **Decoder** reconstruye la entrada a partir de $Z$, produciendo $X'$ con la
  misma dimensión que la entrada original.

La idea central es: ¿qué pasa si la salida de una MLP se conecta como entrada de
otra MLP inversa? (p. 2)

**Formulación general** (p. 3):

$$Z = f(X) = h(XW + b)$$

$$X' = g(Z) = h(ZV + p)$$

El **aprendizaje** consiste en entrenar la red con cualquier método válido para un
MLP, pero poniendo como **target la propia entrada**: para cada patrón $X$, la
salida esperada es el mismo $X$ (p. 4). La red aprende los pesos sinápticos que
generan $X' \approx X$.

La restricción que hace útil al autoencoder —en lugar de ser trivialmente una
función identidad— es el **cuello de botella**: usar menos neuronas en la capa
central fuerza al modelo a encontrar una codificación más eficiente del conjunto
de datos (p. 6).

---

## 2. Función de costo (loss)

### 2.1 MSE — Error cuadrático medio (p. 3)

La loss por defecto del autoencoder es la norma al cuadrado entre entrada y
reconstrucción:

$$L(X, X') = \lVert X - X' \rVert^2$$

Es la loss natural cuando los datos son continuos. Para el AE básico sobre
`font.h` (píxeles binarios representados como flotantes en $[0,1]$) es la opción
más simple y funciona correctamente con salida `logistic`.

### 2.2 BCE — Binary Cross-Entropy (p. 51)

Para datos binarios en $\{0,1\}$ (como píxeles de imágenes de caracteres), la
loss más apropiada es la **entropía cruzada binaria**:

$$H_p(q) = -\frac{1}{N}\sum_{i=1}^{N} \Big[ y_i \log p(y_i) + (1 - y_i)\log(1 - p(y_i)) \Big]$$

donde $y_i$ es el valor verdadero (0 o 1) y $p(y_i)$ es la salida de la red
(probabilidad predicha, en $(0,1)$ con activación sigmoide).

**Ventaja práctica:** con salida sigmoide, el delta de salida en backprop se
simplifica a $(output - t)$, sin el factor $\sigma'(h)$ (se cancela). El
gradiente es limpio y sin problemas de saturación.

**Estabilidad numérica:** clipear el argumento del $\log$ con `eps ≈ 1e-12` para
evitar $\log(0)$.

---

## 3. El cuello de botella (pp. 5–6)

La arquitectura es **simétrica**: las capas del decoder son el espejo de las del
encoder. La capa central —el cuello de botella— tiene muchas menos neuronas que
la entrada. Ejemplo para `font.h` (35 píxeles de entrada, latente 2D):

```
[35 → 30 → 20 → 10 → 2 → 10 → 20 → 30 → 35]
```

El cuello de botella fuerza al encoder a comprimir la información relevante de
los datos. Es la única restricción del AE básico (sin regularización adicional).

**Activación del bottleneck:** puede ser `identity` (espacio latente sin saturar,
permite valores fuera de $[-1,1]$) o `tanh` (acotado, útil para visualizar el
scatter 2D). Ambas se experimentan.

---

## 4. Relación con PCA — Autoencoder lineal (pp. 7–16)

### 4.1 Descomposición espectral y SVD

Si $X$ es una matriz $n \times d$, la descomposición en valores singulares es:

$$\text{SVD}(X) = \tilde{Z}\,\Sigma\,V^T \qquad (p. 8)$$

El autoencoder lineal aprende minimizando $J = \lVert X - ZV^T \rVert$, de modo
que $X \approx ZV^T$ (p. 9).

### 4.2 Lemma: AE lineal == PCA (pp. 10, 15)

**La salida del código interno $Z$ del autoencoder lineal son las proyecciones de
los datos en los componentes principales:**

$$T_{\text{PCA}}(X) = X\,E = Z \qquad (6)$$

donde $E$ es la matriz de autovectores de la matriz de covarianza $\frac{X^T X}{n-1}$.

La demostración usa que si $V^T$ (pesos del decoder) es ortonormal, entonces
$X \approx \text{SVD}(X) = \tilde{Z}\,\Sigma\,V^T$ con $\tilde{Z}\,\Sigma = Z$
(p. 11), y que la matriz de covarianza satisface $\frac{X^T X}{n-1} = V\Sigma^2 V^T \frac{1}{n-1}$,
que coincide con la descomposición espectral $E\,L\,E^T$ con $V = E$ (pp. 12–14).

### 4.3 Extensión no lineal

Cuando se usan funciones de activación **no lineales** (tanh, relu, logistic), el
autoencoder puede verse como una **extensión no lineal de la descomposición en
componentes principales** (p. 16). Es una asunción fuerte, pero útil para
entender la geometría del espacio latente. Comparar el scatter latente del AE con
una proyección PCA 2D es un experimento válido para el informe.

---

## 5. El autoencoder como herramienta (pp. 17–19)

Usos principales más allá de la compresión:

- **Compresión de información** (uso original, pp. 17–18): codificar datos en
  representaciones de menor dimensión.
- **Detección de outliers** (p. 19): entrenar con el conjunto normal; muestras
  anómalas tendrán alto error de reconstrucción $\lVert x_i - x'_i \rVert$.
- **Inicialización de redes profundas** (uso histórico): preentrenar capas como
  autoencoders apilados.

---

## 6. Denoising Autoencoder (DAE) (pp. 20–22)

### 6.1 Motivación

Como el autoencoder genera una aproximación $X \approx X' = ZV^T$, puede
utilizarse para eliminar ruido sobre la entrada $X$ y recuperar el original en
$X'$ (p. 20). La estructura interna del Encoder/Decoder **preserva la información
más relevante** y descarta el ruido (p. 21).

### 6.2 Procedimiento de entrenamiento (p. 22)

La clave es la distinción entre entrada y target:

> **entrada** = $\tilde{X}$ (versión ruidosa), **target** = $X$ (original limpio)

El ruido se **modela y agrega** a los datos antes de presentarlos a la red. La
salida esperada sigue siendo el dato original sin ruido. Esto obliga al encoder a
aprender representaciones robustas que ignoran las perturbaciones.

**Práctica recomendada:** regenerar el ruido en cada época (data augmentation
online), lo que aumenta la variedad de las perturbaciones vistas y mejora la
robustez.

### 6.3 Tipos de ruido

Mencionados en las slides (p. 22):

- **Salt-and-pepper:** invertir (flip) cada píxel con probabilidad $p$. Parámetro:
  tasa de corrupción $p \in [0,1]$.
- **Gaussiano:** $\tilde{x} = x + \mathcal{N}(0, \sigma^2)$, luego clipear a
  $[0,1]$. Parámetro: desvío estándar $\sigma$.
- **Rayleigh:** distribución de probabilidad alternativa para modelar el ruido.
- **Masking noise** (variante práctica): poner a cero una fracción aleatoria de
  píxeles.

### 6.4 Experimento requerido

Barrer niveles de corrupción $p \in \{0.05, 0.1, 0.2, 0.3, \ldots\}$ y medir el
pixel-error de reconstrucción. Comparar **DAE vs AE básico** frente a las mismas
entradas ruidosas.

---

## 7. Autoencoder generativo y "agujeros" del latente (pp. 27–37)

### 7.1 Modelo generativo vs discriminativo (p. 28)

- **Modelo discriminativo:** separa clases, no se preocupa por cómo se generan
  los datos.
- **Modelo generativo:** hipotetiza cómo se generan los datos; permite **samplear**
  nuevas muestras.

### 7.2 Algoritmo del Generative Autoencoder (pp. 31–34)

1. Entrenar el AE para codificar todos los patrones en el espacio latente.
2. Dejar de lado el encoder.
3. Especificar directamente valores de $z_1, z_2, \ldots$ (o $z_1, z_2$ en 2D).
4. Para cada tupla de valores $z_i$, el decoder genera una nueva muestra.

El espacio latente puede explorarse interpolando entre dos códigos conocidos
(morphing), o eligiendo puntos arbitrarios dentro de la nube de códigos existentes.

### 7.3 El problema de los "agujeros" (pp. 35–37)

El AE básico **no regulariza** el espacio latente. Los códigos de los patrones de
entrenamiento quedan como puntos aislados; entre ellos hay regiones del latente
que no fueron vistas durante el entrenamiento y que el decoder puede mapear a
salidas sin sentido ("agujeros"). Visualización en p. 35: los puntos del latente
están dispersos sin estructura continua.

Para poder **samplear** el espacio latente y obtener muestras válidas, se necesita
**estructura estadística** en el latente: en lugar de un único punto $z$ por
muestra, asociar a cada $x$ una **distribución** sobre $z$ (p. 37). Esa es la
motivación del VAE.

---

## 8. Resumen operativo para la implementación

| Componente | Decisión en TP5 |
|-----------|-----------------|
| Arquitectura | Simétrica, cuello 2D: `[35,30,20,10,2,10,20,30,35]` |
| Activación oculta | `tanh` o `relu` (experimentar ambas) |
| Activación salida | `logistic` (píxeles en $(0,1)$) |
| Activación bottleneck | `identity` (principal) o `tanh` |
| Loss | `mse` (baseline), `bce` (experimento) |
| Optimizador | Adam, full-batch, muchas épocas |
| Criterio éxito | `max(pixel_errors(X, reconstruct(X))) <= 1` |
| DAE: ruido | salt-and-pepper, gaussiano, masking |
| DAE: target | siempre $X$ limpio |
| Generación | decodificar puntos interpolados entre códigos |
