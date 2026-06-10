# VAE — Variational Autoencoder — Ancla Teórica

> Fuente principal: `material/clases-teoricas/Autoencoders.pdf` (pp. 54–89, 2026).
> Las citas al pie indican número de diapositiva (p. N).
> Este documento es la **referencia que usa el implementador de `ej2/models/vae.py`**.

---

## 1. Motivación: el problema de los "agujeros" (pp. 35–37)

El autoencoder básico asigna a cada muestra $x$ un único punto $z$ en el espacio
latente. El decoder no ha sido entrenado en las regiones intermedias del latente,
por lo que samplear $z$ arbitrariamente produce salidas sin sentido. Para generar
muestras válidas se necesita **estructura estadística continua** en el espacio
latente.

La solución: en lugar de mapear $x$ a un punto $z$, mapear $x$ a una
**distribución** sobre $z$. Esto es el VAE (p. 37).

---

## 2. Planteo probabilístico (pp. 55–61)

El VAE combina dos ideas (p. 55):

1. **Inferencia variacional:** aproximar una densidad desconocida $p(z/x)$ con
   una familia de distribuciones controlada $q(z/x)$ resolviendo un problema de
   optimización (p. 58).
2. **Dos redes feedforward acopladas:** un encoder que parametriza $q_\phi(z|x)$
   y un decoder que parametriza $p_\theta(x|z)$.

### 2.1 Distribuciones involucradas (pp. 59–60, 74)

| Símbolo | Nombre | Qué es |
|---------|--------|--------|
| $p_\theta(z)$ | prior | Distribución asumida sobre el latente: $\mathcal{N}(0, \mathcal{I})$ |
| $q_\phi(z\|x)$ | posterior aproximado | Lo que el encoder produce: $\mathcal{N}(\mu(x), \Sigma(x))$ con $\Sigma$ diagonal |
| $p_\theta(x\|z)$ | verosimilitud | Lo que el decoder produce dado un $z$ sampleado |
| $p_\theta(z\|x)$ | posterior verdadero | Desconocido; se aproxima con $q_\phi(z\|x)$ |

El objetivo de la inferencia variacional es aproximar $p(z/x)$ con $q(z/x)$
minimizando su divergencia KL (p. 62):

$$\min_{q} \; KL(q(z) \,\Vert\, p(z/x)) = -\sum q(z) \log \frac{p(z/x)}{q(z)}$$

---

## 3. ELBO — Evidence Lower BOund (pp. 62–69)

### 3.1 Derivación

Expandiendo $KL(q(z) \,\Vert\, p(z/x))$ usando $p(z/x) = p(z,x)/p(x)$
(pp. 63–65):

$$KL(q(z)\,\Vert\,p(z/x)) = -\sum_z q(z) \log \frac{p(z,x)}{q(z)} + \log p(x) \qquad (9)$$

Despejando $\log p(x)$ (p. 65):

$$\log p(x) = KL(q(z)\,\Vert\,p(z/x)) + \mathcal{L} \qquad (10, 11)$$

donde $\mathcal{L}$ es el **Variational Lower Bound** (ELBO):

$$\mathcal{L} = \sum_z q(z) \log \frac{p(z,x)}{q(z)} \le \log p(x) \qquad (12)$$

Como $\log p(x)$ está fijo para un $x$ dado, y $KL \ge 0$, maximizar $\mathcal{L}$
equivale a minimizar el KL entre el posterior aproximado y el verdadero.

### 3.2 Descomposición del ELBO (pp. 68–69)

$$\mathcal{L} = \sum_z q(z) \log \frac{p(x,z)}{q(z)}$$

$$= \sum_z q(z) \log p(x/z) + \sum_z q(z) \log \frac{p(z)}{q(z)}$$

$$= \mathbb{E}_{q(z)} \log p(x/z) - KL(q(z)\,\Vert\,p(z))$$

El ELBO tiene dos términos con interpretación directa:

| Término | Interpretación |
|---------|---------------|
| $\mathbb{E}_{q(z)} \log p(x/z)$ | **Reconstrucción:** qué tan bien el decoder recupera $x$ a partir de $z$ sampleado del encoder |
| $-KL(q(z)\,\Vert\,p(z))$ | **Regularización:** qué tan cerca está el posterior aproximado del prior $\mathcal{N}(0,I)$ |

---

## 4. Función de costo a minimizar (pp. 73, 78)

Maximizar $\mathcal{L}$ equivale a minimizar $-\mathcal{L}$ (p. 73):

$$-\mathcal{L} = \underbrace{-\mathbb{E}_{q(z)} \log p(x/z)}_{\text{error de reconstrucción}} + \underbrace{KL(q(z)\,\Vert\,p(z))}_{\text{regularizador}}$$

Con las suposiciones gaussianas (p. 78) y usando MSE como proxy de $-\log p(x/z)$:

$$\min \mathcal{L} = \lVert \tilde{X} - X' \rVert^2 - \frac{1}{2}\sum_k \Big(1 + \Sigma(x) - (\mu(x))^2 - \exp\Sigma(x)\Big)$$

donde $\Sigma$ ya denota $\log\sigma^2$ (ver §5).

En notación compacta (la que se minimiza en el código):

$$\boxed{J = L_{\text{rec}} + \beta \cdot L_{\text{KL}}}$$

con $\beta = 1$ por defecto ($\beta$-VAE cuando $\beta \ne 1$).

---

## 5. KL entre gaussianas con log-varianza (pp. 76–77)

### 5.1 Forma general

Con $p_\theta(z) = \mathcal{N}(0, \mathcal{I})$ y $q_\phi(z) = \mathcal{N}(\mu(x), \Sigma(x))$
diagonal (p. 76):

$$KL = \frac{1}{2}\sum_k \Big(\Sigma_k(x) + (\mu_k(x))^2 - 1 - \log\Sigma_k(x)\Big)$$

### 5.2 Sustitución log-varianza (p. 77)

Por **estabilidad numérica**, la red predice $\log\sigma^2$ en lugar de $\sigma^2$.
Se reemplaza $\Sigma(x) \leftarrow \exp(\Sigma(x))$, de modo que $\Sigma$ pasa a
denotar $\log\sigma^2$:

$$\boxed{KL = -\frac{1}{2}\sum_k \Big(1 + \log\sigma_k^2 - \mu_k^2 - \exp(\log\sigma_k^2)\Big)}$$

Esta es la fórmula que se implementa directamente. Notar el signo: como $-\mathcal{L}$
es lo que se minimiza, el KL aparece sumado positivamente.

**Por qué log-varianza y no varianza directamente:** $\exp(\cdot)$ siempre es
positivo, por lo que la red puede producir cualquier valor real para $\log\sigma^2$
sin restricciones (no hay riesgo de predecir $\sigma^2 < 0$). Es más estable
numéricamente que hacer clipeo.

---

## 6. Truco de reparametrización (pp. 80–83, 45)

### 6.1 El problema

El muestreo $z \sim q_\phi(z|x) = \mathcal{N}(\mu, \sigma^2)$ es un nodo
estocástico: no se puede retropropagar el gradiente a través de una operación de
sampleo.

### 6.2 La solución (p. 80)

Se saca el azar a una **variable externa** $\epsilon \sim \mathcal{N}(0, I)$ y
se expresa $z$ como función determinista de $\mu$, $\log\sigma^2$ y $\epsilon$:

$$\boxed{z = \mu(x) + \sigma(x) \odot \epsilon, \qquad \sigma = e^{\frac{1}{2}\log\sigma^2}, \qquad \epsilon \sim \mathcal{N}(0, I)}$$

En notación de las slides (p. 45, 80):

$$z = h(\tilde{X}) = \epsilon \odot \tilde{\Sigma}(\tilde{X}) + \tilde{\mu}(\tilde{X}) \qquad (8, 14)$$

### 6.3 Consecuencia para la arquitectura (p. 83)

La capa $z$ se comporta como un **perceptrón lineal con activación identidad**:
es una combinación lineal de $\mu$ y $\sigma \odot \epsilon$, y el gradiente fluye
normalmente hacia $\mu$ y $\log\sigma^2$ a través de ella.

Las dos salidas del encoder ($\mu$ y $\log\sigma^2$) actúan como **entradas** a
la capa estocástica. En el grafo de cómputo reparametrizado, la diferenciación es
posible (p. 83):

```
x → [encoder body] → μ, logσ²
                        ↓
ε ~ N(0,I) ──────→  z = μ + exp(0.5·logσ²)·ε  → [decoder] → x̂
```

---

## 7. Backprop del VAE — Receta exacta (pp. 84–89)

Esta es la sección crítica para la implementación. El diagrama de cómputo
(p. 82) muestra dos flujos de gradiente: uno por el decoder (reconstrucción) y
uno analítico (KL).

### Paso 1: dos gradientes

Hay **dos gradientes** a retropropagar (p. 88):
- $\partial L_{\text{rec}} / \partial z$: gradiente de reconstrucción (sube desde la loss del decoder).
- $\partial L_{\text{KL}} / \partial \mu$ y $\partial L_{\text{KL}} / \partial \log\sigma^2$: gradiente de regularización (calculado analíticamente).

### Paso 2: backprop del decoder

Los pesos del **DECODER** se actualizan **exclusivamente** con el gradiente de
reconstrucción (pp. 88, 89). El backprop del decoder es idéntico al de un MLP
normal: se calcula el delta de salida a partir de la loss de reconstrucción y se
retropropaga hacia atrás por las capas del decoder. Al llegar a $z$ se obtiene
$\partial L_{\text{rec}} / \partial z$.

### Paso 3: backprop del encoder — contribución de reconstrucción

El gradiente de reconstrucción que llega al encoder se obtiene multiplicando
$\partial L_{\text{rec}} / \partial z$ por las derivadas del truco de
reparametrización respecto de las dos cabezas del encoder (p. 86):

$$\frac{\partial z}{\partial \mu} = 1$$

$$\frac{\partial z}{\partial \log\sigma^2} = \frac{1}{2}\,\sigma \odot \epsilon = \frac{1}{2}(z - \mu)$$

### Paso 4: gradientes del KL respecto de las cabezas del encoder

Se calculan **analíticamente** a partir de la fórmula del KL (p. 87):

$$\frac{\partial KL}{\partial \mu} = \mu$$

$$\frac{\partial KL}{\partial \log\sigma^2} = \frac{1}{2}\Big(\exp(\log\sigma^2) - 1\Big)$$

El término KL **no depende de la salida del decoder** $\hat{x}$: sólo depende de
$\mu$ y $\log\sigma^2$, variables internas del encoder (p. 82).

### Paso 5 y 6: sumar contribuciones en cada cabeza (p. 89)

Los deltas finales para cada cabeza del encoder son la **suma** de la contribución
de reconstrucción y la del KL:

$$\boxed{\delta_\mu = \frac{\partial L_{\text{rec}}}{\partial z} \cdot 1 + \mu}$$

$$\boxed{\delta_{\log\sigma^2} = \frac{\partial L_{\text{rec}}}{\partial z} \cdot \frac{1}{2}(z - \mu) + \frac{1}{2}\Big(\exp(\log\sigma^2) - 1\Big)}$$

Con estos deltas se continúa el backprop estándar hacia atrás por el **cuerpo del
encoder** (las capas compartidas antes de las dos cabezas).

### Resumen en 6 pasos

| Paso | Qué se hace | Quién recibe el grad |
|------|-------------|---------------------|
| 1 | Forward completo: encoder → reparam → decoder | — |
| 2 | Calcular $L_{\text{rec}}$ y $L_{\text{KL}}$ | — |
| 3 | Backprop decoder con $\partial L_{\text{rec}}$; obtener $\partial L_{\text{rec}}/\partial z$ | Pesos del decoder |
| 4 | Calcular $\partial KL / \partial \mu$ y $\partial KL / \partial \log\sigma^2$ analíticamente | — |
| 5 | Formar $\delta_\mu$ y $\delta_{\log\sigma^2}$ (suma de pasos 3 y 4) | Cabezas del encoder |
| 6 | Backprop estándar por cuerpo del encoder con $\delta_\mu$ y $\delta_{\log\sigma^2}$ | Pesos del encoder |

---

## 8. Generación de muestras nuevas (pp. 80–81)

Una vez entrenado el VAE, para **generar** una muestra nueva:

1. Samplear $z \sim \mathcal{N}(0, I)$ directamente del prior.
2. Pasar $z$ por el **decoder** para obtener $\hat{x}$.

El término KL es el que **fuerza al espacio latente a ser continuo y muestreable**:
penaliza que $q_\phi(z|x)$ se aleje de $\mathcal{N}(0, I)$, lo que hace que el
decoder aprenda a decodificar correctamente cualquier $z$ en la región de alta
densidad del prior. Sin el KL, el latente tendría "agujeros" como el AE básico.

**Interpolación:** dado un punto $z_A = \mu(x_A)$ y $z_B = \mu(x_B)$, recorrer
la recta $z(t) = (1-t)\,z_A + t\,z_B$ para $t \in [0,1]$ y decodificar cada
punto muestra el morphing continuo entre dos muestras.

**Grilla generativa (manifold plot):** para latente 2D, barrer una grilla
$z_1 \times z_2$ usando cuantiles de la $\mathcal{N}(0,1)$ como valores de los
ejes (para cubrir la densidad del prior uniformemente), y decodificar cada punto.

---

## 9. Notas de implementación para `ej2/models/vae.py`

### 9.1 Arquitectura

```
Encoder body:  x (input_dim) → [capas ocultas relu/tanh] → h_enc
Cabeza μ:      h_enc → μ  (lineal, sin activación)
Cabeza logσ²:  h_enc → logσ²  (lineal, sin activación)
Reparam:       z = μ + exp(0.5·logσ²) ⊙ ε,  ε ~ N(0,I)
Decoder:       z (latent_dim) → [capas ocultas relu/tanh] → x̂  (logistic/identity)
```

### 9.2 Estabilidad numérica

- Usar `log_var` (log-varianza) como salida del encoder, nunca $\sigma^2$ directamente.
- Clipear `log_var` en un rango razonable (p. ej. $[-10, 10]$) para evitar
  `exp` con valores extremos.
- En BCE: clipear $p(y_i)$ a $[\epsilon, 1-\epsilon]$ con $\epsilon \approx 10^{-12}$.

### 9.3 Verificación obligatoria: gradient check numérico

Antes de confiar en el entrenamiento, verificar el backprop analítico contra
diferencias finitas en una red pequeña (input_dim=4, latent_dim=2, 1 capa oculta).
Para cada parámetro $\omega_i$:

$$\frac{\partial J}{\partial \omega_i} \approx \frac{J(\omega_i + h) - J(\omega_i - h)}{2h}, \qquad h = 10^{-5}$$

La diferencia relativa debe ser $< 10^{-5}$ para considerar el backprop correcto.

### 9.4 Hiperparámetros clave

| Parámetro | Valor por defecto | Efecto |
|-----------|------------------|--------|
| `latent_dim` | 2 (principal), 8–16 (experimento) | Capacidad del latente |
| `beta` | 1.0 | Peso del KL ($\beta$-VAE) |
| `kl_warmup_epochs` | 0 | Annealing: aumentar $\beta$ gradualmente |
| `recon_loss` | `bce` (imágenes binarias) | Loss de reconstrucción |
| `optimizer` | Adam, lr=1e-3 | Optimizador |

### 9.5 Interfaz de parámetros

Para usar los optimizadores de `shared/` con `optimizer.step(params, grads)`, los
parámetros del VAE se exponen como **lista plana**:

```
params = [W_enc_0, b_enc_0, ..., W_mu, b_mu, W_lv, b_lv,
          W_dec_0, b_dec_0, ..., W_dec_L, b_dec_L]
```

Los gradientes tienen la misma estructura. Esta es la interfaz que espera `Adam.step`.
