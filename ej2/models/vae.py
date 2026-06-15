"""VAE (Variational Autoencoder) en numpy puro — núcleo del TP5 EJ2.

Referencias de teoría (Autoencoders.pdf; el Repaso Capsule.pdf repite el truco
de reparametrización, BCE y KL en sus slides 12, 18 y 19-20):
  - slides 55-71: derivación variacional. Maximizar el ELBO equivale a
    minimizar  -L = -E_q log p(x/z) + KL(q(z)||p(z))  (slides 69 y 73):
    error de reconstrucción + término regularizador.
  - slides 72-74: arquitectura. El Encoder produce los PARÁMETROS (mu, Sigma)
    de q(z/x); se asume p(z) = N(0, I) y q = N(mu(x), Sigma(x)) diagonal.
  - slides 45, 80-83: reparametrization trick  z = eps ⊙ Sigma(x) + mu(x)
    (Eq. 8/14): el sampleo se mueve a la entrada exógena eps ~ N(0,I) para
    poder retropropagar; la capa z actúa como un perceptrón lineal con
    activación identidad (slide 83).
  - slides 76-77: KL en forma cerrada. KL = 1/2 Σ (Sigma + mu² - 1 - log Sigma)
    y, "por razones numéricas, se reemplaza Sigma por exp(Sigma)" (slide 77):
    la red emite LOGVAR y queda KL = 1/2 Σ (exp(logvar) + mu² - 1 - logvar)
    = -1/2 Σ (1 + logvar - mu² - exp(logvar)).
  - slide 78: loss final  min L = ||X - X'|| + 1/2 Σ(...)  con J = L + lambda*R
    (Eq. 13); nuestro `beta` es ese lambda (beta-VAE).
  - slides 84-89: retropropagación por el nodo estocástico (ver backward()).

El MLP de shared/ NO sirve para el VAE: necesita dos cabezas (mu, logsigma^2),
el truco de reparametrización y una loss doble (reconstrucción + KL). Por eso
implementamos una clase propia, reusando de shared/:
  - activate / activate_deriv   (activaciones e identidades de derivadas)
  - initialize_layers           (Glorot/He según activación)
  - Adam                        (optimizer sobre lista plana de params)

Convención de parámetros (IMPORTANTE para optimizer.step(params, grads)):
  Mantenemos los parámetros como una LISTA PLANA en este orden fijo:
     [ encoder body: W0,b0, W1,b1, ...                 ]
     [ cabeza mu:     Wmu, bmu                          ]
     [ cabeza logvar: Wlv, blv                          ]
     [ decoder:       Wd0,bd0, Wd1,bd1, ..., Wout,bout  ]
  get_flat_params()/set_flat_params() exponen exactamente esa lista, y
  backward(...) devuelve los gradientes en EL MISMO orden, de modo que
  Adam.step(params, grads) funciona directo.

Convención de escala del gradiente (igual que el MLP de shared):
  Los deltas se calculan POR MUESTRA (sin el 1/N). TODOS los gradientes de
  parámetros se dividen por N (batch size) al final. beta multiplica las
  contribuciones del KL.

Arquitectura (notación AGENTS §7.4):
  Encoder body:  x (D) -> capas ocultas -> h_enc
  Cabezas LINEALES (identity):
      mu     = h_enc @ Wmu.T + bmu
      logvar = h_enc @ Wlv.T + blv
  Reparam:  z = mu + exp(0.5*logvar) * eps,  eps ~ N(0,I)
  Decoder:  z -> capas ocultas -> x_hat   (salida 'logistic' si recon='bce')

Loss (ambos PROMEDIO sobre el batch):
  L = L_rec + beta * L_KL
  L_rec por muestra = sum_features BCE(x, x_hat)   (o 0.5*||x-x_hat||^2 si mse)
  L_KL por muestra  = -0.5 * sum_k (1 + logvar_k - mu_k^2 - exp(logvar_k))

models/ NO importa matplotlib (convención TP3/TP4).
"""

import os

import numpy as np

from shared.activations import activate, activate_deriv
from shared.initializers import initialize_layers


_BCE_EPS = 1e-12


class VAE:
    def __init__(
        self,
        input_dim,
        encoder_hidden,
        latent_dim,
        decoder_hidden,
        hidden_activation="relu",
        output_activation="logistic",
        recon_loss="bce",
        beta=1.0,
        initializer=None,
        init_scale=0.1,
        seed=42,
    ):
        self.input_dim = int(input_dim)
        self.encoder_hidden = [int(n) for n in encoder_hidden]
        self.latent_dim = int(latent_dim)
        self.decoder_hidden = [int(n) for n in decoder_hidden]
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.recon_loss = recon_loss
        self.beta = float(beta)
        self.seed = int(seed)

        if recon_loss not in ("bce", "mse"):
            raise ValueError(f"recon_loss desconocida: {recon_loss}")
        if recon_loss == "bce" and output_activation != "logistic":
            # BCE requiere salida sigmoide para el gradiente limpio (x_hat - x).
            raise ValueError("recon_loss='bce' requiere output_activation='logistic'")

        # Inicializador por defecto según activación (DL: Glorot/He).
        if initializer is None:
            initializer = "he_normal" if hidden_activation == "relu" else "glorot"
        self.initializer = initializer

        # --- Encoder body: input_dim -> encoder_hidden[-1] ---
        enc_arch = [self.input_dim] + self.encoder_hidden
        # --- Decoder: latent_dim -> decoder_hidden -> input_dim ---
        dec_arch = [self.latent_dim] + self.decoder_hidden + [self.input_dim]

        # h_enc dimensión (entrada de las cabezas)
        self.h_enc_dim = enc_arch[-1] if len(self.encoder_hidden) > 0 else self.input_dim

        rng_seed = np.random.SeedSequence(self.seed)
        s_enc, s_mu, s_lv, s_dec = rng_seed.spawn(4)

        # Pesos del cuerpo del encoder (lista de (W,b))
        if len(self.encoder_hidden) > 0:
            self._enc = initialize_layers(
                enc_arch, method=initializer, scale=init_scale,
                seed=int(s_enc.generate_state(1)[0]),
            )
        else:
            self._enc = []  # encoder body vacío: h_enc = x

        # Cabezas (lineales). Glorot porque son identity.
        self.Wmu, self.bmu = self._init_head(self.h_enc_dim, self.latent_dim, s_mu, init_scale)
        self.Wlv, self.blv = self._init_head(self.h_enc_dim, self.latent_dim, s_lv, init_scale)

        # Decoder
        self._dec = initialize_layers(
            dec_arch, method=initializer, scale=init_scale,
            seed=int(s_dec.generate_state(1)[0]),
        )

        # Activaciones por capa
        self._enc_acts = [hidden_activation] * len(self._enc)
        # Decoder: ocultas con hidden_activation; última con output_activation
        self._dec_acts = [hidden_activation] * len(self.decoder_hidden) + [output_activation]

        # Vistas a las listas para get/set flat params (orden canónico)
        self.n_enc = len(self._enc)
        self.n_dec = len(self._dec)

    @staticmethod
    def _init_head(n_in, n_out, seedseq, scale):
        params = initialize_layers(
            [n_in, n_out], method="glorot", scale=scale,
            seed=int(seedseq.generate_state(1)[0]),
        )
        W, b = params[0]
        return W.copy(), b.copy()

    # ================================================================ params
    def get_flat_params(self):
        """Lista plana en orden canónico [enc..., mu, lv, dec...]."""
        params = []
        for (W, b) in self._enc:
            params.append(W)
            params.append(b)
        params.append(self.Wmu)
        params.append(self.bmu)
        params.append(self.Wlv)
        params.append(self.blv)
        for (W, b) in self._dec:
            params.append(W)
            params.append(b)
        return params

    def set_flat_params(self, params):
        """Inverso de get_flat_params (reasigna in-place desde lista plana)."""
        i = 0
        new_enc = []
        for _ in range(self.n_enc):
            new_enc.append((np.asarray(params[i]), np.asarray(params[i + 1])))
            i += 2
        self._enc = new_enc
        self.Wmu = np.asarray(params[i]); i += 1
        self.bmu = np.asarray(params[i]); i += 1
        self.Wlv = np.asarray(params[i]); i += 1
        self.blv = np.asarray(params[i]); i += 1
        new_dec = []
        for _ in range(self.n_dec):
            new_dec.append((np.asarray(params[i]), np.asarray(params[i + 1])))
            i += 2
        self._dec = new_dec

    # =============================================================== forward
    def _encoder_body(self, X):
        """x -> h_enc. Devuelve (h_enc, cache) con pre/post activaciones."""
        V = X
        Vs = [X]   # Vs[0] = input, Vs[i+1] = salida capa i
        for i, (W, b) in enumerate(self._enc):
            h = V @ W.T + b
            V = activate(h, self._enc_acts[i], 1.0)
            Vs.append(V)
        return V, Vs

    def _decoder(self, Z):
        """z -> x_hat. Devuelve (x_hat, Vs) con activaciones por capa."""
        V = Z
        Vs = [Z]
        for i, (W, b) in enumerate(self._dec):
            h = V @ W.T + b
            V = activate(h, self._dec_acts[i], 1.0)
            Vs.append(V)
        return V, Vs

    def forward(self, X, eps=None, rng=None, sample=True):
        """Forward completo. Si eps se pasa, se usa fijo (para gradcheck).

        sample=False usa z=mu (modo determinístico, para encode/predict).
        Devuelve (x_hat, cache) con todo lo necesario para backward.
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        N = X.shape[0]

        # Encoder: produce los PARÁMETROS de q(z/x) = N(mu, diag(sigma²))
        # (Autoencoders.pdf, slide 72). Las cabezas son lineales (identity):
        # la capa z actúa como perceptrón lineal (slide 83).
        h_enc, enc_Vs = self._encoder_body(X)
        mu = h_enc @ self.Wmu.T + self.bmu
        # La red emite logvar = log(sigma²), no sigma: slide 77 ("por razones
        # numéricas se reemplaza Sigma por exp(Sigma)") — garantiza sigma² > 0.
        logvar = h_enc @ self.Wlv.T + self.blv
        logvar = np.clip(logvar, -30.0, 30.0)  # estabilidad del exp
        std = np.exp(0.5 * logvar)

        if sample:
            if eps is None:
                if rng is None:
                    rng = np.random.default_rng()
                eps = rng.standard_normal(size=mu.shape)
            # Reparametrization trick (slides 45 y 80-83, Eq. 8/14):
            #   z = mu + sigma ⊙ eps,  eps ~ N(0, I)
            # Samplear z~N(mu,sigma²) directo NO es derivable; con eps como
            # entrada exógena, z es función determinística de (mu, logvar) y
            # el error puede retropropagarse hacia el encoder (slide 84).
            z = mu + std * eps
        else:
            eps = np.zeros_like(mu)
            z = mu

        x_hat, dec_Vs = self._decoder(z)

        cache = {
            "X": X, "N": N,
            "enc_Vs": enc_Vs, "h_enc": h_enc,
            "mu": mu, "logvar": logvar, "std": std, "eps": eps, "z": z,
            "dec_Vs": dec_Vs, "x_hat": x_hat,
        }
        return x_hat, cache

    # ================================================================= loss
    def loss(self, cache, beta=None):
        """Devuelve (L_total, L_rec, L_KL), todos PROMEDIO sobre el batch.

        -ELBO = error de reconstrucción + término regularizador KL
        (Autoencoders.pdf, slides 73 y 78). Minimizar esto = maximizar el
        Variational Lower Bound L <= log p(x) (slide 67).
        """
        if beta is None:
            beta = self.beta
        X = cache["X"]; x_hat = cache["x_hat"]
        mu = cache["mu"]; logvar = cache["logvar"]
        N = cache["N"]

        # Error de reconstrucción = -E_q log p(x/z) (slides 73 y 75):
        #  - 'bce': píxeles ~ Bernoulli(x_hat) -> -log p(x/z) es la BCE
        #    (fórmula: slide 51), sumada sobre features.
        #  - 'mse': p(x/z) gaussiana -> -log p(x/z) ∝ ||x - x_hat||² (slide 75).
        if self.recon_loss == "bce":
            p = np.clip(x_hat, _BCE_EPS, 1.0 - _BCE_EPS)
            rec_per = -np.sum(X * np.log(p) + (1.0 - X) * np.log(1.0 - p), axis=1)
        else:  # mse
            rec_per = 0.5 * np.sum((X - x_hat) ** 2, axis=1)
        L_rec = float(np.mean(rec_per))

        # KL(q||p) en forma cerrada con la parametrización logvar (slides
        # 76-77): KL = -1/2 Σ_k (1 + logvar - mu² - exp(logvar)). Tiende a 0
        # cuando q(z/x) -> N(0, I) = p(z) (slide 53: KL(q||q) = 0).
        kl_per = -0.5 * np.sum(1.0 + logvar - mu ** 2 - np.exp(logvar), axis=1)
        L_KL = float(np.mean(kl_per))

        # J = L + lambda*R (slide 78, Eq. 13): beta pondera el regularizador.
        L_total = L_rec + beta * L_KL
        return L_total, L_rec, L_KL

    # ============================================================= backward
    def backward(self, cache, beta=None):
        """Gradientes analíticos en orden canónico (misma lista que params).

        Retropropagación por el nodo estocástico (Autoencoders.pdf, slides
        84-89). Resumen de la receta de las slides 88-89:
          1. Hay DOS gradientes a retropropagar: el de reconstrucción y el de
             regularización (KL).
          2. Los pesos del DECODER se actualizan EXCLUSIVAMENTE con el
             gradiente de reconstrucción ("igual a un perceptrón multicapa",
             slide 88). El KL no depende de la salida del decoder: depende
             sólo de mu y Sigma (slide 82).
          3. Los gradientes de reconstrucción del ENCODER = gradiente que
             viene del decoder (dL/dz) por la derivada del truco de
             reparametrización respecto de mu y de la varianza (slide 86-87:
             dz/dmu = 1, dz/dsigma = eps).
          4. Los gradientes del regularizador del ENCODER = derivadas de la
             KL respecto de mu y la varianza (slide 89).
          5. Ambas contribuciones SE SUMAN en cada paso del encoder (slide 89).

        Receta AGENTS §7.4: deltas por muestra (sin 1/N), grads /N al final.
        beta escala las contribuciones del KL. Verificado contra diferencias
        finitas centrales en gradcheck() (error relativo ~1e-10).
        """
        if beta is None:
            beta = self.beta
        X = cache["X"]; N = cache["N"]
        mu = cache["mu"]; logvar = cache["logvar"]
        std = cache["std"]; eps = cache["eps"]; z = cache["z"]
        h_enc = cache["h_enc"]
        enc_Vs = cache["enc_Vs"]; dec_Vs = cache["dec_Vs"]
        x_hat = cache["x_hat"]

        grads = [None] * (2 * self.n_enc + 4 + 2 * self.n_dec)

        # --------------------------------------------------------------------
        # 1) DECODER = MLP normal (slide 88: "igual a un perceptrón multicapa",
        #    sólo gradiente de reconstrucción). delta de salida (por muestra).
        # --------------------------------------------------------------------
        if self.recon_loss == "bce":
            # bce + sigmoide -> delta_out = (x_hat - x): el sigma'(h) se
            # cancela con el denominador de dBCE/dy (BCE: slide 51).
            delta = (x_hat - X)
        else:  # mse
            # 0.5||x-x_hat||^2 -> dL/dx_hat = (x_hat - x); * sigma'(h) si logistic
            delta = (x_hat - X)
            if self.output_activation != "identity":
                delta = delta * activate_deriv(x_hat, self.output_activation, 1.0)

        dec_grads = [None] * (2 * self.n_dec)
        # Backprop estándar por el decoder.
        for li in reversed(range(self.n_dec)):
            V_in = dec_Vs[li]          # entrada a la capa li
            W, b = self._dec[li]
            gW = (delta.T @ V_in) / N
            gb = np.sum(delta, axis=0) / N
            dec_grads[2 * li] = gW
            dec_grads[2 * li + 1] = gb
            if li > 0:
                # propagar a la capa anterior, aplicar deriv de su activación
                V_prev = dec_Vs[li]    # salida de capa li-1 == entrada capa li
                delta = (delta @ W) * activate_deriv(V_prev, self._dec_acts[li - 1], 1.0)
            else:
                # delta llega a z: dL_rec/dz — el "dL/dz que corresponde a la
                # retropropagación del error en el decoder" (slide 86), listo
                # para entrar al nodo de reparametrización.
                dL_rec_dz = delta @ W

        # 2) El decoder se actualiza SOLO con grad de reconstrucción (ya hecho,
        #    slide 88). El KL no aporta nada al decoder (slide 82).

        # --------------------------------------------------------------------
        # 3) Cabezas del encoder: reconstrucción + KL, por muestra (slides
        #    86-87 y 89). Derivación con z = mu + exp(0.5*logvar)*eps:
        #
        #    Rama mu (slide 86):   dz/dmu = 1
        #      delta_mu = dL_rec/dz * 1 + beta * dKL/dmu,   dKL/dmu = mu
        #      (dKL/dmu sale de derivar 1/2*Σ(mu²+...) de la slide 77).
        #
        #    Rama varianza (slide 87): en la slide, dz/dsigma = eps porque ahí
        #      z = eps⊙Sigma + mu con Sigma = desvío. Nosotros emitimos LOGVAR
        #      (slide 77), así que la regla de la cadena da:
        #        dz/dlogvar = eps * d(exp(0.5*logvar))/dlogvar
        #                   = eps * 0.5*exp(0.5*logvar) = 0.5*(z - mu)
        #      y dKL/dlogvar = 0.5*(exp(logvar) - 1)  (derivada de slide 77).
        # --------------------------------------------------------------------
        # delta_mu     = dL_rec/dz * 1          + beta * mu
        # delta_logvar = dL_rec/dz * 0.5*(z-mu) + beta * 0.5*(exp(logvar) - 1)
        delta_mu = dL_rec_dz + beta * mu
        delta_lv = dL_rec_dz * (0.5 * (z - mu)) + beta * 0.5 * (np.exp(logvar) - 1.0)

        # --------------------------------------------------------------------
        # 4) Backprop por las cabezas y el cuerpo del encoder, SUMANDO las dos
        #    ramas (slide 89, punto 5: "se suman estos valores anteriores en
        #    cada paso del ENCODER").
        # --------------------------------------------------------------------
        gWmu = (delta_mu.T @ h_enc) / N
        gbmu = np.sum(delta_mu, axis=0) / N
        gWlv = (delta_lv.T @ h_enc) / N
        gblv = np.sum(delta_lv, axis=0) / N

        # delta que baja a h_enc desde ambas cabezas (rec+KL ya mezclados)
        dL_dh = delta_mu @ self.Wmu + delta_lv @ self.Wlv

        enc_grads = [None] * (2 * self.n_enc)
        if self.n_enc > 0:
            # aplicar la derivada de la activación del cuerpo del encoder (última capa)
            delta = dL_dh * activate_deriv(enc_Vs[self.n_enc], self._enc_acts[self.n_enc - 1], 1.0)
            for li in reversed(range(self.n_enc)):
                V_in = enc_Vs[li]
                W, b = self._enc[li]
                gW = (delta.T @ V_in) / N
                gb = np.sum(delta, axis=0) / N
                enc_grads[2 * li] = gW
                enc_grads[2 * li + 1] = gb
                if li > 0:
                    V_prev = enc_Vs[li]
                    delta = (delta @ W) * activate_deriv(V_prev, self._enc_acts[li - 1], 1.0)

        # Ensamblar en orden canónico
        i = 0
        for li in range(self.n_enc):
            grads[i] = enc_grads[2 * li]; i += 1
            grads[i] = enc_grads[2 * li + 1]; i += 1
        grads[i] = gWmu; i += 1
        grads[i] = gbmu; i += 1
        grads[i] = gWlv; i += 1
        grads[i] = gblv; i += 1
        for li in range(self.n_dec):
            grads[i] = dec_grads[2 * li]; i += 1
            grads[i] = dec_grads[2 * li + 1]; i += 1

        return grads

    # ============================================================ training
    def fit(self, X, optimizer, epochs=500, batch_size=32, beta=None,
            beta_schedule=None, seed=42, verbose=False, log_every=50):
        """Entrena con mini-batch + Adam. Trackea loss total/rec/KL por época.

        beta_schedule: opcional, función epoch -> beta (KL warmup/annealing).
                       Si se pasa, tiene prioridad sobre `beta`.
        Devuelve dict de historiales {total, rec, kl, beta}.
        """
        X = np.asarray(X, dtype=np.float64)
        N = X.shape[0]
        rng = np.random.default_rng(seed)
        bs = N if (batch_size is None or batch_size <= 0) else int(batch_size)

        hist = {"total": [], "rec": [], "kl": [], "beta": []}
        for ep in range(epochs):
            if beta_schedule is not None:
                cur_beta = float(beta_schedule(ep))
            else:
                cur_beta = self.beta if beta is None else float(beta)

            idx = rng.permutation(N)
            ep_total = ep_rec = ep_kl = 0.0
            n_batches = 0
            for start in range(0, N, bs):
                b_idx = idx[start:start + bs]
                Xb = X[b_idx]
                eps = rng.standard_normal(size=(Xb.shape[0], self.latent_dim))
                x_hat, cache = self.forward(Xb, eps=eps, sample=True)
                Lt, Lr, Lk = self.loss(cache, beta=cur_beta)
                grads = self.backward(cache, beta=cur_beta)
                params = self.get_flat_params()
                new_params = optimizer.step(params, grads)
                self.set_flat_params(new_params)
                ep_total += Lt; ep_rec += Lr; ep_kl += Lk
                n_batches += 1

            hist["total"].append(ep_total / n_batches)
            hist["rec"].append(ep_rec / n_batches)
            hist["kl"].append(ep_kl / n_batches)
            hist["beta"].append(cur_beta)
            if verbose and (ep % log_every == 0 or ep == epochs - 1):
                print(f"  época {ep:4d}  L={hist['total'][-1]:.4f}  "
                      f"rec={hist['rec'][-1]:.4f}  KL={hist['kl'][-1]:.4f}  "
                      f"beta={cur_beta:.3f}")
        return hist

    # ============================================================ inference
    def encode(self, X):
        """Devuelve mu(x) (modo determinístico, z=mu). Para el scatter latente."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        h_enc, _ = self._encoder_body(X)
        mu = h_enc @ self.Wmu.T + self.bmu
        return mu

    def encode_params(self, X):
        """Devuelve (mu, logvar)."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        h_enc, _ = self._encoder_body(X)
        mu = h_enc @ self.Wmu.T + self.bmu
        logvar = h_enc @ self.Wlv.T + self.blv
        logvar = np.clip(logvar, -30.0, 30.0)  # mismo clip que forward(): que el
        # sigma reportado coincida con el que la red usa internamente (exp estable)
        return mu, logvar

    def decode(self, Z):
        """z -> x_hat (decoder determinístico)."""
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        x_hat, _ = self._decoder(Z)
        return x_hat

    def reconstruct(self, X):
        """decode(encode(X)) usando z=mu (determinístico)."""
        return self.decode(self.encode(X))

    def generate(self, n, seed=None):
        """Samplea z~N(0,I) y decodifica. Devuelve (samples, Z).

        Generación (consigna 2.c): como el entrenamiento forzó q(z/x) hacia el
        prior p(z) = N(0, I) (slide 74) vía el KL, samplear z del prior y
        pasarlo por el Decoder p(x/z) produce muestras nuevas del conjunto
        (Generative Model, slides 79 y 34).
        """
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal(size=(n, self.latent_dim))
        return self.decode(Z), Z

    # ============================================================= persistence
    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        sd = {
            "input_dim": self.input_dim,
            "encoder_hidden": np.array(self.encoder_hidden),
            "latent_dim": self.latent_dim,
            "decoder_hidden": np.array(self.decoder_hidden),
            "hidden_activation": self.hidden_activation,
            "output_activation": self.output_activation,
            "recon_loss": self.recon_loss,
            "beta": self.beta,
            "initializer": self.initializer,
            "seed": self.seed,
        }
        params = self.get_flat_params()
        for i, p in enumerate(params):
            sd[f"p{i}"] = np.asarray(p)
        sd["n_params"] = len(params)
        np.savez(path, **sd)

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=True)
        vae = cls(
            input_dim=int(d["input_dim"]),
            encoder_hidden=[int(n) for n in d["encoder_hidden"]],
            latent_dim=int(d["latent_dim"]),
            decoder_hidden=[int(n) for n in d["decoder_hidden"]],
            hidden_activation=str(d["hidden_activation"]),
            output_activation=str(d["output_activation"]),
            recon_loss=str(d["recon_loss"]),
            beta=float(d["beta"]),
            initializer=str(d["initializer"]),
            seed=int(d["seed"]),
        )
        n = int(d["n_params"])
        params = [d[f"p{i}"] for i in range(n)]
        vae.set_flat_params(params)
        return vae


# =========================================================================
# GRADIENT-CHECK (obligatorio): analítico vs diferencias finitas centrales.
# =========================================================================
def gradcheck(seed=0, eps_fd=1e-5, tol=1e-4, verbose=True):
    """Compara grad analítico de L_total (rec + beta*KL) vs diferencias finitas.

    Verificación numérica del backward de las slides 84-89: para cada
    parámetro w, (L(w+h) - L(w-h)) / 2h debe coincidir con el gradiente
    analítico. Con el epsilon de reparam FIJO (muestreado una sola vez) la
    loss es determinística y la comparación es válida. Asserta error
    relativo máx < tol. Devuelve el error relativo máximo (float).
    """
    rng = np.random.default_rng(seed)

    # Red y datos chicos. beta != 1 para ejercitar la rama KL.
    D = 6
    vae = VAE(
        input_dim=D,
        encoder_hidden=[5, 4],
        latent_dim=3,
        decoder_hidden=[4, 5],
        hidden_activation="tanh",   # suave: derivadas continuas para FD
        output_activation="logistic",
        recon_loss="bce",
        beta=1.7,
        initializer="glorot",
        init_scale=0.5,
        seed=seed,
    )
    N = 4
    X = (rng.random((N, D)) > 0.5).astype(np.float64)
    eps = rng.standard_normal(size=(N, vae.latent_dim))  # FIJO

    def total_loss():
        _, cache = vae.forward(X, eps=eps, sample=True)
        Lt, _, _ = vae.loss(cache)
        return Lt

    # Gradiente analítico
    _, cache = vae.forward(X, eps=eps, sample=True)
    ana = vae.backward(cache)
    params = vae.get_flat_params()

    max_rel = 0.0
    worst = None
    for pi, p in enumerate(params):
        flat = np.asarray(p, dtype=np.float64).ravel()
        g_flat = np.asarray(ana[pi], dtype=np.float64).ravel()
        for k in range(flat.size):
            orig = flat[k]
            flat[k] = orig + eps_fd
            Lp = total_loss()
            flat[k] = orig - eps_fd
            Lm = total_loss()
            flat[k] = orig
            num = (Lp - Lm) / (2.0 * eps_fd)
            den = max(1.0, abs(num) + abs(g_flat[k]))
            rel = abs(num - g_flat[k]) / den
            if rel > max_rel:
                max_rel = rel
                worst = (pi, k, num, g_flat[k])

    if verbose:
        print(f"[gradcheck] error relativo máximo = {max_rel:.3e} (tol={tol:.0e})")
        if worst is not None:
            pi, k, num, ana_v = worst
            print(f"[gradcheck] peor param idx={pi} elem={k}  "
                  f"num={num:.6e}  ana={ana_v:.6e}")
    assert max_rel < tol, (
        f"gradient-check FALLÓ: error relativo {max_rel:.3e} >= tol {tol:.0e}"
    )
    return max_rel


if __name__ == "__main__":
    gradcheck()
