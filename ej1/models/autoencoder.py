"""Autoencoder: wrapper FINO sobre el MLP de shared/.

Referencias de teoría (Autoencoders.pdf):
  - slide 3: arquitectura = dos MLPs, el decoder con la distribución de
    neuronas INVERTIDA y salida de la misma dimensión que la entrada:
        Z  = f(X) = h(XW + b)   (Encoder)
        X' = g(Z) = h(ZV + p)   (Decoder)
  - slide 4: se entrena "por cualquier método de aprendizaje que sirva para
    un MLP" poniendo como salida esperada el MISMO patrón X (target = input).
  - slide 5: diagrama Encoder | Latent Code | Decoder (la función identidad).
  - slide 6: la restricción que hace útil al AE es el CUELLO DE BOTELLA:
    menos neuronas en la capa central fuerzan una codificación eficiente.

La arquitectura es SIMÉTRICA con un cuello de botella 2D, p.ej.
    [35, 60, 40, 20, 2, 20, 40, 60, 35]
Se entrena como un único MLP con target = input (full-batch).

Convención de índices (importante):
- En el MLP, cada "capa" i transforma V_list[i] -> V_list[i+1] aplicando
  weights[i], biases[i] y self._activations[i].
- El cuello es la capa k (0-based) tal que architecture[k+1] == 2.
- encode(X) corre las capas 0..k y devuelve V_list[k+1] (la activación 2D).
- decode(Z) corre las capas k+1..n_layers-1 partiendo de Z.

Para fijar la activación del cuello (identity o tanh) sin tocar shared/, en el
constructor pisamos self.mlp._activations[k] = bottleneck_activation. El MLP
aplica hidden_activation a TODAS las capas ocultas, así que esta línea es la que
diferencia un cuello lineal de uno tanh. activate_deriv soporta identity/tanh,
de modo que el backward del MLP sigue siendo correcto.

models/ NO importa matplotlib.
"""

import numpy as np

from shared.mlp import MLP
from shared.activations import activate


class Autoencoder:
    """Wrapper fino sobre MLP con cuello de botella 2D y encode/decode/reconstruct."""

    def __init__(
        self,
        architecture,
        hidden_activation="tanh",
        output_activation="logistic",
        bottleneck_activation="identity",
        beta=1.0,
        initializer="glorot",
        init_scale=0.1,
        seed=42,
        weight_decay=0.0,
        loss_name="mse",
        bottleneck_size=2,
    ):
        architecture = [int(n) for n in architecture]
        self.architecture = architecture
        self.bottleneck_size = int(bottleneck_size)
        self.bottleneck_activation = bottleneck_activation

        self.mlp = MLP(
            architecture=architecture,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            beta=beta,
            initializer=initializer,
            init_scale=init_scale,
            seed=seed,
            weight_decay=weight_decay,
            loss_name=loss_name,
        )

        # Índice de la capa-cuello: k tal que architecture[k+1] == bottleneck_size.
        # Tomamos la PRIMERA aparición (el cuello más estrecho del medio).
        self.bottleneck_layer = self._find_bottleneck(architecture, self.bottleneck_size)

        # Pisar la activación del cuello (identity/tanh) sin tocar shared/.
        self.mlp._activations[self.bottleneck_layer] = bottleneck_activation

    @staticmethod
    def _find_bottleneck(architecture, size):
        # architecture[k+1] == size  ->  V_list[k+1] es el cuello.
        for k in range(len(architecture) - 1):
            if architecture[k + 1] == size:
                return k
        raise ValueError(
            f"No hay capa de tamaño {size} (cuello) en {architecture}"
        )

    # ------------------------------------------------------------------ API
    def encode(self, X):
        """Encoder Z = f(X) (Autoencoders.pdf, slide 3).

        Corre las capas 0..k y devuelve la activación del cuello (N, 2): la
        representación de los datos en el espacio latente (consigna 1.a.3).
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        V = X
        for i in range(self.bottleneck_layer + 1):
            h = V @ self.mlp.weights[i].T + self.mlp.biases[i]
            V = activate(h, self.mlp._activations[i], self.mlp.beta)
        return V

    def decode(self, Z):
        """Decoder X' = g(Z) (Autoencoders.pdf, slide 3).

        Corre las capas k+1..n_layers-1 partiendo del código 2D Z. Llamado con
        un Z arbitrario (no proveniente de encode) implementa el "Algoritmo del
        Autoencoder Generativo" (slide 34): especificar z1, z2 a mano y dejar
        que el decoder genere una muestra nueva (consigna 1.a.4).
        """
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        V = Z
        for i in range(self.bottleneck_layer + 1, self.mlp.n_layers):
            h = V @ self.mlp.weights[i].T + self.mlp.biases[i]
            V = activate(h, self.mlp._activations[i], self.mlp.beta)
        return V

    def reconstruct(self, X):
        """decode(encode(X)): reconstrucción completa (salida sigmoide en (0,1))."""
        return self.decode(self.encode(X))

    def reconstruct_binary(self, X, threshold=0.5):
        """Reconstrucción binarizada a {0,1} (para contar pixel-error)."""
        return (self.reconstruct(X) >= threshold).astype(int)

    # --------------------------------------------------------- entrenamiento
    def train_epoch(self, X, t, optimizer, batch_size=0, shuffle=True, rng=None):
        """Delega en MLP.train_epoch.

        Aprendizaje del AE (Autoencoders.pdf, slide 4): mismo backprop que un
        MLP, con t = X (AE básico) o t = X limpio con entrada ruidosa (DAE,
        slide 22). Por eso el AE no necesita un backward propio.
        """
        return self.mlp.train_epoch(X, t, optimizer, batch_size=batch_size,
                                    shuffle=shuffle, rng=rng)

    def predict(self, X):
        return self.mlp.predict(X)

    def get_params(self):
        return self.mlp.get_params()

    def set_params(self, params):
        self.mlp.set_params(params)

    # ---------------------------------------------------------- persistencia
    def save(self, path):
        """Guarda el MLP subyacente + metadatos del cuello en un .npz."""
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        save_dict = {
            "architecture": np.array(self.architecture),
            "hidden_activation": self.mlp.hidden_activation,
            "output_activation": self.mlp.output_activation,
            "bottleneck_activation": self.bottleneck_activation,
            "bottleneck_size": self.bottleneck_size,
            "bottleneck_layer": self.bottleneck_layer,
            "beta": self.mlp.beta,
            "weight_decay": self.mlp.weight_decay,
            "loss_name": self.mlp.loss_name,
        }
        for i in range(self.mlp.n_layers):
            save_dict[f"W{i}"] = self.mlp.weights[i]
            save_dict[f"b{i}"] = self.mlp.biases[i]
        np.savez(path, **save_dict)

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=True)
        architecture = [int(n) for n in data["architecture"]]
        ae = cls(
            architecture=architecture,
            hidden_activation=str(data["hidden_activation"]),
            output_activation=str(data["output_activation"]),
            bottleneck_activation=str(data["bottleneck_activation"]),
            beta=float(data["beta"]),
            weight_decay=float(data["weight_decay"]),
            loss_name=str(data["loss_name"]),
            bottleneck_size=int(data["bottleneck_size"]),
        )
        for i in range(ae.mlp.n_layers):
            ae.mlp.weights[i] = data[f"W{i}"]
            ae.mlp.biases[i] = data[f"b{i}"]
        return ae
