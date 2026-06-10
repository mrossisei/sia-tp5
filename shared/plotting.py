"""Helpers de figuras compartidos. SOLO se importa desde `analysis/` y `main_*`,
nunca desde `models/` (convención TP3/TP4: los modelos no conocen matplotlib).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "primary": "steelblue",
    "secondary": "tomato",
    "accent": "slateblue",
    "highlight": "crimson",
    "neutral": "dimgray",
    "positive": "#2a9d8f",
    "negative": "#e76f51",
}

DEFAULT_DPI = 150


def save_fig(fig, path, dpi=DEFAULT_DPI):
    """Guarda una figura creando el directorio si hace falta y cerrándola."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def _to_runs(data):
    """Normaliza una corrida (lista de floats) o varias (lista de listas) a 2D.
    Corridas más cortas se rellenan con su último valor."""
    if data is None:
        return None
    if isinstance(data[0], (int, float, np.floating, np.integer)):
        data = [data]
    max_len = max(len(r) for r in data)
    padded = [list(r) + [r[-1]] * (max_len - len(r)) for r in data]
    return np.array(padded, dtype=float)


def plot_learning_curves(epoch_train, val_losses=None, title="Curva de aprendizaje",
                         path=None, ylabel="Loss", logy=False):
    """Curva de loss por época con bandas media ± std si hay varias corridas."""
    train_arr = _to_runs(epoch_train)
    val_arr = _to_runs(val_losses)
    epochs = np.arange(1, train_arr.shape[1] + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    train_mean = train_arr.mean(axis=0)
    train_std = train_arr.std(axis=0)
    ax.plot(epochs, train_mean, label="Train", color=PALETTE["primary"])
    if train_arr.shape[0] > 1:
        ax.fill_between(epochs, train_mean - train_std, train_mean + train_std,
                        alpha=0.2, color=PALETTE["primary"])
    if val_arr is not None:
        val_mean = val_arr.mean(axis=0)
        ax.plot(epochs, val_mean, label="Validation", color=PALETTE["secondary"])

    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Época")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if path:
        save_fig(fig, path)
    return fig
