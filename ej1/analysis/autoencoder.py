"""Figuras del AE basico (EJ1.a). Consume el Autoencoder entrenado.

Genera (en ej1/results/basic/):
  - latent_scatter.png        : espacio latente 2D anotado con cada caracter (1.a.3)
  - latent_scatter_pca.png    : comparacion con PCA 2D (numpy, svd sobre X centrada)
  - reconstruction_grid.png   : original vs reconstruido (7x5) resaltando errores
  - pixel_error_hist.png      : histograma de pixel-error por patron
  - learning_curve.png        : curva de loss por epoca
  - new_letter_generation.png : generacion de letra nueva interpolando en el latente (1.a.4)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.fonts import ROWS, COLS
from shared.metrics import pixel_errors, pixel_error_summary
from shared.plotting import save_fig, PALETTE


def _grid(vec):
    """(35,) -> (7,5)."""
    return np.asarray(vec).reshape(ROWS, COLS)


# --------------------------------------------------------------- 1.a.3 latente
def plot_latent_scatter(ae, X, labels, path):
    Z = ae.encode(X)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(Z[:, 0], Z[:, 1], c=PALETTE["primary"], s=60, alpha=0.6,
               edgecolors="k", linewidths=0.5, zorder=2)
    for i, lab in enumerate(labels):
        ax.annotate(lab, (Z[i, 0], Z[i, 1]), fontsize=11, fontweight="bold",
                    ha="center", va="center", zorder=3)
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_title("Espacio latente 2D del Autoencoder (32 caracteres)")
    ax.grid(True, alpha=0.3)
    save_fig(fig, path)
    return Z


def plot_latent_pca(X, labels, path):
    """PCA 2D en numpy: svd sobre X centrada. Comparacion con el latente del AE.

    Por qué comparar: el AE LINEAL es equivalente a PCA (lemma en
    Autoencoders.pdf, slides 10 y 15, vía SVD slides 7-14); con activaciones
    no lineales el AE es una "extensión no lineal" de PCA (slide 16). PCA 2D
    retiene poca varianza acá, lo que justifica el AE profundo no lineal.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = Xc @ Vt[:2].T  # proyeccion a 2 componentes principales
    var_ratio = (S[:2] ** 2) / np.sum(S ** 2)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(pcs[:, 0], pcs[:, 1], c=PALETTE["accent"], s=60, alpha=0.6,
               edgecolors="k", linewidths=0.5, zorder=2)
    for i, lab in enumerate(labels):
        ax.annotate(lab, (pcs[i, 0], pcs[i, 1]), fontsize=11, fontweight="bold",
                    ha="center", va="center", zorder=3)
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}% var)")
    ax.set_title("PCA 2D (lineal) de los 32 caracteres — baseline vs latente AE")
    ax.grid(True, alpha=0.3)
    save_fig(fig, path)
    return pcs, var_ratio


# ---------------------------------------------------- grilla de reconstruccion
def plot_reconstruction_grid(ae, X, labels, path):
    rec = ae.reconstruct(X)
    rec_bin = (rec >= 0.5).astype(int)
    errs = pixel_errors(X, rec)
    n = X.shape[0]
    ncols = 8
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows * 2, ncols, figsize=(ncols * 1.6, nrows * 3.0))
    for i in range(n):
        block_row = (i // ncols) * 2
        col = i % ncols
        ax_o = axes[block_row, col]
        ax_r = axes[block_row + 1, col]
        # Original
        ax_o.imshow(_grid(X[i]), cmap="Greys", vmin=0, vmax=1)
        ax_o.set_title(f"'{labels[i]}'", fontsize=9)
        ax_o.set_xticks([]); ax_o.set_yticks([])
        # Reconstruido + resaltar pixeles erroneos en rojo
        ax_r.imshow(_grid(rec_bin[i]), cmap="Greys", vmin=0, vmax=1)
        diff = (_grid(rec_bin[i]) != _grid(X[i]))
        yy, xx = np.where(diff)
        ax_r.scatter(xx, yy, marker="s", s=40, facecolors="none",
                     edgecolors=PALETTE["highlight"], linewidths=1.5)
        color = PALETTE["positive"] if errs[i] == 0 else PALETTE["negative"]
        ax_r.set_title(f"err={errs[i]}", fontsize=9, color=color)
        ax_r.set_xticks([]); ax_r.set_yticks([])
    # Apagar ejes sobrantes
    for j in range(n, nrows * ncols):
        block_row = (j // ncols) * 2
        col = j % ncols
        axes[block_row, col].axis("off")
        axes[block_row + 1, col].axis("off")
    fig.suptitle("Original (arriba) vs Reconstruido (abajo). Rojo = pixel erroneo",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    save_fig(fig, path)
    return errs


def plot_pixel_error_hist(ae, X, path):
    errs = pixel_errors(X, ae.reconstruct(X))
    fig, ax = plt.subplots(figsize=(7, 5))
    maxv = max(int(errs.max()), 2)
    bins = np.arange(0, maxv + 2) - 0.5
    ax.hist(errs, bins=bins, color=PALETTE["primary"], edgecolor="k", alpha=0.8)
    ax.axvline(1.5, color=PALETTE["highlight"], ls="--", lw=1.5,
               label="umbral objetivo (max<=1)")
    ax.set_xlabel("pixeles incorrectos por patron")
    ax.set_ylabel("cantidad de caracteres")
    ax.set_xticks(range(0, maxv + 1))
    ax.set_title("Histograma de pixel-error (32 caracteres)")
    ax.legend()
    save_fig(fig, path)


def plot_learning_curve(losses, path, best_epoch=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.arange(1, len(losses) + 1), losses, color=PALETTE["primary"])
    if best_epoch is not None:
        # best_epoch viene 0-based desde EarlyStopping; +1 para alinear con el eje.
        ax.axvline(best_epoch + 1, color=PALETTE["highlight"], ls="--", lw=1.2,
                   label=f"mejor epoca (params guardados) = {best_epoch}")
        ax.legend()
    ax.set_yscale("log")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Loss (escala log)")
    ax.set_title("Curva de aprendizaje del Autoencoder (BCE)")
    ax.grid(True, alpha=0.3)
    save_fig(fig, path)


# ---------------------------------------------------- 1.a.4 generar letra nueva
def plot_new_letter_generation(ae, X, labels, path, char_a="`", char_b="p",
                               n_steps=9):
    """Interpola en linea recta entre los codigos de dos letras y decodifica,
    VERIFICANDO que el punto medio sea una letra genuinamente nueva.

    Es el "Algoritmo del Autoencoder Generativo" (Autoencoders.pdf, slide 34):
    dejamos de lado el Encoder, especificamos valores de (z1, z2) de manera
    directa y cada tupla genera una muestra nueva via el Decoder. Interpolar
    entre dos codigos conocidos elige puntos "seguros" del latente.

    Cuidado (1.a.4): que la COORDENADA latente del punto medio sea nueva NO
    garantiza que la MUESTRA decodificada lo sea — el decoder puede mapear esa
    coordenada de vuelta, tras binarizar, a un patron identico a una letra del
    set (p.ej. el par 'c'->'i' colapsa a 'c'). Por eso medimos la novedad como
    la distancia minima en pixeles del patron generado a las 32 letras y, si el
    par pedido colapsa a una letra existente (dmin=0), buscamos de forma
    determinista el par cuyo punto medio sea el MAS novedoso.
    """
    Xi = np.asarray(X).astype(int)

    def nearest(pat):
        """(dist minima en pixeles, label) del patron binario pat (35,) al set."""
        d = np.sum(Xi != pat[None, :], axis=1)
        j = int(np.argmin(d))
        return int(d[j]), labels[j]

    Z = ae.encode(X)

    def midpoint_pattern(a, b):
        return (ae.decode(0.5 * (Z[a] + Z[b])) >= 0.5).astype(int)[0]

    ia, ib = labels.index(char_a), labels.index(char_b)
    dmid, near_mid = nearest(midpoint_pattern(ia, ib))

    # Verificacion de NOVEDAD: si el punto medio del par pedido coincide con una
    # letra del set, buscar (determinista) el par cuyo punto medio tenga la mayor
    # distancia minima al set, descartando patrones degenerados (casi vacios/llenos).
    if dmid == 0:
        best = None  # (dist, i, k)
        for i in range(len(labels)):
            for k in range(i + 1, len(labels)):
                pat = midpoint_pattern(i, k)
                if not (3 <= int(pat.sum()) <= 25):
                    continue
                d, _ = nearest(pat)
                if best is None or d > best[0]:
                    best = (d, i, k)
        if best is not None:
            _, ia, ib = best
            char_a, char_b = labels[ia], labels[ib]
            dmid, near_mid = nearest(midpoint_pattern(ia, ib))

    za, zb = Z[ia], Z[ib]
    alphas = np.linspace(0.0, 1.0, n_steps)
    codes = np.array([(1 - a) * za + a * zb for a in alphas])
    gen = (ae.decode(codes) >= 0.5).astype(int)
    mid_idx = n_steps // 2

    fig = plt.figure(figsize=(n_steps * 1.3, 3.4))
    gs = fig.add_gridspec(2, n_steps, height_ratios=[3, 1])
    for j in range(n_steps):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(gen[j].reshape(ROWS, COLS), cmap="Greys", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_title(f"'{char_a}'", fontsize=10)
        elif j == n_steps - 1:
            ax.set_title(f"'{char_b}'", fontsize=10)
        elif j == mid_idx:
            if dmid > 0:
                ax.set_title(f"nueva\n(+{dmid}px vs '{near_mid}')",
                             fontsize=9, color=PALETTE["highlight"])
            else:
                ax.set_title(f"= '{near_mid}'", fontsize=9,
                             color=PALETTE["negative"])
    # Recta en el latente
    ax_l = fig.add_subplot(gs[1, :])
    ax_l.scatter(Z[:, 0], Z[:, 1], c="lightgray", s=20, zorder=1)
    ax_l.plot(codes[:, 0], codes[:, 1], "-o", color=PALETTE["highlight"],
              ms=4, zorder=2)
    ax_l.scatter([za[0], zb[0]], [za[1], zb[1]], c=PALETTE["primary"], s=50,
                 zorder=3)
    ax_l.set_title("Recta de interpolacion en el espacio latente", fontsize=9)
    ax_l.set_xticks([]); ax_l.set_yticks([])
    novel = (f"punto medio a {dmid}px de la letra mas cercana ('{near_mid}') "
             f"-> no pertenece al set" if dmid > 0
             else f"punto medio = '{near_mid}' (no es nuevo)")
    fig.suptitle(f"Generacion de letra nueva: interpolacion '{char_a}' -> "
                 f"'{char_b}'  |  {novel}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, path)
    return codes


def generate_all(ae, X, labels, out_dir, losses=None, best_epoch=None):
    """Genera todas las figuras del AE basico en out_dir. Devuelve dict de resumen."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    p = lambda name: os.path.join(out_dir, name)

    Z = plot_latent_scatter(ae, X, labels, p("latent_scatter.png"))
    plot_latent_pca(X, labels, p("latent_scatter_pca.png"))
    errs = plot_reconstruction_grid(ae, X, labels, p("reconstruction_grid.png"))
    plot_pixel_error_hist(ae, X, p("pixel_error_hist.png"))
    if losses is not None:
        plot_learning_curve(losses, p("learning_curve.png"), best_epoch=best_epoch)
    plot_new_letter_generation(ae, X, labels, p("new_letter_generation.png"))

    summary = pixel_error_summary(X, ae.reconstruct(X))
    return summary
