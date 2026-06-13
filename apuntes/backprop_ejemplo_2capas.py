"""Apunte de estudio: backward pass paso a paso con UNA CAPA OCULTA.

Extiende el ejemplo de una neurona de los apuntes de clase (y = tanh(w·x+b),
E = (y'-y)^2, con w=1/2, b=1/4 y tupla (x, y') = (1/2, 1/2)) a una red 1-1-1:
los MISMOS numeros, pero ahora la neurona del apunte es la CAPA OCULTA y su
salida v1 alimenta una segunda neurona identica (la capa de salida). Sirve
para ver:
  - que se guarda durante el forward (h y V de cada capa: h_list, V_list),
  - como nace delta en la salida y como retrocede capa por capa,
  - como cada peso individual recibe su gradiente (delta_destino * V_origen),
  - el update final y la verificacion de que E efectivamente baja.

Genera apuntes/backprop_ejemplo_2capas.png y verifica TODOS los gradientes:
  1. contra diferencias finitas centrales (mismo espiritu que VAE.gradcheck());
  2. contra el backward real de shared/mlp.py, que debe dar exactamente la
     MITAD: el repo usa E = 1/2*(y'-y)^2 y el apunte usa E = (y'-y)^2.

Uso:  python3 apuntes/backprop_ejemplo_2capas.py
"""

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse

from shared.mlp import MLP


# ============================================================ 1. EL EJEMPLO
# Mismos valores del apunte de clase, duplicados en cadena.
x, t = 0.5, 0.5            # tupla de entrenamiento (x, y')
w1, b1 = 0.5, 0.25         # capa 1 (oculta)  -- la "neurona del apunte"
w2, b2 = 0.5, 0.25         # capa 2 (salida)  -- otra igual, en serie
lr = 0.1                   # eta para el update final

# ---- FORWARD (guardando todo lo que el backward va a necesitar) ----
u1 = w1 * x                # producto    (el "phi" del apunte)
h1 = u1 + b1               # pre-activacion capa 1  (el "m" del apunte)
v1 = np.tanh(h1)           # salida capa 1          <-- SE GUARDA (V_list)
u2 = w2 * v1
h2 = u2 + b2               # pre-activacion capa 2
y = np.tanh(h2)            # salida de la red       <-- SE GUARDA
E = (t - y) ** 2           # loss del apunte

# ---- BACKWARD (regla de la cadena, de atras hacia adelante) ----
dE_dE = 1.0                                  # (1) semilla
dE_dy = -2.0 * (t - y)                       # (2) derivada de la loss
d2 = dE_dy * (1.0 - y ** 2)                  # (3) delta2 = dE/dh2 (tanh' = 1-y^2)
dE_dw2 = d2 * v1                             # (4) gradiente de w2  (usa v1 guardada!)
dE_db2 = d2 * 1.0                            # (4) gradiente de b2
dE_dv1 = d2 * w2                             # (5) la culpa cruza a la capa 1 por w2
d1 = dE_dv1 * (1.0 - v1 ** 2)                # (6) delta1 = dE/dh1
dE_dw1 = d1 * x                              # (7) gradiente de w1  (usa x guardada!)
dE_db1 = d1 * 1.0                            # (7) gradiente de b1

grads = {"w1": dE_dw1, "b1": dE_db1, "w2": dE_dw2, "b2": dE_db2}

# ---- (8) UPDATE: theta <- theta - eta * dE/dtheta ----
w1n, b1n = w1 - lr * dE_dw1, b1 - lr * dE_db1
w2n, b2n = w2 - lr * dE_dw2, b2 - lr * dE_db2

# ---- (9) CHEQUEO: nuevo forward con los pesos actualizados ----
v1n = np.tanh(w1n * x + b1n)
yn = np.tanh(w2n * v1n + b2n)
En = (t - yn) ** 2


# ================================== 2. VERIFICACION POR DIFERENCIAS FINITAS
def forward_E(params):
    a, c, d, e = params
    vv = np.tanh(a * x + c)
    yy = np.tanh(d * vv + e)
    return (t - yy) ** 2


p0 = np.array([w1, b1, w2, b2])
eps = 1e-6
num = []
for i in range(4):
    pp, pm = p0.copy(), p0.copy()
    pp[i] += eps
    pm[i] -= eps
    num.append((forward_E(pp) - forward_E(pm)) / (2 * eps))
ana = [dE_dw1, dE_db1, dE_dw2, dE_db2]

print("=== Verificacion 1: backprop a mano vs diferencias finitas ===")
ok_fd = True
for name, a_, n_ in zip(["dE/dw1", "dE/db1", "dE/dw2", "dE/db2"], ana, num):
    rel = abs(a_ - n_) / max(abs(a_), abs(n_), 1e-12)
    ok_fd &= rel < 1e-7
    print(f"  {name}: analitico {a_:+.8f} | numerico {n_:+.8f} | err.rel {rel:.1e}")
assert ok_fd, "backprop a mano NO coincide con el gradiente numerico"
print("  -> backprop a mano == gradiente numerico  OK\n")

# ============================ 3. VERIFICACION CONTRA EL MLP REAL DEL REPO
# Nuestro shared/mlp.py con mse usa E = 1/2*(y'-y)^2  ->  gradientes = MITAD.
mlp = MLP([1, 1, 1], hidden_activation="tanh", output_activation="tanh",
          loss_name="mse", initializer="random_normal", init_scale=0.1, seed=0)
mlp.weights[0] = np.array([[w1]])
mlp.biases[0] = np.array([b1])
mlp.weights[1] = np.array([[w2]])
mlp.biases[1] = np.array([b2])
out, cache = mlp._forward(np.array([[x]]))
gW, gb = mlp._backward(np.array([[t]]), cache)
mlp_grads = {"w1": gW[0][0, 0], "b1": gb[0][0], "w2": gW[1][0, 0], "b2": gb[1][0]}

print("=== Verificacion 2: shared/mlp.py reproduce el ejemplo (x 1/2) ===")
ok_mlp = True
for k in ["w1", "b1", "w2", "b2"]:
    ok_mlp &= np.isclose(2.0 * mlp_grads[k], grads[k], rtol=1e-12)
    print(f"  dE/d{k}: MLP {mlp_grads[k]:+.8f} * 2 = {2*mlp_grads[k]:+.8f}"
          f"  vs apunte {grads[k]:+.8f}")
assert ok_mlp, "el MLP del repo NO coincide con el ejemplo a mano"
assert np.isclose(out[0, 0], y), "forward del MLP distinto al del ejemplo"
print("  -> mismo forward y mismos gradientes (convencion 1/2 aparte)  OK\n")

print("=== Resumen numerico del ejemplo ===")
print(f"  forward : u1={u1:.6f} h1={h1:.6f} v1={v1:.6f} "
      f"u2={u2:.6f} h2={h2:.6f} y={y:.6f} E={E:.6f}")
print(f"  backward: dE/dy={dE_dy:+.6f} d2={d2:+.6f} dE/dv1={dE_dv1:+.6f} d1={d1:+.6f}")
print(f"  grads   : w1={dE_dw1:+.6f} b1={dE_db1:+.6f} w2={dE_dw2:+.6f} b2={dE_db2:+.6f}")
print(f"  update  : w1 {w1}->{w1n:.4f} | b1 {b1}->{b1n:.4f} | "
      f"w2 {w2}->{w2n:.4f} | b2 {b2}->{b2n:.4f}")
print(f"  chequeo : E {E:.6f} -> {En:.6f}  ({'BAJO' if En < E else 'NO bajo'})\n")


# ============================================================ 4. EL DIAGRAMA
ROJO = "#c62828"       # data (forward), como en el apunte
VIOLETA = "#5e35b1"    # gradientes (backward)
NARANJA = "#e65100"    # marcador "esto quedo guardado"
GRIS = "#666666"
GRISOP = "#d9d9d9"


def m(v, dec=3):
    """Formatea con signo menos unicode (como en los apuntes)."""
    return f"{v:.{dec}f}".replace("-", "−")


fig, ax = plt.subplots(figsize=(17.0, 10.72))
ax.set_xlim(0, 18.4)
ax.set_ylim(0, 11.6)
ax.set_aspect("equal", adjustable="box")
ax.axis("off")

HBOX = 0.86


def caja(cx, cy, formula, data, grad, w=2.5, fs=8.2, guardado=False, fs_f=None):
    """Caja estilo apunte: tres compartimentos (formula | data | gradiente)."""
    x0, y0 = cx - w / 2, cy - HBOX / 2
    ax.add_patch(Rectangle((x0, y0), w, HBOX, fc="white", ec="black",
                           lw=1.2, zorder=3))
    xa, xb = x0 + w * 0.46, x0 + w * 0.73
    ax.plot([xa, xa], [y0, y0 + HBOX], color="black", lw=0.8, zorder=4)
    ax.plot([xb, xb], [y0, y0 + HBOX], color="black", lw=0.8, zorder=4)
    yh = y0 + HBOX - 0.06
    ax.text(x0 + w * 0.23, yh, "fórmula", ha="center", va="top",
            fontsize=5.2, color=GRIS, zorder=5)
    ax.text(x0 + w * 0.595, yh, "data", ha="center", va="top",
            fontsize=5.2, color=GRIS, zorder=5)
    ax.text(x0 + w * 0.865, yh, "grad", ha="center", va="top",
            fontsize=5.2, color=GRIS, zorder=5)
    yc = y0 + 0.30
    ax.text(x0 + w * 0.23, yc, formula, ha="center", va="center",
            fontsize=fs_f or fs, zorder=5)
    ax.text(x0 + w * 0.595, yc, data, ha="center", va="center", color=ROJO,
            fontsize=fs + 0.6, fontweight="bold", zorder=5)
    ax.text(x0 + w * 0.865, yc, grad, ha="center", va="center", color=VIOLETA,
            fontsize=fs, fontweight="bold", zorder=5)
    if guardado:
        ax.add_patch(Circle((x0 + 0.14, y0 + HBOX - 0.14), 0.075,
                            fc=NARANJA, ec="none", zorder=6))


def op(cx, cy, simbolo, fs=13):
    ax.add_patch(Circle((cx, cy), 0.27, fc=GRISOP, ec="black", lw=1.1, zorder=3))
    ax.text(cx, cy, simbolo, ha="center", va="center", fontsize=fs,
            zorder=4, fontweight="bold")


def fn_tanh(cx, cy):
    ax.add_patch(Ellipse((cx, cy), 1.25, 0.6, fc=GRISOP, ec="black",
                         lw=1.1, zorder=3))
    ax.text(cx, cy, "tanH", ha="center", va="center", fontsize=9.5,
            zorder=4, fontweight="bold")


def arr(x1, y1, x2, y2, color="black", rad=0.0, lw=1.1, ls="-", ms=11):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle=ls, mutation_scale=ms,
                                connectionstyle=f"arc3,rad={rad}",
                                shrinkA=2, shrinkB=2))


def badge(cx, cy, n, r=0.155, fs=8.5):
    ax.add_patch(Circle((cx, cy), r, fc=VIOLETA, ec="none", zorder=6))
    ax.text(cx, cy, str(n), ha="center", va="center", color="white",
            fontsize=fs, fontweight="bold", zorder=7)


# ----------------------------------------------------------- titulo y leyenda
ax.text(0.25, 11.36, "Backward pass con una capa oculta",
        fontsize=15.5, fontweight="bold", va="center")
ax.text(0.25, 11.0,
        "y = tanh( w₂·v₁ + b₂ ),   "
        "v₁ = tanh( w₁·x + b₁ ),   E = (y′ − y)²",
        fontsize=10, va="center")
ax.text(0.25, 10.66,
        "θ = { w₁ = ½,  b₁ = ¼,  w₂ = ½,  "
        "b₂ = ¼ }     tupla de entrenamiento (x, y′) = (½, ½)"
        "     η = 0.1",
        fontsize=10, va="center")
ax.text(7.6, 11.36, "(los mismos números de tu apunte, dos neuronas en cadena)",
        fontsize=9.5, color=GRIS, va="center")

ax.add_patch(Rectangle((14.55, 10.18), 3.7, 1.27, fc="#f5f5f5", ec="#bbbbbb",
                       lw=0.8, zorder=2))
ax.text(14.75, 11.22, "data del forward (rojo)", fontsize=8.2, color=ROJO,
        va="center", fontweight="bold")
ax.text(14.75, 10.88, "gradiente del backward (violeta)", fontsize=8.2,
        color=VIOLETA, va="center", fontweight="bold")
ax.add_patch(Circle((14.85, 10.5), 0.075, fc=NARANJA, ec="none", zorder=3))
ax.text(15.02, 10.5, "= guardado en el forward (h_list, V_list)",
        fontsize=8.2, color="black", va="center")

# ------------------------------------------------------- BANDA 1: capa oculta
Y1 = 9.45
ax.text(0.25, 9.83, "CAPA 1 (oculta)", fontsize=9, color=GRIS,
        fontweight="bold", rotation=90, va="top", ha="center")

caja(1.55, 9.95, "x", m(x), "—", w=2.0, fs=7.6, guardado=True)
caja(1.55, 8.85, "w₁", m(w1), m(dE_dw1), w=2.0, fs=7.6)
op(3.15, Y1, "×")
caja(5.05, Y1, "w₁·x", m(u1), m(d1), w=2.5)
caja(5.75, 8.35, "b₁", m(b1), m(dE_db1), w=2.0, fs=7.6)
op(7.0, Y1, "+", fs=15)
caja(8.9, Y1, "w₁·x+b₁", m(h1), m(d1), w=2.6, guardado=True)
ax.text(8.9, 10.02, "δ₁ ≡ dE/dh₁",
        fontsize=7.5, color=VIOLETA, ha="center", style="italic")
fn_tanh(11.05, Y1)
caja(13.05, Y1, "tanh(h₁)", m(v1), m(dE_dv1), w=2.5, guardado=True)
ax.text(13.05, 10.02, "v₁ (la salida oculta)",
        fontsize=7.5, color=GRIS, ha="center", style="italic")

arr(2.55, 9.95, 2.95, 9.6)         # x -> (x)
arr(2.55, 8.85, 2.95, 9.3)         # w1 -> (x)
arr(3.42, Y1, 3.8, Y1)             # (x) -> u1
arr(6.3, Y1, 6.73, Y1)             # u1 -> (+)
arr(6.75, 8.5, 6.92, 9.2)          # b1 -> (+)
arr(7.27, Y1, 7.6, Y1)             # (+) -> h1
arr(10.2, Y1, 10.42, Y1)           # h1 -> tanh
arr(11.68, Y1, 11.8, Y1)           # tanh -> v1

# regla de oro (a la derecha de la banda 1)
ax.add_patch(Rectangle((14.7, 8.85), 3.5, 1.2, fc="#ede7f6", ec=VIOLETA,
                       lw=1.2, zorder=2))
ax.text(16.45, 9.78, "REGLA DE ORO", fontsize=9, color=VIOLETA,
        fontweight="bold", ha="center", va="center")
ax.text(16.45, 9.42, "dE/d(peso) = δ(neurona destino) · V(origen)",
        fontsize=8.3, ha="center", va="center")
ax.text(16.45, 9.08, "(por eso guardamos las V del forward)",
        fontsize=7.8, color=GRIS, ha="center", va="center")

# ------------------------------------------------------- BANDA 2: capa salida
Y2 = 5.55
ax.text(0.25, 5.93, "CAPA 2 (salida)", fontsize=9, color=GRIS,
        fontweight="bold", rotation=90, va="top", ha="center")

caja(1.75, 6.05, "v₁ ¡guardada!", m(v1), m(dE_dv1), w=2.2, fs_f=6.8,
     guardado=True)
caja(1.75, 4.95, "w₂", m(w2), m(dE_dw2), w=2.0, fs=7.6)
op(3.55, Y2, "×")
caja(5.35, Y2, "w₂·v₁", m(u2), m(d2), w=2.5)
caja(6.05, 4.45, "b₂", m(b2), m(dE_db2), w=2.0, fs=7.6)
op(7.3, Y2, "+", fs=15)
caja(9.2, Y2, "w₂·v₁+b₂", m(h2), m(d2), w=2.6, guardado=True)
ax.text(9.2, 6.12, "δ₂ ≡ dE/dh₂ («culpa»)",
        fontsize=7.5, color=VIOLETA, ha="center", style="italic")
fn_tanh(11.35, Y2)
caja(13.35, Y2, "tanh(h₂)", m(y), m(dE_dy), w=2.5, guardado=True)
ax.text(13.35, 6.12, "y (salida de la red)",
        fontsize=7.5, color=GRIS, ha="center", style="italic")
caja(16.45, Y2, "(y′−y)²", m(E, 4), "1", w=2.6)
caja(16.45, 6.9, "y′", m(t), "—", w=2.0, fs=7.6)

arr(2.85, 6.05, 3.35, 5.7)         # v1 -> (x)
arr(2.75, 4.95, 3.35, 5.4)         # w2 -> (x)
arr(3.82, Y2, 4.1, Y2)             # (x) -> u2
arr(6.6, Y2, 7.03, Y2)             # u2 -> (+)
arr(7.0, 4.62, 7.2, 5.28)          # b2 -> (+)
arr(7.57, Y2, 7.9, Y2)             # (+) -> h2
arr(10.5, Y2, 10.72, Y2)           # h2 -> tanh
arr(11.98, Y2, 12.1, Y2)           # tanh -> y
arr(14.6, Y2, 15.15, Y2)           # y -> E
arr(16.45, 6.47, 16.45, 5.98)      # y' -> E

# ----------------------------------------- el cable v1: forward baja, grad sube
arr(12.5, 9.0, 2.3, 6.52, color="#888888", rad=-0.13, lw=1.4, ls="--", ms=14)
ax.text(7.9, 7.62,
        "forward ↓  v₁ = 0.462 queda guardada (V_list) y entra a la capa 2\n"
        "backward ↑  dE/dv₁ = −0.042 vuelve a la capa 1 por el mismo cable",
        fontsize=8.0, ha="center", va="center", rotation=12.5,
        color="black", zorder=6,
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#bbbbbb", lw=0.7))

# --------------------------------------------- camino del backward (badges)
# banda 2 (debajo / derecha)
badge(15.45, 4.7, 1)
ax.text(15.68, 4.7, "dE/dE = 1 (semilla)", fontsize=8.3, va="center")
badge(12.0, 4.7, 2)
ax.text(12.23, 4.7, "dE/dy = −2(y′−y) = −0.106",
        fontsize=8.3, va="center")
badge(8.35, 4.62, 3)
ax.text(8.58, 4.62, "δ₂ = dE/dy · (1−y²) = −0.085",
        fontsize=8.3, va="center")
badge(1.0, 4.18, 4)
ax.text(1.23, 4.18,
        "dE/dw₂ = δ₂·v₁ = −0.039      "
        "dE/db₂ = δ₂ = −0.085",
        fontsize=8.3, va="center")
badge(2.0, 6.78, 5)
ax.text(2.23, 6.78,
        "dE/dv₁ = δ₂·w₂ = −0.042  "
        "→ la culpa cruza a la capa 1",
        fontsize=8.3, va="center")

# flechas violetas del camino (banda 2, de derecha a izquierda)
arr(15.3, 5.06, 14.25, 5.06, color=VIOLETA, rad=0.35, lw=1.3)  # E -> y
arr(12.35, 5.04, 10.3, 5.04, color=VIOLETA, rad=0.28, lw=1.3)  # y -> h2 (x tanh')
arr(8.2, 5.08, 6.3, 4.82, color=VIOLETA, rad=-0.25, lw=1.3)    # hacia w2/b2

# banda 1 (arriba)
badge(10.6, 10.32, 6)
ax.text(10.83, 10.32,
        "δ₁ = dE/dv₁ · (1−v₁²) = −0.033",
        fontsize=8.3, va="center")
badge(3.6, 10.35, 7)
ax.text(3.83, 10.35,
        "dE/dw₁ = δ₁·x = −0.017      "
        "dE/db₁ = δ₁ = −0.033",
        fontsize=8.3, va="center")

# ------------------------------------------------------------ panel inferior
ax.plot([0.25, 18.15], [3.62, 3.62], color="#cccccc", lw=0.9)
ax.text(0.25, 3.38, "EL CAMINO COMPLETO, EN ORDEN "
        "(regla de la cadena, siempre de atrás hacia adelante):",
        fontsize=9.3, fontweight="bold", va="center")

pasos = [
    "dE/dE = 1 — la semilla del backward.",
    "dE/dy = −2(y′−y) = −0.106 — derivada de la loss.",
    "δ₂ = dE/dy · tanh′(h₂) = −0.106 · 0.800 = "
    "−0.085 — culpa de la neurona de salida  (tanh′ = 1−y²).",
    "dE/dw₂ = δ₂·v₁ = −0.039     dE/db₂ = "
    "δ₂·1 = −0.085 — ¡acá se usa la v₁ "
    "guardada del forward!",
    "dE/dv₁ = δ₂·w₂ = −0.042 — la culpa cruza "
    "a la capa anterior POR EL PESO que las conecta.",
    "δ₁ = dE/dv₁ · tanh′(h₁) = −0.042 · "
    "0.786 = −0.033 — culpa de la neurona oculta.",
    "dE/dw₁ = δ₁·x = −0.017     dE/db₁ = "
    "δ₁·1 = −0.033 — y ya tenemos los 4 gradientes.",
]
yp = 3.0
for i, paso in enumerate(pasos, start=1):
    badge(0.42, yp, i, r=0.14, fs=7.8)
    ax.text(0.66, yp, paso, fontsize=8.4, va="center")
    yp -= 0.36

xr = 11.6
badge(xr - 0.18, 3.0, 8, r=0.14, fs=7.8)
ax.text(xr + 0.06, 3.0, "UPDATE:  θ ← θ − η·dE/dθ"
        "   (η = 0.1)", fontsize=8.4, va="center", fontweight="bold")
ax.text(xr + 0.06, 2.64,
        f"w₁: 0.500 → {w1n:.4f}        b₁: 0.250 → {b1n:.4f}",
        fontsize=8.4, va="center")
ax.text(xr + 0.06, 2.28,
        f"w₂: 0.500 → {w2n:.4f}        b₂: 0.250 → {b2n:.4f}",
        fontsize=8.4, va="center")
ax.text(xr + 0.06, 1.92,
        "(los 4 gradientes son negativos → los 4 parámetros suben,\n"
        " porque y = 0.447 quedó corto frente a y′ = 0.5)",
        fontsize=7.9, va="center", color=GRIS)
badge(xr - 0.18, 1.38, 9, r=0.14, fs=7.8)
ax.text(xr + 0.06, 1.38,
        f"CHEQUEO: forward con los pesos nuevos → "
        f"E: {E:.5f} → {En:.5f}  ✓ bajó",
        fontsize=8.4, va="center", fontweight="bold")
ax.text(xr + 0.06, 0.92,
        "En el repo (shared/mlp.py):  ③⑥ es  delta = g′(h) · "
        "(delta_sig · W_sig)   [linea 147]\n"
        "④⑦ es  grad_W = deltaᵀ · V_anterior / N  (acá "
        "N = 1)   [linea 155]",
        fontsize=7.9, va="center", color=GRIS)

out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "backprop_ejemplo_2capas.png")
fig.savefig(out_png, dpi=165, facecolor="white", bbox_inches="tight")
plt.close(fig)
print(f"Diagrama guardado en: {out_png}")
