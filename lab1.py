import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp
import math


def Rot2D(X, Y, Alpha):  # rotates point (X,Y) on angle alpha with respect to Origin
    RX = X * np.cos(Alpha) - Y * np.sin(Alpha)
    RY = X * np.sin(Alpha) + Y * np.cos(Alpha)
    return RX, RY


# defining t as a symbol (it will be the independent variable)
t = sp.Symbol('t')
phi = 6.5 * t + 1.2 * sp.cos(6 * t)
# here x, y, Vx, Vy, Wx, Wy, xC are functions of 't'
x = (2 + sp.sin(6 * t)) * sp.cos(phi)
y = (2 + sp.sin(6 * t)) * sp.sin(phi)

Vx = sp.simplify(sp.diff(x, t))
print(Vx)

Vy = sp.simplify(sp.diff(y, t))
print(Vy)

Vmod = sp.simplify(sp.sqrt(Vx * Vx + Vy * Vy))
Wx = sp.simplify(sp.diff(Vx, t))
print(Wx)
Wy = sp.simplify(sp.diff(Vy, t))
print(Wy)
Wmod = sp.sqrt(Wx * Wx + Wy * Wy)
# and here really we could escape integrating, just don't forget that it's absolute value of V here we should differentiate
Wtau = sp.diff(Vmod, t)
# this is the value of rho but in the picture you should draw the radius, don' t forget!
rho = (Vmod * Vmod) / sp.sqrt(Wmod * Wmod - Wtau * Wtau)

# constructing corresponding arrays
T = np.linspace(0, 20, 200)
X = np.zeros_like(T)
Y = np.zeros_like(T)
VX = np.zeros_like(T)
VY = np.zeros_like(T)
WY = np.zeros_like(T)
WX = np.zeros_like(T)
Rho = np.zeros_like(T)
Phi = np.zeros_like(T)

# filling arrays with corresponding values
for i in np.arange(len(T)):
    X[i] = sp.Subs(x, t, T[i])
    Y[i] = sp.Subs(y, t, T[i])
    VX[i] = sp.Subs(Vx, t, T[i])
    VY[i] = sp.Subs(Vy, t, T[i])
    WY[i] = sp.Subs(Wy, t, T[i])
    WX[i] = sp.Subs(Wx, t, T[i])
    Rho[i] = sp.Subs(rho, t, T[i])
    Phi[i] = sp.Subs(phi, t, T[i])

# here we start to plot
fig = plt.figure()

ax1 = fig.add_subplot(1, 1, 1)
ax1.axis('equal')

# plotting a trajectory
ax1.plot(X, Y)

# plotting initial positions

# of the point A on the disc
P, = ax1.plot(X[0], Y[0], marker='o')
# of the velocity vector of this point (line)
VLine, = ax1.plot([X[0], X[0] + VX[0]], [Y[0], Y[0] + VY[0]], 'r')
WLine, = ax1.plot([X[0], X[0] + WX[0]], [Y[0], Y[0] + WY[0]], 'g')
Rline, = ax1.plot([X[0], X[0] - Rho[0] * (-VY[0]) / math.sqrt(math.pow(-VX[0], 2) + math.pow(-VY[0], 2))],
                  [Y[0], Y[0] - Rho[0] * (-VX[0]) / math.sqrt(np.power(-VX[0], 2) + math.pow(-VY[0], 2))], 'b')
R = math.sqrt(math.pow(X[0], 2) + math.pow(Y[0], 2))
# of the velocity vector of this point (arrow)
ArrowX = np.array([-0.2 * R, 0, -0.2 * R])
ArrowY = np.array([0.1 * R, 0, -0.1 * R])
RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[0], VX[0]))
VArrow, = ax1.plot(RArrowX + X[0] + VX[0], RArrowY + Y[0] + VY[0], 'r')

WArrowX = np.array([-0.2 * R, 0, -0.2 * R])
WArrowY = np.array([0.1 * R, 0, -0.1 * R])
RWArrowX, RWArrowY = Rot2D(WArrowX, WArrowY, math.atan2(WY[0], WX[0]))
WArrow, = ax1.plot(RWArrowX + X[0] + WX[0], RWArrowY + Y[0] + WY[0], 'g')


def Rot2D(X, Y, Alpha):  # rotates point (X,Y) on angle alpha with respect to Origin
    RX = X * np.cos(Alpha) - Y * np.sin(Alpha)
    RY = X * np.sin(Alpha) + Y * np.cos(Alpha)
    return RX, RY


def anima(i):
    P.set_data(X[i], Y[i])
    VLine.set_data([X[i], X[i] + VX[i]], [Y[i], Y[i] + VY[i]])
    Rline.set_data([X[i], X[i] + Rho[i] * (-VY[i]) / math.sqrt(math.pow(-VX[i], 2) + math.pow(-VY[i], 2))],
                   [Y[i], Y[i] - Rho[i] * (-VX[i]) / math.sqrt(math.pow(-VX[i], 2) + math.pow(-VY[i], 2))])
    WLine.set_data([X[i], X[i] + WX[i]], [Y[i], Y[i] + WY[i]])
    RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[i], VX[i]))
    VArrow.set_data(RArrowX + X[i] + VX[i], RArrowY + Y[i] + VY[i])
    RWArrowX, RWArrowY = Rot2D(WArrowX, WArrowY, math.atan2(WY[i], WX[i]))
    WArrow.set_data(RWArrowX + X[i] + WX[i], RWArrowY + Y[i] + WY[i])
    return P, VLine, Rline, VArrow, WLine, WArrow,


anim = FuncAnimation(fig, anima, frames=800, interval=100, blit=True)

plt.show()
