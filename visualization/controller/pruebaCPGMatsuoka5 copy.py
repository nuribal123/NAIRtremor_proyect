import numpy as np
import matplotlib.pyplot as plt

class MatsuokaOscillator:
    def __init__(self, Kf=0.22, tau1=0.5, tau2=0.5, beta=2.5, h=2.0, Le=0, R=1.0, Ts=0.005):
        self.Kf = Kf
        self.tau1 = tau1
        self.tau2 = tau2
        self.beta = beta
        self.h = h
        self.Le = Le
        self.R = R
        self.Ts = Ts

        # Estado inicial
        self.X = np.array([0.1, -0.1])  # Diferencia inicial
        self.V = np.array([0.0, 0.0])

    def step(self):
        x1, x2 = self.X
        v1, v2 = self.V

        f1 = max(x1, 0)
        f2 = max(x2, 0)

        dx1 = (1 / (self.Kf * self.tau1)) * (-x1 - self.beta * v1 - self.h * f2 + self.Le + self.R)
        dv1 = (1 / self.tau2) * (-v1 + f1)

        dx2 = (1 / (self.Kf * self.tau1)) * (-x2 - self.beta * v2 - self.h * f1 - self.Le + self.R)
        dv2 = (1 / self.tau2) * (-v2 + f2)

        x1 += self.Ts * dx1
        v1 += self.Ts * dv1
        x2 += self.Ts * dx2
        v2 += self.Ts * dv2

        self.X = np.array([x1, x2])
        self.V = np.array([v1, v2])

        y1 = max(x1, 0)
        y2 = max(x2, 0)
        return y1, y2

# Simulación
osc = MatsuokaOscillator()
y1_vals = []
y2_vals = []

for _ in range(2000):
    y1, y2 = osc.step()
    y1_vals.append(y1)
    y2_vals.append(y2)

# Graficar
plt.plot(y1_vals, label='y1')
plt.plot(y2_vals, label='y2')
plt.xlabel('Tiempo (pasos)')
plt.ylabel('Salida')
plt.title('Oscilador de Matsuoka')
plt.legend()
plt.grid(False)
plt.show()
