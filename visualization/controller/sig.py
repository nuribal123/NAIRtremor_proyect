import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ------------ Modelo original ------------
class MatsuokaOscillator:
    def __init__(self, tau1=0.1, tau2=0.1, B=2.5, A=5, h=2.5, rosc=1, Ts=0.005, Kf=0.3):
        self.tau1 = tau1
        self.tau2 = tau2
        self.B = B
        self.A = A
        self.h = h
        self.rosc = rosc
        self.Ts = Ts
        self.Kf = Kf
        self.X = np.random.rand(2)
        self.V = np.random.rand(2)

    def step(self, s1=0, s2=0):
        x1 = self.X[0] + self.Ts * ((1 / (self.Kf * self.tau1)) * (
            -self.X[0] - self.B * self.V[0] - self.h * max(self.X[1], 0) + self.A * s1 + self.rosc))
        y1 = max(x1, 0)
        v1 = self.V[0] + self.Ts * ((1 / (self.Kf * self.tau2)) * (-self.V[0] + max(self.X[0], 0)))

        x2 = self.X[1] + self.Ts * ((1 / (self.Kf * self.tau1)) * (
            -self.X[1] - self.B * self.V[1] - self.h * max(self.X[0], 0) - self.A * s2 + self.rosc))
        y2 = max(x2, 0)
        v2 = self.V[1] + self.Ts * ((1 / (self.Kf * self.tau2)) * (-self.V[1] + max(self.X[1], 0)))

        self.X = np.array([x1, x2])
        self.V = np.array([v1, v2])
        return np.array([y1, y2])

# ------------ Simulación: barrido de Kf ------------
Ts = 0.005 #0,005
timesteps = 1000
kf_vals = np.linspace(0.183, 0.419, 90)
freqs_dom = []

for kf in kf_vals:
    osc = MatsuokaOscillator(Ts=Ts, Kf=kf)
    output = []

    for _ in range(timesteps):
        y = osc.step()
        output.append(y[0])

    signal = np.array(output) - np.mean(output)
    fft_vals = np.abs(rfft(signal))
    freqs = rfftfreq(len(signal), Ts)
    fft_vals[0] = 0
    freq_dom = freqs[np.argmax(fft_vals)]
    freqs_dom.append(freq_dom)

kf_vals = np.array(kf_vals)
freqs_dom = np.array(freqs_dom)

# ------------ Ajuste de curva ------------
def modelo(kf, a, b):
    return a / kf + b

params, _ = curve_fit(modelo, kf_vals, freqs_dom)
a_fit, b_fit = params
print(f"Ajuste: f = {a_fit:.3f} / Kf + {b_fit:.3f}")

# ------------ Visualización del ajuste ------------
kf_fino = np.linspace(min(kf_vals), max(kf_vals), 200)
f_ajustada = modelo(kf_fino, a_fit, b_fit)

plt.figure(figsize=(10, 5))
plt.plot(kf_vals, freqs_dom, 'o', label='Datos simulados')
plt.plot(kf_fino, f_ajustada, '-', label='Ajuste empírico')
plt.xlabel("Kf")
plt.ylabel("Frecuencia dominante [Hz]")
plt.title("Ajuste empírico de frecuencia vs. Kf")
plt.grid(True)
plt.legend()
plt.show()

# ------------ Predicción inversa ------------
freq_deseada = 6.5
kf_estimado = a_fit / (freq_deseada - b_fit)
print(f"Para una frecuencia deseada de {freq_deseada:.2f} Hz, el Kf estimado es ≈ {kf_estimado:.4f}")
