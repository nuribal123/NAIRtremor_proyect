import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks

class MatsuokaOscillator:
    def __init__(self, Kf=0.3, tau1=0.1, tau2=0.1, beta=2.5, h=2.4, Le=0, R=1, Ts=0.005):
        self.Kf = Kf
        self.tau1 = tau1
        self.tau2 = tau2
        self.beta = beta
        self.h = h
        self.Le = Le
        self.R = R
        self.Ts = Ts

        # Estados iniciales
        self.X = np.array([np.random.normal(0.5, 0.25), np.random.normal(0.5, 0.25)])
        self.V = np.array([np.random.normal(0.5, 0.25), np.random.normal(0.5, 0.25)])
        self.Y = np.maximum(self.X, 0)

    def step(self, s1=0, s2=0):
        Ts_div_tau1 = self.Ts / (self.Kf * self.tau1)
        Ts_div_tau2 = self.Ts / self.tau2

        f = lambda x: max(x, 0)
        x = np.zeros(2)
        v = np.zeros(2)

        # Neurona 1
        x[0] = self.X[0] + Ts_div_tau1 * (-self.X[0] - self.beta * self.V[0] - self.h * f(self.X[1]) + self.Le + self.R)
        v[0] = self.V[0] + Ts_div_tau2 * (-self.V[0] + f(self.X[0]))

        # Neurona 2
        x[1] = self.X[1] + Ts_div_tau1 * (-self.X[1] - self.beta * self.V[1] - self.h * f(self.X[0]) - self.Le + self.R)
        v[1] = self.V[1] + Ts_div_tau2 * (-self.V[1] + f(self.X[1]))

        self.X = x
        self.V = v
        self.Y = np.maximum(self.X, 0)

        return self.Y

# --- Parámetros ---
Ts = 0.005
T_total = 3.0  # segundos
timesteps = int(T_total / Ts)
fs = 1 / Ts
t = np.linspace(0, T_total, timesteps)
Kf_values = [0.10, 0.051, 0.05, 0.018 , 0.036, 0.006]

plt.figure(figsize=(14, 10))
kf_validos = []

for i, kf in enumerate(Kf_values, 1):
    osc = MatsuokaOscillator(Kf=kf, Ts=Ts)
    output = []

    for _ in range(timesteps):
        y = osc.step()
        output.append(y[0])  # Y1

    signal = np.array(output)
    signal_detrended = signal - np.mean(signal)

    # --- FFT ---
    fft_vals = np.abs(rfft(signal_detrended))
    freqs = rfftfreq(len(signal), Ts)
    fft_vals[0] = 0
    freq_dom = freqs[np.argmax(fft_vals)]

    # --- Por picos ---
    peaks, _ = find_peaks(signal, height=0.3, distance=int(0.1 / Ts))
    if len(peaks) > 1:
        periodo_medio = np.mean(np.diff(peaks))
        freq_picos = fs / periodo_medio
    else:
        freq_picos = np.nan

    print(f"Kf = {kf:.3f} → Frecuencia FFT: {freq_dom:.2f} Hz | Picos: {freq_picos:.2f} Hz")

    if 4 <= freq_dom <= 9:
        kf_validos.append((kf, freq_dom))

    # --- Gráfica de la señal ---
    plt.subplot(len(Kf_values), 2, 2 * i - 1)
    plt.plot(t, signal, label=f"Kf={kf} | FFT={freq_dom:.2f} Hz")
    if len(peaks) > 1:
        plt.plot(t[peaks], signal[peaks], 'rx', label="Picos")
    plt.grid(True)
    plt.ylabel("Y1")
    plt.legend()
    if i == 1:
        plt.title("Salida Y1 vs tiempo")

    # --- Gráfica de espectro ---
    plt.subplot(len(Kf_values), 2, 2 * i)
    plt.plot(freqs, fft_vals)
    plt.xlim(0, 20)
    plt.grid(True)
    plt.xlabel("Hz")
    plt.ylabel("Magnitud")
    if i == 1:
        plt.title("Espectro FFT")

plt.tight_layout()
plt.show()

# --- Imprimir Kf válidos ---
print("\nValores de Kf con frecuencia FFT entre 4 y 9 Hz:")
for kf, freq in kf_validos:
    print(f"Kf = {kf:.3f} -> Frecuencia = {freq:.2f} Hz")
