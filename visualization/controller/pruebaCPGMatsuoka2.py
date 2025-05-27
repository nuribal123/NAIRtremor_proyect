import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks

class MatsuokaOscillator:
    def __init__(self, Kf=0.3, tau1=0.1, tau2=0.1, B=2.5, A=5, h=2.3, rosc=1.0, Ts=0.005):
        self.tau1 = tau1
        self.tau2 = tau2
        self.B = B
        self.A = A
        self.h = h
        self.rosc = rosc
        self.Ts = Ts
        self.Kf = Kf
        self.X = np.array([np.random.normal(0.5, 0.25), np.random.normal(0.5, 0.25)])
        self.V = np.array([np.random.normal(0.5, 0.25), np.random.normal(0.5, 0.25)])

    def step(self, s1=0, s2=0):
        x1 = self.X[0] + self.Ts * (1 / (self.Kf * self.tau1)) * (-self.X[0] - self.B * self.V[0] - self.h * max(self.X[1], 0) + self.A * s1 + self.rosc)
        y1 = max(x1, 0)
        v1 = self.V[0] + self.Ts * (1 / (self.Kf * self.tau2)) * (-self.V[0] + max(self.X[0], 0))

        x2 = self.X[1] + self.Ts * (1 / (self.Kf * self.tau1)) * (-self.X[1] - self.B * self.V[1] - self.h * max(self.X[0], 0) - self.A * s2 + self.rosc)
        y2 = max(x2, 0)
        v2 = self.V[1] + self.Ts * (1 / (self.Kf * self.tau2)) * (-self.V[1] + max(self.X[1], 0))

        self.X = np.array([x1, x2])
        self.V = np.array([v1, v2])
        return np.array([y1, y2])

# --- Parámetros ---
Ts = 0.005
T_total = 3.0  # segundos
timesteps = int(T_total / Ts)
fs = 1 / Ts
t = np.linspace(0, T_total, timesteps)
Kf_values = [0.20, 0.21, 0.22, 0.54, 0.50, 0.45]

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
    plt.grid(False)
    plt.ylabel("Y1")
    plt.legend()
    if i == 1:
        plt.title("Numerical y1 output vs Time")
    plt.xlabel("Time (s)")

    # --- Gráfica de espectro ---
    plt.subplot(len(Kf_values), 2, 2 * i)
    plt.plot(freqs, fft_vals)
    plt.xlim(0, 20)
    plt.grid(False)
    plt.xlabel("Hz")
    plt.ylabel("Magnitude")
    if i == 1:

        plt.title("FFT Spectrum")

plt.tight_layout()
plt.show()

# --- Imprimir Kf válidos ---
print("\nValores de Kf con frecuencia FFT entre 4 y 9 Hz:")
for kf, freq in kf_validos:
    print(f"Kf = {kf:.3f} -> Frecuencia = {freq:.2f} Hz")
