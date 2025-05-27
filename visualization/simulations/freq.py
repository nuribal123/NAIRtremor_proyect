import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt
import glob
import os

# --- Parámetros ---
#carpeta = "D:/ingenieriabiomedica/sconeGym/sconegym/sim_dataAmpFreq2"
carpeta = "D:/ingenieriabiomedica/sconeGym/sconegym/sim_data_B"
archivos_csv = sorted(glob.glob(os.path.join(carpeta, "*.csv")))[:6]  # Solo los primeros 6
filtro_corte = 2.0  # Hz, para eliminar componente lenta
limite_grafico = 20  # Hz para el eje X

# --- Funciones auxiliares ---
def butter_highpass_filter(data, cutoff, fs, order=4):
    b, a = butter(order, cutoff / (0.5 * fs), btype='high', analog=False)
    return filtfilt(b, a, data)

def calcular_fft(signal, dt):
    signal = signal - np.mean(signal)
    freqs = rfftfreq(len(signal), dt)
    fft_vals = np.abs(rfft(signal))
    fft_vals[0] = 0  # ignorar DC
    return freqs, fft_vals

# --- Crear figura ---
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(14, 10))
axs = axs.flatten()

# --- Procesar cada CSV ---
for i, archivo in enumerate(archivos_csv):
    df = pd.read_csv(archivo)
    nombre = os.path.basename(archivo)

    time = df["time"].values
    dt = time[1] - time[0]
    fs = 1 / dt

    pos = df["wrist_hand_r3_pos"].values
    vel = df["wrist_hand_r3_vel"].values

    # Filtrar posición y velocidad
    pos_filtrada = butter_highpass_filter(pos, filtro_corte, fs)
    vel_filtrada = butter_highpass_filter(vel, filtro_corte, fs)

    # FFTs
    freqs_p, fft_p = calcular_fft(pos_filtrada, dt)
    freqs_v, fft_v = calcular_fft(vel_filtrada, dt)

    # Frecuencias dominantes
    freq_dom_pos = freqs_p[np.argmax(fft_p)]
    freq_dom_vel = freqs_v[np.argmax(fft_v)]

    # Graficar
    axs[i].plot(freqs_p, fft_p, label=f'Posición ({freq_dom_pos:.2f} Hz)', color='orange')
    axs[i].plot(freqs_v, fft_v, label=f'Velocidad ({freq_dom_vel:.2f} Hz)', color='green')
    axs[i].set_xlim(0, limite_grafico)
    axs[i].set_title(f'{nombre}')
    axs[i].set_xlabel('Frecuencia (Hz)')
    axs[i].set_ylabel('Magnitud')
    axs[i].legend()
    axs[i].grid(True)

# Ajustar y mostrar
plt.tight_layout()
plt.suptitle("Espectros de frecuencia (posición y velocidad de muñeca)", y=1.02, fontsize=16)
plt.show()
