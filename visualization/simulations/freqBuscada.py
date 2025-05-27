import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt
import glob
import os

# --- Parámetros ---
#carpeta = "D:/ingenieriabiomedica/sconeGym/sconegym/sim_dataAmpFreqTryingFreq12"
#carpeta = "D:/ingenieriabiomedica/sconeGym/sconegym/sim_data_01_05_2025_5_9_mifrec"
#carpeta = "D:/ingenieriabiomedica/sconeGym/sconegym/sim_data_A"
#filename = f"D:/ingenieriabiomedica/sconeGym/sconegym/sim_data_MATSUOKA6_sinsconethings/twin_{i}.csv"
filename = f"D:/ingenieriabiomedica/sconeGym/sconegym/sim_data_MATSUOKA6_sinsconethingsBIEN/twin_"
archivos_csv = sorted(glob.glob(os.path.join(carpeta, "*.csv")))
filtro_corte = 2.0  # Hz
limite_grafico = 20  # Hz
frecuencia_min = 5
frecuencia_max = 7
max_archivos_graficar = 6

# --- Funciones auxiliares ---
def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def calcular_fft(signal, dt):
    signal = signal - np.mean(signal)
    freqs = rfftfreq(len(signal), dt)
    fft_vals = np.abs(rfft(signal))
    fft_vals[0] = 0  # quitar componente DC
    return freqs, fft_vals

# --- Seleccionar archivos con frecuencia dominante en rango ---
archivos_filtrados = []
dom_frequencies = []

for archivo in archivos_csv:
    try:
        df = pd.read_csv(archivo)
        time = df["time"].values
        dt = time[1] - time[0]
        fs = 1 / dt

        pos = df["wrist_hand_r3_pos"].values
        pos_filtrada = butter_bandpass_filter(pos, 4.0, 10.0, fs)

        freqs_p, fft_p = calcular_fft(pos_filtrada, dt)
        freq_dom_pos = freqs_p[np.argmax(fft_p)]

        if frecuencia_min <= freq_dom_pos <= frecuencia_max:
            archivos_filtrados.append((archivo, freq_dom_pos))
            if len(archivos_filtrados) >= max_archivos_graficar:
                break
    except Exception as e:
        print(f"Error procesando {archivo}: {e}")

# --- Graficar ---
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(14, 10))
axs = axs.flatten()

for i, (archivo, freq_dom) in enumerate(archivos_filtrados):
    df = pd.read_csv(archivo)
    nombre = os.path.basename(archivo)

    time = df["time"].values
    dt = time[1] - time[0]
    fs = 1 / dt

    pos = butter_bandpass_filter(df["wrist_hand_r3_pos"].values, 4.0, 10.0, fs)
    vel = butter_bandpass_filter(df["wrist_hand_r3_vel"].values, 4.0, 10.0, fs)

    freqs_p, fft_p = calcular_fft(pos, dt)
    freqs_v, fft_v = calcular_fft(vel, dt)

    axs[i].plot(freqs_p, fft_p, label=f'Posición ({freq_dom:.2f} Hz)', color='orange')
    axs[i].plot(freqs_v, fft_v, label='Velocidad', color='green')
    axs[i].set_xlim(0, limite_grafico)
    axs[i].set_title(f'{nombre}')
    axs[i].set_xlabel('Frecuencia (Hz)')
    axs[i].set_ylabel('Magnitud')
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
plt.suptitle("Archivos con frecuencia dominante entre 4-9 Hz (posición)", y=1.02, fontsize=16)
plt.show()
