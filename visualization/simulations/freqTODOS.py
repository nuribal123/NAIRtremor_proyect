
#----------------PARA TODOS---------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt
import glob
import os

# --- Parámetros ---
#carpeta = "D:/ingenieriabiomedica/sconeGym/sconegym/sim_data_01_05_2025_5_9_mifrec"
#carpeta = "D:/ingenieriabiomedica/sconeGym/sconegym/sim_data_03_05_2025_4_9_sumando"
carpeta = "D:/ingenieriabiomedica/sconeGym/sconegym/sim_data_MATSUOKA6_sinsconethings"
archivos_csv = sorted(glob.glob(os.path.join(carpeta, "*.csv")))
filtro_corte = 2.0  # Hz
bins = [(0, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, np.inf)]
etiquetas = ['<4 Hz', '4-5 Hz', '5-6 Hz', '6-7 Hz', '7-8 Hz', '8-9 Hz', '>9 Hz']
conteo_bins = dict.fromkeys(etiquetas, 0)

# --- Funciones ---
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
    fft_vals[0] = 0
    return freqs, fft_vals

# --- Procesamiento ---
frecuencias_dominantes = []

for archivo in archivos_csv:
    try:
        df = pd.read_csv(archivo)
        time = df["time"].values
        dt = time[1] - time[0]
        fs = 1 / dt

        pos = df["wrist_hand_r3_pos"].values
        pos_filtrada = butter_bandpass_filter(pos, 4.0, 10.0, fs)

        freqs, fft_vals = calcular_fft(pos_filtrada, dt)
        freq_dom = freqs[np.argmax(fft_vals)]
        frecuencias_dominantes.append(freq_dom)

        # Clasificar en bin
        for (min_f, max_f), etiqueta in zip(bins, etiquetas):
            if min_f <= freq_dom < max_f:
                conteo_bins[etiqueta] += 1
                break

    except Exception as e:
        print(f"Error procesando {archivo}: {e}")

# --- Gráfico ---
valores = list(conteo_bins.values())

plt.figure(figsize=(10, 6))
plt.bar(etiquetas, valores, color='skyblue')
plt.title("Distribución de frecuencias dominantes (posición de muñeca)")
plt.xlabel("Rango de frecuencia (Hz)")
plt.ylabel("Número de simulaciones")
for i, v in enumerate(valores):
    plt.text(i, v + 1, str(v), ha='center', fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# --- Extra: Mostrar conteos por consola ---
print("\nConteo por rango de frecuencia:")
for etiqueta in etiquetas:
    print(f"{etiqueta}: {conteo_bins[etiqueta]}")



