#Con pandas para ver si se puede leer el archivo sin errores, si contiente datos y si contiene las columnas esperadas

#Se lee el lua correspondiente y extraemos Kf, x1, v1, x2, v2, h buscando los placeholders en el controlador correspondiente y se extraen

#se muestran gráficamente los parámetros para cada simulación corrupta

import os
import pandas as pd
import matplotlib.pyplot as plt
import re

csv_dir = "D:/ingenieriabiomedica/sconeGym/sconegym/sim_data_B"
corruptos = []

parametros_extraidos = []

# Función para extraer parámetros de un archivo .lua
def extraer_parametros_lua(lua_path):
    with open(lua_path, "r") as f:
        texto = f.read()

    valores = {}
    patrones = {
        "base_freq": r"base_freq\s*=\s*([0-9.]+)",
        "h": r"h\s*=\s*([0-9.]+)",
        "x1": r"x1\s*=\s*([0-9.]+)",
        "v1": r"v1\s*=\s*([0-9.]+)",
        "x2": r"x2\s*=\s*([0-9.]+)",
        "v2": r"v2\s*=\s*([0-9.]+)",
    }

    for nombre, patron in patrones.items():
        m = re.search(patron, texto)
        if m:
            valores[nombre] = float(m.group(1))
        else:
            valores[nombre] = None  # Por si hay valores aún con placeholder

    if valores["base_freq"] is not None:
        valores["Kf"] = 1.698 / (0.237 * valores["base_freq"])
    else:
        valores["Kf"] = None

    return valores

# Revisar todos los .csv
for file in os.listdir(csv_dir):
    if file.endswith(".csv") and file.startswith("twin_"):
        csv_path = os.path.join(csv_dir, file)
        try:
            df = pd.read_csv(csv_path)

            # Condiciones de corrupción:
            if df.empty:
                raise ValueError("CSV vacío.")
            if "time" not in df.columns:
                raise ValueError("Falta columna 'time'.")
            if df.shape[1] < 5:
                raise ValueError("Muy pocas columnas.")

        except Exception as e:
            print(f"❌ Corrupto: {file} — {e}")
            corruptos.append(file)

            # Extraer número
            num = re.findall(r"twin_(\d+)", file)
            if num:
                numero = num[0]
                lua_file = os.path.join(csv_dir, f"NewMatsuoka_{numero}.lua")
                if os.path.exists(lua_file):
                    params = extraer_parametros_lua(lua_file)
                    params["sim"] = numero
                    parametros_extraidos.append(params)

# ==========================
# 📊 Mostrar parámetros gráficamente
# ==========================
if parametros_extraidos:
    df_params = pd.DataFrame(parametros_extraidos)

    df_params_sorted = df_params.sort_values("sim")

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    axs = axs.ravel()
    claves = ["Kf", "x1", "v1", "x2", "v2", "h"]

    for i, clave in enumerate(claves):
        axs[i].bar(df_params_sorted["sim"], df_params_sorted[clave])
        axs[i].set_title(clave)
        axs[i].set_xlabel("Simulación")
        axs[i].set_ylabel(clave)

    plt.suptitle("Parámetros de controladores asociados a CSV corruptos")
    plt.tight_layout()
    plt.show()
else:
    print("🎉 No se detectaron CSV corruptos.")

#--------INTEGRIDAD SEÑAL----------
import pandas as pd, numpy as np

def chequear_integridad(df, dt=0.01, duration=6.0):
    ok = True
    if df.isna().any().any():
        print("❌ Hay NaN en los datos."); ok = False
    if not np.isfinite(df.values).all():
        print("❌ Hay Inf o -Inf."); ok = False
    t = df["time"].values
    dt_m = np.median(np.diff(t))
    if abs(dt_m - dt) > 1e-6:
        print(f"❌ Paso de tiempo irregular: {dt_m:.4f}")
        ok = False
    if abs(t.max() - duration) > dt:
        print(f"❌ Duración inesperada: {t.max():.3f} vs {duration}")
        ok = False
    return ok


#--------CONTROL RANGOS FISIOLÓGICOS----------
rangos = {
  "ECRL_exc": (0.0, 1.0),
  "wrist_hand_r3_pos": (-1.0, 1.0),   # rad o m según unidad
  "wrist_hand_r3_vel": (-10.0, 10.0), # rad/s o m/s
  # … añade todas las señales que te interesen
}
#CADA COLUMNA DENTRO DE RANGO?:

def chequear_rangos(df, rangos):
    all_ok = True
    for col, (mn, mx) in rangos.items():
        if col in df:
            colmin, colmax = df[col].min(), df[col].max()
            if colmin < mn or colmax > mx:
                print(f"⚠️ {col}: [{colmin:.3f}, {colmax:.3f}] fuera de [{mn}, {mx}]")
                all_ok = False
    return all_ok

#-----------------DETECCIÓN ANOMALÍAS EN DINÁMICAS----------------
#Saturaciones
def chequear_saturacion(df):
    # por ejemplo, activaciones >0.8 o =0 repetido
    exc_cols = [c for c in df if c.endswith("_exc")]
    for c in exc_cols:
        pct_max = (df[c] >= 0.8).mean()
        pct_min = (df[c] <= 0.0).mean()
        if pct_max > 0.05:
            print(f"⚠️ {c} saturada arriba el {pct_max*100:.1f}% del tiempo")
        if pct_min > 0.05:
            print(f"⚠️ {c} saturada abajo el {pct_min*100:.1f}% del tiempo")

#Estabilidad/periodicidad
from scipy.fft import rfft, rfftfreq

def frecuencia_dominante(signal, dt):
    sig = signal - signal.mean()
    fft = np.abs(rfft(sig))
    freqs = rfftfreq(len(sig), dt)
    return freqs[np.argmax(fft)]

# extrae frecuencia en ventana tardía, p. ej. t∈[2,6]
sig = df.loc[df.time>2, "ECRL_exc"].values
f = frecuencia_dominante(sig, dt=0.01)
print("Freq dom tardía:", f)


#Flujo de QC completo
import os, pandas as pd

def qc_un_archivo(csv_path, rangos):
    df = pd.read_csv(csv_path)
    ok1 = chequear_integridad(df)
    ok2 = chequear_rangos(df, rangos)
    chequear_saturacion(df)
    f_late = frecuencia_dominante(df.loc[df.time>2, "ECRL_exc"], 0.01)
    print("→ Freq tardía:", f_late)
    return ok1 and ok2

def qc_carpeta(csv_dir, rangos):
    malos = []
    for f in os.listdir(csv_dir):
        if f.startswith("twin_") and f.endswith(".csv"):
            path = os.path.join(csv_dir, f)
            print("=== QC", f, "===")
            if not qc_un_archivo(path, rangos):
                malos.append(f)
    return malos

rangos = {
  "ECRL_exc": (0,1), "FCU_exc": (0,1),
  "wrist_hand_r3_pos": (-1,1), "wrist_hand_r3_vel": (-10,10)
}

malos = qc_carpeta("ruta/a/sim_data_A", rangos)
print("Simulaciones con problemas:", malos)







