import os
import pandas as pd

# Ruta donde están los CSV (ajusta según tu estructura)
csv_dir = "D:/ingenieriabiomedica/sconeGym/sconegym/sim_data_MATSUOKA6_sinsconethingsBIEN"  # o sim_data_B, etc.

# Recorre todos los archivos en la carpeta
for root, dirs, files in os.walk(csv_dir):
    for file in files:
        if file.endswith(".csv") and file.startswith("twin_"):
            csv_path = os.path.join(root, file)
            df = pd.read_csv(csv_path)

            # Limita la duración 
            df = df[df["time"] <= 2.41]

            # Guarda el CSV sobreescribiendo el original
            df.to_csv(csv_path, index=False)
            print(f"✅ Procesado: {csv_path}")