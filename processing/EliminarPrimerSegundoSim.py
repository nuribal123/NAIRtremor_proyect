# Elimino el primer segundo de simulación de todos los .csv y hago que le nuevo segundo 1 sea el anterior segundo 2

# Se carga cada archivo .csv que comienza con twin_
# Elimina todas las filas donde time <= 1.00
# Se resta 1.0 a la columna time para que el nuevo tiempo comience en 0
# Guarda el archivo sobrescribiendo el original

import os
import pandas as pd

# Ruta donde están los CSV 
#csv_dir = "D:/ingenieriabiomedica/sconeGym/sconegym/sim_data_MATSUOKA6_sinsconethings_largoBIEN"  # o sim_data_B, etc.
csv_dir = "generation/data/data_replicate_processed"

# Recorre todos los archivos en la carpeta
for root, dirs, files in os.walk(csv_dir):
    for file in files:
        if file.endswith(".csv") and file.startswith("twin_"):
            csv_path = os.path.join(root, file)
            df = pd.read_csv(csv_path)

            # Filtra filas a partir de t > 1.0
            df = df[df["time"] > 1.00].copy()

            # Resta 1 segundo al tiempo para reindexar desde 0
            df["time"] = df["time"] - 1.00

            # Guarda el CSV sobreescribiendo el original
            df.to_csv(csv_path, index=False)
            print(f"✅ Procesado: {csv_path}")
