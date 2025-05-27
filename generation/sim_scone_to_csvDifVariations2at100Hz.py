#En este código el objetivo es fijar un valor base de frecuencia entre 4 y 9Hz para cada simulación, que representa el valor predominante (70%)
#Se introduce una modulación muy leve alrededor de ese valor para simular variaciones fisiológicas sin desplazar la freq. base
#se añaden variaciones de fase con 2 componentes: 1 de baja freq. para simular una deriva lenta y otra de mayor freq con baja amplit. para imitar saltos o peques fluctuaciones en la sintonización
#se usa NewMatsuoka3_template2.lua
#TODO ELLO EN LUA!

import sys
sys.path.append("C:\\Users\\balba\\OneDrive\\Documentos\\SCONE\\SconePy")
from sconetools import sconepy # type: ignore
import pandas as pd
import numpy as np
import csv
import time
import os
import shutil
from multiprocessing import Pool


# Función para generar archivo Lua a partir de una plantilla.
def generate_lua(seed, base_freq, h_value, base_lua_path, output_path):
    with open(base_lua_path, "r", encoding="utf-8") as file:
        lua_template = file.read()

    # Genera estados iniciales aleatorios entre 0 y 1 para x1, v1, x2, v2
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0.0, 1.0)
    v1 = rng.uniform(0.0, 1.0)
    x2 = rng.uniform(0.0, 1.0)
    v2 = rng.uniform(0.0, 1.0)
    # base_freq = rng.uniform(2, 4)# 2.25-5.28

    # Reemplaza los placeholders:
    # FREQ_PLACEHOLDER: valor que se desea como frecuencia base (4-9 Hz)
    # H_PLACEHOLDER: valor base para la amplitud del oscilador
    # X1, V1, X2, V2: estados iniciales
    lua_code = (
        lua_template
        .replace("FREQ_PLACEHOLDER", str(base_freq))
        .replace("H_PLACEHOLDER", str(h_value))
        .replace("X1_PLACEHOLDER", str(x1))
        .replace("V1_PLACEHOLDER", str(v1))
        .replace("X2_PLACEHOLDER", str(x2))
        .replace("V2_PLACEHOLDER", str(v2))
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(lua_code)

    #print(f"✅ LUA generado correctamente para seed {seed}")
    #print("Lua generado:\n", lua_code)


# Función para generar archivo SCONE a partir de una plantilla.
def generate_scone(seed, base_scone_path, output_path, lua_path):
    with open(base_scone_path, "r", encoding="utf-8") as file:
        scone_template = file.read()

    lua_filename = os.path.basename(lua_path)
    updated_scone = scone_template.replace("$LUA_PATH$", lua_filename)

    with open(output_path, "w") as file:
        file.write(updated_scone)
    #print(f"✅ SCONE generado correctamente para seed {seed}")


# Función wrapper para ejecutar una simulación completa.
def run_simulation_wrapper(args):
    seed, output_dir, base_lua, base_scone = args



    try:
        #print(f"🌱 [PID {os.getpid()}] Empezando sim {seed}", flush=True)

        os.makedirs(output_dir, exist_ok=True)

        # Genera parámetros aleatorios para cada simulación
        rng = np.random.default_rng(seed)
        base_freq = rng.uniform(0.183, 0.416) #antes de 2.75 a 4.83, después 2.05, 5.33
        h_value = 2.5 #estaba en 2.2 a 2.4...

        # Paths para los archivos generados
        lua_path = os.path.join(output_dir, f"NewMatsuoka_{seed}.lua")
        scone_path = os.path.join(output_dir, f"Simulation_{seed}.scone")

        # Genera los archivos de configuración
        #print(f"📄 Generando LUA para seed {seed}")
        generate_lua(seed, base_freq, h_value, base_lua, lua_path)

        #print(f"🧩 Generando SCONE para seed {seed}")
        generate_scone(seed, base_scone, scone_path, lua_path)

        # Copia el archivo de delays si no existe
        delay_src = "D:/ingenieriabiomedica/sconeGym/sconegym/sconegym/data/neural_delays_FEA_v4.zml"
        delay_dst = os.path.join(output_dir, "neural_delays_FEA_v4.zml")
        if not os.path.exists(delay_dst):
            shutil.copy(delay_src, delay_dst)

        # Ejecuta la simulación
        #print(f"🚀 Ejecutando simulación para seed {seed}")
        model = sconepy.load_model(scone_path)
        run_simulation(model, store_data=False, random_seed=seed, output_dir=output_dir, file_id=seed)

        #print(f"✅ [PID {os.getpid()}] Simulación {seed} completada", flush=True)

    except Exception as e:
        print(f"❌ [PID {os.getpid()}] Error en seed {seed}: {e}", flush=True)


# Función de simulación (ejemplo simplificado)
def run_simulation(model, store_data, random_seed, output_dir, file_id, max_time=4): #antes puesto en 6
    model.reset()
    model.set_store_data(store_data)

    rng = np.random.default_rng(random_seed)
    # Inicialización aleatoria de activaciones y posición inicial.
    muscle_activations = 0.1 + 0.4 * rng.random(len(model.muscles()))
    model.init_muscle_activations(muscle_activations)

    dof_positions = model.dof_position_array()
    dof_positions += 0.1 * rng.random(len(dof_positions)) - 0.05
    model.set_dof_positions(dof_positions)
    model.init_state_from_dofs()

    # Nombre del CSV de salida
    csv_path = os.path.join(output_dir, f"twin_{file_id}.csv")
    actuators = model.actuators()
    muscles = model.muscles()
    wrist_dofs = [d for d in model.dofs() if "wrist_hand_r3" in d.name().lower()]

    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["time"] +
                        [a.name() for a in actuators] +
                        [f"{a.name()}_min" for a in actuators] +
                        [f"{a.name()}_max" for a in actuators] +
                        [m.name() + "_exc" for m in muscles] +
                        [m.name() + "_act" for m in muscles] +
                        [d.name() + "_pos" for d in wrist_dofs] +
                        [d.name() + "_vel" for d in wrist_dofs])

        for t in np.arange(0, max_time, 0.01):
            # Actualiza entradas, por ejemplo combinando varias señales.

            #mus_in = model.muscle_force_array()
            #mus_in += model.muscle_fiber_length_array() - 1
            #mus_in += 0.2 * model.muscle_fiber_velocity_array()
            #model.set_actuator_inputs(mus_in)

            mus_in = model.delayed_muscle_force_array()
            mus_in += model.delayed_muscle_fiber_length_array() - 1.2
            mus_in += 0.1 * model.delayed_muscle_fiber_velocity_array()
            model.set_delayed_actuator_inputs(mus_in)

            inputs = [a.input() for a in actuators]
            min_inputs = [a.min_input() for a in actuators]
            max_inputs = [a.max_input() for a in actuators]
            muscle_excitations = model.muscle_excitation_array()
            muscle_activations = model.muscle_activation_array()
            wrist_positions = [d.pos() for d in wrist_dofs]
            wrist_velocities = [d.vel() for d in wrist_dofs]

            writer.writerow([t] + inputs + min_inputs + max_inputs +
                            list(muscle_excitations) + list(muscle_activations) +
                            wrist_positions + wrist_velocities)

            model.advance_simulation_to(t)

    print(f"✅ Simulación {file_id} completada y guardada en {csv_path}")


# Punto de entrada principal
if __name__ == "__main__":
    #output_dir = "D:/ingenieriabiomedica/sconeGym/sconegym/sim_dataAmpFreqTryingFreq49noNoise"
    # output_dir = "D:/ingenieriabiomedica/sconeGym/sconegym/sim_data_MATSUOKA6_sinsconethings" udsado con NewMatsuoka6_template.lua
    output_dir = "D:/ingenieriabiomedica/sconeGym/sconegym/sim_data_MATSUOKA6segundapruebadelays_sinsconethings" #usado con NewMatsuoka6_template.lua
    #base_lua = "C:/Users/balba/OneDrive/Documentos/SCONE/Tutorials/controllers/NewMatsuoka3_template2.lua"
    base_lua = "C:/Users/balba/OneDrive/Documentos/SCONE/Tutorials/controllers/NewMatsuoka6_template.lua"
    base_scone = "D:/ingenieriabiomedica/sconeGym/sconegym/sconegym/data/H0918_S2_template.scone"

    seeds = list(range(200, 7200))  # Número de simulaciones a lanzar
    args = [(seed, output_dir, base_lua, base_scone) for seed in seeds]

    #usado para debug (sin multiprocessing)
    #for seed_args in args:
    #    run_simulation_wrapper(seed_args)

    #Con multiprocessing para paralelizar
    with Pool(processes=os.cpu_count()) as pool:
        pool.map(run_simulation_wrapper, args)

#Comentarios del código:

#Frecuencia predominante:

#El valor FREQ_PLACEHOLDER (elegido en el intervalo 4–9 Hz) determina la frecuencia base para la simulación.

#Se utiliza como base para calcular tau1 y tau2 (definidos en forma inversa: tau = 1 / base_freq).

#La función update introduce una pequeña variación (±5% de la base) que garantiza que la mayor parte de la señal (cerca del 70% de la energía) permanezca en el valor predominante.

#Modulación de amplitud:

#El valor H_PLACEHOLDER es el valor base para la amplitud del oscilador.

#Se modula con un factor muy pequeño (±5% usando smooth_noise), de manera que la variación en la fuerza del temblor es sutil.

#Variaciones de fase:

#Se han introducido dos componentes de variación de fase.

#Una oscilación a 0.5 Hz con una amplitud de 0.2 radianes simula una deriva o fluctuación de fase más rápida.

#Otra a 0.07 Hz con 0.1 radianes simula cambios de fase más lentos, que en conjunto aportan la variabilidad y las pequeñas irregularidades observadas en estudios del temblor esencial.

#El parámetro resultante phase_offset se suma al tiempo en la señal del control voluntario, alterando la fase de la señal modulada.