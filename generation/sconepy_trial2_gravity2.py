import numpy as np
import csv
import time
import sys
sys.path.append("C:\\Users\\balba\\OneDrive\\Documentos\\SCONE\\SconePy")
from sconetools import sconepy
import pandas as pd
import os

def run_simulation(model, store_data, random_seed, max_time=3, min_com_height=-10):
    model.reset()
    model.set_store_data(store_data)

    rng = np.random.default_rng(random_seed)

    muscle_activations = 0.1 + 0.4 * rng.random((len(model.muscles())))
    model.init_muscle_activations(muscle_activations)

    dof_positions = model.dof_position_array()
    dof_positions += 0.1 * rng.random(len(dof_positions)) - 0.05
    model.set_dof_positions(dof_positions)
    model.init_state_from_dofs()

    actuators = model.actuators()
    muscles = model.muscles()
    wrist_dofs = [d for d in model.dofs() if "wrist_hand_r3" in d.name().lower()]

    filename = "gravity_check.csv"
    filepath = os.path.abspath(filename)
    print(f'Guardando los datos CSV en {filepath}')
    file = open(filename, "w", newline="")
    writer = csv.writer(file)

    writer.writerow(["time", "com_y"] +
                    [d.name() + "_pos" for d in wrist_dofs] +
                    [d.name() + "_vel" for d in wrist_dofs])

    for t in np.arange(0, max_time, 0.01):
        # NO aplicamos entradas al modelo -> sin controlador
        model.set_actuator_inputs(np.zeros(len(actuators)))

        model.advance_simulation_to(t)

        com_y = model.com_pos().y
        wrist_positions = [d.pos() for d in wrist_dofs]
        wrist_velocities = [d.vel() for d in wrist_dofs]

        writer.writerow([t, com_y] + wrist_positions + wrist_velocities)

        print(f"[t={t:.2f}] COM Y = {com_y:.4f}")  # Imprime para ver si cae

        if com_y < min_com_height:
            print(f"Abortando simulación: COM Y por debajo de {min_com_height}")
            break

    if store_data:
        dirname = 'sconepy_gravity_test_' + model.name()
        filename = model.name() + f'_gravity_{random_seed}_{model.time():0.3f}_{model.com_pos().y:0.3f}'
        model.write_results(dirname, filename)
        print(f'Resultados escritos en {dirname}/{filename}')

    file.close()

if sconepy.is_supported('ModelOpenSim4'):
    model = sconepy.load_model('D:/ingenieriabiomedica/sconeGym/sconegym/sconegym/data/H0918_S2.scone')

    print("Gravedad del modelo:", model.gravity())
    run_simulation(model, True, 1)
    print("Prueba completada — revisa el archivo gravity_check.csv")
