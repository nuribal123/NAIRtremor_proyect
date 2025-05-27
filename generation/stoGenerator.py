#import matplotlib.pyplot as plt PARA QUE FUNCIONE CON ESTO NO PUEDE HABER PROBLEMAS DE DEPENDENCIAS!!
import numpy as np
import csv
import time
import sys
sys.path.append("C:\\Users\\balba\\OneDrive\\Documentos\\SCONE\\SconePy")
from sconetools import sconepy
import pandas as pd
import os

def run_simulation(model, store_data, random_seed, max_time=6, min_com_height=-10):
    model.reset()
    model.set_store_data(store_data)

    # Inicializar el generador de números aleatorios
    rng = np.random.default_rng(random_seed)

    # Inicializar activaciones musculares con valores aleatorios
    muscle_activations = 0.1 + 0.4 * rng.random((len(model.muscles())))
    model.init_muscle_activations(muscle_activations)

    # Modificar la pose inicial del modelo
    dof_positions = model.dof_position_array()
    dof_positions += 0.1 * rng.random(len(dof_positions)) - 0.05
    model.set_dof_positions(dof_positions)

    model.init_state_from_dofs()

    # Obtener los actuadores y músculos

    actuators = model.actuators()
    muscles = model.muscles()

    #obtener DOFs que me interesan
    wrist_dofs = [d for d in model.dofs() if "wrist_hand_r3" in d.name().lower()]

    # Abrir un archivo CSV para guardar los datos
    filename = "comprobacion.csv"
    filepath = os.path.abspath(filename)
    print(f'guardando los datos csv en {filepath}')
    file = open("dataforLSTM.csv", "w", newline="")
    writer = csv.writer(file)

    actuators = model.actuators()
    muscles = model.muscles()

    #Para ver qué está escribiendo
    writer.writerow(["time"] +
                    [a.name() for a in actuators] +  # Inputs de actuadores
                    [f"{a.name()}_min" for a in actuators] +
                    [f"{a.name()}_max" for a in actuators] +
                    [m.name() + "_exc" for m in muscles] +  # Excitaciones musculares
                    [m.name() + "_act" for m in muscles] +  # Activaciones musculares
                    [d.name() + "_pos" for d in wrist_dofs] +  # Posición de la muñeca
                    [d.name() + "_vel" for d in wrist_dofs])  # Velocidad de la muñeca

    # Ejecutar la simulación
    for t in np.arange(0, max_time, 0.01):

        # Set actuator_inputs based on muscle force, length and velocity
        mus_in = model.delayed_muscle_force_array()
        mus_in += model.delayed_muscle_fiber_length_array() - 1.2
        mus_in += 0.1 * model.delayed_muscle_fiber_velocity_array()
        model.set_delayed_actuator_inputs(mus_in)

        # Advance the simulation to time t
        # Internally, this performs as many simulations steps as required
        # The internal step size is variable, and determined by the 'accuracy'
        # setting in the .scone file
        # Obtener y almacenar valores de los actuadores

        inputs = [a.input() for a in actuators]
        min_inputs = [a.min_input() for a in actuators]
        max_inputs = [a.max_input() for a in actuators]
        muscle_excitations = model.muscle_excitation_array()
        muscle_activations = model.muscle_activation_array()
        wrist_positions = [d.pos() for d in wrist_dofs]
        wrist_velocities = [d.vel() for d in wrist_dofs]

        # Escribir en el archivo CSV
        writer.writerow([t] + inputs + min_inputs + max_inputs +
                        list(muscle_excitations) + list(muscle_activations) +
                        wrist_positions + wrist_velocities)

        # Avanzar la simulación
        model.advance_simulation_to(t)


    # Para imprimir los valores en la consola :)
    print(f"t={t:.2f}, wrist_pos={wrist_positions}, wrist_vel={wrist_velocities}")

    if store_data:
        dirname = 'sconepy_stocomprobation_' + model.name()
        filename = model.name() + f'_{random_seed}_{model.time():0.3f}_{model.com_pos().y:0.3f}'
        model.write_results(dirname, filename)
        print(f'Results written to {dirname}/{filename}; please use SCONE Studio to replay the .sto file.',
              flush=True)

        # Aplica la conversión de nombres de columna
        input_sto_path = f"C:/Users/balba/OneDrive/Documentos/SCONE/results/{dirname}/{filename}.sto"
        print(input_sto_path)
        print(os.path.exists(input_sto_path))
        if os.path.exists(input_sto_path):
            print('yay')
                # rename_sto_columns(input_sto_path)

    file.close()


if sconepy.is_supported('ModelOpenSim4'):
    i = 8
    model = sconepy.load_model(f'D:/ingenieriabiomedica/sconeGym/sconegym/sim_data_ENHANCED_ANTIOVERFIT/Simulation_{i}.scone')
    #model = sconepy.load_model('D:/ingenieriabiomedica/CSICtesis/SegundaPrueba/new.scone')
    #model = sconepy.load_model('D:/ingenieriabiomedica/CSICtesis/SegundaPrueba/AtrialARMscone.scone')

    for i in range(1, 6):
        run_simulation(model, True, 4)
    print("Please open the .sto files in SCONE studio")