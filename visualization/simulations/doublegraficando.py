import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Cargar el CSV
filename = "D:/ingenieriabiomedica/sconeGym/sconegym/sim_data_MATSUOKA6_sinsconethings_copia/twin_6.csv"
df = pd.read_csv(filename)

# Lista de músculos
muscles = [col.replace("_exc", "") for col in df.columns if "_exc" in col]

# Variables articulares
position = df["wrist_hand_r3_pos"]
velocity = df["wrist_hand_r3_vel"]
time = df["time"]

for muscle in muscles:
    excitation = df[muscle + "_exc"]
    activation = df[muscle + "_act"]
    actuator_input = df[muscle]

    # Crear figura con 2 filas (subplots apilados)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f"Respuesta del músculo {muscle}", "Movimiento articular (muñeca)")
    )

    # --- Subplot 1: Activación muscular ---
    fig.add_trace(go.Scatter(x=time, y=excitation, mode='lines', name='Excitación',
                             line=dict(dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=time, y=activation, mode='lines', name='Activación'), row=1, col=1)
    fig.add_trace(go.Scatter(x=time, y=actuator_input, mode='lines', name='Input Actuador',
                             line=dict(dash='dot')), row=1, col=1)

    # --- Subplot 2: Movimiento articular ---
    fig.add_trace(go.Scatter(x=time, y=position, mode='lines', name='Posición muñeca',
                             line=dict(color='orange')), row=2, col=1)
    fig.add_trace(go.Scatter(x=time, y=velocity, mode='lines', name='Velocidad muñeca',
                             line=dict(color='green')), row=2, col=1)

    # Layout general
    fig.update_layout(
        height=700,
        showlegend=True,
        template='simple_white',
        xaxis2_title='Tiempo (s)',  # Solo en el segundo gráfico
        yaxis1_title='Activación / Excitación',
        yaxis2_title='Ángulo / Velocidad'
    )

    # Mostrar figura interactiva
    fig.show()
