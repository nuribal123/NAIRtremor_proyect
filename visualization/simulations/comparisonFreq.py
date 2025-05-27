# Para imprimir la frecuencia y flexo-extensión de 15 sims diferentes en una misma pantalla

import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Crear figura 5x3 para 15 simulaciones
fig = make_subplots(rows=5, cols=3, subplot_titles=[f"Simulación {i+1}" for i in range(15)])

row, col = 1, 1

for i in range(0,15):
    # Cargar el archivo de simulación i
    #filename = f"D:/ingenieriabiomedica/sconeGym/sconegym/sim_data_MATSUOKA6_sinsconethings/twin_{i}.csv"
    #filename = f"D:/ingenieriabiomedica/sconeGym/sconegym/sim_data_MATSUOKA6_sinsconethings/twin_{i}.csv"
    filename = f"D:/ingenieriabiomedica/sconeGym/sconegym/sim_data_ENHANCED_ANTIOVERFIT/twin_{i}.csv"
    df = pd.read_csv(filename)
    
    time = df["time"]
    position = df["wrist_hand_r3_pos"]
    velocity = df["wrist_hand_r3_vel"]

    # Agregar trazas
    fig.add_trace(go.Scatter(x=time, y=position, mode='lines', name='Posición',
                             line=dict(color='orange'), showlegend=False), row=row, col=col)
    fig.add_trace(go.Scatter(x=time, y=velocity, mode='lines', name='Velocidad',
                             line=dict(color='green'), showlegend=False), row=row, col=col)

    # Controlar posición
    col += 1
    if col > 3:
        col = 1
        row += 1

# Layout
fig.update_layout(
    height=1500,
    width=1200,
    title_text="Movimiento de flexo-extensión de muñeca - Primeras 15 simulaciones",
    template='simple_white'
)

fig.show()
