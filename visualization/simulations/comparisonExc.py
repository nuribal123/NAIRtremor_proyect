import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

muscles_of_interest = ['ECRL', 'FCU']

# Crear figura 5x3 para 15 simulaciones
fig = make_subplots(rows=5, cols=3, subplot_titles=[f"Simulación {i+1}" for i in range(15)])

row, col = 1, 1

for i in range(0, 15):
    #filename = f"D:/ingenieriabiomedica/sconeGym/sconegym/sim_data_MATSUOKA6variaciones_sinsconethings/twin_{i}.csv"
    #filename = f"D:/ingenieriabiomedica/sconeGym/sconegym/sim_data_MATSUOKA6_sinsconethings/twin_{i}.csv"
    #filename = f"D:/ingenieriabiomedica/sconeGym/sconegym/sim_data_ENHANCED_ANTIOVERFIT/twin_{i}.csv"
    filename = f"D:/ingenieriabiomedica/sconeGym/sconegym/sim_data_MATSUOKA6at60Hz_sinsconethings_amplitude/twin_{i}.csv"
    #filename = f"D:/ingenieriabiomedica/sconeGym/sconegym/sim_dataAmpFreqMatsuoka5/twin_{i}.csv"
    df = pd.read_csv(filename)
    time = df["time"]

    # Concatenar gráficas de ECRL y FCU en el mismo subplot
    for muscle in muscles_of_interest:
        excitation = df[f"{muscle}_exc"]
        activation = df[f"{muscle}_act"]
        actuator_input = df[muscle]

        fig.add_trace(go.Scatter(x=time, y=excitation, mode='lines',
                                 name=f'{muscle} Excitación', line=dict(dash='dash'),
                                 showlegend=False), row=row, col=col)
        fig.add_trace(go.Scatter(x=time, y=activation, mode='lines',
                                 name=f'{muscle} Activación',
                                 showlegend=False), row=row, col=col)
        fig.add_trace(go.Scatter(x=time, y=actuator_input, mode='lines',
                                 name=f'{muscle} Input Actuador', line=dict(dash='dot'),
                                 showlegend=False), row=row, col=col)

    col += 1
    if col > 3:
        col = 1
        row += 1

# Layout
fig.update_layout(
    height=1500,
    width=1200,
    title_text="Excitación, activación e input (ECRL y FCU) - Primeras 15 simulaciones",
    template='simple_white'
)

fig.show()
