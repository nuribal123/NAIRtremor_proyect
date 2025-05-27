import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Parámetros
fs = 100
cutoff = 10

def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Precalcular min y max globales para el eje Y
ymins, ymaxs = [], []
for i in range(200, 215):
    filename = f"D:/ingenieriabiomedica/sconeGym/sconegym/sim_data_MATSUOKA6_sinsconethingsBIEN/twin_{i}.csv"
    df = pd.read_csv(filename)
    time = df["time"]
    velocity = df["wrist_hand_r3_vel"]
    if isinstance(velocity.iloc[0], str):
        velocity = velocity.apply(eval).apply(lambda x: x[0])
    else:
        velocity = velocity.astype(float)
    acc = np.gradient(velocity, time)
    acc_filtered = butter_lowpass_filter(acc, cutoff=cutoff, fs=fs)
    acc_g = acc_filtered / 9.81
    ymins.append(np.min(acc_g))
    ymaxs.append(np.max(acc_g))

ymin = min(ymins)
ymax = max(ymaxs)
# Añade margen para la leyenda
yrange = ymax - ymin
ymax += 0.15 * yrange

# Crear figura
fig = make_subplots(
    rows=5, cols=3,
    subplot_titles=[],
    vertical_spacing=0.08
)

row, col = 1, 1

for i in range(200, 215):
    filename = f"D:/ingenieriabiomedica/sconeGym/sconegym/sim_data_MATSUOKA6_sinsconethingsBIEN/twin_{i}.csv"
    df = pd.read_csv(filename)
    time = df["time"]
    velocity = df["wrist_hand_r3_vel"]
    if isinstance(velocity.iloc[0], str):
        velocity = velocity.apply(eval).apply(lambda x: x[0])
    else:
        velocity = velocity.astype(float)
    acc = np.gradient(velocity, time)
    acc_filtered = butter_lowpass_filter(acc, cutoff=cutoff, fs=fs)
    acc_g = acc_filtered / 9.81

    fig.add_trace(go.Scatter(
        x=time, y=acc_g, mode='lines',
        name=f'Aceleración - Sim {i}',
        line=dict(color='purple', width=1.5),
        showlegend=False
    ), row=row, col=col)

    # Coordenadas para la leyenda local (en unidades de datos)
    x_legend = time.iloc[-1] - 0.12 * (time.iloc[-1] - time.iloc[0])
    y_legend_top = ymax - 0.05 * yrange
    y_legend_bottom = y_legend_top - 0.12 * yrange

    # Rectángulo de fondo
    fig.add_shape(
        type="rect",
        xref=f"x{(row-1)*3+col}" if (row, col) != (1, 1) else "x",
        yref=f"y{(row-1)*3+col}" if (row, col) != (1, 1) else "y",
        x0=x_legend - 0.18 * (time.iloc[-1] - time.iloc[0]), x1=x_legend,
        y0=y_legend_bottom, y1=y_legend_top,
        fillcolor="white", line_color="black", layer="above",
        row=row, col=col
    )
    # Línea de color
    fig.add_shape(
        type="line",
        xref=f"x{(row-1)*3+col}" if (row, col) != (1, 1) else "x",
        yref=f"y{(row-1)*3+col}" if (row, col) != (1, 1) else "y",
        x0=x_legend - 0.16 * (time.iloc[-1] - time.iloc[0]),
        x1=x_legend - 0.08 * (time.iloc[-1] - time.iloc[0]),
        y0=(y_legend_top + y_legend_bottom) / 2,
        y1=(y_legend_top + y_legend_bottom) / 2,
        line=dict(color='purple', width=3),
        row=row, col=col
    )

    # Rectángulo pequeño para el texto
    text_x0 = x_legend - 0.075 * (time.iloc[-1] - time.iloc[0])
    text_x1 = x_legend - 0.01 * (time.iloc[-1] - time.iloc[0])
    text_y0 = (y_legend_top + y_legend_bottom) / 2 - 0.03 * yrange
    text_y1 = (y_legend_top + y_legend_bottom) / 2 + 0.03 * yrange

    fig.add_shape(
        type="rect",
        xref=f"x{(row-1)*3+col}" if (row, col) != (1, 1) else "x",
        yref=f"y{(row-1)*3+col}" if (row, col) != (1, 1) else "y",
        x0=text_x0, x1=text_x1,
        y0=text_y0, y1=text_y1,
        fillcolor="white", line_color="black", layer="above",
        row=row, col=col
    )

    # Texto
    fig.add_annotation(
        text=f"Filtered acceleration (G) - Sim {i}",
        x=x_legend - 0.07 * (time.iloc[-1] - time.iloc[0]),
        y=(y_legend_top + y_legend_bottom) / 2,
        xanchor="left", yanchor="middle",
        showarrow=False,
        font=dict(size=12, color="black"),
        xref=f"x{(row-1)*3+col}" if (row, col) != (1, 1) else "x",
        yref=f"y{(row-1)*3+col}" if (row, col) != (1, 1) else "y",
        row=row, col=col
    )

    col += 1
    if col > 3:
        col = 1
        row += 1

fig.update_layout(
    paper_bgcolor="white",
    height=2000,
    width=2500,
    title_text="Filtered acceleration - First 15 simulations",
    font=dict(family="Arial", size=12),
    margin=dict(t=60, b=40, l=60, r=20),
    template='simple_white'
)

fig.update_xaxes(title_text="Time (s)")
fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
fig.update_yaxes(title_text="Acceleration (G)", range=[ymin, ymax])
fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)

fig.show()

