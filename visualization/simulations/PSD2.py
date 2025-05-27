from scipy.signal import butter, filtfilt, welch
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Parámetros
fs = 100
cutoff = 10

def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Precalcular min y max globales para el eje Y (PSD)
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
    f, Pxx = welch(acc_g, fs=fs, nperseg=256)
    ymins.append(np.min(Pxx))
    ymaxs.append(np.max(Pxx))

ymin = min(ymins)
ymax = max(ymaxs)
ymax = 15
yrange = ymax - ymin
#ymax += 0.15 * yrange  # Espacio para la leyenda

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

    # Calcular PSD con Welch
    f, Pxx = welch(acc_g, fs=fs, nperseg=256)

    fig.add_trace(go.Scatter(
        x=f, y=Pxx, mode='lines',
        name=f'PSD - Sim {i}',
        line=dict(color='purple', width=1.5),
        showlegend=False
    ), row=row, col=col)

    # Leyenda local (ajustable con x_legend)
    x_min, x_max = 0, 30  # Nuevo rango de x
    y_min, y_max = 0, 15  # Rango de y (igual que antes)

    x_legend = x_max - 0.12 * (x_max - x_min)
    y_legend_top = y_max - 0.10 * (y_max - y_min)
    y_legend_bottom = y_legend_top - 0.12 * (y_max - y_min)

    # Rectángulo de fondo (blanco)
    fig.add_shape(
        type="rect",
        xref=f"x{(row-1)*3+col}" if (row, col) != (1, 1) else "x",
        yref=f"y{(row-1)*3+col}" if (row, col) != (1, 1) else "y",
        x0=x_legend - 0.18 * (x_max - x_min), x1=x_legend,
        y0=y_legend_bottom, y1=y_legend_top,
        fillcolor="white", line_color="black", layer="above",
        row=row, col=col
    )
    # Línea de color
    fig.add_shape(
        type="line",
        xref=f"x{(row-1)*3+col}" if (row, col) != (1, 1) else "x",
        yref=f"y{(row-1)*3+col}" if (row, col) != (1, 1) else "y",
        x0=x_legend - 0.16 * (x_max - x_min),
        x1=x_legend - 0.08 * (x_max - x_min),
        y0=(y_legend_top + y_legend_bottom) / 2,
        y1=(y_legend_top + y_legend_bottom) / 2,
        line=dict(color='purple', width=3),
        row=row, col=col
    )
    # Rectángulo pequeño para el texto (blanco con borde negro)
    text_x0 = x_legend - 0.03 * (x_max - x_min)
    text_x1 = x_legend - 0.01 * (x_max - x_min)
    text_y0 = (y_legend_top + y_legend_bottom) / 2 - 0.03 * (y_max - y_min)
    text_y1 = (y_legend_top + y_legend_bottom) / 2 + 0.03 * (y_max - y_min)

    fig.add_shape(
        type="rect",
        xref=f"x{(row-1)*3+col}" if (row, col) != (1, 1) else "x",
        yref=f"y{(row-1)*3+col}" if (row, col) != (1, 1) else "y",
        x0=text_x0, x1=text_x1,
        y0=text_y0, y1=text_y1,
        fillcolor="white", line_color="black", layer="above",
        row=row, col=col
    )

    # Texto (negro)
    fig.add_annotation(
        text=f"PSD Sim {i}",
        x=(text_x0 + text_x1) / 2,
        y=(text_y0 + text_y1) / 2,
        xanchor="center", yanchor="middle",
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
    title_text="PSD - First 15 simulations",
    font=dict(family="Arial", size=12),
    margin=dict(t=60, b=40, l=60, r=20),
    template='simple_white'
)

fig.update_xaxes(title_text="Frequency (Hz)")
fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
fig.update_xaxes(range=[0, 30], dtick=2) 

fig.update_yaxes(title_text="PSD", range=[ymin, ymax])
fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)

fig.show()