import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import joblib
import seaborn as sns

# ---------- CONFIGURATION ----------
input_len = 60
output_len = 100
sequence_len = input_len + output_len
#csv_file = "D:/ingenieriabiomedica/sconeGym/sconegym/sim_data_MATSUOKA6_sinsconethings/simlarga/ruta_csv_largo.csv"
csv_file = "D:/ingenieriabiomedica/sconeGym/sconegym/sim_data_MATSUOKA6_sinsconethings/simlargasinseg/twin_207.csv"
results_dir = "animation_results"
os.makedirs(results_dir, exist_ok=True)

# Load the trained model
class SeqLSTMModel(torch.nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=2, num_layers=2, dropout_prob=0.2):
        super(SeqLSTMModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = torch.nn.Linear(hidden_dim, output_len * output_dim)
        self.dropout = torch.nn.Dropout(dropout_prob)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        hidden_last = hidden[-1]
        hidden_last = self.dropout(hidden_last)
        out = self.fc(hidden_last)
        out = out.view(-1, output_len, 2)
        return out

model = SeqLSTMModel()
#model.load_state_dict(torch.load("LSTMpth/best_emg_model_seq_LSTM_LSTMact7better.pth", map_location=torch.device("cpu")))
model.load_state_dict(torch.load("LSTMpth/best_emg_model_seq_LSTM_LSTMact7better.pth"))
model.eval()

# Load scalers
scaler_input = joblib.load("scaler_input.pkl")
scaler_output = joblib.load("scaler_output.pkl")

# ---------- LOAD AND PROCESS DATA ----------
df = pd.read_csv(csv_file)
if not all(col in df.columns for col in ["wrist_hand_r3_pos", "ECRL_act", "FCR_act"]):
    raise ValueError("CSV must contain 'wrist_hand_r3_pos', 'ECRL_act', and 'FCR_act' columns.")

input_signal = df["wrist_hand_r3_pos"].values.astype(np.float32)
real_ecrl = df["ECRL_act"].values.astype(np.float32)
real_fcr = df["FCR_act"].values.astype(np.float32)

# Scale input signal
input_scaled = scaler_input.transform(input_signal.reshape(-1, 1)).flatten()

# Generate predictions using sliding windows
predictions = []
for i in range(0, len(input_scaled) - input_len, output_len):
    input_window = input_scaled[i:i + input_len]
    input_tensor = torch.tensor(input_window).unsqueeze(0).unsqueeze(-1).float()
    with torch.no_grad():
        pred = model(input_tensor).squeeze(0).numpy()
        predictions.append(pred)

predictions = np.concatenate(predictions, axis=0)
predictions_descaled = scaler_output.inverse_transform(predictions)

# ---------- CREATE ANIMATION ----------
def create_animation(input_signal, predictions, real_ecrl, real_fcr, input_len, output_len):
    plt.style.use("ggplot")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [1, 2]})

    # Flexo-extension (top)
    line_signal, = ax1.plot([], [], label="Flexion-Extension (Input)", color="blue", linewidth=1)
    line_window, = ax1.plot([], [], label="Sliding Window", color="cyan", linewidth=1)
    ax1.set_ylabel("Flexion-extension", fontsize=14)
    ax1.legend(fontsize=10, frameon=False)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Muscle activations (bottom)
    line_pred_ecrl, = ax2.plot([], [], label="Predicted ECRL activation", color="black", linestyle="--", linewidth=1)
    line_pred_fcr, = ax2.plot([], [], label="Predicted FCR activation", color="gray", linestyle="--", linewidth=1)
    line_real_ecrl, = ax2.plot([], [], label="Baseline ECRL", color="black", alpha=0.5, linewidth=1)
    line_real_fcr, = ax2.plot([], [], label="Baseline FCR", color="gray", alpha=0.5, linewidth=1)
    ax2.set_xlabel("Time (s)", fontsize=14)
    ax2.set_ylabel("Muscle activation", fontsize=14)
    ax2.legend(fontsize=10, frameon=False)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Set x axis in seconds
    ax2.set_xlim(0, len(input_signal) * 0.01)
    ax1.set_xlim(0, len(input_signal) * 0.01)

    def init():
        line_signal.set_data([], [])
        line_window.set_data([], [])
        line_pred_ecrl.set_data([], [])
        line_pred_fcr.set_data([], [])
        line_real_ecrl.set_data([], [])
        line_real_fcr.set_data([], [])
        return line_signal, line_window, line_pred_ecrl, line_pred_fcr, line_real_ecrl, line_real_fcr

    last_pred_ecrl = []
    last_pred_fcr = []
    last_x_pred = []

    def update(frame):
        start = frame
        end = frame + input_len

        # X axis in seconds
        x_full = np.arange(len(input_signal)) * 0.01
        x_window = np.arange(start, end) * 0.01

        # Flexo-extension
        line_signal.set_data(x_full, input_signal)
        line_window.set_data(x_window, input_signal[start:end])

        # --- Rectángulo azul claro para la sliding window (input) ---
        [p.remove() for p in ax1.patches]
        ax1.axvspan(x_window[0], x_window[-1], color='lightblue', alpha=0.3)

        # --- Rectángulo gris claro para la predicción (output) ---
        [p.remove() for p in ax2.patches]
        if frame >= input_len:
            pred_idx = (frame - input_len) // output_len
            if (frame - input_len) % output_len == 0:
                pred_start = pred_idx * output_len
                pred_end = pred_start + output_len
                if pred_end <= len(predictions):
                    last_x_pred[:] = np.arange(frame, frame + output_len) * 0.01
                    last_pred_ecrl[:] = predictions[pred_start:pred_end, 0]
                    last_pred_fcr[:] = predictions[pred_start:pred_end, 1]
                else:
                    last_x_pred[:] = []
                    last_pred_ecrl[:] = []
                    last_pred_fcr[:] = []
            # Dibuja el rectángulo gris claro detrás de la predicción
            if last_x_pred:
                ax2.axvspan(last_x_pred[0], last_x_pred[-1], color='lightgrey', alpha=0.3)
            line_pred_ecrl.set_data(last_x_pred, last_pred_ecrl)
            line_pred_fcr.set_data(last_x_pred, last_pred_fcr)
        else:
            line_pred_ecrl.set_data([], [])
            line_pred_fcr.set_data([], [])

        line_real_ecrl.set_data(x_full, real_ecrl)
        line_real_fcr.set_data(x_full, real_fcr)

        # Autoscale axes
        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()

        return line_signal, line_window, line_pred_ecrl, line_pred_fcr, line_real_ecrl, line_real_fcr
    num_frames = len(input_signal) - input_len - output_len
    ani = FuncAnimation(fig, update, frames=range(0, num_frames, 5), init_func=init, blit=True, interval=100)
    #ani.save(os.path.join(results_dir, "real_time_animation.mp4"), writer="ffmpeg", fps=10)
    plt.show()

# Run animation
create_animation(input_signal, predictions_descaled, real_ecrl, real_fcr, input_len, output_len)