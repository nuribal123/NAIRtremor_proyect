import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import joblib
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------- CONFIGURATION ----------

# Cargar los índices de test
name = "701515_120epochs_optunafullBIEN" #name of the LSTMnextScale
RESULTS_DIR = os.path.join("LSTMact7fullBIEN", name) #RESULTS_DIR of the LSTMnextScale
test_idx = np.load(os.path.join(RESULTS_DIR, f"test_indices_{name}.npy"))

input_len = 60
output_len = 100
sequence_len = input_len + output_len
csv_folder = "D:/ingenieriabiomedica/sconeGym/sconegym/sim_data_MATSUOKA6_sinsconethings_largoBIEN"
results_dir = "animation_resultsMetrics"
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

# Para almacenar errores por ventana de todos los twins
all_window_errors = []

for idx in test_idx:
    csv_file = os.path.join(csv_folder, f"twin_{idx}.csv")
    if not os.path.exists(csv_file):
        print(f"Archivo {csv_file} no encontrado, lo salto.")
        continue

    df = pd.read_csv(csv_file)
    if not all(col in df.columns for col in ["wrist_hand_r3_pos", "ECRL_act", "FCR_act"]):
        continue

    input_signal = df["wrist_hand_r3_pos"].values.astype(np.float32)
    real_ecrl = df["ECRL_act"].values.astype(np.float32)
    real_fcr = df["FCR_act"].values.astype(np.float32)

    input_scaled = scaler_input.transform(input_signal.reshape(-1, 1)).flatten()

    predictions = []
    for i in range(0, len(input_scaled) - input_len, output_len):
        input_window = input_scaled[i:i + input_len]
        input_tensor = torch.tensor(input_window).unsqueeze(0).unsqueeze(-1).float()
        with torch.no_grad():
            pred = model(input_tensor).squeeze(0).numpy()
            predictions.append(pred)

    if not predictions:
        continue

    predictions = np.concatenate(predictions, axis=0)
    predictions_descaled = scaler_output.inverse_transform(predictions)

    y_true = np.stack([real_ecrl, real_fcr], axis=1)[:predictions_descaled.shape[0]]
    y_pred = predictions_descaled

    # Calcular errores por ventana
    num_windows = (len(input_signal) - input_len) // output_len
    twin_window_errors = []
    for i in range(num_windows):
        start = i * output_len
        end = start + output_len
        err = y_pred[start:end] - y_true[start:end]  # (output_len, 2)
        mean_err = np.mean(np.abs(err), axis=0)      # (2,)
        twin_window_errors.append(mean_err)
    all_window_errors.append(twin_window_errors)

    # Opcional: guardar para scatter plot
    if idx == test_idx[0]:
        all_y_true = y_true
        all_y_pred = y_pred
    else:
        all_y_true = np.vstack([all_y_true, y_true])
        all_y_pred = np.vstack([all_y_pred, y_pred])

# Convertir a array para fácil manejo
all_window_errors = np.array(all_window_errors)  # (num_twins, num_windows, 2)

# Calcular media y std por ventana globalmente
mean_per_window = np.nanmean(all_window_errors, axis=0)  # (num_windows, 2)
std_per_window = np.nanstd(all_window_errors, axis=0)    # (num_windows, 2)

# Graficar
plt.figure(figsize=(10, 4))
plt.plot(mean_per_window[:, 0], label="Mean abs error ECRL")
plt.plot(mean_per_window[:, 1], label="Mean abs error FCR")
plt.fill_between(np.arange(mean_per_window.shape[0]), mean_per_window[:, 0] - std_per_window[:, 0], mean_per_window[:, 0] + std_per_window[:, 0], alpha=0.2, label="Std ECRL")
plt.fill_between(np.arange(mean_per_window.shape[0]), mean_per_window[:, 1] - std_per_window[:, 1], mean_per_window[:, 1] + std_per_window[:, 1], alpha=0.2, label="Std FCR")
plt.xlabel("Window (across all test twins)")
plt.ylabel("Error")
plt.title("Mean abs error and std per window (test twins)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "mean_error_std_per_window_test_twins.png"))
plt.show()

# Scatter plots solo para test twins
for i, label in enumerate(["ECRL", "FCR"]):
    plt.figure(figsize=(5, 5))
    plt.scatter(all_y_true[:, i], all_y_pred[:, i], alpha=0.4, label=label)
    min_val = min(all_y_true[:, i].min(), all_y_pred[:, i].min())
    max_val = max(all_y_true[:, i].max(), all_y_pred[:, i].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    plt.xlabel("Valor real")
    plt.ylabel("Predicción")
    plt.title(f"Scatter plot - {label} (test twins)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"scatter_{label}_test_twins.png"))
    plt.show()