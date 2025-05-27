import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Parámetros ----------
name = "701515_KINETICONLY_optunafullBIEN"
run_name = "701515_KINETICONLY"
IMU_DIR = "realIMU"
CSV_FILE = "real_imu_input_100Hz.csv"
PTH_DIR = f"LSTMpth/best_emg_model_seq_LSTM_{name}.pth"
RESULTS_LSTM_DIR = os.path.join("LSTMact7fullBIEN", name)
RESULTS_DIR = os.path.join("LastResults", run_name)
os.makedirs(RESULTS_DIR, exist_ok=True)

input_len = 60
output_len = 100
sequence_len = input_len + output_len

# ---------- Cargar CSV ----------
csv_path = os.path.join(IMU_DIR, CSV_FILE)
data = pd.read_csv(csv_path)
imu_signal = data.iloc[:, 1].values.astype(np.float32)

# ---------- Normalizar ----------
scaler_input = joblib.load("scaler_input.pkl")
imu_scaled = scaler_input.transform(imu_signal.reshape(-1, 1)).flatten()

# ---------- Crear secuencias ----------
X_real, Y_real = [], []
for i in range(len(imu_scaled) - sequence_len + 1):
    X_real.append(imu_scaled[i : i+input_len])
    Y_real.append(imu_signal[i+input_len : i+input_len+output_len])  # sin escalar para comparar

X_real = np.array(X_real)
Y_real = np.array(Y_real)

X_real_tensor = torch.tensor(X_real).unsqueeze(-1).float().to(device)  # (N, input_len, 1)

# ---------- Definir modelo ----------
class SeqLSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=50, output_dim=1, num_layers=2, dropout_prob=0.153248867):
        super(SeqLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_len * output_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        hidden_last = hidden[-1]
        hidden_last = self.dropout(hidden_last)
        out = self.fc(hidden_last)
        out = out.view(-1, output_len, 1)
        return out

# ---------- Cargar modelo ----------
model = SeqLSTMModel(output_dim=1)
model.load_state_dict(torch.load(PTH_DIR, map_location=device))
model.to(device)
model.eval()

# ---------- Predicción ----------
with torch.no_grad():
    predictions = model(X_real_tensor)  # (N, output_len, 1)

# ---------- Reescalar salida ----------
scaler_output = joblib.load("scaler_output.pkl")
pred_np = predictions.squeeze(-1).cpu().numpy()  # (N, output_len)
pred_rescaled = scaler_output.inverse_transform(pred_np)

# ---------- Visualizar ----------
plt.figure()
plt.plot(pred_rescaled[0], label='Predicted Output')
plt.title('Primera Predicción LSTM sobre IMU real')
plt.xlabel('Timestep (modelo entrenado a 100Hz)')
plt.ylabel('Posición estimada')
plt.legend()
plt.grid()
plt.show()

# ---------- Gráfica 1: Predicción vs Real ----------
plt.figure()
plt.plot(pred_rescaled[0], label='Predicción')
plt.plot(Y_real[0], label='Real')
plt.title("Primera predicción vs señal real")
plt.xlabel("Timestep")
plt.ylabel("Posición")
plt.legend()
plt.grid()
plt.show()

# ---------- Gráfica 2: Promedio predicho vs promedio real ----------
mean_pred = np.mean(pred_rescaled, axis=0)
mean_real = np.mean(Y_real, axis=0)

plt.figure()
plt.plot(mean_real, label='Promedio Real', color='blue')
plt.plot(mean_pred, label='Promedio Predicho', color='orange')
plt.title('Promedio de secuencias reales vs predichas')
plt.xlabel('Timestep')
plt.ylabel('Posición')
plt.legend()
plt.grid()
plt.show()

# ---------- Gráfica 3: Error de una predicción ----------
error_0 = Y_real[0] - pred_rescaled[0]
plt.figure()
plt.plot(error_0, label='Error (Real - Predicho)', color='red')
plt.title('Error de la primera predicción')
plt.xlabel('Timestep')
plt.ylabel('Error (posición)')
plt.legend()
plt.grid()
plt.show()

# ---------- Gráfica 4: Error absoluto medio ----------
abs_errors = np.abs(Y_real - pred_rescaled)
mean_abs_error = np.mean(abs_errors, axis=0)

plt.figure()
plt.plot(mean_abs_error, label='Error absoluto medio', color='green')
plt.title('Error absoluto medio por timestep')
plt.xlabel('Timestep')
plt.ylabel('Error absoluto')
plt.legend()
plt.grid()
plt.show()

# ---------- Gráfica 5: Error medio con desviación estándar ----------
errores = pred_rescaled - Y_real
mean_error = np.mean(errores, axis=0)
std_error = np.std(errores, axis=0)

plt.figure()
plt.plot(mean_error, label='Error medio', color='purple')
plt.fill_between(np.arange(output_len), mean_error - std_error, mean_error + std_error,
                 color='purple', alpha=0.3, label='±1 desviación estándar')
plt.title("Error medio y desviación típica por timestep")
plt.xlabel("Timestep")
plt.ylabel("Error")
plt.legend()
plt.grid()
plt.show()
