import os
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import time
import joblib
import json


#----------NOMBRE DEL ARCHIVO----------
name = "701515_KINETICONLY_optunafullBIEN"
RESULTS_DIR = os.path.join("LSTMact7fullBIEN", name)
PTH_DIR= f"LSTMpth/best_emg_model_seq_LSTM_{name}.pth"
os.makedirs(RESULTS_DIR, exist_ok=True)
DATA_PTH = r"D:/ingenieriabiomedica/sconeGym/sconegym/sim_data_MATSUOKA6_sinsconethingsBIEN"

# ---------- NOTA EN RESULTS_DIR ----------

nota = """
Experimento LSTM - 701515_120epochs_optunafullBIEN

- CAMBIO: quitando el break de early stopping, se entrena hasta el final pero con 120 epochs
- Datos: sim_data_MATSUOKA6_sinsconethingsBIEN
- input_len=60, output_len=100, hidden_dim=50, num_layers=2, dropout=0.153
- Optimizer: Adam, lr=0.001, batch_size=16, epochs=70, patience=10
- Descripción: se ha cambiado la configuración de 80 10 10 a 70 15 15 para el entrenamiento, validación y test respectivamente.
"""

with open(os.path.join(RESULTS_DIR, "nota.txt"), "w", encoding="utf-8") as f:
    f.write(nota)

# ---------- CONFIGURACIÓN ----------

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Cambia el número según tu GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")
torch.backends.cudnn.benchmark = True  # Mejora el rendimiento en GPUs con tamaño de batch fijo
torch.backends.cudnn.deterministic = True  # Hace que el entrenamiento sea reproducible
torch.manual_seed(42)  # Fija la semilla para reproducibilidad
if device.type == 'cuda':
    torch.cuda.manual_seed(42)  # Fija la semilla para reproducibilidad en GPU
    print("Usando GPU")

# ---------- PARÁMETROS ----------
input_len = 60
output_len = 100
sequence_len = input_len + output_len
batch_size = 16

# ---------- CARGA Y PROCESADO ----------
data_path = DATA_PTH
csv_files = glob.glob(os.path.join(data_path, "twin_*.csv"))

input_signals = []
output_signals = []
input_scalers = []
output_scalers = []

for file in csv_files:
    df = pd.read_csv(file)
    if all(col in df.columns for col in ["wrist_hand_r3_pos", "ECRL_act", "FCR_act"]):
        signal = df["wrist_hand_r3_pos"].values.astype(np.float32)#[80:]
        input_signals.append(signal)
        output_signals.append(signal)

# División por paciente
indices = np.arange(len(input_signals))
train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42)
val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)

# Divide the signals based on training indices
input_signals_train = [input_signals[i] for i in train_idx]
output_signals_train = [output_signals[i] for i in train_idx]

# Concatenate all training signals (after optionally discarding the first 80 samples)
all_train_inputs = np.concatenate([signal[80:].reshape(-1, 1) for signal in input_signals_train])
all_train_outputs = np.concatenate([signal[80:].reshape(-1, 1) for signal in output_signals_train])

scaler_input = MinMaxScaler()
scaler_output = MinMaxScaler()

scaler_input.fit(all_train_inputs)
scaler_output.fit(all_train_outputs)


def create_sequences(indices):
    X, y = [], []

    for idx in indices:
        # Get raw simulation data and apply the [80:] offset
        input_seq = input_signals[idx][80:]
        output_seq = output_signals[idx][80:]
        
        # Only create sequences if there are enough timesteps
        if len(input_seq) < sequence_len:
            print(f"Simulation {idx} skipped: insufficient timesteps (has {len(input_seq)}, needs {sequence_len})")
            continue

        # Scale using the global scalers (already fitted)
        input_scaled = scaler_input.transform(input_seq.reshape(-1, 1)).flatten()
        output_scaled = scaler_output.transform(output_seq.reshape(-1, 1)).flatten()
        
        # Generate sequences without crossing boundaries
        for i in range(len(input_scaled) - sequence_len + 1):
            X.append(input_scaled[i:i+input_len])
            y.append(output_scaled[i+input_len:i+input_len+output_len])
    return np.array(X), np.array(y)

#saving the scalers
joblib.dump(scaler_input, "scaler_input.pkl")
joblib.dump(scaler_output, "scaler_output.pkl")
#loading the scalers
scaler_input = joblib.load("scaler_input.pkl")
scaler_output = joblib.load("scaler_output.pkl")

X_train, y_train = create_sequences(train_idx)
X_val, y_val = create_sequences(val_idx)
X_test, y_test = create_sequences(test_idx)  # Necesitamos estos para revertir

# ---------- DATASET ----------
class EMGDataset(Dataset):
    def __init__(self, X, y, scalers=None):
        self.X = torch.tensor(X).unsqueeze(-1).float()
        self.y = torch.tensor(y).unsqueeze(-1).float() #batch, output_len, 1

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(EMGDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(EMGDataset(X_val, y_val), batch_size=batch_size)
test_loader = DataLoader(EMGDataset(X_test, y_test), batch_size=batch_size)

# ---------- MODELO ----------
#sequence to sequence

# Model: Encoder using the final hidden state only mapped to output sequence
class SeqLSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=50, output_dim=1, num_layers=2, dropout_prob=0.153248867):
        super(SeqLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob)
        # Map the last hidden state to (output_len * output_dim) total outputs
        self.fc = nn.Linear(hidden_dim, output_len * output_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # x: (batch_size, input_len, 1)
        _, (hidden, _) = self.lstm(x)         # hidden: (num_layers, batch_size, hidden_dim)
        hidden_last = hidden[-1]              # (batch_size, hidden_dim)
        hidden_last = self.dropout(hidden_last)
        out = self.fc(hidden_last)            # (batch_size, output_len * output_dim)
        out = out.view(-1, output_len, 1)       # (batch_size, output_len, 2)
        return out

model = SeqLSTMModel(output_dim=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

#----------- COMPROBACIONES ----------


#overlap entre índices
print("Train indices:", train_idx)
print("Validation indices:", val_idx)
print("Test indices:", test_idx)

# ---------- ENTRENAMIENTO ----------
patience = 10
counter = 0
best_val_loss = float('inf')

train_losses = []
val_losses = []

start_time = time.time()
for epoch in range(70):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        pred = model(batch_X)
        loss = criterion(pred, batch_y)
        optimizer.zero_grad()


        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for val_X, val_y in val_loader:
            val_X, val_y = val_X.to(device), val_y.to(device)
            val_pred = model(val_X)
            val_loss += criterion(val_pred, val_y).item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss) #para guardar la pérdida de validación

    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Batch Loss = {loss.item():.4f}")

    # Visualization of predictions for debugging
    #plt.plot(batch_y[0].cpu().numpy(), label="True")  # Replace y_batch with batch_y
    #plt.plot(pred[0].detach().cpu().numpy(), label="Predicted")
    #plt.legend()
    #plt.show()

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), PTH_DIR)
        print("✅ Modelo guardado")
        counter = 0
         # Guarda la mejor epoch cada vez que mejora
        with open(os.path.join(RESULTS_DIR, "early_stopping_epoch.txt"), "w") as f:
            f.write(str(epoch + 1))  # +1 para que sea 1-based
    else:
        counter += 1
        if counter >= patience:
            print("⏳ Early stopping")
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"⏱️ Tiempo de entrenamiento: {elapsed_time:.2f} segundos")
            with open(os.path.join(RESULTS_DIR, "training_time.txt"), "w") as f:
                f.write(f"Training time (s): {elapsed_time:.2f}")
            break

# Guardando los datos para plots en otro file de las curvas de aprendizaje
np.save(os.path.join(RESULTS_DIR, f"train_losses_{name}.npy"), np.array(train_losses))
np.save(os.path.join(RESULTS_DIR, f"val_losses_{name}.npy"), np.array(val_losses))

# Graficar las curvas de aprendizaje
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Pérdida de entrenamiento", color="blue")
plt.plot(val_losses, label="Pérdida de validación", color="orange")
plt.title("Curvas de aprendizaje")
plt.xlabel("Época")
plt.ylabel("Pérdida (Loss)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, f"learning_curve_{name}.png"))
plt.show()


# ---------- EVALUACIÓN ----------
def evaluate_and_plot(loader, title="Evaluación"):
    model.eval()
    all_preds, all_trues = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            all_preds.append(pred.cpu().numpy())
            all_trues.append(y_batch.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_trues)

    # Use global scaler for inverse transformation
    pred_inv = scaler_output.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)
    true_inv = scaler_output.inverse_transform(y_true.reshape(-1, 1)).reshape(y_true.shape)

    mae = mean_absolute_error(true_inv.reshape(-1), pred_inv.reshape(-1))
    rmse = np.sqrt(mean_squared_error(true_inv.reshape(-1), pred_inv.reshape(-1)))
    r2 = r2_score(true_inv.reshape(-1), pred_inv.reshape(-1))

    print(f"\n📊 {title} EMG Prediction:")
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

    # Plot examples
    def plot_example(i):
        y_real = true_inv[i]
        y_pred_plot = pred_inv[i]
        t = np.linspace(0, 1, output_len)

        plt.figure(figsize=(10, 4))
        plt.plot(t, y_real[:, 0], label='real', color='purple')
        plt.plot(t, y_pred_plot[:, 0], '--', label='predicho', color='violet')
        plt.title(f"{title} - Ejemplo #{i}")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Activación")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    for i in [0, 10, 30]:
        if i < len(true_inv):
            plot_example(i)

    # Plot error histogram
    errors = np.abs(pred_inv - true_inv)
    plt.figure(figsize=(6, 4))
    plt.hist(errors[:, :, 0].flatten(), bins=50, alpha=0.6, label="ECRL")
    plt.title("Histograma de errores absolutos")
    plt.xlabel("Error")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return y_pred, y_true

# Evaluación
y_pred, y_true = evaluate_and_plot(test_loader, "Test")

# ---------- INFERENCIA RÁPIDA ----------
def medir_tiempo_inferencia(model, input_len=60):
    model.eval()
    dummy_input = torch.randn(1, input_len, 1).to(device)  # Simulando una entrada de tamaño (1, input_len, 1)

    start_time = time.time()
    with torch.no_grad():
        output = model(dummy_input)
    end_time = time.time()

    duracion = end_time - start_time
    print(f"⏱️ Tiempo de inferencia: {duracion*1000:.2f} ms")
    print(f"📈 Output shape: {output.shape}")
    return output

pred = medir_tiempo_inferencia(model)
futuro_ms = np.arange(10, 1010, 10)
plt.plot(futuro_ms, pred.cpu().numpy()[0, :, 0], label='ECRL predicho')
plt.xlabel("Tiempo en el futuro (ms)")
plt.ylabel("Activación estimada")
plt.title("Predicción de activación muscular 1s hacia el futuro")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ---------- ERROR POR TIMESTEP ----------
errors = np.abs(y_pred - y_true)
mean_errors = errors.mean(axis=0)

t = np.arange(output_len)
plt.figure(figsize=(10, 4))
plt.plot(t, mean_errors[:, 0], label="ECRL", color='green')
plt.title("Error promedio por paso futuro")
plt.xlabel("Timestep (0 = inmediato, 100 = 1s futuro)")
plt.ylabel("Error absoluto medio")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


#Debugging

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

print(f"y_train min: {y_train.min()}, max: {y_train.max()}")

print(f"Input scaler min: {scaler_input.data_min_}, max: {scaler_input.data_max_}")
print(f"Output scaler min: {scaler_output.data_min_}, max: {scaler_output.data_max_}")

y_pred_train, y_true_train = evaluate_and_plot(train_loader, "Train")


#-------------TEST------------------
# Evaluate on test set
def evaluate(loader):
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            all_preds.append(pred.cpu().numpy())
            all_trues.append(y_batch.cpu().numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_trues)
    return y_pred, y_true

y_pred_test, y_true_test = evaluate(test_loader)

# Save predictions for later analysis/plotting
np.save(os.path.join(RESULTS_DIR, f"y_pred_test_{name}.npy"), y_pred_test)
np.save(os.path.join(RESULTS_DIR, f"y_true_test_{name}.npy"), y_true_test)

settings = {
    "batch_size": batch_size,
    "optimizer": "Adam",
    "learning_rate": 0.001,  # or your variable if it's not hardcoded
    "num_layers": 2,         # or your variable
    "dropout": 0.153248867,  # or your variable
    "hidden_dim": 50,        # or your variable
    "input_len": input_len,
    "output_len": output_len,
    "epochs": 120,
    "patience": patience,
    "seed": 42
}

with open(os.path.join(RESULTS_DIR, f"settings_{name}.json"), "w") as f:
    json.dump(settings, f, indent=4)

np.save(os.path.join(RESULTS_DIR, f"test_indices_{name}.npy"), test_idx)
np.save(os.path.join(RESULTS_DIR, f"train_indices_{name}.npy"), train_idx)
np.save(os.path.join(RESULTS_DIR, f"val_indices_{name}.npy"), val_idx)