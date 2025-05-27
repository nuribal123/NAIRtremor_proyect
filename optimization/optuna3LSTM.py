
import optuna
import optunahub
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
import time
import json
import joblib
from torch.utils.tensorboard import SummaryWriter
from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna.visualization.matplotlib import plot_parallel_coordinate, plot_contour
from sklearn.model_selection import train_test_split

#% pip install optuna-dashboard
#% optuna-dashboard sqlite:///db.sqlite3

#def objective(trial: optuna.Trial) -> float:
#   x = trial.suggest_float("x", -5, 5)
#   y = trial.suggest_float("y", -5, 5)
#   return x**2 + y**2
#module = optunahub.load_module(package="samplers/auto_sampler"...


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

results_dir = "optuna_resultsLSTMBIEN"
os.makedirs(results_dir, exist_ok=True)

# ---------- PARÁMETROS ----------
input_len = 60
output_len = 100
sequence_len = input_len + output_len
batch_size = 64 #menos mejores resultados pero tarda más
data_path = r"/home/nair-group/nuria/NAIRtremor-main/sim_data"

scaler_input = MinMaxScaler()
scaler_output = MinMaxScaler()


# ---------- CARGA Y PROCESADO ----------
data_path = r"D:/ingenieriabiomedica/sconeGym/sconegym/sim_data_MATSUOKA6_sinsconethingsBIEN"
csv_files = glob.glob(os.path.join(data_path, "twin_*.csv"))

input_signals = []
output_signals = []
input_scalers = []
output_scalers = []

for file in csv_files:
    df = pd.read_csv(file)
    if all(col in df.columns for col in ["wrist_hand_r3_pos", "ECRL_act", "FCR_act"]):
        input_signal = df["wrist_hand_r3_pos"].values.astype(np.float32)#[80:]
        emg = np.stack([df["ECRL_act"], df["FCR_act"]], axis=1).astype(np.float32)#[80:]
        input_signals.append(input_signal)
        output_signals.append(emg)

# División por paciente
indices = np.arange(len(input_signals))
train_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42)
val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)

# Divide the signals based on training indices
input_signals_train = [input_signals[i] for i in train_idx]
output_signals_train = [output_signals[i] for i in train_idx]

# Concatenate all training signals (after optionally discarding the first 80 samples)
all_train_inputs = np.concatenate([signal[80:].reshape(-1, 1) for signal in input_signals_train])
all_train_outputs = np.concatenate([signal[80:] for signal in output_signals_train])

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
        output_scaled = scaler_output.transform(output_seq)
        
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
        self.y = torch.tensor(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(EMGDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(EMGDataset(X_val, y_val), batch_size=batch_size)
test_loader = DataLoader(EMGDataset(X_test, y_test), batch_size=batch_size)


# ---------- MODELO ----------
class SeqLSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=2, num_layers=2, dropout_prob=0.2):
        super(SeqLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_len * output_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        hidden_last = hidden[-1]
        hidden_last = self.dropout(hidden_last)
        out = self.fc(hidden_last)
        out = out.view(-1, output_len, 2)
        return out

#-------------OPTUNA IMPLEMENTATION---------------
def objective(trial):
    # Hiperparámetros sugeridos por Optuna
    hidden_dim = trial.suggest_categorical("hidden_dim", [20, 35, 50])  # Neuronas
    num_layers = trial.suggest_categorical("num_layers", [1, 2])  # Capas
    lr = trial.suggest_categorical("lr", [0.001, 0.0005, 0.0001])  # Tasa de aprendizaje
    dropout_prob = trial.suggest_float("dropout_prob", 0.1, 0.5)  # Dropout
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])  # Tamaño del batch
    
    train_loader = DataLoader(EMGDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(EMGDataset(X_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(EMGDataset(X_test, y_test), batch_size=batch_size)


    # Modelo basado en LSTMnext2
    model = SeqLSTMModel(hidden_dim=hidden_dim, num_layers=num_layers, dropout_prob=dropout_prob).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Entrenamiento en un pequeño subconjunto de datos
    for epoch in range(10):  # Reducido para Optuna
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

    # Validación
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for val_X, val_y in val_loader:
            val_X, val_y = val_X.to(device), val_y.to(device)
            val_pred = model(val_X)
            val_loss += criterion(val_pred, val_y).item()

    val_loss /= len(val_loader)
    return val_loss

# ---------- EJECUCIÓN DEL STUDY ----------
study = optuna.create_study(direction="minimize")
start_time = time.time()
study.optimize(objective, n_trials=70)
end_time = time.time()
elapsed_time = end_time - start_time

# ---------- RESULTADO MEJOR MODELO ----------
print("🎯 Mejor conjunto de hiperparámetros:")
print(study.best_params)

best_params_path = os.path.join(results_dir, "best_params.json")
with open(best_params_path, "w") as f:
    json.dump(study.best_params, f, indent =4)

print(f"Mejores parámetros guardados en {best_params_path}")

trials_path = os.path.join(results_dir, "trials.csv")
df_trials = study.trials_dataframe()
df_trials.to_csv(trials_path, index=False)
print(f"Resultados de los ensayos guardados en {trials_path}")

# Entrenar con los mejores parámetros encontrados
best_params = study.best_params
model = SeqLSTMModel(
    hidden_dim=best_params["hidden_dim"],
    num_layers=best_params["num_layers"]
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
criterion = nn.MSELoss()

# Historial de optimización
fig_optimization_history = plot_optimization_history(study)
fig_optimization_history.write_image(os.path.join(results_dir, "optimization_history.png"))

# Importancia de los hiperparámetros
fig_param_importances = plot_param_importances(study)
fig_param_importances.write_image(os.path.join(results_dir, "param_importances.png"))

# Coordenadas paralelas
fig_parallel_coordinate = plot_parallel_coordinate(study)
fig_parallel_coordinate.figure.savefig(os.path.join(results_dir, "parallel_coordinate.png"))

# Contorno
fig_contour = plot_contour(study)
if isinstance(fig_contour, np.ndarray):
    # Si es un array de ejes, toma la figura del primer eje
    fig_contour[0].figure.savefig(os.path.join(results_dir, "contour.png"))
else:
    # Si es un solo eje
    fig_contour.figure.savefig(os.path.join(results_dir, "contour.png"))

print(f"Gráficas guardadas en: {results_dir}")


#FALTA ENTRENAMIENTO COMPLETO Y EVALUACIÓN
# Entrenamiento completo
#for epoch in range(30):  # Entrenamiento completo
    #model.train()
    #total_loss = 0
    #for batch_X, batch_y in train_loader:
     #   batch_X, batch_y = batch_X.to(device), batch_y.to(device)
       # pred = model(batch_X)
      #  loss = criterion(pred, batch_y)
        #optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()
        #total_loss += loss.item()

    # Validación
    #model.eval()
    #val_loss = 0
    #with torch.no_grad():
    #    for val_X, val_y in val_loader:
    #        val_X, val_y = val_X.to(device), val_y.to(device)
    #        val_pred = model(val_X)
    #        val_loss += criterion(val_pred, val_y).item()

    #val_loss /= len(val_loader)
    #print(f"Epoch {epoch+1}: Train Loss = {total_loss:.4f}, Val Loss = {val_loss:.4f}")

summary_path = os.path.join(results_dir, "optuna_summary.txt")
with open(summary_path, "w") as f:
    f.write(f"Mejor conjunto de hiperparámetros:\n{json.dumps(study.best_params, indent=4)}\n\n")
    f.write(f"Mejor valor objetivo: {study.best_value:.6f}\n")
    f.write(f"Número de trial ganador: {study.best_trial.number}\n")
    f.write(f"Tiempo total de optimización: {elapsed_time/60:.2f} minutos ({elapsed_time:.1f} segundos)\n")
    f.write(f"Resultados de los trials guardados en: {trials_path}\n")
    f.write(f"Gráficas guardadas en: {results_dir}\n")
print(f"Resumen guardado en {summary_path}")
joblib.dump(study, os.path.join(results_dir, "optuna_study.pkl"))

study = optuna.create_study(direction="minimize", storage="sqlite:///db.sqlite3", study_name="my_study", load_if_exists=True)
