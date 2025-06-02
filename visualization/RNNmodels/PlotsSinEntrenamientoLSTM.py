#File to plot the results from the LSTM using the pth
#Ensure the settings of the LSTM are the same used in the training (number of layers, hidden dimensions, learning rate and )


import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import joblib
import time
import matplotlib as mpl
import seaborn as sns
import json

# ----------- CONFIGURATION GPU -----------

# Configuración de GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# ----------- CHANGE DEPENDING ON THE DIRECTORIES GIVEN FOR THE LSTM TRAINING -----------

#name = "701515_200epochs_optunafullBIEN"
name = "701515_LSTMact7guardandoentrena2_optunafullBIEN_condatatrain"
run_name = "701515_LSTMplotsBIEN"  # Name for the results to be saved
RESULTS_DIR = os.path.join("LastResultsTRIALS", run_name)
RESULTS_LSTM_DIR = os.path.join("LSTMact7fullBIEN", name)
os.makedirs(RESULTS_DIR, exist_ok=True)
PTH_DIR= f"LSTMpth/best_emg_model_seq_LSTM_{name}.pth"
DATA_PTH = r"D:/ingenieriabiomedica/sconeGym/sconegym/sim_data_MATSUOKA6_sinsconethingsBIEN"

# ---------------- GRAPH STYLES -----------------

plt.style.use("seaborn-v0_8-paper")  
color = sns.color_palette("colorblind") 

mpl.rcParams['font.size'] = 14  
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 16
mpl.rcParams['figure.titlesize'] = 14

# ---------- CONFIGURATION ----------

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Change the number depending on your GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")
torch.backends.cudnn.benchmark = True  # Mejora el rendimiento en GPUs con tamaño de batch fijo
torch.backends.cudnn.deterministic = True  # Hace que el entrenamiento sea reproducible
torch.manual_seed(42)  # Fija la semilla para reproducibilidad
if device.type == 'cuda':
    torch.cuda.manual_seed(42)  # Fija la semilla para reproducibilidad en GPU
    print("Usando GPU")

# ---------- PARAMETERS ----------
input_len = 60
output_len = 100
sequence_len = input_len + output_len
batch_size = 16

#comprobar las configuraciones

#with open(os.path.join(RESULTS_LSTM_DIR, f"settings_{name}.json"), "r") as f:
#    settings = json.load(f)

#print("LSTM Settings:")
#for k, v in settings.items():
 #   print(f"{k}: {v}")

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
        input_signal = df["wrist_hand_r3_pos"].values.astype(np.float32)#[80:]
        emg = np.stack([df["ECRL_act"], df["FCR_act"]], axis=1).astype(np.float32)#[80:]
        input_signals.append(input_signal)
        output_signals.append(emg)

# División por paciente
indices = np.arange(len(input_signals))
train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42)
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
#sequence to sequence

# Model: Encoder using the final hidden state only mapped to output sequence
class SeqLSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=50, output_dim=2, num_layers=2, dropout_prob=0.153524):
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
        out = out.view(-1, output_len, 2)       # (batch_size, output_len, 2)
        return out

model = SeqLSTMModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003426)
criterion = nn.MSELoss()


# Carga del modelo entrenado
model = SeqLSTMModel()
# Primero mueve el modelo al dispositivo
model.to(device)
# Luego carga los pesos - NOTA: Modificado para cargar al dispositivo correcto
#model.load_state_dict(torch.load("D:/ingenieriabiomedica/CSICtesis/modelsGRU/best_emg_model_seqGRU.pth", 
#                                  map_location=device))
#model.load_state_dict(torch.load("LSTMpth/best_emg_model_seq_LSTM_LSTMact7better.pth", map_location=device))
model.load_state_dict(torch.load(PTH_DIR))
model.eval()



# Configuración para mostrar gráficas
plt.ion()  # Modo interactivo de matplotlib
plt.show()  # Para asegurar que las gráficas se muestren


# Modificación de las funciones de ploteo para mostrar y guardar

def evaluate_and_plot(loader, title="Evaluación"):
    model.eval()
    all_preds, all_trues = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            pred = model(X_batch)
            all_preds.append(pred.cpu().numpy())
            all_trues.append(y_batch.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_trues)

    flat_pred = y_pred.reshape(-1, 2)
    flat_true = y_true.reshape(-1, 2)

    pred_inv = scaler_output.inverse_transform(flat_pred)
    true_inv = scaler_output.inverse_transform(flat_true)

    mae = mean_absolute_error(true_inv, pred_inv)
    rmse = np.sqrt(mean_squared_error(true_inv, pred_inv))
    r2 = r2_score(true_inv, pred_inv)

    metrics = f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}\n"
    with open(os.path.join(RESULTS_DIR, "metrics.txt"), "w") as f:
        f.write(metrics)

    print(f"\n📊 {title} EMG Prediction:")
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

    def plot_example(i):
        y_real = scaler_output.inverse_transform(y_true[i])
        y_pred_plot = scaler_output.inverse_transform(y_pred[i])
        t = np.arange(output_len)

        plt.figure(figsize=(10, 4)) #10, 4
        plt.plot(t, y_real[:, 0], label='ECRL real', color='purple')
        plt.plot(t, y_pred_plot[:, 0], '--', label='ECRL pred', color='violet')
        plt.plot(t, y_real[:, 1], label='FCR real', color='sandybrown')
        plt.plot(t, y_pred_plot[:, 1], '--', label='FCR pred', color='brown')
        plt.title(f"{title} - Example #{i}", fontsize=16)
        plt.xlabel("Timestep (100Hz)")
        plt.ylabel("Activation")
        #plt.legend(fontsize = 12, loc = "best", frameon=False)
        plt.grid(False)

        plt.legend(
            loc="upper right",
            fontsize=14,
            frameon=True,
            fancybox=True,
            framealpha=0.8,
            borderpad=1.2
        )
        plt.subplots_adjust(top=0.82) 

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"SinEntrenamientoExample_{run_name}{i}.png"))
        plt.show()  # Añadido para mostrar la gráfica
        #plt.subplots_adjust(top=0.85)
        #plt.legend(loc='upper right', fontsize=12, frameon=False)
        plt.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), fontsize=12, frameon=False)

    # Mostrar 3 ejemplos
    for i in [0, 49, 62]:
        plot_example(i)

    # Calcula el error absoluto medio por ejemplo (N ejemplos)
    # y_pred, y_true tienen shape (N, output_len, 2)
    abs_errors = np.abs(y_pred - y_true)  # (N, 100, 2)
    mean_abs_error_per_example = abs_errors.mean(axis=(1, 2))  # (N,)

    idx_worst = np.argmax(mean_abs_error_per_example)
    plot_example(idx_worst)


    # Histograma de errores
    errors = np.abs(pred_inv - true_inv)
    plt.figure(figsize=(6, 3))
    plt.hist(errors[:, 0], bins=50, alpha=0.6, label="ECRL")
    plt.hist(errors[:, 1], bins=50, alpha=0.6, label="FCR")
    plt.title("Absolute Error Histogram", fontsize=16)
    plt.xlabel("Error", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    #plt.legend(fontsize=12, loc="best", frameon=False)

    plt.legend(
        loc="upper right",
        fontsize=14,
        frameon=True,
        fancybox=True,
        framealpha=0.8,
        borderpad=1.2
    )
    plt.subplots_adjust(top=0.82) 

    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"SinEntrenamientoErrors_histogram{run_name}.png"))
    plt.show()  # Añadido para mostrar la gráfica

    return y_pred, y_true

def plot_input_output_pair(index=0):
    # Entrada de flexo-extensión (60 puntos = 600ms a 100Hz)
    input_raw = X_test[index]  # ya está escalado
    input_real = scaler_input.inverse_transform(input_raw.reshape(-1, 1)).flatten()

    # Predicción EMG (100 puntos = 1s)
    pred_emg = scaler_output.inverse_transform(y_pred[index])
    # Suponiendo que tienes también la señal real EMG para el test (puedes usar y_true si ya la tienes)
    real_emg = scaler_output.inverse_transform(y_test[index])

    t_input = np.arange(-input_len, 0) * 10   # de -600ms a 0ms
    t_output = np.arange(0, output_len) * 10    # de 0ms a 1000ms

    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Graficar la señal de flexo-extensión en ax1
    ax1.plot(t_input, input_real, color='blue', label='Flexo-extension (real)', linewidth=1)
    ax1.set_ylabel("Wrist Flexoextension Angle", color='blue', fontsize=14)
    ax1.set_xlabel("Time (ms)", fontsize=14)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(False)
    
    # Crear un segundo eje para las activaciones predichas y reales
    ax2 = ax1.twinx()
    # Graficar las activaciones predichas con líneas discontinuas
    ax2.plot(t_output, pred_emg[:, 0], 'purple', linestyle='--', label='ECRL predicted', linewidth=1)
    ax2.plot(t_output, pred_emg[:, 1], 'brown', linestyle='--', label='FCR predicted', linewidth=1)
    # Graficar las activaciones reales (si dispones de ellas) como línea continua
    ax2.plot(t_output, real_emg[:, 0], 'purple', label='ECRL real', alpha=0.7, linewidth=1)
    ax2.plot(t_output, real_emg[:, 1], 'brown', label='FCR real', alpha=0.7, linewidth=1)

    ax2.set_ylabel("Muscle Activation", color='black', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='black')
    
    # Dar espacio separando la leyenda del gráfico. Por ejemplo, usamos bbox_to_anchor
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    #ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', bbox_to_anchor=(1.15, 1.0), fontsize=12, frameon=False)
    
    ax2.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper right",
        fontsize=10,
        frameon=True,
        fancybox=True,
        framealpha=0.8,
        borderpad=1.2,
        ncol=1
    )
    fig.subplots_adjust(top=0.82)

    plt.title("Movement Input + predicted vs. real muscle activations", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"Input_output_pair_{index}_{run_name}.png"))
    plt.show()
    

# Modificación de las funciones de ploteo para error por timestep y mu/sigma
def plot_error_analysis(y_pred, y_true):
    # Error por timestep
    errors_by_step = np.abs(y_pred - y_true)  # (N, 100, 2)
    mean_errors = errors_by_step.mean(axis=0)  # (100, 2)

    t = np.arange(output_len)
    plt.figure(figsize=(10, 4))
    plt.plot(t, mean_errors[:, 0], label="ECRL", color='purple')
    plt.plot(t, mean_errors[:, 1], label="FCR", color='brown')
    plt.title("Mean error per future timestep", fontsize=16)
    plt.xlabel("Timestep", fontsize=14)
    plt.ylabel("Mean Absolute Error (MAE)", fontsize=14)
    plt.legend(fontsize=12, loc="best", frameon=False)
    plt.grid(False)

    plt.legend(
        loc="upper right",
        fontsize=14,
        frameon=True,
        fancybox=True,
        framealpha=0.8,
        borderpad=1.2
    )
    plt.subplots_adjust(top=0.82) 

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"SinEntrenamientoError_by_timestep_{run_name}.png"))
    plt.show()

    # Media y desviación estándar del error por timestep
    errors = y_pred - y_true  # (N, 100, 2)
    mu = errors.mean(axis=0)  # (100, 2)
    sigma = errors.std(axis=0)  # (100, 2)
    t = np.arange(output_len)

    plt.figure(figsize=(10, 4))
    for i, label in enumerate(["ECRL", "FCR"]):
        plt.plot(t, mu[:, i], label=f"μ {label}", linestyle='-')
        plt.fill_between(t, mu[:, i] - sigma[:, i], mu[:, i] + sigma[:, i], alpha=0.3, label=f"μ±σ {label}")

    plt.title("Mean Error and standard deviation per timestep", fontsize=16)
    plt.xlabel("Timestep", fontsize=14)
    plt.ylabel("Error", fontsize=14)
    #plt.legend(fontsize=12, loc="best", frameon=False)

    plt.legend(
        loc="upper right",
        fontsize=10,
        frameon=True,
        fancybox=True,
        framealpha=0.6,
        borderpad=1.0
    )
    plt.subplots_adjust(top=0.82) 

    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"SinEntrenamientoError_mu_sigma_{run_name}.png"))
    plt.show()

# Ejecución principal
# Evaluar en conjunto de test
y_pred, y_true = evaluate_and_plot(test_loader, "Test")
# Evaluar en conjunto de validación
y_pred_val, y_true_val = evaluate_and_plot(val_loader, "Validation")

# Análisis de errores
plot_error_analysis(y_pred, y_true)

# Visualizar par de entrada-salida
plot_input_output_pair(index=5)

# Visualizar predicción futura
futuro_ms = np.arange(10, 1010, 10)  # 100 pasos de 10 ms
plt.figure(figsize=(10, 4))
plt.plot(futuro_ms, y_pred[0, :, 0], label='Predicted ECRL')
plt.plot(futuro_ms, y_pred[0, :, 1], label='Predicted FCR')
plt.xlabel("Time in the future (ms)", fontsize=14)
plt.ylabel("Estimated Activation", fontsize=14)
plt.title("Muscle Predicted Activation 1s in the Future", fontsize=16)
plt.legend(fontsize=12, loc="best", frameon=False)
plt.grid(False)
plt.savefig(os.path.join(RESULTS_DIR, f"SinEntrenamientoFuture_prediction_{run_name}.png"))
plt.show()

print(f"\n✅ Proceso completado. Resultados guardados en: {RESULTS_DIR}")


def plot_learning_curves(train_losses, val_losses, results_dir, run_name):
    # Graficar las curvas de aprendizaje
    plt.figure(figsize=(7, 6))
    plt.plot(train_losses, label="Training Loss LSTM", color="blue")
    plt.plot(val_losses, label="Validation Loss LSTM", color="orange")
    plt.title("Learning Curves", fontsize=18)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Loss" ,fontsize=16)
    #plt.legend(fontsize=12, loc="best", frameon=False)

    plt.legend(
        loc="upper right",
        fontsize=16,
        frameon=True,
        fancybox=True,
        framealpha=0.8,
        borderpad=1.2
    )
    plt.subplots_adjust(top=0.82) 

    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"learning_curve_{run_name}.png"))
    plt.show()

train_losses = np.load(os.path.join(RESULTS_LSTM_DIR, f"train_losses_{name}.npy"))
val_losses = np.load(os.path.join(RESULTS_LSTM_DIR, f"val_losses_{name}.npy"))
plot_learning_curves(train_losses, val_losses, RESULTS_DIR, name)

#Scatter plot para comparar predicciones y valores reales
def scatter_plot(y_pred, y_true):
    flat_pred = y_pred.reshape(-1, 2)
    flat_true = y_true.reshape(-1, 2)

    pred_inv = scaler_output.inverse_transform(flat_pred)
    true_inv = scaler_output.inverse_transform(flat_true)

    for i, label in enumerate(['ECRL', 'FCR']):
        plt.figure(figsize=(5, 5))
        plt.scatter(true_inv[:, i], pred_inv[:, i], alpha=0.4, label=label, color=color[i])
        plt.plot([0, 1], [0, 1], 'k--')  # Línea diagonal

        #Adding shaded regions
        x = np.linspace(0, 1, 100)
        plt.fill_between(x, x * 0.8, x * 1.2, color='lightgray', alpha=0.2, label="80% Precision")
        plt.fill_between(x, x * 0.9, x * 1.1, color='lightgray', alpha=0.4, label="90% Precision")


        plt.xlabel("Muscle Activation Simulation")
        plt.ylabel("Muscle Activation Prediction")
        plt.title(f"Scatter plot LSTM prediction- {label}")
        plt.legend(
            loc="upper right",
            fontsize=14,
            frameon=True,
            fancybox=True,
            framealpha=0.8,
            borderpad=1.2
        )
        plt.subplots_adjust(top=0.82) 
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"Scatter_{label}_{run_name}.png"))
        plt.show()

scatter_plot(y_pred, y_true)

def scatter_plot(y_pred, y_true):
    flat_pred = y_pred.reshape(-1, 2)
    flat_true = y_true.reshape(-1, 2)

    pred_inv = scaler_output.inverse_transform(flat_pred)
    true_inv = scaler_output.inverse_transform(flat_true)

    for i, label in enumerate(['ECRL', 'FCR']):
        plt.figure(figsize=(5, 5))
        plt.scatter(true_inv[:, i], pred_inv[:, i], alpha=0.4, label=label, color=color[i])
        plt.plot([0, 1], [0, 1], 'k--')  # Línea diagonal

        plt.xlabel("Muscle Activation Simulation")
        plt.ylabel("Muscle Activation Prediction")
        plt.title(f"Scatter plot LSTM prediction- {label}")
        plt.legend(
            loc="upper right",
            fontsize=14,
            frameon=True,
            fancybox=True,
            framealpha=0.8,
            borderpad=1.2
        )
        plt.subplots_adjust(top=0.82) 
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"Scatter_{label}_{run_name}.png"))
        plt.show()

scatter_plot(y_pred, y_true)


# Boxplot errores absolutos
def boxplot_abs_errors(y_pred, y_true):
    flat_pred = y_pred.reshape(-1, 2)
    flat_true = y_true.reshape(-1, 2)

    pred_inv = scaler_output.inverse_transform(flat_pred)
    true_inv = scaler_output.inverse_transform(flat_true)

    abs_errors = np.abs(pred_inv - true_inv)
    
    plt.figure(figsize=(6, 4))
    plt.boxplot([abs_errors[:, 0], abs_errors[:, 1]], labels=["ECRL", "FCR"])
    plt.title("Boxplot absolute errors", fontsize=16)
    plt.ylabel("Absolute Error", fontsize=14)
    plt.xlabel("Muscle Activation", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"Boxplot_Errors_{run_name}.png"))
    plt.show()

def sensitivity_analysis(index=0, perturbation=0.05):
    original_input = X_test[index].copy()
    perturbed_input = original_input + perturbation  # Aumentamos entrada

    # Inferencia
    model.eval()
    with torch.no_grad():
        X_orig = torch.tensor(original_input).unsqueeze(0).unsqueeze(-1).to(device)
        X_pert = torch.tensor(perturbed_input).unsqueeze(0).unsqueeze(-1).to(device)

        y_orig = model(X_orig).cpu().numpy().squeeze()
        y_pert = model(X_pert).cpu().numpy().squeeze()

    y_orig = scaler_output.inverse_transform(y_orig)
    y_pert = scaler_output.inverse_transform(y_pert)

    t = np.arange(output_len)

    plt.figure(figsize=(10, 4))
    plt.plot(t, y_orig[:, 0], label="Original ECRL", color='purple')
    plt.plot(t, y_pert[:, 0], '--', label="Perturbed ECRL", color='violet')
    plt.plot(t, y_orig[:, 1], label="Original FCR", color='brown')
    plt.plot(t, y_pert[:, 1], '--', label="Perturbed FCR", color='sandybrown')
    plt.title("Sensitivity Analysis - Modified Input")
    plt.xlabel("Timestep")
    plt.ylabel("Activation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"Sensitivity_{run_name}_{index}.png"))
    plt.show()

def robustness_test_noise(index=0, noise_std=0.1):
    noisy_input = X_test[index] + np.random.normal(0, noise_std, size=X_test[index].shape)

    model.eval()
    with torch.no_grad():
        X_orig = torch.tensor(X_test[index]).unsqueeze(0).unsqueeze(-1).to(device)
        X_noisy = torch.tensor(noisy_input).unsqueeze(0).unsqueeze(-1).to(device)

        y_orig = model(X_orig).cpu().numpy().squeeze()
        y_noisy = model(X_noisy).cpu().numpy().squeeze()

    y_orig = scaler_output.inverse_transform(y_orig)
    y_noisy = scaler_output.inverse_transform(y_noisy)

    t = np.arange(output_len)

    plt.figure(figsize=(10, 4))
    plt.plot(t, y_orig[:, 0], label="Original ECRL", color='purple')
    plt.plot(t, y_noisy[:, 0], '--', label="ECRL with noise", color='violet')
    plt.plot(t, y_orig[:, 1], label="Original FCR", color='brown')
    plt.plot(t, y_noisy[:, 1], '--', label="FCR with noise", color='sandybrown')
    plt.title("Robustness Test - Input Noise")
    plt.xlabel("Timestep")
    plt.ylabel("Activation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"Robustness_noise_{run_name}_{index}.png"))
    plt.show()



def plot_learning_curves(train_losses, val_losses, results_dir, run_name):
    # Cargar la epoch de early stopping
    early_stopping_epoch = None
    early_stopping_path = os.path.join(RESULTS_LSTM_DIR, "early_stopping_epoch.txt")
    if os.path.exists(early_stopping_path):
        with open(early_stopping_path, "r") as f:
            early_stopping_epoch = int(f.read().strip())

    plt.figure(figsize=(7, 6))
    plt.plot(train_losses, label="Training Loss LSTM", color="blue")
    plt.plot(val_losses, label="Validation Loss LSTM", color="orange")
    if early_stopping_epoch is not None:
        plt.axvline(x=early_stopping_epoch-1, color='red', linestyle='--', label=f"Early stopping (epoch {early_stopping_epoch})")
    plt.title("Learning Curves", fontsize=18)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend(
        loc="upper right",
        fontsize=16,
        frameon=True,
        fancybox=True,
        framealpha=0.8,
        borderpad=1.2
    )
    plt.subplots_adjust(top=0.82)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"learning_curve_{run_name}.png"))
    plt.show()

def compute_metrics(y_pred, y_true, scaler_output):
    flat_pred = y_pred.reshape(-1, 2)
    flat_true = y_true.reshape(-1, 2)
    pred_inv = scaler_output.inverse_transform(flat_pred)
    true_inv = scaler_output.inverse_transform(flat_true)
    mae = mean_absolute_error(true_inv, pred_inv)
    rmse = np.sqrt(mean_squared_error(true_inv, pred_inv))
    r2 = r2_score(true_inv, pred_inv)
    return mae, rmse, r2

# Evaluar en conjunto de entrenamiento
y_pred_train, y_true_train = evaluate_and_plot(train_loader, "Train")
# Evaluar en conjunto de validación
y_pred_val, y_true_val = evaluate_and_plot(val_loader, "Validation")
# Evaluar en conjunto de test
y_pred, y_true = evaluate_and_plot(test_loader, "Test")

# Calcula métricas para cada conjunto
mae_train, rmse_train, r2_train = compute_metrics(y_pred_train, y_true_train, scaler_output)
mae_val, rmse_val, r2_val = compute_metrics(y_pred_val, y_true_val, scaler_output)
mae_test, rmse_test, r2_test = compute_metrics(y_pred, y_true, scaler_output)

# Guarda en metrics.txt
with open(os.path.join(RESULTS_DIR, "metrics.txt"), "w") as f:
    f.write("MAE, RMSE, R2\n")
    f.write(f"Train: {mae_train:.4f}, {rmse_train:.4f}, {r2_train:.4f}\n")
    f.write(f"Val:   {mae_val:.4f}, {rmse_val:.4f}, {r2_val:.4f}\n")
    f.write(f"Test:  {mae_test:.4f}, {rmse_test:.4f}, {r2_test:.4f}\n")

    
def plot_kinematic_and_emg_with_windows(index=0, input_len=60, output_len=100):
    # 1. Recuperar y desescalar la entrada y salida
    input_raw = X_test[index]  # (input_len,)
    input_real = scaler_input.inverse_transform(input_raw.reshape(-1, 1)).flatten()
    real_emg = scaler_output.inverse_transform(y_test[index])  # (output_len, 2)

    # 2. Predicción
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(input_raw).unsqueeze(0).unsqueeze(-1).to(device)
        y_pred = model(X_tensor).cpu().numpy().squeeze()
    pred_emg = scaler_output.inverse_transform(y_pred)

    # 3. Ejes temporales
    t_input = np.arange(-input_len, 0) * 10   # de -600ms a 0ms
    t_output = np.arange(0, output_len) * 10  # de 0ms a 1000ms

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

    # 4. Gráfica superior: movimiento cinético
    ax1.plot(np.concatenate([t_input, t_output]), 
             np.concatenate([input_real, [np.nan]*output_len]), 
             color='black', label='Wrist flexoextension angle')
    # Rectángulo azul claro para la ventana de entrada
    ax1.axvspan(t_input[0], t_input[-1], color='lightblue', alpha=0.3, label='Input window')
    ax1.set_ylabel("Wrist Angle", fontsize=14)
    ax1.legend(loc="Upper left", fontsize=12, frameon=True)
    ax1.set_title("Kinematic Input and Muscle Activation Prediction", fontsize=16)

    # 5. Gráfica inferior: activaciones musculares
    # Rectángulo gris claro para la ventana de predicción
    ax2.axvspan(t_output[0], t_output[-1], color='lightgrey', alpha=0.3, label='Prediction window')
    # Activaciones reales
    ax2.plot(t_output, real_emg[:, 0], color='royalblue', label='ECRL real')
    ax2.plot(t_output, real_emg[:, 1], color='tan', label='FCR real')
    # Activaciones predichas (superpuestas)
    ax2.plot(t_output, pred_emg[:, 0], '--', color='royalblue', label='ECRL pred')
    ax2.plot(t_output, pred_emg[:, 1], '--', color='tan', label='FCR pred')
    ax2.set_ylabel("Muscle Activation", fontsize=14)
    ax2.set_xlabel("Time (ms)", fontsize=14)
    ax2.legend(loc="upper left", fontsize=12, frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"InputOutput_windows_{index}_{run_name}.png"))
    plt.show()



def plot_kinematic_and_emg_with_windows(index=0, input_len=60, output_len=100):
    # 1. Recuperar y desescalar la entrada y salida
    input_raw = X_test[index]  # (input_len,)
    input_real = scaler_input.inverse_transform(input_raw.reshape(-1, 1)).flatten()
    real_emg = scaler_output.inverse_transform(y_test[index])  # (output_len, 2)

    # 2. Predicción
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(input_raw).unsqueeze(0).unsqueeze(-1).to(device)
        y_pred = model(X_tensor).cpu().numpy().squeeze()
    pred_emg = scaler_output.inverse_transform(y_pred)

    # 3. Ejes temporales
    t_input = np.arange(-input_len, 0) * 10   # de -600ms a 0ms
    t_output = np.arange(0, output_len) * 10  # de 0ms a 1000ms

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

    # 4. Gráfica superior: movimiento cinético
    ax1.plot(np.concatenate([t_input, t_output]), 
             np.concatenate([input_real, [np.nan]*output_len]), 
             color='black', label='Wrist flexoextension angle')
    # Rectángulo azul claro para la ventana de entrada
    ax1.axvspan(t_input[0], t_input[-1], color='lightblue', alpha=0.3, label='Input window')
    ax1.set_ylabel("Wrist Angle", fontsize=14)
    ax1.legend(loc="upper left", fontsize=12, frameon=True)
    ax1.set_title("Kinematic Input and Muscle Activation Prediction", fontsize=16)

    # 5. Gráfica inferior: activaciones musculares
    # Rectángulo gris claro para la ventana de predicción
    ax2.axvspan(t_output[0], t_output[-1], color='lightgrey', alpha=0.3, label='Prediction window')
    # Activaciones reales
    ax2.plot(t_output, real_emg[:, 0], color='blue', label='ECRL real')
    ax2.plot(t_output, real_emg[:, 1], color='brown', label='FCR real')
    # Activaciones predichas (superpuestas)
    ax2.plot(t_output, pred_emg[:, 0], '--', color='blue', label='ECRL pred')
    ax2.plot(t_output, pred_emg[:, 1], '--', color='brown', label='FCR pred')
    ax2.set_ylabel("Muscle Activation", fontsize=14)
    ax2.set_xlabel("Time (ms)", fontsize=14)
    ax2.legend(loc="upper left", fontsize=12, frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"InputOutput_windows_{index}_{run_name}.png"))
    plt.show()

# Ejemplo de uso:
plot_kinematic_and_emg_with_windows(index=0, input_len=input_len, output_len=output_len)

def plot_kinematic_and_emg_split(index=0, input_len=60, output_len=100):
    # Recuperar la simulación completa
    sim_idx = test_idx[index // (len(X_test) // len(test_idx))]
    df = pd.read_csv(csv_files[sim_idx])
    input_signal_full = df["wrist_hand_r3_pos"].values.astype(np.float32)[80:]
    emg_full = np.stack([df["ECRL_act"], df["FCR_act"]], axis=1).astype(np.float32)[80:]

    input_raw = X_test[index]
    input_real = scaler_input.inverse_transform(input_raw.reshape(-1, 1)).flatten()
    real_emg = scaler_output.inverse_transform(y_test[index])
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(input_raw).unsqueeze(0).unsqueeze(-1).to(device)
        y_pred = model(X_tensor).cpu().numpy().squeeze()
    pred_emg = scaler_output.inverse_transform(y_pred)

    t_full = np.arange(len(input_signal_full)) * 10
    t_input = np.arange(-input_len, 0) * 10 + 800
    t_output = np.arange(0, output_len) * 10 + 800 + input_len * 10

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1]})

    # Superior: cinética
    ax1.plot(t_full, input_signal_full, color='black', label='Wrist flexoextension angle')
    ax1.axvspan(t_input[0], t_input[-1], color='lightblue', alpha=0.3, label='Input window')
    ax1.set_ylabel("Wrist Angle", fontsize=14)
    ax1.legend(loc="upper left", fontsize=12, frameon=True)
    ax1.set_title("Kinematic Input and Muscle Activation Prediction", fontsize=16)

    # Medio: ECRL
    ax2.plot(t_full, emg_full[:, 0], color='blue', label='ECRL real')
    ax2.axvspan(t_output[0], t_output[-1], color='lightgrey', alpha=0.3, label='Prediction window')
    ax2.plot(t_output, pred_emg[:, 0], '--', color='blue', label='ECRL pred')
    ax2.set_ylabel("ECRL Activation", fontsize=14)
    ax2.legend(loc="upper left", fontsize=12, frameon=True)

    # Inferior: FCR
    ax3.plot(t_full, emg_full[:, 1], color='brown', label='FCR real')
    ax3.axvspan(t_output[0], t_output[-1], color='lightgrey', alpha=0.3, label='Prediction window')
    ax3.plot(t_output, pred_emg[:, 1], '--', color='brown', label='FCR pred')
    ax3.set_ylabel("FCR Activation", fontsize=14)
    ax3.set_xlabel("Time (ms)", fontsize=14)
    ax3.legend(loc="upper left", fontsize=12, frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"InputOutput_windows_split_{index}_{run_name}.png"))

    
    plt.show()

plot_kinematic_and_emg_with_windows(index=0, input_len=input_len, output_len=output_len)
plot_kinematic_and_emg_split(index=0, input_len=input_len, output_len=output_len)
plot_kinematic_and_emg_with_windows_full(index=0, input_len=input_len, output_len=output_len)
plot_kinematic_and_emg_split(index=0, input_len=input_len, output_len=output_len)
