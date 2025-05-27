import sys
sys.path.append("C:\\Users\\balba\\OneDrive\\Documentos\\SCONE\\SconePy")
from sconetools import sconepy # type: ignore
import pandas as pd
import numpy as np
import csv
import time
import os
import shutil
from multiprocessing import Pool

class TremorParameterGenerator:
    """
    Generador de parámetros para evitar overfitting manteniendo relaciones fisiológicas
    
    """
    
    def __init__(self):
        # Distribución estratificada para 7000 simulaciones
        self.strategies = {
            'frequency_progression': 1800,    # Progresión sistemática de frecuencia
            'amplitude_correlation': 1400,    # Correlación freq-amplitud fisiológica
            'matsuoka_variation': 1400,       # Variación parámetros Matsuoka
            'phase_modulation': 1200,         # Modulación de fase temporal
            'reflex_sensitivity': 1200        # Variación sensibilidad muscular
        }
    
    def generate_parameters(self, seed):
        """
        Genera parámetros específicos basados en el seed
        Mantiene consistencia pero evita overfitting
        """
        rng = np.random.default_rng(seed)
        
        # Determina estrategia basada en el seed
        strategy = self._get_strategy_for_seed(seed)
        
        if strategy == 'frequency_progression':
            return self._frequency_progression_params(seed, rng)
        elif strategy == 'amplitude_correlation':
            return self._amplitude_correlation_params(seed, rng)
        elif strategy == 'matsuoka_variation':
            return self._matsuoka_variation_params(seed, rng)
        elif strategy == 'phase_modulation':
            return self._phase_modulation_params(seed, rng)
        elif strategy == 'reflex_sensitivity':
            return self._reflex_sensitivity_params(seed, rng)
    
    def _get_strategy_for_seed(self, seed):
        """Determina estrategia basada en el seed para distribución uniforme"""
        total = 0
        for strategy, count in self.strategies.items():
            total += count
            if seed < total:
                return strategy
        return 'frequency_progression'  # fallback
    
    def _frequency_progression_params(self, seed, rng):
        """Progresión sistemática de frecuencia con correlaciones"""
        # Progresión lineal en el rango de seeds asignados
        progress = (seed % self.strategies['frequency_progression']) / self.strategies['frequency_progression']
        
        # Mapeo no lineal para mejor cobertura del espacio de frecuencias
        base_freq = 0.181 + (0.414 - 0.181) * (progress ** 0.8)
        
        # Correlación fisiológica: frecuencias altas → amplitudes menores
        h_base = 2.8 - (progress * 0.6)  # 2.8 a 2.2
        
        return {
            'base_freq': base_freq,
            'h_value': h_base + rng.uniform(-0.1, 0.1),  # Pequeña variación
            'tau_modifier': 1.0,
            'beta_modifier': 1.0,
            'strategy': 'frequency_progression'
        }
    
    def _amplitude_correlation_params(self, seed, rng):
        """Correlación amplitud-frecuencia con variación temporal"""
        base_freq = rng.uniform(0.181, 0.414)
        
        # Correlación inversa freq-amplitud (fisiológicamente realista)
        freq_normalized = (base_freq - 0.181) / (0.414 - 0.181)
        h_correlated = 2.8 - (freq_normalized * 0.8)  # 2.8 a 2.0
        
        # Añade modulación temporal de amplitud
        h_modulation_freq = rng.uniform(0.05, 0.3)  # Modulación lenta
        h_modulation_depth = rng.uniform(0.1, 0.4)
        
        return {
            'base_freq': base_freq,
            'h_value': h_correlated,
            'h_modulation_freq': h_modulation_freq,
            'h_modulation_depth': h_modulation_depth,
            'tau_modifier': 1.0,
            'beta_modifier': 1.0,
            'strategy': 'amplitude_correlation'
        }
    
    def _matsuoka_variation_params(self, seed, rng):
        """Variación de parámetros del oscilador Matsuoka"""
        base_freq = rng.uniform(0.181, 0.414)
        h_value = rng.uniform(2.2, 2.8)
        
        # Variación de parámetros del oscilador
        tau_modifier = rng.uniform(0.7, 1.3)  # Varía constante de tiempo
        beta_modifier = rng.uniform(0.8, 1.2)  # Varía acoplamiento inhibitorio
        
        return {
            'base_freq': base_freq,
            'h_value': h_value,
            'tau_modifier': tau_modifier,
            'beta_modifier': beta_modifier,
            'strategy': 'matsuoka_variation'
        }
    
    def _phase_modulation_params(self, seed, rng):
        """Modulación de fase para asimetría temporal"""
        base_freq = rng.uniform(0.181, 0.414)
        h_value = rng.uniform(2.2, 2.8)
        
        # Parámetros para modulación de fase
        phase_mod_freq = rng.uniform(0.3, 1.0)    # Frecuencia de modulación
        phase_mod_depth = rng.uniform(0.1, 0.5)   # Profundidad de modulación
        phase_drift_rate = rng.uniform(0.02, 0.1) # Drift lento de fase
        
        return {
            'base_freq': base_freq,
            'h_value': h_value,
            'phase_mod_freq': phase_mod_freq,
            'phase_mod_depth': phase_mod_depth,
            'phase_drift_rate': phase_drift_rate,
            'tau_modifier': 1.0,
            'beta_modifier': 1.0,
            'strategy': 'phase_modulation'
        }
    
    def _reflex_sensitivity_params(self, seed, rng):
        """Variación de sensibilidad de reflejos musculares"""
        base_freq = rng.uniform(0.181, 0.414)
        h_value = rng.uniform(2.2, 2.8)
        
        # Simulación de diferentes niveles de rigidez/sensibilidad
        reflex_gain = rng.uniform(0.6, 1.4)
        muscle_stiffness = rng.uniform(0.8, 1.2)
        
        return {
            'base_freq': base_freq,
            'h_value': h_value,
            'reflex_gain': reflex_gain,
            'muscle_stiffness': muscle_stiffness,
            'tau_modifier': 1.0,
            'beta_modifier': 1.0,
            'strategy': 'reflex_sensitivity'
        }

# Instancia global del generador
param_generator = TremorParameterGenerator()

def generate_lua_enhanced(seed, base_lua_path, output_path):
    """
    Versión mejorada que incluye variaciones anti-overfitting
    """
    with open(base_lua_path, "r", encoding="utf-8") as file:
        lua_template = file.read()

    # Genera estados iniciales aleatorios
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0.0, 1.0)
    v1 = rng.uniform(0.0, 1.0)
    x2 = rng.uniform(0.0, 1.0)
    v2 = rng.uniform(0.0, 1.0)
    
    # Obtiene parámetros específicos para esta simulación
    params = param_generator.generate_parameters(seed)
    
    # Plantilla base del LUA mejorada
    lua_enhanced = f"""
function init(model, par)
    -- Buscar músculos
    ECRL = model:find_muscle("ECRL")
    FCU = model:find_muscle("FCU")

    -- Parámetros base con variaciones anti-overfitting
    base_freq = {params['base_freq']:.6f}
    
    -- Parámetros Matsuoka con variaciones
    tau1 = 0.1 * {params['tau_modifier']:.3f}
    tau2 = 0.1 * {params['tau_modifier']:.3f}
    beta = 2.4 * {params['beta_modifier']:.3f}
    
    h = {params['h_value']:.3f}
    L = 0
    R = 1.0
    e = 0
    Kf = base_freq

    voluntary_drive = par:create_from_mean_std("voluntary_drive", 0.0, 0.1, 0.0, 0.5)

    -- Estados iniciales
    x1 = {x1:.6f}
    v1 = {v1:.6f}
    x2 = {x2:.6f}
    v2 = {v2:.6f}
    
    -- Parámetros específicos de la estrategia
    strategy = "{params['strategy']}"
"""
    
    # Añade parámetros específicos según la estrategia
    if params['strategy'] == 'amplitude_correlation':
        lua_enhanced += f"""
    h_mod_freq = {params['h_modulation_freq']:.6f}
    h_mod_depth = {params['h_modulation_depth']:.3f}
"""
    elif params['strategy'] == 'phase_modulation':
        lua_enhanced += f"""
    phase_mod_freq = {params['phase_mod_freq']:.6f}
    phase_mod_depth = {params['phase_mod_depth']:.3f}
    phase_drift_rate = {params['phase_drift_rate']:.6f}
"""
    elif params['strategy'] == 'reflex_sensitivity':
        lua_enhanced += f"""
    reflex_gain = {params['reflex_gain']:.3f}
    muscle_stiffness = {params['muscle_stiffness']:.3f}
"""
    
    lua_enhanced += """
end

function max0(x)
    return math.max(x, 0)
end

function smooth_noise(t, freq)
    return math.sin(2 * math.pi * freq * t + 1.5) +
           0.5 * math.sin(2 * math.pi * 0.5 * freq * t + 0.7)
end

function voluntary_control(model, u_ECRL, u_FCU, t, phase_offset)
    local voluntary_input = math.sin((t + phase_offset) * 2 * math.pi * 1)
    u_ECRL = u_ECRL + voluntary_drive * voluntary_input
    u_FCU = u_FCU + voluntary_drive * voluntary_input
    return u_ECRL, u_FCU
end

function update(model)
    local t = model:time()
    local dt = model:delta_time()

    -- Variaciones específicas por estrategia
    local freq_variation = 0
    local h_mod = h
    local phase_offset = 0
    
    if strategy == "amplitude_correlation" then
        -- Modulación temporal de amplitud
        h_mod = h * (1.0 + h_mod_depth * math.sin(2 * math.pi * h_mod_freq * t))
        freq_variation = base_freq * 0.03 * math.sin(2 * math.pi * 0.8 * t)
        
    elseif strategy == "phase_modulation" then
        -- Modulación de fase compleja
        phase_offset = phase_mod_depth * math.sin(2 * math.pi * phase_mod_freq * t) +
                      0.5 * phase_mod_depth * math.sin(2 * math.pi * phase_drift_rate * t)
        freq_variation = base_freq * 0.02 * smooth_noise(t, 0.3)
        h_mod = h * (1.0 + 0.05 * smooth_noise(t, 0.2))
        
    elseif strategy == "matsuoka_variation" then
        -- Variación de parámetros internos del oscilador
        freq_variation = base_freq * 0.04 * math.sin(2 * math.pi * 0.6 * t + 0.5)
        h_mod = h * (1.0 + 0.08 * math.cos(2 * math.pi * 0.4 * t))
        
    elseif strategy == "reflex_sensitivity" then
        -- Simulación de diferentes sensibilidades
        h_mod = h * muscle_stiffness * (1.0 + 0.03 * smooth_noise(t, 0.15))
        freq_variation = base_freq * 0.025 * math.sin(2 * math.pi * 0.7 * t)
        
    else -- frequency_progression
        -- Variación mínima para mantener frecuencia base
        freq_variation = base_freq * 0.02 * math.sin(2 * math.pi * 0.5 * t)
        h_mod = h * (1.0 + 0.03 * smooth_noise(t, 0.1))
    end
    
    local inst_freq = base_freq + freq_variation
    local local_kf = 1 / (0.1051 * base_freq)

    -- Dinámica del oscilador Matsuoka
    local dx1 = (-x1 - beta * v1 - h_mod * max0(x2) + L * e + R) * (dt * local_kf / tau1)
    local dv1 = (-v1 + max0(x1)) * (dt * local_kf / tau2)
    local dx2 = (-x2 - beta * v2 - h_mod * max0(x1) - L * e + R) * (dt * local_kf / tau1)
    local dv2 = (-v2 + max0(x2)) * (dt * local_kf / tau2)

    x1 = x1 + dx1
    v1 = v1 + dv1
    x2 = x2 + dx2
    v2 = v2 + dv2

    local y1 = max0(x1)
    local y2 = max0(x2)

    -- Control voluntario con modulación de fase
    local u_ECRL, u_FCU = voluntary_control(model, y1, y2, t, phase_offset)
    
    -- Aplicar ganancia de reflejos si corresponde
    if strategy == "reflex_sensitivity" then
        u_ECRL = u_ECRL * reflex_gain
        u_FCU = u_FCU * reflex_gain
    end

    -- Limitar la salida
    u_ECRL = math.max(0, math.min(0.8, u_ECRL))
    u_FCU = math.max(0, math.min(0.8, u_FCU))

    ECRL:add_input(u_ECRL)
    FCU:add_input(u_FCU)
end
"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(lua_enhanced)

# Función original modificada para usar la versión mejorada
def generate_lua(seed, base_freq, h_value, base_lua_path, output_path):
    """Wrapper para mantener compatibilidad - ahora usa generate_lua_enhanced"""
    return generate_lua_enhanced(seed, base_lua_path, output_path)

# Función de simulación modificada para reflejos variables
def run_simulation(model, store_data, random_seed, output_dir, file_id, max_time=4):
    model.reset()
    model.set_store_data(store_data)

    # Obtiene parámetros para inicialización específica
    params = param_generator.generate_parameters(random_seed)
    
    rng = np.random.default_rng(random_seed)
    
    # Inicialización de activaciones con variación según estrategia
    if params['strategy'] == 'reflex_sensitivity':
        # Variación más amplia para simular diferentes sensibilidades
        muscle_activations = 0.05 + 0.5 * rng.random(len(model.muscles()))
        muscle_activations *= params.get('muscle_stiffness', 1.0)
    else:
        muscle_activations = 0.1 + 0.4 * rng.random(len(model.muscles()))
        
    model.init_muscle_activations(muscle_activations)

    dof_positions = model.dof_position_array()
    dof_positions += 0.1 * rng.random(len(dof_positions)) - 0.05
    model.set_dof_positions(dof_positions)
    model.init_state_from_dofs()

    # Resto del código original de simulación...
    csv_path = os.path.join(output_dir, f"twin_{file_id}.csv")
    actuators = model.actuators()
    muscles = model.muscles()
    wrist_dofs = [d for d in model.dofs() if "wrist_hand_r3" in d.name().lower()]

    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["time"] +
                        [a.name() for a in actuators] +
                        [f"{a.name()}_min" for a in actuators] +
                        [f"{a.name()}_max" for a in actuators] +
                        [m.name() + "_exc" for m in muscles] +
                        [m.name() + "_act" for m in muscles] +
                        [d.name() + "_pos" for d in wrist_dofs] +
                        [d.name() + "_vel" for d in wrist_dofs])

        for t in np.arange(0, max_time, 0.01667):
            # Aplicar ganancia de reflejos si corresponde
            if params['strategy'] == 'reflex_sensitivity':
                reflex_gain = params.get('reflex_gain', 1.0)
                mus_in = model.delayed_muscle_force_array() * reflex_gain
                mus_in += (model.delayed_muscle_fiber_length_array() - 1.2) * reflex_gain
                mus_in += 0.1 * model.delayed_muscle_fiber_velocity_array() * reflex_gain
            else:
                mus_in = model.delayed_muscle_force_array()
                mus_in += model.delayed_muscle_fiber_length_array() - 1.2
                mus_in += 0.1 * model.delayed_muscle_fiber_velocity_array()
                
            model.set_delayed_actuator_inputs(mus_in)

            inputs = [a.input() for a in actuators]
            min_inputs = [a.min_input() for a in actuators]
            max_inputs = [a.max_input() for a in actuators]
            muscle_excitations = model.muscle_excitation_array()
            muscle_activations = model.muscle_activation_array()
            wrist_positions = [d.pos() for d in wrist_dofs]
            wrist_velocities = [d.vel() for d in wrist_dofs]

            writer.writerow([t] + inputs + min_inputs + max_inputs +
                            list(muscle_excitations) + list(muscle_activations) +
                            wrist_positions + wrist_velocities)

            model.advance_simulation_to(t)

    print(f"✅ Simulación {file_id} ({params['strategy']}) completada y guardada en {csv_path}")

# Función original sin cambios para mantener compatibilidad
def generate_scone(seed, base_scone_path, output_path, lua_path):
    with open(base_scone_path, "r", encoding="utf-8") as file:
        scone_template = file.read()

    lua_filename = os.path.basename(lua_path)
    updated_scone = scone_template.replace("$LUA_PATH$", lua_filename)

    with open(output_path, "w") as file:
        file.write(updated_scone)

def run_simulation_wrapper(args):
    seed, output_dir, base_lua, base_scone = args
    
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Paths para los archivos generados
        lua_path = os.path.join(output_dir, f"NewMatsuoka_{seed}.lua")
        scone_path = os.path.join(output_dir, f"Simulation_{seed}.scone")

        # Genera los archivos de configuración usando la versión mejorada
        generate_lua_enhanced(seed, base_lua, lua_path)
        generate_scone(seed, base_scone, scone_path, lua_path)

        # Copia el archivo de delays si no existe
        delay_src = "D:/ingenieriabiomedica/sconeGym/sconegym/sconegym/data/neural_delays_FEA_v4.zml"
        delay_dst = os.path.join(output_dir, "neural_delays_FEA_v4.zml")
        if not os.path.exists(delay_dst):
            shutil.copy(delay_src, delay_dst)

        # Ejecuta la simulación
        model = sconepy.load_model(scone_path)
        run_simulation(model, store_data=False, random_seed=seed, output_dir=output_dir, file_id=seed)

    except Exception as e:
        print(f"❌ [PID {os.getpid()}] Error en seed {seed}: {e}", flush=True)

# Punto de entrada principal - sin cambios
if __name__ == "__main__":
    output_dir = "D:/ingenieriabiomedica/sconeGym/sconegym/sim_data_ENHANCED_ANTIOVERFIT"
    base_lua = "C:/Users/balba/OneDrive/Documentos/SCONE/Tutorials/controllers/NewMatsuoka3_template.lua"
    base_scone = "D:/ingenieriabiomedica/sconeGym/sconegym/sconegym/data/H0918_S2_template.scone"

    seeds = list(range(0, 7000))  # Ajustado a 7000 simulaciones
    args = [(seed, output_dir, base_lua, base_scone) for seed in seeds]

    # Imprime estadísticas de la distribución
    print("Distribución de estrategias anti-overfitting:")
    for strategy, count in param_generator.strategies.items():
        print(f"- {strategy}: {count} simulaciones")
    print(f"Total: {sum(param_generator.strategies.values())} simulaciones")
    
    start_time = time.time()
    
    # Con multiprocessing para paralelizar
    with Pool(processes=os.cpu_count()) as pool:
        pool.map(run_simulation_wrapper, args)

    end_time = time.time()
    elapsed = end_time - start_time

    with open(os.path.join(output_dir, "tiempo_total_sims.txt"), "w") as f:
        f.write(f"Tiempo total de ejecución de las simulaciones: {elapsed: .2f} segundos\n")
        f.write("Estrategias aplicadas:\n")
        for strategy, count in param_generator.strategies.items():
            f.write(f"- {strategy}: {count} simulaciones\n")