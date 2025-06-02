-- tratando de hacer las sims


-- Estoy incluyendo la parte de NewMatsuoka3_template3.lua pero limpiando a solamente la parte de CPG Matsuoka

-- La plantilla “NewMatsuoka3_template.lua”
function init(model, par)
    -- Buscar músculos
    ECRL = model:find_muscle("ECRL")
    FCU = model:find_muscle("FCU")

    -- La frecuencia base se obtiene de FREQ_PLACEHOLDER (entre 4 y 9Hz)
    base_freq = FREQ_PLACEHOLDER

    tau1 = 0.1
    tau2 = 0.1
    beta = 2.4
    -- La amplitud base se fija con H_PLACEHOLDER; para mantener cambios sutiles,
    -- se modula con ±5% (ajustable)
    h = H_PLACEHOLDER
    L = 0
    R = 1.0
    e = 0
    -- Kf = 1.704 / (base_freq + 0.262)
    Kf = base_freq

    voluntary_drive = par:create_from_mean_std("voluntary_drive", 0.0, 0.1, 0.0, 0.5)

    -- Estados iniciales del oscilador, entre 0 y 1
    x1 = X1_PLACEHOLDER
    v1 = V1_PLACEHOLDER
    x2 = X2_PLACEHOLDER
    v2 = V2_PLACEHOLDER
end

function max0(x)
    return math.max(x, 0)
end

-- Función de ruido suave para modulación
function smooth_noise(t, freq)
    return math.sin(2 * math.pi * freq * t + 1.5) +
           0.5 * math.sin(2 * math.pi * 0.5 * freq * t + 0.7)
end


function update(model)
    local t = model:time()
    local dt = model:delta_time()

    -- ===============================================================
    -- Control de FRECUENCIA:
    --
    -- La idea es que cada simulación tenga un FREQ_BASE predominante.
    -- Se introduce una variación muy leve (±5% de la base) para mantener
    -- alrededor del 70% de la energía en la frecuencia predeterminada.
    --
    local freq_variation = base_freq * 0.05 * math.sin(2 * math.pi * 0.8 * t)
    local inst_freq = base_freq + freq_variation
    -- local local_kf = 1 / 0.1051 * inst_freq
    -- local local_kf = 1.704 / (base_freq + 0.262)
    local_kf = base_freq

    -- ===============================================================
    -- Control de AMPLITUD:
    --
    -- Se modula con un ±5% (puedes ajustar este factor si quieres menos o más variación).
    local h_mod = h * (1.0 + 0.05 * smooth_noise(t, 0.2))

    -- ===============================================================
    -- Introducir VARIACIONES DE FASE:
    --
    -- Se usan dos componentes:
    --   1. Una modulación de frecuencia media (ej. 0.5Hz) para drift lento.
    --   2. Una modulación de muy baja frecuencia (ej. 0.07Hz) para saltos lentos de fase.
    local phase_jitter_medium = 0.2 * math.sin(2 * math.pi * 0.5 * t + 0.3)
    local phase_jitter_slow   = 0.1 * math.sin(2 * math.pi * 0.07 * t)
    local phase_offset = phase_jitter_medium + phase_jitter_slow

    -- ===============================================================


    -- Dinámica del oscilador Matsuoka
    local dx1 = (-x1 - beta * v1 - h_mod * max0(x2) + L * e + R) * (dt * 1 / (local_kf * tau1))
    local dv1 = (-v1 + max0(x1)) * (dt * 1 / (local_kf * tau2))
    local dx2 = (-x2 - beta * v2 - h_mod * max0(x1) - L * e + R) * (dt * 1 / (local_kf * tau1))
    local dv2 = (-v2 + max0(x2)) * (dt * 1 /(local_kf * tau2))

    x1 = x1 + dx1
    v1 = v1 + dv1
    x2 = x2 + dx2
    v2 = v2 + dv2 -- si puedo, debug para comprobar que se están actualizando

    local y1 = max0(x1)
    local y2 = max0(x2)


    local u_ECRL, u_FCU = y1, y2

    -- Limitar la salida
    u_ECRL = math.max(0, math.min(0.8, u_ECRL))
    u_FCU = math.max(0, math.min(0.8, u_FCU))

    ECRL:add_input(u_ECRL)
    FCU:add_input(u_FCU)
end