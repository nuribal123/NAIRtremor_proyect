-- Inicialización del controlador
function init(model, par)

    -- Definir los músculos correctamente
    ECRL = model:find_muscle("ECRL")
    FCU = model:find_muscle("FCU")
    PQ = model:find_muscle("PQ")
    SUP = model:find_muscle("SUP")

    -- Parámetros ajustables
    gain_flexor = par:create_from_mean_std("gain_flexor", 0.5, 0.1, 0.0, 1.0)
    gain_extensor = par:create_from_mean_std("gain_extensor", 0.5, 0.1, 0.0, 1.0)
    gain_pronator = par:create_from_mean_std("gain_pronator", 0.5, 0.1, 0.0, 1.0)
    gain_supinator = par:create_from_mean_std("gain_supinator", 0.5, 0.1, 0.0, 1.0)

    -- Inicialización del CPG (oscilador)
    j1 = 0
    Kf = 2
    R = Kf
    phi_ref = math.rad(10)
    psi_ref = math.rad(70)

    -- Estados del oscilador
    X = {math.random(), math.random()}
    V = {math.random(), math.random()}
end

-- Función de actualización
function update(model)

    -- Obtener DOFs de la muñeca
    local wrist_dof_flexion = model:find_dof("wrist_hand_r1")
    local wrist_dof_prosup = model:find_dof("wrist_hand_r3")

    -- Obtener posiciones articulares
    local wrist_flexion = wrist_dof_flexion:position()
    local pro_sup = wrist_dof_prosup:position()

    -- Convertir a grados
    local eps_phi = math.deg(wrist_flexion)
    local eps_psi = math.deg(pro_sup)

    -- Función de inhibición recíproca (sigmoide hiperbólica)
    local ALPHA1 = (-0.5 * (math.exp(eps_phi) - math.exp(-eps_phi)) / (math.exp(eps_phi) + math.exp(-eps_phi))) + 0.5
    local ALPHA2 = (0.5 * (math.exp(eps_phi) - math.exp(-eps_phi)) / (math.exp(eps_phi) + math.exp(-eps_phi))) + 0.5
    local ALPHA3 = (0.5 * (math.exp(eps_psi) - math.exp(-eps_psi)) / (math.exp(eps_psi) + math.exp(-eps_psi))) + 0.5
    local ALPHA4 = (-0.5 * (math.exp(eps_psi) - math.exp(-eps_psi)) / (math.exp(eps_psi) + math.exp(-eps_psi))) + 0.5

    -- Variación de la frecuencia del temblor cada 5000 pasos
    if j1 % 5000 == 0 then
        Tosc = 1 / 8  -- Simulación de variación de frecuencia
        Kf = Tosc / 0.1051
        R = Kf
        phi_ref = math.rad(10)
        psi_ref = math.rad(70)
    end
    j1 = j1 + 1

    -- Parámetros del oscilador
    local tau1 = 0.1
    local tau2 = 0.1
    local B = 2.5
    local A = 5
    local h = 2.5
    local rosc = 1
    local dh = 0.001  -- Paso de integración

    -- Señales musculares iniciales
    local s1 = 0  -- model:muscle("ECRL"):activation()
    local s2 = 0  -- model:muscle("FCU"):activation()

    -- Método de Euler para resolver las ecuaciones diferenciales
    local x1 = X[1] + dh * ((1 / (Kf * tau1)) * (-X[1] - B * V[1] - h * math.max(X[2], 0) + A * s1 + rosc))
    local y1 = math.max(x1, 0)
    local v1 = V[1] + dh * ((1 / (Kf * tau2)) * (-V[1] + math.max(X[1], 0)))

    local x2 = X[2] + dh * ((1 / (Kf * tau1)) * (-X[2] - B * V[2] - h * math.max(X[1], 0) - A * s2 + rosc))
    local y2 = math.max(x2, 0)
    local v2 = V[2] + dh * ((1 / (Kf * tau2)) * (-V[2] + math.max(X[2], 0)))

    -- Actualizar estados del oscilador
    X = {x1, x2}
    V = {v1, v2}
    Y1 = {y1, y2}

    -- Salidas del oscilador (temblor)
    local du_1 = Y1[1]
    local du_2 = Y1[2]

    -- Control de transición
    local time = model:time()
    local transition_time = 1.0
    local control_signal = (time > transition_time) and 0.5 or 1.0

    -- Calcular excitaciones corregidas con CPG
    local u_ECRL, u_FCU, u_PQ, u_SUP

    if time < 2 then -- esto eran 10 probablemente por el du_1 ... dar tiempo a calcularlo
        u_ECRL = ALPHA1 * (gain_flexor * ECRL:excitation())
        u_FCU  = ALPHA2 * (gain_extensor * FCU:excitation())
        u_PQ   = ALPHA3 * (gain_pronator * PQ:excitation())
        u_SUP  = ALPHA4 * (gain_supinator * SUP:excitation())
    else
        u_ECRL = ALPHA1 * (gain_flexor * ECRL:excitation() + 0.2 * du_1)
        u_FCU  = ALPHA2 * (gain_extensor * FCU:excitation() + 0.2 * du_2)
        u_PQ   = ALPHA3 * (gain_pronator * PQ:excitation() + 0.2 * du_2)
        u_SUP  = ALPHA4 * (gain_supinator * SUP:excitation() + 0.2 * du_1)
    end

    -- Saturación de los valores (clamp entre 0 y 1)
    u_ECRL = math.max(0, math.min(1, u_ECRL))
    u_FCU  = math.max(0, math.min(1, u_FCU))
    u_PQ   = math.max(0, math.min(1, u_PQ))
    u_SUP  = math.max(0, math.min(1, u_SUP))

    -- Aplicar excitación muscular con temblor integrado
    ECRL:add_input(u_ECRL)
    FCU:add_input(u_FCU)
    PQ:add_input(u_PQ)
    SUP:add_input(u_SUP)

end
