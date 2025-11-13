
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import pandas as pd
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fsolve




M = 0.028# Massa molar do gás [kg/mol]
Ta = 301 # Temperatura da região anular [K]
La = 1500  # Comprimento da região anular [m]
ro = 1000# Densidade do óleo no reservatório [kg/m³]
Tw = 305 # Temperatura no tubo [K]
Lw = 1500  # Comprimento do tubo [m]
Lr = 500  # Distância do reservatório até o ponto de injeção [m]
  # Área da seção transversal abaixo do ponto injeção [m²]
 # Área da seção transversal acima do ponto injeção [m²]
Hw = 1500  # [m]  

Dw= 0.121
Dr =0.121
u= 1# Exemplo de valor para u
u1= 50**(u - 1) 

Dbh = 0.121
Da = 0.189
Hr = 500
Hbh = 500
Lbh = 500
Tr = 301
PI = 0.22e-5
Cpc = 2e-3 # [m²]
Civ =  10e-4   # [m²]
Cr = 2.6e-4  # [m²]
Crh = 10e-3
Pr = 15e6  # Pressão no reservatório longe da cabeça do poço [Pa]
Ps = 2e6 # Pressão no manifold [Pa]

vo = 1 / ro  # Volume específico do óleo  
g = 9.81  # Aceleração da gravidade [m/s²]
Riser_p_fric = 0.5
R = 8.314  # Constante dos gases [J/mol.K]
GOR =0.1
dp_t_fric = 1.0e5 #'atm';
dp_bh_fric = 5.0e4 #'atm';

Ha = 1000
# Cálculo das áreas
Aw = (ca.pi * (Dw ** 2)) / 4  # Área da seção transversal do poço [m²]
Aa = (ca.pi * (Da ** 2)) / 4 - (ca.pi * (Dw ** 2)) / 4  # Área da seção transversal do anel [m²]
Ar = (ca.pi * (Dr ** 2)) / 4  # Área da seção transversal abaixo do ponto de injeção [m²]
Va = La * Aa  # Volume da região anular [m³]versal abaixo do ponto de injeção [m²]
Abh = (ca.pi * (Dbh ** 2)) / 4
def fun(t, x, par, z):
    
    x1, x2, x3 = x[0], x[1], x[2]
    par, u1, GOR = par[0], par[1], par[2]
    # Variáveis algébricas (indexação, não desempacotamento)
    Pai = z[0]
    Pwh = z[1]
    Pwi = z[2]
    Pwb = z[3]
    wiv = z[4]
    wro = z[5]
    wpc = z[6]
    wpg = z[7]
    wpo = z[8]
    wrg = z[9]
    ro_w = z[10]
    ro_a = z[11]

    
    dx1 = par - wiv
    dx2 = wiv + wrg - wpg
    dx3 = wro - wpo

    return ca.vertcat(dx1, dx2, dx3)


def modelo(x, par, z):
    par, u1, GOR = par[0], par[1], par[2]
    Pai = z[0]
    Pwh = z[1]
    Pwi = z[2]
    Pwb = z[3]
    wiv = z[4]
    wro = z[5]
    wpc = z[6]
    wpg = z[7]
    wpo = z[8]
    wrg = z[9]
    ro_w = z[10]
    ro_a = z[11]

    return {
        'Pai': Pai,
        'Pwh': Pwh,
        'Pwi': Pwi,
        'Pwb': Pwb,
        'wiv': wiv,
        'wro': wro,
        'wpc': wpc,
        'wpg': wpg,
        'wpo': wpo,
        'wrg': wrg,
        'ro_w': ro_w,
        'ro_a': ro_a
    }



def equacoes_algebricas(x, par,  z):
    Pai = z[0]
    Pwh = z[1]
    Pwi = z[2]
    Pwb = z[3]
    wiv = z[4]
    wro = z[5]
    wpc = z[6]
    wpg = z[7]
    wpo = z[8]
    wrg = z[9]
    ro_w = z[10]
    ro_a = z[11]

    x1, x2, x3 = x[0], x[1], x[2]
    par, u1, GOR = par[0], par[1], par[2]

    # Auxiliares simbólicos
    D = ca.sqrt(Aw * 4 / np.pi)
    y1 = ca.fmax(1e-9, Pai - Pwi)
    y2 = ca.fmax(1e-9, Pr - Pwb)
    y3 = ca.fmax(1e-9, Pwh - Ps)
    veloc = wpc / Aw
    
    # Equações implícitas Fi = 0
    F1  = Pai - ((R * Ta / (Va * M)) + ((g * La) / Va)) * x1
    F2  = Pwh - (R * Tw / M) * (x2 / (Lw * Aw - (vo * x3)))
    F3  = ro_w - ((x2 + x3) / (Lw * Aw))
    F4  = Pwi - (Pwh + ((x3 + x2) * g / Aw))
    F5  = Pwb - (Pwi + (ro * g * Hr))
    F6  = ro_a - ((M * Pai) / (R * Ta))
    F7  = wiv - Civ * ca.sqrt(ro_a * y1)
    F8  = wpc - Cpc * ca.sqrt(ro_w * y3) * 50 ** (u1 - 1)
    F9  = wpg - ((x2 / (x2 + x3 +1e-9)) * wpc)
    F10 = wpo - ((x3 / (x2 + x3+ 1e-9)) * wpc)
    F11 = wro - Cr * ca.sqrt(ro * y2)
    F12 = wrg - (GOR * wro)

    return ca.vertcat(F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12)



def fun_integrador(dt):
    x_sym = ca.SX.sym('x', 3)
    z_sym = ca.SX.sym('z', 12)
    par_sym = ca.SX.sym('par', 3)

    dx_sym = fun(0, x_sym, par_sym, z_sym)
    F_alg = equacoes_algebricas(x_sym, par_sym, z_sym)

    dae = {'x': x_sym, 'z': z_sym, 'p': par_sym, 'ode': dx_sym, 'alg': F_alg}

    opts = {
        'tf': dt,
        'abstol': 1e-4,
        'reltol': 1e-4,
        'max_num_steps': 20000,
        'disable_internal_warnings': False,
    }

    integrador = ca.integrator('integrador', 'idas', dae, opts)
    return integrador, modelo



def simular(integrador, modelo, params, y0, z0_vals, t):
    tolerance=1e-8
    passos_estaveis=100
    x_current = np.array(y0)
    z_current = np.array(z0_vals)
    
    results = []
    contador = 0
    stable_found = False
    w_inj, u1, GOR = params

    for i, ti in enumerate(t):
        res = integrador(x0=x_current, z0=z_current, p=params)
        x_next = res['xf'].full().flatten()
        z_next = res['zf'].full().flatten()
        model_output = modelo(x_next, params, z_next)

        
        output_step = {key: float(model_output[key]) for key in model_output}
        output_step['x1'], output_step['x2'], output_step['x3'] = float(x_next[0]), float(x_next[1]), float(x_next[2])
        results.append(output_step)
        if i > 0:
            delta = abs(x_next[0] - x_current[0])
            if delta <= tolerance:
                contador += 1
            else:
                contador = 0

            if contador >= passos_estaveis :
                stable_found = True
                break

 

        x_current = x_next
        z_current = z_next
    if u1 <= 0.2:
        # Força o estado '0' (estável) se u1 for baixo
        estado = '0'
    else:
        estado = '0' if stable_found else '1'

    if stable_found:
        last = results[-1]
        last['estado'] = estado
        return last
    else:
        meio = len(results) // 2
        media = {k: np.mean([d[k] for d in results[meio:]]) for k in results[0]}
        media['estado'] = estado
        return  media
    
    

def run_simulation(integrador, modelo, w_inj, u1, gor_val, y0, z0_vals, t):
    params = [w_inj, u1, gor_val]
    resultado = simular(integrador, modelo, params, y0, z0_vals, t)

    resultado['w_inj'] = w_inj
    resultado['u1'] = u1
    resultado['GOR'] = gor_val
    return resultado

def simular_2 (integrador, modelo, params, y0, z0_vals, t):
    x_current = np.array(y0)
    z_current = np.array(z0_vals)
    results = {key: [] for key in [
        'Pai', 'Pwh', 'Pwi', 'Pwb', 'wiv', 'wro', 'wpc', 'wpg', 'wpo', 'wrg', 'ro_w', 'ro_a','x1', 'x2','x3'
    ]}

    for ti in t:
        res = integrador(x0=x_current, z0=z_current, p=params)
        x_next = res['xf'].full().flatten()
        z_next = res['zf'].full().flatten()

        model_output = modelo(x_next, params, z_next)
        results['x1'].append(float(x_next[0]))
        results['x2'].append(float(x_next[1]))
        results['x3'].append(float(x_next[2]))

        # Salva variáveis algébricas
        for key in results:
            if key not in ['x1', 'x2', 'x3']:
                results[key].append(float(model_output[key]))

        x_current = x_next
        z_current = z_next
    for key in results:
        results[key] = np.array(results[key])

    return results




def fun_fsolve(x, par, u1, GOR):
    epsilon = 1e-9 

    
    x1, x2, x3 = x[0], x[1], x[2]

    # Pressão no anular
    Pai = ((R * Ta / (Va * M)) + ((g * La) / Va)) * x1

    # Cálculo da pressão no topo
    gas_volume = np.maximum(epsilon, (Lw * Aw - (vo * x3)))
    Pwh = (R * Tw / M) * (x2 / gas_volume)

    # Densidade total (gás + óleo)
    ro_w = (x2 + x3) / (Lw * Aw + epsilon)

    # Diferença de pressão no choke
    y3 = np.maximum(epsilon, (Pwh - Ps))

    # Vazão no choke (com proteção contra raiz negativa)
    wpc_arg = np.maximum(epsilon, ro_w * y3)
    wpc = Cpc * np.sqrt(wpc_arg) * 50 ** (u1 - 1)

    # Pressões intermediárias
    Pwi = Pwh + ((x3 + x2) * g) / Aw
    Pwb = Pwi + (ro * g * Hr)

    # Densidade do gás
    ro_a = (M * Pai) / (R * Ta + epsilon)

    # Diferenças de pressão (com proteção)
    y1 = np.maximum(epsilon, (Pai - Pwi))
    y2 = np.maximum(epsilon, (Pr - Pwb))

    # Vazões através das válvulas
    wiv_arg = np.maximum(epsilon, ro_a * y1)
    wiv = Civ * np.sqrt(wiv_arg)

    wro_arg = np.maximum(epsilon, ro * y2)
    wro = Cr * np.sqrt(wro_arg)

    wrg = GOR * wro

    # Vazões separadas (gás e óleo)
    wpg = (x2 / (x2 + x3 + epsilon)) * wpc
    wpo = (x3 / (x2 + x3 + epsilon)) * wpc

    # Equações diferenciais (balanços de massa)
    dx1 = par - wiv
    dx2 = wiv + wrg - wpg
    dx3 = wro - wpo
    return np.array([dx1, dx2, dx3])


def get_outputs_fsolve(x, par, u1, GOR):
    epsilon = 1e-9 

    
    x1, x2, x3 = x[0], x[1], x[2]

    # Pressão no anular
    Pai = ((R * Ta / (Va * M)) + ((g * La) / Va)) * x1

    # Cálculo da pressão no topo
    gas_volume = np.maximum(epsilon, (Lw * Aw - (vo * x3)))
    Pwh = (R * Tw / M) * (x2 / gas_volume)

    # Densidade total (gás + óleo)
    ro_w = (x2 + x3) / (Lw * Aw + epsilon)

    # Diferença de pressão no choke
    y3 = np.maximum(0.0, (Pwh - Ps))

    # Vazão no choke (com proteção contra raiz negativa)
    wpc_arg = np.maximum(epsilon, ro_w * y3)
    wpc = Cpc * np.sqrt(wpc_arg) * 50 ** (u1 - 1)

    # Pressões intermediárias
    Pwi = Pwh + ((x3 + x2) * g) / Aw
    Pwb = Pwi + (ro * g * Hr)

    # Densidade do gás
    ro_a = (M * Pai) / (R * Ta + epsilon)

    # Diferenças de pressão (com proteção)
    y1 = np.maximum(epsilon, (Pai - Pwi))
    y2 = np.maximum(epsilon, (Pr - Pwb))

    # Vazões através das válvulas
    wiv_arg = np.maximum(epsilon, ro_a * y1)
    wiv = Civ * np.sqrt(wiv_arg)

    wro_arg = np.maximum(epsilon, ro * y2)
    wro = Cr * np.sqrt(wro_arg)

    wrg = GOR * wro

    # Vazões separadas (gás e óleo)
    wpg = (x2 / (x2 + x3 + epsilon)) * wpc
    wpo = (x3 / (x2 + x3 + epsilon)) * wpc

    # Equações diferenciais (balanços de massa)
    dx1 = par - wiv
    dx2 = wiv + wrg - wpg
    dx3 = wro - wpo
    dx_sym = ca.vertcat(dx1, dx2, dx3)

    outputs = {
        'Pai': Pai, 'Pwh': Pwh, 'Pwi': Pwi, 'Pwb': Pwb,
        'wiv': wiv, 'wro': wro, 'wpc': wpc, 'wpg': wpg,
        'wpo': wpo, 'wrg': wrg, 'ro_w': ro_w, 'ro_a': ro_a
    }
    return outputs


# ... (todo o seu código DAE existente: M, Ta, La, fun, modelo, equacoes_algebricas, fun_integrador, simular, etc.) ...
# ...
# ...
# ===============================================================
# 4. FUNÇÕES PARA OTIMIZAÇÃO (ODE EXPLÍCITO)
# (Adicione esta seção no final do seu arquivo)
# ===============================================================

def fun_casadi(x_sym, p_sym):
    """
    Cria o modelo ODE explícito (dx = f(x,p))
    e a função de saídas (y = g(x,p)) para a otimização.
    Baseado na sua lógica 'fun_fsolve'.
    """
    

    epsilon = 1e-9

    # Desempacotar estados (x) e parâmetros (p)
    x1, x2, x3 = x_sym[0], x_sym[1], x_sym[2]
    par, u1, GOR = p_sym[0], p_sym[1], p_sym[2]

    
    Pai = ((R * Ta / (Va * M)) + ((g * La) / Va)) * x1
    gas_volume = ca.fmax(epsilon, (Lw * Aw - (vo * x3)))
    Pwh = (R * Tw / M) * (x2 / gas_volume)
    ro_w = (x2 + x3) / (Lw * Aw + epsilon)
    y3 = ca.fmax(0.0, (Pwh - Ps))
    wpc_arg = ca.fmax(epsilon, ro_w * y3)
    wpc = Cpc * ca.sqrt(wpc_arg) * 50 ** (u1 - 1)
    Pwi = Pwh + ((x3 + x2) * g) / Aw
    Pwb = Pwi + (ro * g * Hr)
    ro_a = (M * Pai) / (R * Ta + epsilon)
    y1 = ca.fmax(epsilon, (Pai - Pwi))
    y2 = ca.fmax(epsilon, (Pr - Pwb))
    wiv_arg = ca.fmax(epsilon, ro_a * y1)
    wiv = Civ * ca.sqrt(wiv_arg)
    wro_arg = ca.fmax(epsilon, ro * y2)
    wro = Cr * ca.sqrt(wro_arg)
    wrg = GOR * wro
    wpg = (x2 / (x2 + x3 + epsilon)) * wpc
    wpo = (x3 / (x2 + x3 + epsilon)) * wpc

    # --- Equações Diferenciais ---
    dx1 = par - wiv
    dx2 = wiv + wrg - wpg
    dx3 = wro - wpo
    
    dx_sym = ca.vertcat(dx1, dx2, dx3)
    
    # --- Saídas ---
    # (Retorna um dicionário de expressões simbólicas)
    model_outputs = {
        'Pai': Pai, 'Pwh': Pwh, 'Pwi': Pwi, 'Pwb': Pwb,
        'wiv': wiv, 'wro': wro, 'wpc': wpc, 'wpg': wpg,
        'wpo': wpo, 'wrg': wrg, 'ro_w': ro_w, 'ro_a': ro_a
    }
    
    return dx_sym, model_outputs