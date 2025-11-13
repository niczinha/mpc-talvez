
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



def simular (integrador, modelo, params, y0, z0_vals, t):
    x_current = np.array(y0)
    z_current = np.array(z0_vals)
    results = {key: [] for key in [
        'Pai', 'Pwh', 'Pwi', 'Pwb', 'wiv', 'wro', 'wpc', 'wpg', 'wpo', 'wrg', 'ro_w', 'ro_a'
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



 
