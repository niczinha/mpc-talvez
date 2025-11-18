
import numpy as np
import matplotlib.pyplot as plt
from ModeloClassificado import fun_integrador, simular_2, fun_fsolve
from scipy.optimize import root
import casadi as ca
import pandas as pd

from ModeloClassificado import fun_casadi 



x_sym_eq = ca.SX.sym('x_eq', 3)
p_sym_eq = ca.SX.sym('p_eq', 3) # [w_inj, u1, GOR]

#
dx_sym_eq, modelo_sym_eq = fun_casadi(x_sym_eq, p_sym_eq)

#  Minimizar a soma dos quadrados das derivadas
J = ca.dot(dx_sym_eq, dx_sym_eq)

nlp_problem = {
    'f': J,
    'x': x_sym_eq,
    'p': p_sym_eq
}

opts_nlp = {
    'ipopt.print_level': 0,
    'print_time': 0,
    'ipopt.sb': 'yes' # Suprime o banner do Ipopt
}


solver_nlp = ca.nlpsol('solver_nlp', 'ipopt', nlp_problem, opts_nlp)

# Criar funções numéricas para obter saídas e resíduos
output_names = list(modelo_sym_eq.keys())
output_func = ca.Function('output_func', 
                          [x_sym_eq, p_sym_eq], 
                          [ca.vertcat(*modelo_sym_eq.values())])

g_check = ca.Function('g_check', [x_sym_eq, p_sym_eq], [dx_sym_eq])

# Limites (são constantes)
lbx_eq = [0.0, 0.0, 0.0]
ubx_eq = [ca.inf, ca.inf, ca.inf]



chutes_df = pd.read_csv("resultados_estabilidade.csv", sep=';', decimal=',')


chutes_excepcionais = {

    (0.2, 0.9, 0.01): [1501.08, 300.62, 272.66],
    
   
    (0.2, 1.0, 0.01): [2000.0102885748443, 253.48, 7871.77],
    

    (0.4, 1.0, 0.01): [2000.0102885748443, 253.48, 7871.77]
}
print(f"Definidas {len(chutes_excepcionais)} exceções de chute.")


all_results = []
count = 0
total = len(chutes_df)
print(f"\nIniciando {total} simulações de equilíbrio (com CasADi/Ipopt)...\n")

for index, row in chutes_df.iterrows():
    count += 1
    
    w_inj = float(row["w_inj"])
    u1 = float(row["u1"])
    gor_val = float(row["GOR"])
    original_state = row.get('estado', 0)
    

    ponto_atual = (w_inj, u1, gor_val)

    print(f"🔹 Simulação {count}/{total}: w_inj={w_inj:.2f}, u1={u1:.2f}, GOR={gor_val:.3f}")


    if ponto_atual in chutes_excepcionais:
        x_guess = chutes_excepcionais[ponto_atual]
        print(f" Usando chute EXCEPCIONAL = {[round(g, 2) for g in x_guess]}")
    else:
        
        x_guess = [
            float(row["x1"]),
            float(row["x2"]),
            float(row["x3"]),
        ]
        print(f" Usando chute do CSV = {[round(g, 2) for g in x_guess]}")
    

    try:
        sol = solver_nlp(
            x0=x_guess,
            p=[w_inj, u1, gor_val],
            lbx=lbx_eq,
            ubx=ubx_eq
        )
        
        x_eq = sol['x'] 
        final_objective = float(sol['f'])

        if final_objective > 1e-6:
             print(f"Não convergiu: J={final_objective:.2e} (alto)")
             status = 'não convergiu (J > 1e-6)'
             x_eq_vals = [np.nan, np.nan, np.nan]
             outputs_dict = {name: np.nan for name in output_names}
        else:
            x1_val = float(x_eq[0])
            x2_val = float(x_eq[1])
            x3_val = float(x_eq[2])
            
            print(f" Convergiu: x_eq = {[round(x1_val, 2), round(x2_val, 2), round(x3_val, 2)]}")
            status = 'convergiu'
            x_eq_vals = [x1_val, x2_val, x3_val] 

            outputs_val = output_func(x_eq, [w_inj, u1, gor_val]).full().flatten()
            outputs_dict = {name: val for name, val in zip(output_names, outputs_val)}

        result_dict = {
            'w_inj': w_inj, 'u1': u1, 'GOR': gor_val,
            'x1_eq': x_eq_vals[0],
            'x2_eq': x_eq_vals[1],
            'x3_eq': x_eq_vals[2],
            'estado': original_state,
            'status': status,
            'residual_norm': final_objective,
            **outputs_dict
        }
        all_results.append(result_dict)

    except Exception as e:
        print(f" ERRO CRÍTICO no Ipopt: {e}")
        all_results.append({
            'w_inj': w_inj, 'u1': u1, 'GOR': gor_val,
            'x1_eq': np.nan, 'x2_eq': np.nan, 'x3_eq': np.nan,
            'estado': original_state,
            'status': f'erro: {e}',
            'residual_norm': np.nan,
            **{name: np.nan for name in output_names}
        })


df = pd.DataFrame(all_results)
df.to_csv("mapa_equilibrio_casadi.csv", sep=';', decimal=',', index=False)

