import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# ------------------------ Problema 1 ------------------------------------

# Datos del problema 1
y_data = np.array([10, 20, 30, 40, 50])
t_data = np.array([3.5, 5.2, 6.0, 7.3, 7.9])

# --- Gráfico 1: Datos originales ---
plt.figure(figsize=(8, 6))
plt.plot(t_data, y_data, marker='o', linestyle='-', color='b')
plt.xlabel('Tiempo t (s)')
plt.ylabel('Distancia y (m)')
plt.title('Caída de Objetos en la Luna: Datos Originales')
plt.grid(True)
plt.savefig('Latex/plot_hoja_calculo_1_1.png') # Guarda el gráfico
#plt.show()

# --- Gráfico 2: Datos en escala log-log y regresión ---
log_t_data = np.log(t_data)
log_y_data = np.log(y_data)

# Realizar regresión lineal
slope, intercept, r_value, p_value, std_err = linregress(log_t_data, log_y_data)
C_estimado = slope
log_B_estimado = intercept
B_estimado = np.exp(log_B_estimado)

print(f"Problema 1 - Estimaciones:")
print(f"  Pendiente (C): {C_estimado:.3f}")
print(f"  Ordenada al origen (ln(B)): {log_B_estimado:.3f}")
print(f"  B: {B_estimado:.3f}")
print(f"  Coeficiente de correlación R^2: {r_value**2:.4f}")


plt.figure(figsize=(8, 6))
plt.plot(log_t_data, log_y_data, marker='s', linestyle='-', color='r', label='Datos Log-Log')
plt.plot(log_t_data, intercept + slope * log_t_data, 'g--', label=f'Ajuste lineal: ln(y) = {slope:.2f}ln(t) + ({intercept:.2f})')
plt.xlabel('ln(t)')
plt.ylabel('ln(y)')
plt.title('Caída de Objetos en la Luna: Escala Log-Log y Ajuste Lineal')
plt.legend()
plt.grid(True)
plt.savefig('Latex/plot_hoja_calculo_1_2.png') # Guarda el gráfico
#plt.show()

# ------------------------ Problema 2 ------------------------------------

# Datos del problema 2
P_data = np.array([2.10, 4.19, 9.14, 10.82, 16.85])
t_data_acciones = np.array([1, 6, 11, 16, 21]) # Años desde 1980

# --- Gráfico 1: Datos originales ---
plt.figure(figsize=(8, 6))
plt.plot(t_data_acciones, P_data, marker='o', linestyle='-', color='b')
plt.xlabel('Años desde 1980 (t)')
plt.ylabel('Precio P (dólares)')
plt.title('Valor de Acciones: Datos Originales')
plt.grid(True)
plt.savefig('Latex/plot_hoja_calculo_2_1.png')
#plt.show()

# --- Gráfico 2: Datos en escala log-log y regresión ---
log_t_data_acciones = np.log(t_data_acciones)
log_P_data = np.log(P_data)

# Realizar regresión lineal
slope, intercept, r_value, p_value, std_err = linregress(log_t_data_acciones, log_P_data)
C_estimado_acc = slope
log_B_estimado_acc = intercept
B_estimado_acc = np.exp(log_B_estimado_acc)

print(f"\nProblema 2 - Estimaciones:")
print(f"  Pendiente (C): {C_estimado_acc:.3f}")
print(f"  Ordenada al origen (ln(B)): {log_B_estimado_acc:.3f}")
print(f"  B: {B_estimado_acc:.3f}")
print(f"  Coeficiente de correlación R^2: {r_value**2:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(log_t_data_acciones, log_P_data, marker='s', linestyle='-', color='r', label='Datos Log-Log')
plt.plot(log_t_data_acciones, intercept + slope * log_t_data_acciones, 'g--', label=f'Ajuste lineal: ln(P) = {slope:.2f}ln(t) + ({intercept:.2f})')
plt.xlabel('ln(t)')
plt.ylabel('ln(P)')
plt.title('Valor de Acciones: Escala Log-Log y Ajuste Lineal')
plt.legend()
plt.grid(True)
plt.savefig('Latex/plot_hoja_calculo_2_2.png')
#plt.show()

# ------------------------ Problema 3 ------------------------------------

# Datos del problema 3
T_data = np.array([0.44, 1.61, 3.88, 7.89]) # Periodo T (años)
r_data = np.array([0.088, 0.208, 0.374, 0.600]) # Radio r (Gm)

# --- Gráfico 1: Datos originales ---
plt.figure(figsize=(8, 6))
plt.plot(r_data, T_data, marker='o', linestyle='-', color='b')
plt.xlabel('Radio r (Gm)')
plt.ylabel('Periodo T (años)')
plt.title('Órbitas de Satélites: Datos Originales')
plt.grid(True)
plt.savefig('Latex/plot_hoja_calculo_3_1.png')
#plt.show()

# --- Gráfico 2: Datos en escala log-log y regresión ---
log_r_data = np.log(r_data)
log_T_data = np.log(T_data)

# Realizar regresión lineal (ln(T) vs ln(r))
# ln(T) = n * ln(r) + ln(C)
# n es la pendiente, ln(C) es la ordenada al origen
slope, intercept, r_value, p_value, std_err = linregress(log_r_data, log_T_data)
n_estimado = slope
log_C_estimado = intercept
C_estimado_sat = np.exp(log_C_estimado)

print(f"\nProblema 3 - Estimaciones:")
print(f"  Pendiente (n): {n_estimado:.3f}")
print(f"  Ordenada al origen (ln(C)): {log_C_estimado:.3f}")
print(f"  C: {C_estimado_sat:.3f}")
print(f"  Coeficiente de correlación R^2: {r_value**2:.4f}")


plt.figure(figsize=(8, 6))
plt.plot(log_r_data, log_T_data, marker='s', linestyle='-', color='r', label='Datos Log-Log')
plt.plot(log_r_data, intercept + slope * log_r_data, 'g--', label=f'Ajuste lineal: ln(T) = {slope:.2f}ln(r) + ({intercept:.2f})')
plt.xlabel('ln(r)')
plt.ylabel('ln(T)')
plt.title('Órbitas de Satélites: Escala Log-Log y Ajuste Lineal')
plt.legend()
plt.grid(True)
plt.savefig('Latex/plot_hoja_calculo_3_2.png')
#plt.show()