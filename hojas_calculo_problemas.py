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
