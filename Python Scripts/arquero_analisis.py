# arquero_analisis.py
# Versión básica de análisis, diseñada para claridad

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# --- 1. CONSTANTES Y PARÁMETROS DEL ESCENARIO ---
g = 9.81
x_c1, y_c1, R1 = 40.0, 23.0, 4.0
x_c2, y_c2, R2 = 65.0, 19.0, 6.0

# --- 2. BÚSQUEDA DE SOLUCIONES (Lógica Básica con Bucles For) ---
# Definimos los rangos de velocidad y ángulo con pasos gruesos.
paso = 0.5
rango_velocidad = np.arange(10, 60, paso)
rango_angulo = np.arange(0, 90, paso)

# Determinamos el número de filas y columnas para nuestra matriz de resultados.
num_filas = len(rango_velocidad)  # Cada fila representa una velocidad
num_columnas = len(rango_angulo)   # Cada columna representa un ángulo

# Creamos la matriz de resultados, inicialmente llena de ceros (fallos).
resultados = np.zeros((num_filas, num_columnas))

# ======================= Análisis de Soluciones ================================

# Bucle for tradicional que itera sobre los ÍNDICES de las filas (velocidades).
for idx_v in range(num_filas):
    
    # Bucle for anidado que itera sobre los ÍNDICES de las columnas (ángulos).
    for idx_a in range(num_columnas):
        
        # Obtenemos el valor de la velocidad y el ángulo correspondientes a los índices actuales.
        # Este paso es ahora explícito y claro.
        velocidad = rango_velocidad[idx_v]
        angulo_deg = rango_angulo[idx_a]
        
        # --- Lógica Física Directa ---
        angulo_rad = np.radians(angulo_deg)
        cos_theta = np.cos(angulo_rad)
        tan_theta = np.tan(angulo_rad)
        
        if cos_theta < 1e-6:
            continue
        
        # =========================================================================
        # INICIO DEL SISTEMA DE INECUACIONES
        # =========================================================================

        y_x1 = x_c1 * tan_theta - (g * x_c1**2) / (2 * velocidad**2 * cos_theta**2)
        exito_anillo1 = abs(y_x1 - y_c1) < R1
        
        y_x2 = x_c2 * tan_theta - (g * x_c2**2) / (2 * velocidad**2 * cos_theta**2)
        exito_anillo2 = abs(y_x2 - y_c2) < R2
        
        if exito_anillo1 and exito_anillo2:
            # Si es una solución, usamos los índices (idx_v, idx_a) para marcar
            # la celda correspondiente en nuestra matriz.
            resultados[idx_v, idx_a] = 1

        # =========================================================================
        # FIN DEL SISTEMA DE INECUACIONES
        # =========================================================================

# ===================== FIN Análisis de Soluciones ================================


# --- 3. VISUALIZACIÓN ---
fig, ax = plt.subplots(figsize=(12, 9))

# La visualización con imshow es la misma
cax = ax.imshow(resultados, origin='lower', aspect='auto',
                extent=[rango_angulo[0], rango_angulo[-1], rango_velocidad[0], rango_velocidad[-1]],
                cmap='Greens', vmin=0, vmax=1.5)

# --- Estilo del Gráfico ---
ax.set_title("Espacio de Soluciones", fontsize=16)
ax.set_xlabel("Ángulo de Lanzamiento (°)", fontsize=12)
ax.set_ylabel("Velocidad Inicial (m/s)", fontsize=12)
ax.set_xlim((20, 70))
ax.set_ylim((15, 55))

# Etiquetas de números en los ejes
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_major_locator(MultipleLocator(5))

# Rejilla cada 0.5 unidades
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))

ax.grid(which='major', linestyle='--', linewidth='0.8', color='gray', alpha=0.5)
ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.3)

fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

plt.show()