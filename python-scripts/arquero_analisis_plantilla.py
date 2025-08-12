# arquero_analisis_plantilla.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# --- 1. CONFIGURACIÓN DEL GRÁFICO ---

x_lim_vis = (20, 70)
y_lim_vis = (15, 55)

fig, ax = plt.subplots(figsize=(12, 9))

# --- 2. Estilo del Gráfico (Sin datos) ---
ax.set_title("Espacio de Soluciones", fontsize=16)
ax.set_xlabel("Ángulo de Lanzamiento (°)", fontsize=12)
ax.set_ylabel("Velocidad Inicial (m/s)", fontsize=12)

# Aplicar los límites de visualización
ax.set_xlim(x_lim_vis)
ax.set_ylim(y_lim_vis)

# --- 3. Configuración de la Rejilla ---
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))

ax.grid(which='major', linestyle='--', linewidth='0.8', color='black', alpha=0.6)
ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black', alpha=0.4)

# Mensaje para el estudiante
ax.text(0.95, 0.90, 'Usa esta rejilla para pintar\nlas soluciones que encuentres\ncon la simulación interactiva.',
        ha='right', va='top', transform=ax.transAxes,
        fontsize=12, color='gray', alpha=0.7,
        bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.5'))

# Mostrar la plantilla
plt.show()

# fig.savefig("plantilla_rejilla.png", dpi=300)