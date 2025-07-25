# bola_en_escalera_sim_corregido.py
# Simulación interactiva del problema de la bola en la escalera (versión corregida)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# --- 1. Parámetros y Constantes ---
g = 9.81
NUM_ESCALONES = 25

# --- 2. Configuración de la Figura y los Ejes ---
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(left=0.1, bottom=0.35)

# Artistas gráficos
escalera_line, = ax.plot([], [], 'k-', lw=2)
projectile, = ax.plot([], [], 'o', color='red', markersize=10)
trace, = ax.plot([], [], ':', color='red', alpha=0.6)
resultado_text = ax.text(0.5, 1.02, "Ajusta los parámetros y presiona LANZAR", 
                         ha='center', va='bottom', transform=ax.transAxes, 
                         fontsize=12)

# Configuración del gráfico
ax.set_title("Simulación: La Bola en la Escalera")
ax.set_xlabel("Distancia Horizontal (m)")
ax.set_ylabel("Altura Vertical (m)")
ax.grid(True, linestyle='--')

# --- 3. Funciones de la Interfaz ---

def dibujar_escalera(w, h):
    x_coords = [0]
    y_coords = [0]
    for n in range(1, NUM_ESCALONES + 1):
        x_coords.append(n * w)
        y_coords.append(-(n - 1) * h)
        x_coords.append(n * w)
        y_coords.append(-n * h)
    
    escalera_line.set_data(x_coords, y_coords)
    ax.set_xlim(-w, NUM_ESCALONES * w * 1.05)
    ax.set_ylim(-(NUM_ESCALONES * h * 1.05), h)
    ax.set_aspect('equal', 'box')

def launch(event):
    v0 = v0_slider.val
    w = w_slider.val
    h = h_slider.val
    
    dibujar_escalera(w, h)
    projectile.set_data([], [])
    trace.set_data([], [])
    
    if w < 1e-6:
        resultado_text.set_text("El ancho (w) no puede ser cero.")
        return
        
    n_aterrizaje = np.ceil((2 * h * v0**2) / (g * w**2))
    resultado_text.set_text(f"Cálculo teórico: Aterriza en el escalón n = {int(n_aterrizaje)}")

    # Calcular la trayectoria hasta un poco más allá del impacto
    tiempo_de_vuelo = np.sqrt(2 * n_aterrizaje * h / g)
    t_points = np.linspace(0, tiempo_de_vuelo * 1.05, 100)
    x_points = v0 * t_points
    y_points = -0.5 * g * t_points**2
    
    x_trace, y_trace = [], []
    for i in range(len(t_points)):
        # ******** LA CORRECCIÓN ESTÁ AQUÍ ********
        # Pasamos los valores como listas de un solo elemento
        projectile.set_data([x_points[i]], [y_points[i]])
        # *****************************************
        
        x_trace.append(x_points[i])
        y_trace.append(y_points[i])
        trace.set_data(x_trace, y_trace)

        # Detener la animación si la bola ya está por debajo de la escalera
        if y_points[i] < -(NUM_ESCALONES * h):
            break

        plt.pause(0.01)

# --- 4. Creación de los Widgets ---
ax_v0 = plt.axes([0.25, 0.20, 0.6, 0.03])
ax_w = plt.axes([0.25, 0.15, 0.6, 0.03])
ax_h = plt.axes([0.25, 0.1, 0.6, 0.03])
ax_launch = plt.axes([0.8, 0.025, 0.1, 0.04])

v0_slider = Slider(ax=ax_v0, label='Velocidad Inicial v₀ (m/s)', valmin=0.5, valmax=10, valinit=3.0)
w_slider = Slider(ax=ax_w, label='Ancho Escalón w (m)', valmin=0.05, valmax=1.0, valinit=0.28)
h_slider = Slider(ax=ax_h, label='Altura Escalón h (m)', valmin=0.05, valmax=1.0, valinit=0.18)

launch_button = Button(ax_launch, 'LANZAR', hovercolor='limegreen')

# --- 5. Conectar los Widgets a las Funciones ---
launch_button.on_clicked(launch)

# --- 6. Estado Inicial y Ejecución ---
dibujar_escalera(w_slider.valinit, h_slider.valinit)
plt.show()