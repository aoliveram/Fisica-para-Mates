# bola_en_escalera_sim_final_con_revelacion.py
# Fusión del script funcional con la mejora de mostrar el resultado al final.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# --- 1. Parámetros y Constantes ---
g = 9.81
NUM_ESCALONES = 40

# --- 2. Configuración de la Figura y los Ejes ---
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(left=0.1, bottom=0.35, top=0.9)

# --- Artistas Gráficos ---
escalera_line, = ax.plot([], [], 'k-', lw=2)
projectile, = ax.plot([], [], 'o', color='red', markersize=10)
trace, = ax.plot([], [], ':', color='red', alpha=0.6)
resultado_text = ax.text(0.95, 0.95, "Ajusta los parámetros y presiona LANZAR", 
                         ha='right', va='top', transform=ax.transAxes, 
                         fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.7))

# --- Configuración del Gráfico ---
ax.set_title("Simulación: La Bola en la Escalera")
ax.set_xlabel("Distancia Horizontal (m)")
ax.set_ylabel("Altura (m)")
ax.grid(True, linestyle='--')

# --- 3. Funciones de la Interfaz ---

def dibujar_escalera(w, h):
    x_coords, y_coords = [0], [0]
    for n in range(1, NUM_ESCALONES + 1):
        x_prev_step = (n - 1) * w
        x_curr_step = n * w
        y_curr_step = -n * h
        x_coords.extend([x_prev_step, x_curr_step])
        y_coords.extend([y_curr_step, y_curr_step])
    
    escalera_line.set_data(x_coords, y_coords)
    ax.set_xlim(-w * 1.1, 25 * w)
    ax.set_ylim(-(25 * h), h * 2)
    ax.set_aspect('equal', 'box')

def launch(event):
    # --- PREPARACIÓN ---
    v0 = v0_slider.val
    w = w_slider.val
    h = h_slider.val
    
    # Resetear el estado visual, incluyendo el texto de resultado.
    dibujar_escalera(w, h)
    projectile.set_data([], [])
    trace.set_data([], [])
    resultado_text.set_text("") # Limpiar el texto anterior.
    
    if w < 1e-6 or h < 1e-6:
        resultado_text.set_text("Dimensiones inválidas.")
        return
        
    # --- CÁLCULO (SIN MOSTRAR AÚN) ---
    # El cálculo se hace al principio para definir la duración de la animación.
    n_aterrizaje = np.ceil((2 * h * v0**2) / (g * w**2)) if v0 > 0 and w > 0 else 1
    
    # --- ANIMACIÓN ---
    tiempo_de_vuelo = np.sqrt(2 * n_aterrizaje * h / g)
    t_points = np.linspace(0, tiempo_de_vuelo, 100)
    x_points = v0 * t_points
    y_points = -0.5 * g * t_points**2
    
    x_trace, y_trace = [], []
    # Bucle de animación
    for i in range(len(t_points)):
        projectile.set_data([x_points[i]], [y_points[i]])
        x_trace.append(x_points[i])
        y_trace.append(y_points[i])
        trace.set_data(x_trace, y_trace)
        plt.pause(0.01)

    # --- REVELACIÓN DEL RESULTADO ---
    # Esta línea ahora se ejecuta DESPUÉS de que el bucle de animación ha terminado.
    resultado_text.set_text(f"Cálculo: n = {int(n_aterrizaje)}")
    fig.canvas.draw_idle() # Forzar un redibujado final para asegurar que el texto aparezca.

# --- 4. Creación de los Widgets ---
ax_v0 = plt.axes([0.275, 0.20, 0.5, 0.03])
ax_w = plt.axes([0.275, 0.15, 0.5, 0.03])
ax_h = plt.axes([0.275, 0.1, 0.5, 0.03])
ax_launch = plt.axes([0.8, 0.025, 0.1, 0.04])

v0_slider = Slider(ax=ax_v0, label='Velocidad Inicial: v₀ (m/s)', valmin=0.5, valmax=10, valinit=3.0)
w_slider = Slider(ax=ax_w, label='Ancho Escalón: w (m)', valmin=0.05, valmax=0.50, valinit=0.28)
h_slider = Slider(ax=ax_h, label='Altura Escalón: h (m)', valmin=0.05, valmax=0.50, valinit=0.18)
launch_button = Button(ax_launch, 'LANZAR', hovercolor='limegreen')

# --- 5. Conectar los Widgets a las Funciones ---
launch_button.on_clicked(launch)

# --- 6. Estado Inicial y Ejecución ---
dibujar_escalera(w_slider.valinit, h_slider.valinit)
plt.show()