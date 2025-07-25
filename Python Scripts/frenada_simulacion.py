# frenada_emergencia_explorador_3_sliders.py
# Versión final con sliders para D, Δv y la desaceleración 'a'.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider, Button

# --- CONSTANTES ---
CAR_LENGTH = 4.5  # Longitud visual de los autos en metros

# --- CONFIGURACIÓN DE LA FIGURA Y EJES (MODO OSCURO) ---
fig, (ax_sim, ax_space) = plt.subplots(2, 1, figsize=(10, 8.5), 
                                     gridspec_kw={'height_ratios': [1, 5]})
plt.subplots_adjust(left=0.1, bottom=0.35, hspace=0.5) # Aumentado el espacio para los widgets

BACKGROUND_COLOR = '#2B2B2B'; PLOT_COLOR = '#3C3C3C'; LIGHT_COLOR = 'white'
SAFE_COLOR = 'limegreen'; CRASH_COLOR = 'red'

fig.patch.set_facecolor(BACKGROUND_COLOR)
for ax in [ax_sim, ax_space]:
    ax.set_facecolor(PLOT_COLOR)
    for spine in ax.spines.values(): spine.set_color(LIGHT_COLOR)
    ax.tick_params(axis='both', colors=LIGHT_COLOR)
    ax.xaxis.label.set_color(LIGHT_COLOR)
    ax.yaxis.label.set_color(LIGHT_COLOR)
    ax.title.set_color(LIGHT_COLOR)

# --- CONFIGURACIÓN DEL GRÁFICO SUPERIOR (SIMULACIÓN 1D) ---
ax_sim.set_title("Simulación 1D en la Carretera")
ax_sim.set_yticks([]); ax_sim.set_xlabel("Posición (m)")
car_A = Rectangle((0, -0.2), CAR_LENGTH, 0.4, color='cyan')
car_B = Rectangle((0, -0.2), CAR_LENGTH, 0.4, color='magenta')
ax_sim.add_patch(car_A); ax_sim.add_patch(car_B)
status_text = ax_sim.text(0.5, 0.5, "Presiona 'SIMULAR SETUP'", ha='center', va='center',
                          transform=ax_sim.transAxes, fontsize=12, color=LIGHT_COLOR)

# --- CONFIGURACIÓN DEL GRÁFICO INFERIOR (ESPACIO DE PARÁMETROS) ---
ax_space.set_title("Explorador del Espacio de Parámetros")
ax_space.set_xlabel("Distancia Inicial, D (m)")
ax_space.set_ylabel("Diferencia de Velocidad, Δv (m/s)")
ax_space.grid(True, linestyle=':', alpha=0.4)
markers_safe = {'x': [], 'y': []}; markers_crash = {'x': [], 'y': []}
safe_plot, = ax_space.plot([], [], 'o', color=SAFE_COLOR, label='Seguro')
crash_plot, = ax_space.plot([], [], 'o', color=CRASH_COLOR, label='Colisión')
ax_space.legend()

# --- LÓGICA Y FUNCIONES DE LA INTERFAZ ---
def check_safety(d, dv, a):
    """Aplica la condición teórica de seguridad: (Δv)² < 2aD"""
    return dv**2 < 2 * a * d

def simulate_setup(event):
    """
    Se ejecuta al presionar "SIMULAR SETUP".
    Versión corregida con cálculo de tiempo de colisión correcto y velocidad de animación constante.
    """
    d = d_slider.val
    dv = dv_slider.val
    a = a_slider.val
    
    is_safe = check_safety(d, dv, a)
    
    # --- Colocar marcador en el espacio de parámetros ---
    if is_safe:
        markers_safe['x'].append(d)
        markers_safe['y'].append(dv)
        safe_plot.set_data(markers_safe['x'], markers_safe['y'])
    else:
        markers_crash['x'].append(d)
        markers_crash['y'].append(dv)
        crash_plot.set_data(markers_crash['x'], markers_crash['y'])
    
    # --- Animación en la carretera ---
    v2 = 10
    v1 = v2 + dv
    
    car_A.set_x(0)
    car_B.set_x(d)
    ax_sim.set_xlim(-5, d + CAR_LENGTH + 40) # Aumentado el margen para ver mejor
    
    # --- CORRECCIÓN 1: Cálculo del Tiempo Final ---
    if is_safe:
        # Si es seguro, la simulación dura hasta que el auto A se detiene.
        t_final = v1 / a
        status_text.set_text("SEGURO")
        status_text.set_color(SAFE_COLOR)
    else:
        # Si hay colisión, usamos la fórmula cuadrática CORRECTA para el tiempo de impacto.
        # (-B + sqrt(B^2 - 4AC)) / 2A  donde A=a/2, B=-dv, C=-d
        discriminant = dv**2 + 2 * a * d
        t_final = (dv + np.sqrt(discriminant)) / a
        status_text.set_text("¡COLISIÓN!")
        status_text.set_color(CRASH_COLOR)

    # --- CORRECCIÓN 2: Velocidad de Animación Constante ---
    # Usamos un paso de tiempo fijo (dt) para asegurar una velocidad de animación consistente.
    dt = 0.05  # La simulación avanzará en pasos de 0.05 segundos.
    num_frames = int(t_final / dt)
    
    # Bucle de animación
    for i in range(num_frames + 1):
        t = i * dt
        
        pos_A = v1 * t - 0.5 * a * t**2
        pos_B = d + v2 * t
        
        # Detener la simulación si los autos ya chocaron
        if not is_safe and pos_A >= pos_B:
            # Colocar el auto A justo detrás del B para mostrar el impacto
            car_A.set_x(pos_B)
            break
        
        car_A.set_x(pos_A)
        car_B.set_x(pos_B)
        plt.pause(0.01) # Pausa corta para el renderizado

def show_full_solution(event):
    """Rellena el espacio de parámetros con la solución teórica."""
    a = a_slider.val # Obtener el valor de 'a' para el cálculo completo
    
    d_range = np.linspace(d_slider.valmin, d_slider.valmax, 150)
    dv_range = np.linspace(dv_slider.valmin, dv_slider.valmax, 150)
    D, dV = np.meshgrid(d_range, dv_range)
    
    is_safe_matrix = check_safety(D, dV, a)
    
    ax_space.imshow(is_safe_matrix, origin='lower', aspect='auto',
                    extent=[d_slider.valmin, d_slider.valmax, dv_slider.valmin, dv_slider.valmax],
                    cmap='RdYlGn', alpha=0.6) # Usar _r invierte el mapa de color
    fig.canvas.draw_idle()

# --- CREACIÓN DE WIDGETS ---
# Ajustar las posiciones verticales para acomodar el nuevo slider
ax_a = plt.axes([0.25, 0.20, 0.65, 0.03])
ax_d = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_dv = plt.axes([0.25, 0.10, 0.65, 0.03])
ax_sim_button = plt.axes([0.3, 0.025, 0.18, 0.04])
ax_sol_button = plt.axes([0.52, 0.025, 0.18, 0.04])

# Crear el nuevo slider para 'a'
a_slider = Slider(ax=ax_a, label='Desaceleración (m/s²)', valmin=1, valmax=12, valinit=8.8)
d_slider = Slider(ax=ax_d, label='Distancia (m)', valmin=5, valmax=100, valinit=50)
dv_slider = Slider(ax=ax_dv, label='Δ Velocidad (m/s)', valmin=1, valmax=30, valinit=15)
sim_button = Button(ax_sim_button, 'SIMULAR SETUP', hovercolor='cyan')
sol_button = Button(ax_sol_button, 'SOLUCIÓN COMPLETA', hovercolor='limegreen')

# Aplicar estilo a todos los sliders
for w in [a_slider, d_slider, dv_slider]:
    w.label.set_color(LIGHT_COLOR)
    w.valtext.set_color(LIGHT_COLOR)
    w.ax.set_facecolor(BACKGROUND_COLOR)
for btn in [sim_button, sol_button]:
    btn.label.set_color('black')

# --- CONECTAR WIDGETS A FUNCIONES ---
sim_button.on_clicked(simulate_setup)
sol_button.on_clicked(show_full_solution)

# --- EJECUCIÓN ---
ax_space.set_xlim(d_slider.valmin, d_slider.valmax)
ax_space.set_ylim(dv_slider.valmin, dv_slider.valmax)
plt.show()