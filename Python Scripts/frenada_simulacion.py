# frenada_emergencia_explorador.py
# Un explorador interactivo del espacio de parámetros para el problema de frenada de emergencia.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider, Button

# --- 1. CONSTANTES Y PARÁMETROS FIJOS ---
A_MAX = 8.8  # Desaceleración máxima constante en m/s^2
CAR_LENGTH = 4.5  # Longitud visual de los autos en metros

# --- 2. CONFIGURACIÓN DE LA FIGURA Y LOS EJES (MODO OSCURO) ---
# Creamos una figura con dos subgráficos verticales.
# El gráfico de arriba (simulación) es más pequeño que el de abajo (espacio de parámetros).
fig, (ax_sim, ax_space) = plt.subplots(2, 1, figsize=(10, 8), 
                                     gridspec_kw={'height_ratios': [1, 5]})
plt.subplots_adjust(left=0.1, bottom=0.3, hspace=0.4)

# Aplicar estilo "Modo Oscuro"
BACKGROUND_COLOR = '#2B2B2B'
PLOT_COLOR = '#3C3C3C'
LIGHT_COLOR = 'white'
SAFE_COLOR = 'limegreen'
CRASH_COLOR = 'red'

fig.patch.set_facecolor(BACKGROUND_COLOR)
for ax in [ax_sim, ax_space]:
    ax.set_facecolor(PLOT_COLOR)
    for spine in ax.spines.values(): spine.set_color(LIGHT_COLOR)
    ax.tick_params(axis='both', colors=LIGHT_COLOR)
    ax.xaxis.label.set_color(LIGHT_COLOR)
    ax.yaxis.label.set_color(LIGHT_COLOR)
    ax.title.set_color(LIGHT_COLOR)

# --- 3. CONFIGURACIÓN DEL GRÁFICO SUPERIOR (SIMULACIÓN 1D) ---
ax_sim.set_title("Simulación 1D en la Carretera")
ax_sim.set_yticks([]) # Ocultar el eje Y, ya que no es relevante
ax_sim.set_xlabel("Posición (m)")

# Creamos los "artistas" para los autos. Son rectángulos que moveremos.
car_A = Rectangle((0, -0.2), CAR_LENGTH, 0.4, color='cyan')
car_B = Rectangle((0, -0.2), CAR_LENGTH, 0.4, color='magenta')
ax_sim.add_patch(car_A)
ax_sim.add_patch(car_B)
status_text = ax_sim.text(0.5, 0.5, "Presiona 'SIMULAR SETUP'", ha='center', va='center',
                          transform=ax_sim.transAxes, fontsize=12, color=LIGHT_COLOR)

# --- 4. CONFIGURACIÓN DEL GRÁFICO INFERIOR (ESPACIO DE PARÁMETROS) ---
ax_space.set_title("Explorador del Espacio de Parámetros")
ax_space.set_xlabel("Distancia Inicial, D (m)")
ax_space.set_ylabel("Diferencia de Velocidad, Δv (m/s)")
ax_space.grid(True, linestyle=':', alpha=0.4)

# Guardamos los puntos de los marcadores para irlos añadiendo
markers_safe = {'x': [], 'y': []}
markers_crash = {'x': [], 'y': []}
# Creamos los artistas para los marcadores. Empezarán vacíos.
safe_plot, = ax_space.plot([], [], 'o', color=SAFE_COLOR, label='Seguro')
crash_plot, = ax_space.plot([], [], 'o', color=CRASH_COLOR, label='Colisión')
ax_space.legend()

# --- 5. LÓGICA Y FUNCIONES DE LA INTERFAZ ---

def check_safety(d, dv, a):
    """
    Función central que aplica la condición teórica de seguridad.
    (Δv)² < 2aD
    """
    return dv**2 < 2 * a * d

def simulate_setup(event):
    """
    Se ejecuta al presionar "SIMULAR SETUP".
    Realiza la animación y coloca un marcador.
    """
    d = d_slider.val
    dv = dv_slider.val
    
    is_safe = check_safety(d, dv, A_MAX)
    
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
    # Para la animación, asumimos una velocidad base para el auto B
    v2 = 10  # m/s (26 km/h)
    v1 = v2 + dv
    
    # Resetear posiciones iniciales
    car_A.set_x(0)
    car_B.set_x(d)
    ax_sim.set_xlim(-5, d + CAR_LENGTH + 20)
    
    # Calcular tiempo de la simulación
    if is_safe:
        # Si es seguro, el auto A se detiene antes de alcanzar al B.
        t_stop_A = v1 / A_MAX
        t_final = t_stop_A
        status_text.set_text("SEGURO")
        status_text.set_color(SAFE_COLOR)
    else:
        # Si hay colisión, calculamos el tiempo del choque
        # resolviendo la ecuación cuadrática de posición.
        # (1/2)a*t^2 - dv*t - d = 0
        discriminant = dv**2 + 2 * A_MAX * d
        t_crash = (dv - np.sqrt(discriminant)) / A_MAX if A_MAX > 0 else 0
        t_final = abs(t_crash) # Tomamos el tiempo positivo
        status_text.set_text("¡COLISIÓN!")
        status_text.set_color(CRASH_COLOR)

    # Bucle de animación
    t_points = np.linspace(0, t_final, 100)
    for t in t_points:
        pos_A = v1 * t - 0.5 * A_MAX * t**2
        pos_B = d + v2 * t
        
        # Detener la animación si el auto A ya se detuvo (en caso seguro)
        if t > v1 / A_MAX and is_safe:
             break
        
        car_A.set_x(pos_A)
        car_B.set_x(pos_B)
        plt.pause(0.01)

def show_full_solution(event):
    """
    Se ejecuta al presionar "SOLUCIÓN COMPLETA".
    Rellena el espacio de parámetros con la solución teórica.
    """
    d_range = np.linspace(d_slider.valmin, d_slider.valmax, 150)
    dv_range = np.linspace(dv_slider.valmin, dv_slider.valmax, 150)
    
    # Crear una malla de puntos (D, Δv)
    D, dV = np.meshgrid(d_range, dv_range)
    
    # Aplicar la condición de seguridad a toda la malla a la vez
    is_safe_matrix = check_safety(D, dV, A_MAX)
    
    # Dibujar la imagen con imshow
    ax_space.imshow(is_safe_matrix, origin='lower', aspect='auto',
                    extent=[d_slider.valmin, d_slider.valmax, dv_slider.valmin, dv_slider.valmax],
                    cmap='RdYlGn', alpha=0.5) # Mapa de color Rojo-Amarillo-Verde
    fig.canvas.draw_idle()

# --- 6. CREACIÓN DE WIDGETS ---
# Definir posiciones para los widgets
ax_d = plt.axes([0.25, 0.15, 0.5, 0.03])
ax_dv = plt.axes([0.25, 0.1, 0.5, 0.03])
ax_sim_button = plt.axes([0.25, 0.025, 0.2, 0.04])
ax_sol_button = plt.axes([0.55, 0.025, 0.2, 0.04])

# Crear sliders
d_slider = Slider(ax=ax_d, label='Distancia (m)', valmin=5, valmax=100, valinit=50)
dv_slider = Slider(ax=ax_dv, label='Δ Velocidad (m/s)', valmin=1, valmax=30, valinit=15)

# Crear botones
sim_button = Button(ax_sim_button, 'SIMULAR SETUP', hovercolor='cyan')
sol_button = Button(ax_sol_button, 'SOLUCIÓN COMPLETA', hovercolor='limegreen')

# Estilo de widgets
for w in [d_slider, dv_slider]:
    w.label.set_color(LIGHT_COLOR)
    w.valtext.set_color(LIGHT_COLOR)
    w.ax.set_facecolor(BACKGROUND_COLOR)
for btn in [sim_button, sol_button]:
    btn.label.set_color('black')

# --- 7. CONECTAR WIDGETS A FUNCIONES ---
sim_button.on_clicked(simulate_setup)
sol_button.on_clicked(show_full_solution)

# --- 8. EJECUCIÓN ---
# Fijar los límites del espacio de parámetros para que no cambien
ax_space.set_xlim(d_slider.valmin, d_slider.valmax)
ax_space.set_ylim(dv_slider.valmin, dv_slider.valmax)
plt.show()