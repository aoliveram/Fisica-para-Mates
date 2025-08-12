# Nueva versión: mantiene la dinámica original pero reemplaza los rectángulos
# por imágenes (SVG convertidas a PNG en memoria) y posiciona los autos por
# el frente (A) y la parte trasera (B).

import io
import numpy as np
import cairosvg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Slider, Button

# --- CONSTANTES ---
CAR_LENGTH = 10.0  # Longitud visual de los autos en metros (usada en coordenadas de datos)
CAR_HEIGHT = 10.0  # Alto visual en coordenadas de datos (para el extent vertical)

# --- FUNCIONES AUXILIARES ---

def load_svg_as_png_array(path_svg):
    """Convierte un SVG a PNG en memoria, y así leer con ax.imshow(..., extent=...).
    """
    png_data = cairosvg.svg2png(url=path_svg)
    img = mpimg.imread(io.BytesIO(png_data), format='png')
    return img

# Cargar autos
car_A_img = load_svg_as_png_array('Python Scripts/car-side-view-A.svg')
car_B_img = load_svg_as_png_array('Python Scripts/car-side-view-B.svg')

# --- CONFIGURACIÓN DE LA FIGURA Y EJES ---
fig, (ax_sim, ax_space) = plt.subplots(2, 1, figsize=(10, 8.5),
                                     gridspec_kw={'height_ratios': [1, 5]})
plt.subplots_adjust(left=0.1, bottom=0.35, hspace=0.5)

BACKGROUND_COLOR = '#2B2B2B'; PLOT_COLOR = "#706C6C"; LIGHT_COLOR = 'white'
SAFE_COLOR = 'limegreen'; CRASH_COLOR = 'red'

fig.patch.set_facecolor(BACKGROUND_COLOR)
for ax in [ax_sim, ax_space]:
    ax.set_facecolor(PLOT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(LIGHT_COLOR)
    ax.tick_params(axis='both', colors=LIGHT_COLOR)
    ax.xaxis.label.set_color(LIGHT_COLOR)
    ax.yaxis.label.set_color(LIGHT_COLOR)
    ax.title.set_color(LIGHT_COLOR)

# --- CONFIGURACIÓN DEL GRÁFICO SUPERIOR (SIMULACIÓN 1D) ---
ax_sim.set_title("Simulación 1D en la Carretera")
ax_sim.set_yticks([]); ax_sim.set_xlabel("Posición (m)")
status_text = ax_sim.text(0.5, 0.5, " ", ha='center', va='center',
                          transform=ax_sim.transAxes, fontsize=12, color=LIGHT_COLOR)

# Dibujaremos las imágenes dentro de la simulación
init_left_A = -CAR_LENGTH
init_right_A = 0
init_bottom = -CAR_HEIGHT / 2
init_top = CAR_HEIGHT / 2
car_A_artist = ax_sim.imshow(car_A_img, extent=[init_left_A, init_right_A, init_bottom, init_top], zorder=5)
car_B_artist = ax_sim.imshow(car_B_img, extent=[50, 50 + CAR_LENGTH, init_bottom, init_top], zorder=5)

# --- CONFIGURACIÓN DEL GRÁFICO INFERIOR (ESPACIO DE PARÁMETROS) ---
ax_space.set_title("Espacio de Parámetros")
ax_space.set_xlabel("D (m)")
ax_space.set_ylabel("Δv (m/s)")
ax_space.grid(True, linestyle=':', alpha=0.4)
markers_safe = {'x': [], 'y': []}; markers_crash = {'x': [], 'y': []}
safe_plot, = ax_space.plot([], [], 'o', color=SAFE_COLOR, label='Seguro')
crash_plot, = ax_space.plot([], [], 'o', color=CRASH_COLOR, label='Colisión')
ax_space.legend()

# --- LÓGICA Y FUNCIONES DE LA INTERFAZ ---

def check_safety(d, dv, a):
    """Aplica la condición teórica de seguridad: (Δv)² < 2 a D."""
    return dv**2 < 2 * a * d


def simulate_setup(event):
    """Se ejecuta al presionar "SIMULAR SETUP."""
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

    # --- Dinámica ---
    v2 = 10.0
    v1 = v2 + dv

    # Posiciones iniciales (definimos pos_A como la coordenada del frente de A,
    # y pos_B como la coordenada de la parte trasera de B). En t=0:
    pos_A_0 = 0.0
    pos_B_0 = d

    # --- Cálculo del tiempo final ---
    if is_safe:
        t_final = v1 / a  # tiempo hasta que A se detiene
        status_text.set_text("SEGURO")
        status_text.set_color(SAFE_COLOR)
    else:
        discriminant = dv**2 + 2 * a * d
        t_final = (dv + np.sqrt(discriminant)) / a
        status_text.set_text("¡COLISIÓN!")
        status_text.set_color(CRASH_COLOR)

    # Ajustar límites X para ver toda la acción (considerando desplazamiento de B)
    max_x = d + v2 * t_final + CAR_LENGTH + 40
    ax_sim.set_xlim(-CAR_LENGTH - 5, max_x)

    # --- Animación con paso fijo ---
    dt = 0.05
    num_frames = max(1, int(np.ceil(t_final / dt)))

    for i in range(num_frames + 1):
        t = i * dt

        # pos_A es la coordenada del frente de A 
        # pos_B es la coordenada de la parte trasera de B.
        pos_A = pos_A_0 + v1 * t - 0.5 * a * t**2
        pos_B = pos_B_0 + v2 * t

        # Detectar colisión
        if (not is_safe) and (pos_A >= pos_B):
            # Fijamos que A quede justo tocando a B
            pos_A = pos_B
            # Actualizar extents finales y terminar la simulación
            car_A_artist.set_extent([pos_A - CAR_LENGTH, pos_A, init_bottom, init_top])
            car_B_artist.set_extent([pos_B, pos_B + CAR_LENGTH, init_bottom, init_top])
            fig.canvas.draw_idle()
            break

        # Actualizar posiciones
        car_A_artist.set_extent([pos_A - CAR_LENGTH, pos_A, init_bottom, init_top])
        car_B_artist.set_extent([pos_B, pos_B + CAR_LENGTH, init_bottom, init_top])

        fig.canvas.flush_events()
        plt.pause(0.01)


def show_full_solution(event):
    """Rellena el espacio de parámetros con la solución teórica."""
    a = a_slider.val
    d_range = np.linspace(d_slider.valmin, d_slider.valmax, 150)
    dv_range = np.linspace(dv_slider.valmin, dv_slider.valmax, 150)
    D, dV = np.meshgrid(d_range, dv_range)

    is_safe_matrix = check_safety(D, dV, a)

    ax_space.imshow(is_safe_matrix, origin='lower', aspect='auto',
                    extent=[d_slider.valmin, d_slider.valmax, dv_slider.valmin, dv_slider.valmax],
                    cmap='RdYlGn', alpha=0.6)
    fig.canvas.draw_idle()


# --- CREACIÓN DE WIDGETS ---
ax_a = plt.axes([0.25, 0.20, 0.65, 0.03])
ax_d = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_dv = plt.axes([0.25, 0.10, 0.65, 0.03])
ax_sim_button = plt.axes([0.3, 0.025, 0.18, 0.04])
ax_sol_button = plt.axes([0.52, 0.025, 0.18, 0.04])

# Sliders de parámetros
a_slider = Slider(ax=ax_a, label='Desaceleración (m/s²)', valmin=1, valmax=12, valinit=8.8, initcolor='none')
d_slider = Slider(ax=ax_d, label='Distancia (m)', valmin=5, valmax=100, valinit=50, initcolor='none')
dv_slider = Slider(ax=ax_dv, label='Δ Velocidad (m/s)', valmin=1, valmax=30, valinit=15, initcolor='none')

sim_button = Button(ax_sim_button, 'SIMULAR SETUP', hovercolor='cyan')
sol_button = Button(ax_sol_button, 'SOLUCIÓN COMPLETA', hovercolor='limegreen')

# Estilo de los sliders y botones
for w in [a_slider, d_slider, dv_slider]:
    w.label.set_color(LIGHT_COLOR)
    w.valtext.set_color(LIGHT_COLOR)
    w.ax.set_facecolor(BACKGROUND_COLOR)
for btn in [sim_button, sol_button]:
    btn.label.set_color('black')

# Conexiones
sim_button.on_clicked(simulate_setup)
sol_button.on_clicked(show_full_solution)

# Ejecución
ax_space.set_xlim(d_slider.valmin, d_slider.valmax)
ax_space.set_ylim(dv_slider.valmin, dv_slider.valmax)
plt.show()