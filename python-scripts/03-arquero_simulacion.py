# arquero_simulacion_final_con_deteccion.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation

# --- 1. Parámetros y Constantes ---
g = 9.81
x1, y1, R1 = 40.0, 23.0, 4.0
x2, y2, R2 = 65.0, 19.0, 4.0
X_LIMIT, Y_LIMIT = 90, 50

# --- 2. Configuración de la Figura (Modo Oscuro) ---
fig, ax = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(left=0.1, bottom=0.3, top=0.9)
BACKGROUND_COLOR = '#2B2B2B'; PLOT_COLOR = '#3C3C3C'; LIGHT_COLOR = 'white'
fig.patch.set_facecolor(BACKGROUND_COLOR); ax.set_facecolor(PLOT_COLOR)
for spine in ax.spines.values(): spine.set_color(LIGHT_COLOR)
ax.tick_params(axis='both', colors=LIGHT_COLOR)
ax.set_xlim(0, X_LIMIT); ax.set_ylim(0, Y_LIMIT); ax.set_aspect('equal', 'box')
ax.grid(True, linestyle=':', alpha=0.4, color=LIGHT_COLOR)
ax.set_title("El Desafío del Arquero", color=LIGHT_COLOR, fontsize=16)
ax.set_xlabel("Distancia (m)", color=LIGHT_COLOR); ax.set_ylabel("Altura (m)", color=LIGHT_COLOR)

# --- 3. Crear Artistas Gráficos ---
ring1_patch = Circle((x1, y1), R1, fc='none', ec='cyan', lw=2, ls='--'); ax.add_patch(ring1_patch)
ring2_patch = Circle((x2, y2), R2, fc='none', ec='cyan', lw=2, ls='--'); ax.add_patch(ring2_patch)
cannon = Rectangle((-2, -0.5), 4, 1, fc='gray', ec='white'); ax.add_patch(cannon)
projectile, = ax.plot([], [], 'o', color='orange', markersize=10, zorder=10)
trace, = ax.plot([], [], ':', color='orange', alpha=0.7)
status_text = ax.text(0.5, 0.7, "Apunta y presiona Lanzar!", ha='center', va='bottom', 
                      transform=ax.transAxes, fontsize=14, color=LIGHT_COLOR)
ani = None

# --- 4. Funciones de la Interfaz ---
def update_aim(val):
    theta_rad = np.radians(theta_slider.val)
    transform = plt.matplotlib.transforms.Affine2D().rotate(theta_rad) + ax.transData
    cannon.set_transform(transform)
    fig.canvas.draw_idle()

def launch(event):
    global ani
    v0 = v0_slider.val
    theta_rad = np.radians(theta_slider.val)
    
    # Resetear estado visual antes de cada lanzamiento
    ring1_patch.set_edgecolor('cyan')
    ring2_patch.set_edgecolor('cyan')
    status_text.set_text("")
    
    t_flight = (2 * v0 * np.sin(theta_rad)) / g if np.sin(theta_rad) > 0 else 0
    t_points = np.linspace(0, t_flight, num=150)
    x_points = v0 * np.cos(theta_rad) * t_points
    y_points = v0 * np.sin(theta_rad) * t_points - 0.5 * g * t_points**2
    trace_data = ([], [])

    # banderas para asegurar que cada anillo se compruebe solo una vez
    checked_ring1 = False
    checked_ring2 = False

    def init_anim():
        projectile.set_data([], [])
        trace.set_data([], [])
        return projectile, trace, ring1_patch, ring2_patch

    def animate_frame(i):
        nonlocal checked_ring1, checked_ring2

        # Detener animación si sale de los límites
        if y_points[i] < 0 or x_points[i] > X_LIMIT:
            ani.event_source.stop()
            return projectile, trace, ring1_patch, ring2_patch
        
        # Mover proyectil y dibujar traza
        projectile.set_data([x_points[i]], [y_points[i]])
        trace_data[0].append(x_points[i])
        trace_data[1].append(y_points[i])
        trace.set_data(trace_data[0], trace_data[1])
        
        # ******** LÓGICA DE DETECCIÓN EN TIEMPO REAL ********
        # Comprobar anillo 1 solo cuando el proyectil lo cruza
        if not checked_ring1 and x_points[i] >= x1:
            # Usamos la fórmula de la trayectoria para la máxima precisión
            y_at_x1 = x1 * np.tan(theta_rad) - (g * x1**2) / (2 * v0**2 * np.cos(theta_rad)**2 + 1e-6)
            hit1 = abs(y_at_x1 - y1) < R1
            ring1_patch.set_edgecolor('limegreen' if hit1 else 'red')
            checked_ring1 = True # Marcar como comprobado

        # Comprobar anillo 2 solo cuando el proyectil lo cruza
        if not checked_ring2 and x_points[i] >= x2:
            y_at_x2 = x2 * np.tan(theta_rad) - (g * x2**2) / (2 * v0**2 * np.cos(theta_rad)**2 + 1e-6)
            hit2 = abs(y_at_x2 - y2) < R2
            ring2_patch.set_edgecolor('limegreen' if hit2 else 'red')
            checked_ring2 = True # Marcar como comprobado
        # ******************************************************
        
        # Devolver todos los artistas que cambian para que blit funcione
        return projectile, trace, ring1_patch, ring2_patch

    ani = animation.FuncAnimation(fig, animate_frame, frames=len(t_points),
                                  init_func=init_anim, blit=True, interval=25, repeat=False)

    # Verificación final para el mensaje de texto
    def check_result_text(*args):
        y_at_x1 = np.interp(x1, x_points, y_points, left=Y_LIMIT, right=-1)
        y_at_x2 = np.interp(x2, x_points, y_points, left=Y_LIMIT, right=-1)
        hit1 = abs(y_at_x1 - y1) < R1
        hit2 = abs(y_at_x2 - y2) < R2
        if hit1 and hit2: 
            status_text.set_text("¡ÉXITO!")
            status_text.set_color('lime')
        else: 
            status_text.set_text("FALLO")
            status_text.set_color('red')
        fig.canvas.draw_idle()

    fig.canvas.new_timer(interval=len(t_points) * 25 + 200, callbacks=[(check_result_text, [], {})]).start()
    fig.canvas.draw_idle()


# --- 5. Creación y Conexión de Widgets ---
ax_v0 = plt.axes([0.25, 0.15, 0.6, 0.03]);
v0_slider = Slider(ax=ax_v0, label=r'$v_0$ (m/s)', valmin=10, valmax=60, valinit=35, initcolor='none')

ax_theta = plt.axes([0.25, 0.1, 0.6, 0.03]);
theta_slider = Slider(ax=ax_theta, label=r'$\theta$ (°)', valmin=0, valmax=90, valinit=45, initcolor='none')

ax_launch = plt.axes([0.8, 0.025, 0.1, 0.04]);
launch_button = Button(ax_launch, 'Lanzar!', hovercolor='limegreen')

for w in [v0_slider, theta_slider]: 
    w.label.set_color(LIGHT_COLOR)
    w.valtext.set_color(LIGHT_COLOR)
    w.ax.set_facecolor(BACKGROUND_COLOR)
launch_button.label.set_color('black')

theta_slider.on_changed(update_aim)
launch_button.on_clicked(launch)

# --- 6. Estado Inicial y Ejecución ---
update_aim(None)
plt.show()