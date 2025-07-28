# lanzamiento_mecanico_con_zonas.py
# Simulación interactiva con la capacidad de visualizar las zonas alcanzables.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation
from matplotlib.patches import Circle

# --- 1. PARÁMETROS Y CONSTANTES DEL ESCENARIO ---
g = 9.81
v0 = 35.0

X_OBJETIVO = 70.0
Y_OBJETIVO = 50.0
TAMANO_OBJETIVO = 1.5

# --- 2. CONFIGURACIÓN DE LA FIGURA Y LOS EJES ---
fig, ax = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(left=0.1, bottom=0.25)

ax.set_title("Simulación: El Lanzamiento del Mecánico")
ax.set_xlabel("Distancia Horizontal (m)")
ax.set_ylabel("Altura Vertical (m)")
ax.grid(True, linestyle='--', zorder=1) # Grid en capa 1
ax.set_xlim(-5, 90)
ax.set_ylim(-5, 70)
ax.set_aspect('equal', 'box')

# --- 3. CREACIÓN DE LOS "ARTISTAS" DE MATPLOTLIB ---
cannon, = ax.plot([], [], 'k-', lw=5, zorder=5)
projectile, = ax.plot([], [], 'o', color='gray', markersize=8, zorder=10)
trace, = ax.plot([], [], ':', color='gray', alpha=0.7, zorder=9)
target_ball = Circle((X_OBJETIVO, Y_OBJETIVO), TAMANO_OBJETIVO/2, color='deepskyblue', zorder=5)
ax.add_patch(target_ball)
status_text = ax.text(0.5, 1.02, "Ajusta el ángulo y presiona LANZAR", 
                      ha='center', va='bottom', transform=ax.transAxes, fontsize=14)
ani = None
# Artista para las zonas de color (inicialmente vacío)
zone_fill = None

# --- 4. FUNCIONES DE LA INTERFAZ ---

def update_aim(val):
    """Actualiza la dirección del cañón."""
    theta_rad = np.radians(theta_slider.val)
    end_x = 4 * np.cos(theta_rad)
    end_y = 4 * np.sin(theta_rad)
    cannon.set_data([0, end_x], [0, end_y])
    fig.canvas.draw_idle()

def launch(event):
    """Prepara y comienza la animación del proyectil."""
    global ani
    if ani and ani.event_source: ani.event_source.stop()

    theta_rad = np.radians(theta_slider.val)
    trace.set_data([], []); projectile.set_data([], []); status_text.set_text("")
    
    t_max_flight = (2 * v0) / g; sim_duration = t_max_flight * 1.2
    num_frames = 200; t_points = np.linspace(0, sim_duration, num_frames)
    x_proj = v0 * np.cos(theta_rad) * t_points
    y_proj = v0 * np.sin(theta_rad) * t_points - 0.5 * g * t_points**2
    
    trace_data = ([], [])
    interception = False

    def animate_frame(i):
        nonlocal interception
        if interception: return projectile, trace
        xp, yp = x_proj[i], y_proj[i]
        
        if yp < -5:
            if not interception: status_text.set_text("FALLO"); status_text.set_color('red')
            ani.event_source.stop()
            return projectile, trace

        projectile.set_data([xp], [yp]); trace_data[0].append(xp); trace_data[1].append(yp)
        trace.set_data(trace_data[0], trace_data[1])
        
        distance = np.sqrt((xp - X_OBJETIVO)**2 + (yp - Y_OBJETIVO)**2)
        
        if distance < TAMANO_OBJETIVO / 2:
            interception = True
            status_text.set_text("¡IMPACTO!"); status_text.set_color('limegreen')
            ani.event_source.stop()
        
        if i == num_frames - 1 and not interception:
            status_text.set_text("FALLO"); status_text.set_color('red')

        return projectile, trace

    ani = animation.FuncAnimation(fig, animate_frame, frames=num_frames,
                                  interval=20, blit=False, repeat=False)
    fig.canvas.draw_idle()

def show_zones(event):
    """Dibuja las zonas alcanzable e inalcanzable en el gráfico."""
    global zone_fill
    # Si ya hay zonas dibujadas, las borra para poder redibujar (en caso de que v0 cambiara)
    if zone_fill is not None:
        for fill in zone_fill:
            fill.remove()
    
    # Rango de X para dibujar la parábola de seguridad
    x_range = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 400)
    
    # Ecuación de la Parábola de Seguridad
    y_parabola = (v0**2 / (2 * g)) - (g / (2 * v0**2)) * x_range**2
    
    # Dibujar la zona alcanzable (verde)
    fill_safe = ax.fill_between(x_range, y_parabola, ax.get_ylim()[0], 
                                color='green', alpha=0.2, zorder=0)
                                
    # Dibujar la zona inalcanzable (roja)
    fill_unsafe = ax.fill_between(x_range, y_parabola, ax.get_ylim()[1], 
                                  color='red', alpha=0.2, zorder=0)
    
    # Guardar la referencia a los objetos de relleno
    zone_fill = [fill_safe, fill_unsafe]
    
    status_text.set_text("Zonas de alcance mostradas para v₀ = {:.1f} m/s".format(v0))
    status_text.set_color('black')
    fig.canvas.draw_idle()


# --- 5. CREACIÓN DE WIDGETS ---
# Ajustar posiciones para el nuevo botón
ax_theta = plt.axes([0.25, 0.1, 0.6, 0.03])
ax_launch = plt.axes([0.65, 0.025, 0.1, 0.04])
ax_zones = plt.axes([0.8, 0.025, 0.15, 0.04]) # Eje para el nuevo botón

theta_slider = Slider(ax=ax_theta, label='Ángulo de Lanzamiento (°)', valmin=0, valmax=90, valinit=45)
launch_button = Button(ax_launch, 'LANZAR', hovercolor='limegreen')
zones_button = Button(ax_zones, 'Mostrar Zonas', hovercolor='cyan') # Nuevo botón

# --- 6. CONEXIÓN DE WIDGETS ---
theta_slider.on_changed(update_aim)
launch_button.on_clicked(launch)
zones_button.on_clicked(show_zones) # Conectar el nuevo botón a su función

# --- 7. INICIO ---
update_aim(None)
plt.show()