import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation

# --- 1. Parámetros del Escenario y Gráfico Fijo ---
g = 9.81
x1, y1, R1 = 30.0, 10.0, 2.0
x2, y2, R2 = 60.0, 8.0, 2.0

# Límites fijos para el gráfico
X_LIMIT = 90
Y_LIMIT = 40

# --- 2. Configuración de la Figura y los Ejes (Modo Oscuro) ---
fig, ax = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(left=0.1, bottom=0.3, top=0.9)

# Estilo "Modo Oscuro"
BACKGROUND_COLOR = '#2B2B2B'
PLOT_COLOR = '#3C3C3C'
LIGHT_COLOR = 'lightgray'

fig.patch.set_facecolor(BACKGROUND_COLOR)
ax.set_facecolor(PLOT_COLOR)
ax.spines['top'].set_color(LIGHT_COLOR)
ax.spines['bottom'].set_color(LIGHT_COLOR)
ax.spines['left'].set_color(LIGHT_COLOR)
ax.spines['right'].set_color(LIGHT_COLOR)
ax.xaxis.label.set_color(LIGHT_COLOR)
ax.yaxis.label.set_color(LIGHT_COLOR)
ax.title.set_color(LIGHT_COLOR)
ax.tick_params(axis='x', colors=LIGHT_COLOR)
ax.tick_params(axis='y', colors=LIGHT_COLOR)

# Configuración inicial del gráfico
ax.set_xlim(0, X_LIMIT)
ax.set_ylim(0, Y_LIMIT)
ax.set_title("El Desafío del Arquero", fontsize=16)
ax.set_xlabel("Distancia Horizontal (m)", fontsize=12)
ax.set_ylabel("Altura Vertical (m)", fontsize=12)
ax.set_aspect('equal', adjustable='box')
ax.grid(True, linestyle=':', alpha=0.4, color=LIGHT_COLOR)

# Dibujar los anillos (permanecen estáticos)
ring1 = Circle((x1, y1), R1, facecolor='none', edgecolor='cyan', linewidth=2, linestyle='--')
ring2 = Circle((x2, y2), R2, facecolor='none', edgecolor='cyan', linewidth=2, linestyle='--')
ax.add_patch(ring1)
ax.add_patch(ring2)

# --- 3. Elementos Dinámicos y de Animación ---
# Cañón (un rectángulo que rotará)
cannon_width = 4.0
cannon_height = 1.0
cannon = Rectangle((-cannon_width/2, -cannon_height/2), cannon_width, cannon_height, facecolor='gray', edgecolor='white')
ax.add_patch(cannon)

# Objetos para la animación (inicialmente vacíos)
projectile, = ax.plot([], [], 'o', color='orange', markersize=8)
trace, = ax.plot([], [], '--', color='orange', alpha=0.7)
status_text = ax.text(0.5, 1.02, "Ajusta los parámetros y presiona LANZAR", 
                      ha='center', va='bottom', transform=ax.transAxes, 
                      fontsize=14, color='white')

# Variable para almacenar la animación para que no sea eliminada por el recolector de basura
ani = None

# --- 4. Funciones de la Interfaz ---

def update_aim(val):
    """Actualiza la rotación del cañón cuando los sliders se mueven."""
    theta_deg = theta_slider.val
    # La transformación se aplica desde el centro del rectángulo (0,0)
    transform = plt.matplotlib.transforms.Affine2D().rotate_deg(theta_deg) + ax.transData
    cannon.set_transform(transform)
    fig.canvas.draw_idle()

def launch(event):
    """Inicia la animación de lanzamiento cuando se presiona el botón."""
    global ani

    v0 = v0_slider.val
    theta_deg = theta_slider.val
    theta_rad = np.radians(theta_deg)

    # Limpiar la trayectoria anterior
    trace.set_data([], [])
    
    # Calcular la trayectoria completa
    t_flight = (2 * v0 * np.sin(theta_rad)) / g
    t_points = np.linspace(0, t_flight, num=150) # 150 frames para la animación
    x_points = v0 * np.cos(theta_rad) * t_points
    y_points = v0 * np.sin(theta_rad) * t_points - 0.5 * g * t_points**2
    
    # Asegurarse de que la animación no continúe bajo tierra
    y_points[y_points < 0] = 0

    def init_animation():
        """Prepara el proyectil para el inicio de la animación."""
        projectile.set_data([], [])
        status_text.set_text("")
        return projectile, trace, status_text

    def animate(i):
        """Función que se llama para cada cuadro de la animación."""
        # Mover el proyectil
        projectile.set_data(x_points[i], y_points[i])
        # Dibujar la estela
        trace.set_data(x_points[:i+1], y_points[:i+1])
        
        # Comprobación de éxito/fallo en tiempo real (opcional pero genial)
        if abs(x_points[i] - x1) < 0.5:
            hit1 = abs(y_points[i] - y1) < R1
            ring1.set_edgecolor('green' if hit1 else 'red')
        if abs(x_points[i] - x2) < 0.5:
            hit2 = abs(y_points[i] - y2) < R2
            ring2.set_edgecolor('green' if hit2 else 'red')
        
        # Al final, mostrar el resultado global
        if i == len(t_points) - 1:
            # Re-verificar con el cálculo exacto
            y_at_x1 = x1 * np.tan(theta_rad) - (g * x1**2) / (2 * v0**2 * np.cos(theta_rad)**2 + 1e-6)
            y_at_x2 = x2 * np.tan(theta_rad) - (g * x2**2) / (2 * v0**2 * np.cos(theta_rad)**2 + 1e-6)
            final_hit1 = abs(y_at_x1 - y1) < R1
            final_hit2 = abs(y_at_x2 - y2) < R2
            
            if final_hit1 and final_hit2:
                status_text.set_text("¡ÉXITO!")
                status_text.set_bbox(dict(facecolor='green', alpha=0.8))
            else:
                status_text.set_text("FALLO")
                status_text.set_bbox(dict(facecolor='red', alpha=0.8))

        return projectile, trace, status_text, ring1, ring2

    # Crear y ejecutar la animación
    ani = animation.FuncAnimation(fig, animate, frames=len(t_points),
                                  init_func=init_animation, blit=True, interval=20, repeat=False)
    fig.canvas.draw_idle()


# --- 5. Creación de los Widgets de Matplotlib ---
# Ejes para los sliders y botones
ax_v0 = plt.axes([0.25, 0.15, 0.5, 0.03], facecolor=BACKGROUND_COLOR)
ax_theta = plt.axes([0.25, 0.1, 0.5, 0.03], facecolor=BACKGROUND_COLOR)
ax_launch = plt.axes([0.8, 0.025, 0.1, 0.05])

# Sliders
v0_slider = Slider(ax=ax_v0, label='Velocidad (m/s)', valmin=10, valmax=60, valinit=35, valstep=0.5, color=LIGHT_COLOR)
theta_slider = Slider(ax=ax_theta, label='Ángulo (°)', valmin=1, valmax=89.5, valinit=30, valstep=0.5, color=LIGHT_COLOR)

# Botón
launch_button = Button(ax_launch, 'LANZAR', color='gray', hovercolor='limegreen')

# --- 6. Conectar los Widgets a las Funciones ---
theta_slider.on_changed(update_aim)
launch_button.on_clicked(launch)

# --- 7. Estado Inicial y Mostrar Ventana ---
update_aim(None) # Poner el cañón en la posición inicial
plt.show()