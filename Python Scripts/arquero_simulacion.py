import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Slider, Button # Importar los widgets de Matplotlib

# --- 1. Definición de Constantes y Parámetros del Escenario ---
g = 9.81  # Aceleración de la gravedad (m/s^2)

# Propiedades del primer anillo
x1, y1 = 30.0, 10.0  # Coordenadas del centro (m)
R1 = 2.0             # Radio (m)

# Propiedades del segundo anillo
x2, y2 = 60.0, 8.0   # Coordenadas del centro (m)
R2 = 2.0             # Radio (m)

# --- 2. Creación de la Figura y los Ejes del Gráfico ---
# Ajustamos el layout de la figura para dejar espacio para los sliders en la parte inferior
fig, ax = plt.subplots(figsize=(10, 7))
plt.subplots_adjust(left=0.1, bottom=0.3) # Dejar 30% del espacio inferior para widgets

# --- 3. Función Principal de Simulación y Ploteo ---
# Esta función ahora será llamada por la función de actualización de los sliders
def draw_plot(v0, theta_deg):
    """
    Función de dibujo. No necesita limpiar el gráfico, eso lo hará la función 'update'.
    """
    ax.clear()

    theta_rad = np.radians(theta_deg)
    range_estimate = (v0**2 * np.sin(2 * theta_rad)) / g
    x_max = max(x2 + 20, range_estimate * 1.2)
    x_points = np.linspace(0, x_max, 500)
    
    cos_theta = np.cos(theta_rad)
    tan_theta = np.tan(theta_rad)
    y_points = x_points * tan_theta - (g * x_points**2) / (2 * v0**2 * cos_theta**2 + 1e-6)

    y_at_x1 = x1 * tan_theta - (g * x1**2) / (2 * v0**2 * cos_theta**2 + 1e-6)
    y_at_x2 = x2 * tan_theta - (g * x2**2) / (2 * v0**2 * cos_theta**2 + 1e-6)

    hit1 = abs(y_at_x1 - y1) < R1
    hit2 = abs(y_at_x2 - y2) < R2

    ax.plot(x_points, y_points, 'b-', label='Trayectoria')

    color1 = 'green' if hit1 else 'red'
    color2 = 'green' if hit2 else 'red'
    
    if hit1 and hit2:
        status_text = "¡ÉXITO! Pasó por ambos anillos."
        status_color = 'green'
    elif hit1:
        status_text = "FALLO: Pasó por el primero, pero no por el segundo."
        status_color = 'orange'
    else:
        status_text = "FALLO: No pasó por ningún anillo."
        status_color = 'red'

    ring1 = Circle((x1, y1), R1, facecolor='none', edgecolor=color1, linewidth=2, linestyle='--')
    ring2 = Circle((x2, y2), R2, facecolor='none', edgecolor=color2, linewidth=2, linestyle='--')
    ax.add_patch(ring1)
    ax.add_patch(ring2)
    
    # Redibujamos la leyenda cada vez
    ax.legend(['Trayectoria', 'Anillo 1', 'Anillo 2'])


    ax.set_title("El Desafío del Arquero", fontsize=16)
    ax.set_xlabel("Distancia Horizontal (m)", fontsize=12)
    ax.set_ylabel("Altura Vertical (m)", fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlim(0, x_max)
    # Evitar que el límite Y se vuelva negativo
    ymax = max(y_points.max() * 1.2 if len(y_points) > 0 else 0, y1 + R1 * 2, y2 + R2 * 2)
    ax.set_ylim(0, ymax)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    
    ax.text(0.5, 1.02, status_text, ha='center', va='bottom', transform=ax.transAxes, 
            fontsize=12, color='white', bbox=dict(facecolor=status_color, alpha=0.8))

    # Forzar el redibujado del canvas
    fig.canvas.draw_idle()


# --- 4. Creación de los ejes para los sliders ---
ax_v0 = plt.axes([0.15, 0.15, 0.65, 0.03]) # [izquierda, abajo, ancho, alto]
ax_theta = plt.axes([0.15, 0.1, 0.65, 0.03])

# --- 5. Creación de los Sliders de Matplotlib ---
v0_slider = Slider(
    ax=ax_v0,
    label='Velocidad (m/s)',
    valmin=10,
    valmax=60,
    valinit=35,
    valstep=0.5
)

theta_slider = Slider(
    ax=ax_theta,
    label='Ángulo (°)',
    valmin=1,
    valmax=89.5,
    valinit=30,
    valstep=0.5
)

# --- 6. Función de Actualización ---
# Esta función se conectará al evento 'on_changed' de los sliders
def update(val):
    # Obtiene los valores actuales de los sliders
    v0 = v0_slider.val
    theta_deg = theta_slider.val
    # Llama a la función de dibujo con los nuevos valores
    draw_plot(v0, theta_deg)

# --- 7. Conectar los sliders a la función de actualización ---
v0_slider.on_changed(update)
theta_slider.on_changed(update)

# --- Bonus: Botón de Reset ---
ax_reset = plt.axes([0.8, 0.025, 0.1, 0.04])
button_reset = Button(ax_reset, 'Reset', hovercolor='0.975')

def reset(event):
    v0_slider.reset()
    theta_slider.reset()

button_reset.on_clicked(reset)


# --- 8. Dibujo Inicial y Mostrar la Ventana ---
# Dibuja el estado inicial antes de mostrar la ventana
draw_plot(v0_slider.valinit, theta_slider.valinit)

# Muestra la ventana de Matplotlib. El script se pausará aquí hasta que la ventana se cierre.
plt.show()