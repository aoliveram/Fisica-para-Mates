# --- Bibliotecas Necesarias ---
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import ipywidgets as widgets
from ipywidgets import interactive, VBox, HBox

# --- Habilitar el backend interactivo de Matplotlib en Jupyter ---
# Descomenta la siguiente línea si estás en un Jupyter Notebook clásico
# %matplotlib notebook 
# O usa esta para JupyterLab
# %matplotlib widget

# --- 1. Definición de Constantes y Parámetros del Escenario ---
g = 9.81  # Aceleración de la gravedad (m/s^2)

# Propiedades del primer anillo
x1, y1 = 30.0, 10.0  # Coordenadas del centro (m)
R1 = 2.0             # Radio (m)

# Propiedades del segundo anillo
x2, y2 = 60.0, 8.0   # Coordenadas del centro (m)
R2 = 2.0             # Radio (m)

# --- 2. Creación de la Figura y los Ejes del Gráfico ---
# Se crea una sola vez para que la actualización sea más rápida
fig, ax = plt.subplots(figsize=(10, 6))

# --- 3. Función Principal de Simulación y Ploteo ---
# Esta función se llamará cada vez que un slider se mueva
def plot_trajectory(v0, theta_deg):
    """
    Calcula y dibuja la trayectoria del proyectil para una v0 y theta dadas,
    y verifica si pasa a través de los anillos.
    """
    # Limpiar el gráfico anterior para redibujar
    ax.clear()

    # --- Cálculos Físicos ---
    # Convertir ángulo de grados a radianes para los cálculos de numpy
    theta_rad = np.radians(theta_deg)

    # Calcular la distancia máxima (alcance) para ajustar el eje X
    # Usamos la fórmula del alcance en suelo plano como una buena estimación
    range_estimate = (v0**2 * np.sin(2 * theta_rad)) / g
    x_max = max(x2 + 20, range_estimate * 1.2) # Asegurarse de que el gráfico llegue más allá del último anillo
    
    # Crear un array de puntos x para dibujar la trayectoria
    x_points = np.linspace(0, x_max, 500)
    
    # Ecuación de la Trayectoria (y(x))
    # Se añade una pequeña constante (1e-6) al denominador para evitar división por cero si theta=90
    cos_theta = np.cos(theta_rad)
    tan_theta = np.tan(theta_rad)
    y_points = x_points * tan_theta - (g * x_points**2) / (2 * v0**2 * cos_theta**2 + 1e-6)

    # --- Verificación de Paso por los Anillos ---
    # Calcular la altura 'y' de la flecha en la posición 'x' de cada anillo
    y_at_x1 = x1 * tan_theta - (g * x1**2) / (2 * v0**2 * cos_theta**2 + 1e-6)
    y_at_x2 = x2 * tan_theta - (g * x2**2) / (2 * v0**2 * cos_theta**2 + 1e-6)

    # Comprobar si la distancia vertical al centro es menor que el radio
    hit1 = abs(y_at_x1 - y1) < R1
    hit2 = abs(y_at_x2 - y2) < R2

    # --- Dibujo y Estilo del Gráfico ---
    # Dibujar la trayectoria de la flecha
    ax.plot(x_points, y_points, 'b-', label='Trayectoria')

    # Determinar colores y mensaje de estado
    color1 = 'green' if hit1 else 'red'
    color2 = 'green' if hit2 else 'red'
    
    if hit1 and hit2:
        status_text = "¡ÉXITO! Pasó por ambos anillos."
        status_color = 'green'
    elif hit1:
        status_text = "FALLO: Pasó por el primero, pero no por el segundo."
        status_color = 'orange'
    elif hit2:
        status_text = "FALLO: No pasó por el primero, pero sí por el segundo."
        status_color = 'orange'
    else:
        status_text = "FALLO: No pasó por ningún anillo."
        status_color = 'red'

    # Dibujar los anillos
    ring1 = Circle((x1, y1), R1, facecolor='none', edgecolor=color1, linewidth=2, linestyle='--', label='Anillo 1')
    ring2 = Circle((x2, y2), R2, facecolor='none', edgecolor=color2, linewidth=2, linestyle='--', label='Anillo 2')
    ax.add_patch(ring1)
    ax.add_patch(ring2)

    # Configuraciones estéticas del gráfico
    ax.set_title("El Desafío del Arquero", fontsize=16)
    ax.set_xlabel("Distancia Horizontal (m)", fontsize=12)
    ax.set_ylabel("Altura Vertical (m)", fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, max(y_points.max() * 1.2, y1 + R1 * 2, y2 + R2 * 2))
    ax.axhline(0, color='black', linewidth=0.5) # Línea del suelo
    ax.set_aspect('equal', adjustable='box') # Para que los círculos no se vean como elipses
    ax.legend()
    
    # Mostrar el texto de estado en el gráfico
    ax.text(0.5, 0.9, status_text, ha='center', va='center', transform=ax.transAxes, 
            fontsize=14, color='white', bbox=dict(facecolor=status_color, alpha=0.8))

# --- 4. Creación y Configuración de los Widgets Interactivos ---
v0_slider = widgets.FloatSlider(
    value=35,           # Valor inicial
    min=10,             # Valor mínimo
    max=60,             # Valor máximo
    step=0.5,           # Incremento
    description='Velocidad (m/s):',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='500px')
)

theta_slider = widgets.FloatSlider(
    value=30,           # Valor inicial
    min=1,              # Valor mínimo (evitar 0)
    max=89.5,           # Valor máximo (evitar 90 grados para prevenir división por cero)
    step=0.5,           # Incremento
    description='Ángulo (°):',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='500px')
)

# --- 5. Vincular los Sliders a la Función de Ploteo ---
# 'interactive' crea la interfaz de usuario y llama a la función cuando se cambia un slider
interactive_plot = interactive(plot_trajectory, v0=v0_slider, theta_deg=theta_slider)

# --- 6. Mostrar la Simulación ---
# Organizar los controles y el gráfico en la pantalla
controls = VBox([v0_slider, theta_slider])
display(VBox([controls, fig.canvas]))