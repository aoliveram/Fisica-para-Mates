import plotly.graph_objects as go
import numpy as np

# --- Constantes y Parámetros Iniciales ---
g = 9.81  # Aceleración de la gravedad (m/s^2)
H_cone_visualization = 8.0  # Altura máxima del cono para la visualización (m)

# Valores iniciales para los sliders
initial_h = 5.0  # Altura del bloque desde el vértice (m)
initial_theta_deg = 30.0  # Semiángulo del cono (grados)

# --- Funciones Auxiliares ---

def get_cone_geometry(theta_rad, H_max_vis):
    """Genera las coordenadas X, Y, Z para la superficie de un cono."""
    # El cono tiene el vértice en (0,0,0) y se abre hacia arriba (+Z).
    # El radio en una coordenada z_c es |z_c| * tan(theta_rad).
    z_coords = np.linspace(0, H_max_vis, 50)
    phi_angles = np.linspace(0, 2 * np.pi, 50)
    
    Z_grid, PHI_grid = np.meshgrid(z_coords, phi_angles)
    R_grid = np.abs(Z_grid) * np.tan(theta_rad) # abs(Z_grid) es la altura desde el vértice
    
    X_grid = R_grid * np.cos(PHI_grid)
    Y_grid = R_grid * np.sin(PHI_grid)
    
    return X_grid, Y_grid, Z_grid

def calculate_physics(h_val, theta_rad_val, g_val):
    """Calcula el radio y la velocidad basados en h, theta y g."""
    if np.cos(theta_rad_val) < 1e-6 or h_val <=0: # Evitar tan(90) o h=0
        return 0, 0
    r_val = h_val * np.tan(theta_rad_val)
    # v = tan(theta) * sqrt(g*h)
    v_val = np.tan(theta_rad_val) * np.sqrt(g_val * h_val)
    return r_val, v_val

# --- Creación de la Figura Inicial ---
fig = go.Figure()

# Convertir theta inicial a radianes
theta_rad_current = np.deg2rad(initial_theta_deg)
h_current = initial_h

# Calcular r y v iniciales
r_current, v_current = calculate_physics(h_current, theta_rad_current, g)

# 1. Superficie del Cono (Trace 0)
X_cone, Y_cone, Z_cone = get_cone_geometry(theta_rad_current, H_cone_visualization)
fig.add_trace(go.Surface(
    x=X_cone, y=Y_cone, z=Z_cone,
    opacity=0.4,
    colorscale='sunsetdark',
    showscale=False,
    name="Cono",
    hoverinfo='skip'
))

# 2. Trayectoria Circular del Bloque (Trace 1)
path_angles = np.linspace(0, 2 * np.pi, 100)
x_path = r_current * np.cos(path_angles)
y_path = r_current * np.sin(path_angles)
z_path = np.full_like(x_path, h_current)
fig.add_trace(go.Scatter3d(
    x=x_path, y=y_path, z=z_path,
    mode='lines',
    line=dict(color='darkorange', width=6), # Naranja más oscuro
    name="Trayectoria"
))

# 3. Bloque (Trace 2)
# Colocamos el bloque en (r, 0, h)
fig.add_trace(go.Scatter3d(
    x=[r_current], y=[0], z=[h_current],
    mode='markers',
    marker=dict(size=10, color='saddlebrown', symbol='square'), # Marrón, cuadrado
    name="Bloque"
))

# 4. Línea de Altura 'h' (Trace 3)
fig.add_trace(go.Scatter3d(
    x=[0, 0], y=[0, 0], z=[0, h_current],
    mode='lines',
    line=dict(color='blue', width=4, dash='dash'),
    name="h"
))

# 5. Línea de Radio 'r' (Trace 4)
fig.add_trace(go.Scatter3d(
    x=[0, r_current], y=[0, 0], z=[h_current, h_current],
    mode='lines',
    line=dict(color='green', width=4, dash='dash'),
    name="r"
))

# 6. Generatriz para indicar el ángulo theta (Trace 5)
# Desde el vértice (0,0,0) hasta el bloque (r_current, 0, h_current)
fig.add_trace(go.Scatter3d(
    x=[0, r_current], y=[0, 0], z=[0, h_current],
    mode='lines',
    line=dict(color='purple', width=3),
    name="Generatriz"
))

# 7. Arco para el ángulo theta (Trace 6)
# En el plano XZ (y=0), desde el eje Z hacia +X
arc_display_radius = min(h_current / 2.5, H_cone_visualization / 8) if h_current > 0 else 0.1
theta_arc_line = np.linspace(0, theta_rad_current, 20)
x_theta_arc = arc_display_radius * np.sin(theta_arc_line)
z_theta_arc = arc_display_radius * np.cos(theta_arc_line)
fig.add_trace(go.Scatter3d(
    x=x_theta_arc, y=np.zeros_like(x_theta_arc), z=z_theta_arc,
    mode='lines',
    line=dict(color='red', width=3),
    name="Ángulo θ"
))


# --- Configuración del Layout y Sliders ---
h_slider_values = np.linspace(0.5, H_cone_visualization * 0.8, 20) # Evitar h muy cerca del borde visual
theta_slider_values_deg = np.linspace(5, 85, 17) # Evitar 0 y 90 grados

sliders_list = [
    dict(
        active=np.argmin(np.abs(h_slider_values - initial_h)),
        currentvalue={"prefix": "Altura h (m): ", "suffix": " m", "font": {"size": 16}},
        pad={"t": 50, "b":10}, # Espacio arriba para el texto, abajo para el siguiente slider
        x=0.05, xanchor="left",
        y=0.1, yanchor="top",
        len=0.4,
        steps=[]
    ),
    dict(
        active=np.argmin(np.abs(theta_slider_values_deg - initial_theta_deg)),
        currentvalue={"prefix": "Ángulo θ (grados): ", "suffix": "°", "font": {"size": 16}},
        pad={"t": 50, "b":10},
        x=0.55, xanchor="left",
        y=0.1, yanchor="top",
        len=0.4,
        steps=[]
    )
]

# --- Llenar los pasos de los sliders ---

# Slider para h (el ángulo theta se mantiene fijo al valor inicial del otro slider)
for h_step_val in h_slider_values:
    # Para el slider de h, theta_deg_for_calc se toma del valor inicial del slider de theta
    theta_deg_for_calc = initial_theta_deg # O podría ser theta_slider_values_deg[sliders_list[1]['active']]
    theta_rad_for_calc = np.deg2rad(theta_deg_for_calc)
    
    r_step, v_step = calculate_physics(h_step_val, theta_rad_for_calc, g)
    
    x_path_step = r_step * np.cos(path_angles)
    y_path_step = r_step * np.sin(path_angles)
    z_path_step = np.full_like(x_path_step, h_step_val)
    
    arc_display_radius_step = min(h_step_val / 2.5, H_cone_visualization / 8) if h_step_val > 0 else 0.1
    theta_arc_line_step = np.linspace(0, theta_rad_for_calc, 20) # theta_rad_for_calc no cambia con h_step_val
    x_theta_arc_step = arc_display_radius_step * np.sin(theta_arc_line_step)
    z_theta_arc_step = -arc_display_radius_step * np.cos(theta_arc_line_step)

    annotations_step = [
        dict(text=f"h = {h_step_val:.2f} m", x=0.05, y=0, z=h_step_val/2, showarrow=False, xanchor="left", yanchor="middle", font=dict(color="blue", size=14)),
        dict(text=f"r = {r_step:.2f} m", x=r_step/2, y=0.05, z=h_step_val, showarrow=False, xanchor="center", yanchor="bottom", font=dict(color="green", size=14)),
        dict(text=f"θ = {theta_deg_for_calc:.1f}°", x=arc_display_radius_step * np.sin(theta_rad_for_calc/1.5) * 1.1, y=0, z=arc_display_radius_step * np.cos(theta_rad_for_calc/1.5) * 1.1,
             showarrow=False, xanchor="center", font=dict(color="red", size=14)),
        dict(text=f"<b>Velocidad Requerida: {v_step:.2f} m/s</b>", x=0, y=0, z=0.5, # Posición para el texto de velocidad
             showarrow=False, font=dict(size=16, color="black"), bgcolor="rgba(255,255,255,0.7)")
    ]
    
    sliders_list[0]['steps'].append(dict(
        method="update",
        args=[
            { # Actualizaciones de datos de traces (índices 1 a 6, el cono (0) no cambia con h)
                'x': [None, x_path_step, [r_step], [0,0], [0,r_step], [0,r_step], x_theta_arc_step],
                'y': [None, y_path_step, [0],      [0,0], [0,0],      [0,0],      np.zeros_like(x_theta_arc_step)],
                'z': [None, z_path_step, [h_step_val], [0,h_step_val], [h_step_val,h_step_val], [0,h_step_val], z_theta_arc_step]
            },
            { # Actualizaciones de layout
                "title": f"Bloque en Cono: h={h_step_val:.2f}m, θ={theta_deg_for_calc:.1f}°, v={v_step:.2f}m/s",
                "scene.annotations": annotations_step
            }
        ],
        label=f"{h_step_val:.2f}"
    ))

# Slider para theta (la altura h se mantiene fija al valor inicial del otro slider)
for theta_deg_step_val in theta_slider_values_deg:
    theta_rad_step = np.deg2rad(theta_deg_step_val)
    # Para el slider de theta, h_for_calc se toma del valor inicial del slider de h
    h_for_calc = initial_h # O podría ser h_slider_values[sliders_list[0]['active']]
    
    r_step, v_step = calculate_physics(h_for_calc, theta_rad_step, g)
    
    X_cone_step, Y_cone_step, Z_cone_step = get_cone_geometry(theta_rad_step, H_cone_visualization)
    
    x_path_step = r_step * np.cos(path_angles)
    y_path_step = r_step * np.sin(path_angles)
    z_path_step = np.full_like(x_path_step, h_for_calc)
    
    arc_display_radius_step = min(h_for_calc / 2.5, H_cone_visualization / 8) if h_for_calc > 0 else 0.1
    theta_arc_line_step = np.linspace(0, theta_rad_step, 20)
    x_theta_arc_step = arc_display_radius_step * np.sin(theta_arc_line_step)
    z_theta_arc_step = arc_display_radius_step * np.cos(theta_arc_line_step)

    annotations_step = [
        dict(text=f"h = {h_for_calc:.2f} m", x=0.05, y=0, z=h_for_calc*2/3, showarrow=False, xanchor="left", yanchor="middle", font=dict(color="blue", size=14)),
        dict(text=f"r = {r_step:.2f} m", x=r_step/2, y=0.05, z=h_for_calc, showarrow=False, xanchor="center", yanchor="bottom", font=dict(color="green", size=14)),
        dict(text=f"θ = {theta_deg_step_val:.1f}°", x=arc_display_radius_step * np.sin(theta_rad_step/1.5) * 1.1, y=0, z=arc_display_radius_step * np.cos(theta_rad_step/1.5) * 1.25,
             showarrow=False, xanchor="center", font=dict(color="red", size=14)),
        dict(text=f"<b>Velocidad Requerida: {v_step:.2f} m/s</b>", x=0, y=0, z=0.5,
             showarrow=False, font=dict(size=16, color="black"), bgcolor="rgba(255,255,255,0.7)")
    ]
    
    sliders_list[1]['steps'].append(dict(
        method="update",
        args=[
            { # Actualizaciones de datos de traces (todos los traces pueden cambiar con theta)
                'x': [X_cone_step, x_path_step, [r_step], [0,0], [0,r_step], [0,r_step], x_theta_arc_step],
                'y': [Y_cone_step, y_path_step, [0],      [0,0], [0,0],      [0,0],      np.zeros_like(x_theta_arc_step)],
                'z': [Z_cone_step, z_path_step, [h_for_calc], [0,h_for_calc], [h_for_calc,h_for_calc], [0,h_for_calc], z_theta_arc_step]
            },
            { # Actualizaciones de layout
                "title": f"Bloque en Cono: h={h_for_calc:.2f}m, θ={theta_deg_step_val:.1f}°, v={v_step:.2f}m/s",
                "scene.annotations": annotations_step
            }
        ],
        label=f"{theta_deg_step_val:.1f}"
    ))

# Aplicar configuración inicial de título y anotaciones (para el primer frame antes de mover sliders)
initial_annotations = [
    dict(text=f"h = {h_current:.2f} m", x=0.05, y=0, z=h_current/2, showarrow=False, xanchor="left", yanchor="middle", font=dict(color="blue", size=14)),
    dict(text=f"r = {r_current:.2f} m", x=r_current/2, y=0.05, z=h_current, showarrow=False, xanchor="center", yanchor="bottom", font=dict(color="green", size=14)),
    dict(text=f"θ = {initial_theta_deg:.1f}°", x=arc_display_radius * np.sin(theta_rad_current/1.5) * 1.1, y=0, z=arc_display_radius * np.cos(theta_rad_current/1.5) * 1.1,
         showarrow=False, xanchor="center", font=dict(color="red", size=14)),
    dict(text=f"<b>Velocidad Requerida: {v_current:.2f} m/s</b>", x=0, y=0, z=8,
         showarrow=False, font=dict(size=16, color="black"), bgcolor="rgba(255,255,255,0.7)")
]

fig.update_layout(
    title=f"Bloque en Cono: h={h_current:.2f}m, θ={initial_theta_deg:.1f}°, v={v_current:.2f}m/s",
    scene=dict(
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        zaxis_title='Z (m)',
        aspectmode='data', # Mantiene proporciones relativas; 'cube' fuerza un cubo
        # Para que se parezca más a la imagen, ajustar zaxis.range o aspectratio
        # zaxis=dict(range=[-(H_cone_visualization*1.1), 0.5]), # Asegurar que el vértice y texto V se vean
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=0.8) # Posición de la cámara
        ),
        annotations=initial_annotations
    ),
    margin=dict(l=10, r=10, b=100, t=50), # Más espacio abajo para sliders
    sliders=sliders_list,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

# --- Mostrar Figura ---
fig.show()