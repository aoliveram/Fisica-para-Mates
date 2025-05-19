import plotly.graph_objects as go
import numpy as np

# --- Constantes y Parámetros Iniciales ---
g = 9.81  # Aceleración de la gravedad (m/s^2)
# H_visualization no es tan relevante para la esfera, R_curv define la altura máxima.

# Valores iniciales para los sliders
initial_R_curv = 5.0  # Radio de curvatura del tazón esférico (m)
# La altura inicial h debe ser menor que R_curv
initial_h_sphere = initial_R_curv / 2.0  # Altura del bloque desde el vértice (m)


# --- Funciones Auxiliares para el TAZÓN ESFÉRICO ---

def get_spherical_bowl_geometry(R_curv_val, vis_height_limit):
    """Genera las coordenadas X, Y, Z para la superficie de un tazón esférico."""
    # El tazón tiene el vértice en (0,0,0) y se abre hacia arriba (+Z).
    # La altura máxima del tazón es R_curv_val.
    # vis_height_limit es la altura máxima de la malla z que generaremos.
    
    # Generar z_coords hasta R_curv_val o el límite de visualización si es menor.
    actual_bowl_height = min(R_curv_val, vis_height_limit)
    if actual_bowl_height <= 0: # Evitar errores si R_curv es 0 o negativo
        return np.array([]), np.array([]), np.array([])

    z_coords = np.linspace(0, actual_bowl_height, 70) # Más puntos para la curvatura
    phi_angles = np.linspace(0, 2 * np.pi, 70)
    
    Z_grid, PHI_grid = np.meshgrid(z_coords, phi_angles)
    
    # r^2 = 2*h*R_curv - h^2. Aquí h es Z_grid.
    r_squared_values = 2 * Z_grid * R_curv_val - Z_grid**2
    # Asegurar que r_squared no sea negativo debido a errores de flotante si Z_grid excede R_curv_val
    r_squared_values[r_squared_values < 1e-9] = 0 
    R_grid = np.sqrt(r_squared_values)
    
    X_grid = R_grid * np.cos(PHI_grid)
    Y_grid = R_grid * np.sin(PHI_grid)
    
    return X_grid, Y_grid, Z_grid

def calculate_spherical_bowl_physics(h_val, R_curv_val, g_val):
    """Calcula r, v, y phi para el tazón esférico."""
    if R_curv_val <= 0 or h_val < 0:
        return 0, 0, 0, 0 # r, v, tan_phi, phi_rad

    # Asegurar que h_val no exceda R_curv_val para evitar sqrt de negativos o división por cero.
    h_clipped = min(h_val, R_curv_val * 0.9999) # Clip muy cerca del borde superior
    if h_clipped < 1e-6 : h_clipped = 1e-6 # Evitar h=0 exacto para tan_phi si R_curv > 0

    r_sq_val = 2 * h_clipped * R_curv_val - h_clipped**2
    r_val = np.sqrt(r_sq_val) if r_sq_val > 0 else 0
    
    v_val = np.sqrt(g_val * (R_curv_val - h_clipped)) if g_val * (R_curv_val - h_clipped) >= 0 else 0
    
    # tan(phi) = (R_curv - h) / r
    if r_val > 1e-5: # Evitar división por cero si r es muy pequeño
        tan_phi_val = (R_curv_val - h_clipped) / r_val
        phi_rad_val = np.arctan(tan_phi_val)
    elif h_clipped < R_curv_val * 0.01 : # Cerca del vértice, r es pequeño, normal casi vertical
        tan_phi_val = float('inf') # dr/dh es infinito
        phi_rad_val = 0 # Ángulo de la normal con la vertical es 0
    else: # h_clipped está cerca de R_curv, r es pequeño, normal casi horizontal
        tan_phi_val = 1e-5 # Casi cero
        phi_rad_val = np.pi/2 # Ángulo de la normal con la vertical es pi/2

    # El ángulo phi en el .tex es tal que tan(phi) = |dr/dh|
    # dr/dh = (R-h)/r. Entonces el phi del .tex es el mismo que calculamos aquí.
    
    return r_val, v_val, tan_phi_val, phi_rad_val


# --- Creación de la Figura Inicial ---
fig = go.Figure()

# Valores iniciales actuales
h_current = initial_h_sphere
R_curv_current = initial_R_curv

# Calcular r, v, y phi iniciales
r_current, v_current, tan_phi_current, phi_rad_current = calculate_spherical_bowl_physics(h_current, R_curv_current, g)

# H_visualization: altura máxima de la escena. Para la esfera, debería ser al menos R_curv_current.
H_visualization = max(R_curv_current * 1.1, 5.0) # Asegurar un mínimo de visualización


# 1. Superficie del Tazón Esférico (Trace 0)
X_bowl, Y_bowl, Z_bowl = get_spherical_bowl_geometry(R_curv_current, R_curv_current) # Visualizar hasta R_curv
fig.add_trace(go.Surface(
    x=X_bowl, y=Y_bowl, z=Z_bowl,
    opacity=0.5,
    colorscale='Blues', # Un colorscale diferente
    showscale=False,
    name="Tazón Esférico",
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
    line=dict(color='darkorange', width=6),
    name="Trayectoria"
))

# 3. Bloque (Trace 2) - Usaremos esfera dorada como antes
fig.add_trace(go.Scatter3d(
    x=[r_current], y=[0], z=[h_current],
    mode='markers',
    marker=dict(size=10, color='gold', symbol='circle'), # Esfera dorada
    name="Bloque"
))

# 4. Línea de Altura 'h' (Trace 3)
fig.add_trace(go.Scatter3d(
    x=[0, 0], y=[0, 0], z=[0, h_current],
    mode='lines',
    line=dict(color='cyan', width=4, dash='dash'),
    name="h"
))

# 5. Línea de Radio 'r' (Trace 4)
fig.add_trace(go.Scatter3d(
    x=[0, r_current], y=[0, 0], z=[h_current, h_current],
    mode='lines',
    line=dict(color='lime', width=4, dash='dash'),
    name="r"
))

# 6. Línea Normal (desde el centro de curvatura al bloque) (Trace 5)
normal_line_x = [0, r_current]
normal_line_y = [0, 0]
normal_line_z = [R_curv_current, h_current]
fig.add_trace(go.Scatter3d(
    x=normal_line_x, y=normal_line_y, z=normal_line_z,
    mode='lines',
    line=dict(color='magenta', width=3, dash='dot'), # Magenta punteado para la normal
    name="Normal (Radio Esfera)"
))

# 7. Arco para el ángulo phi (Trace 6)
# phi es el ángulo entre la normal y la vertical.
# La vertical pasa por el bloque en (r_current, 0, h_current) y va hacia (r_current, 0, h_current + longitud_vertical)
# La normal va desde (r_current, 0, h_current) hacia el centro de curvatura (0,0,R_curv_current)
arc_display_radius = min(r_current * 0.5, (R_curv_current - h_current) * 0.5) if r_current > 0 and R_curv_current > h_current else 0.1
arc_display_radius = max(arc_display_radius, 0.1) # Mínimo radio

# El arco estará centrado en el bloque (r_current, 0, h_current)
# El ángulo phi es con la vertical.
# Si la normal apunta hacia abajo y a la izquierda (desde el bloque), el ángulo se mide desde la vertical hacia arriba.
# Vector vertical desde el bloque: (0,0,1)
# Vector normal desde el bloque hacia el centro de curvatura: (-r_current, 0, R_curv_current - h_current)
# El ángulo phi_rad_current ya es el ángulo correcto.
# El arco irá desde la línea vertical (hacia arriba desde el bloque) hacia la línea normal.
# Ángulo inicial (vertical hacia arriba): pi/2 respecto al plano XY
# Ángulo final: pi/2 - phi_rad_current (si la normal está "a la izquierda" de la vertical)
# O pi/2 + phi_rad_current (si la normal está "a la derecha")

# Vector del bloque al centro de curvatura
vec_to_center_x = 0 - r_current
vec_to_center_z = R_curv_current - h_current

# Ángulo que forma este vector con el eje +X (horizontal en el plano XZ del bloque)
# Si usamos el punto (r,0,h) como origen para el arco
# La vertical es (0, L) -> ángulo pi/2
# La normal tiene componentes (-r, R-h) relativas al bloque para ir al centro.
# Pero phi es el ángulo con la vertical.
# Es más fácil dibujar el arco alrededor del punto de intersección de la normal y la vertical si lo tuviéramos,
# o usar el bloque como centro del arco.

# Dibujando el arco desde el bloque:
# Vertical hacia arriba desde el bloque: (r_current, 0, h_current + L_arco)
# Normal desde el bloque: (r_current + L_arco*sin(phi), 0, h_current + L_arco*cos(phi)) - si phi=0 es vertical
# Nuestro phi_rad_current es el ángulo con la vertical.
# El arco comienza en la vertical (ángulo 0 para el arco) y va hasta phi_rad_current.
phi_arc_line = np.linspace(0, phi_rad_current, 20)
# Queremos que el arco esté en el plano XZ que pasa por el bloque.
# Eje X del arco: r_current - arc_display_radius * np.sin(phi_arc_line) (va hacia la izquierda)
# Eje Z del arco: h_current + arc_display_radius * np.cos(phi_arc_line) (va hacia arriba)
x_phi_arc = r_current - arc_display_radius * np.sin(phi_arc_line)
z_phi_arc = h_current + arc_display_radius * np.cos(phi_arc_line)


fig.add_trace(go.Scatter3d(
    x=x_phi_arc, y=np.zeros_like(x_phi_arc), z=z_phi_arc,
    mode='lines',
    line=dict(color='red', width=3),
    name="Ángulo φ"
))


# --- Configuración del Layout y Sliders ---
# Slider para h: de 0.01*R_curv_current a 0.98*R_curv_current
# Necesitamos que los valores del slider h dependan del R_curv_current del otro slider.
# Esto es la parte compleja de la interdependencia de sliders en Plotly sin Dash.
# Por ahora, definiremos los rangos basados en los valores iniciales.

h_slider_min = 0.01 * initial_R_curv # Un pequeño valor > 0
h_slider_max = 0.98 * initial_R_curv # Cerca del borde
h_slider_values = np.linspace(h_slider_min, h_slider_max, 15)

R_curv_slider_values = np.linspace(1.0, 10.0, 10) # Rango para R_curv

sliders_list = [
    dict(
        active=np.argmin(np.abs(h_slider_values - initial_h_sphere)),
        currentvalue={"prefix": "Altura h (m): ", "suffix": " m", "font": {"size": 16}},
        pad={"t": 50, "b":10},
        x=0.05, xanchor="left",
        y=0.1, yanchor="top",
        len=0.4,
        steps=[]
    ),
    dict(
        active=np.argmin(np.abs(R_curv_slider_values - initial_R_curv)),
        currentvalue={"prefix": "Radio Tazón R (m): ", "suffix": " m", "font": {"size": 16}},
        pad={"t": 50, "b":10},
        x=0.55, xanchor="left",
        y=0.1, yanchor="top",
        len=0.4,
        steps=[]
    )
]

# --- Llenar los pasos de los sliders ---

# Slider para h (R_curv se mantiene fijo al valor inicial del otro slider)
for h_step_val in h_slider_values:
    R_curv_for_calc = initial_R_curv # Usar R_curv inicial para este slider
    
    r_step, v_step, tan_phi_step, phi_rad_step = calculate_spherical_bowl_physics(h_step_val, R_curv_for_calc, g)
    
    x_path_step = r_step * np.cos(path_angles)
    y_path_step = r_step * np.sin(path_angles)
    z_path_step = np.full_like(x_path_step, h_step_val)
    
    normal_line_x_step = [0, r_step]
    normal_line_z_step = [R_curv_for_calc, h_step_val]

    arc_disp_rad_step = min(r_step * 0.5, (R_curv_for_calc - h_step_val) * 0.5) if r_step > 0 and R_curv_for_calc > h_step_val else 0.1
    arc_disp_rad_step = max(arc_disp_rad_step, 0.1)
    phi_arc_line_step_pts = np.linspace(0, phi_rad_step, 20)
    x_phi_arc_step = r_step - arc_disp_rad_step * np.sin(phi_arc_line_step_pts)
    z_phi_arc_step = h_step_val + arc_disp_rad_step * np.cos(phi_arc_line_step_pts)


    annotations_step = [
        dict(text=f"h = {h_step_val:.2f} m", x=0.05, y=0, z=h_step_val/2, showarrow=False, xanchor="left", yanchor="middle", font=dict(color="cyan", size=14)),
        dict(text=f"r = {r_step:.2f} m", x=r_step/2 if r_step >0 else 0.01, y=0.05, z=h_step_val, showarrow=False, xanchor="center", yanchor="bottom", font=dict(color="lime", size=14)),
        dict(text=f"R = {R_curv_for_calc:.1f}m", x=0.05, y=0, z=H_visualization*0.85, showarrow=False, xanchor="left", font=dict(color="gray", size=12)),
        dict(text=f"φ = {np.degrees(phi_rad_step):.1f}°",
             x=(r_step - arc_disp_rad_step * np.sin(phi_rad_step/2) * 1.2),
             y=0,
             z=(h_step_val + arc_disp_rad_step * np.cos(phi_rad_step/2) * 1.2),
             showarrow=False, xanchor="center", font=dict(color="red", size=14)),
        dict(text=f"<b>Velocidad: {v_step:.2f} m/s</b><br>tan(φ)={tan_phi_step:.2f}",
             x=0, y=0, z=H_visualization*0.95, # Posición para el texto de velocidad
             showarrow=False, font=dict(size=16, color="black"), bgcolor="rgba(255,255,255,0.7)", align="left")
    ]
    
    sliders_list[0]['steps'].append(dict(
        method="update",
        args=[
            { # Actualizaciones de datos de traces (Superficie (0) no cambia con h solo)
              # Path (1), Block (2), h_line (3), r_line (4), Normal (5), Phi_Arc (6)
                'x': [None, x_path_step, [r_step], [0,0], [0,r_step], normal_line_x_step, x_phi_arc_step],
                'y': [None, y_path_step, [0],      [0,0], [0,0],      [0,0],               np.zeros_like(x_phi_arc_step)],
                'z': [None, z_path_step, [h_step_val], [0,h_step_val], [h_step_val,h_step_val], normal_line_z_step, z_phi_arc_step]
            },
            { # Actualizaciones de layout
                "title.text": f"Tazón Esférico: h={h_step_val:.2f}m, R={R_curv_for_calc:.1f}m, v={v_step:.2f}m/s",
                "scene.annotations": annotations_step
            }
        ],
        label=f"{h_step_val:.2f}"
    ))

# Slider para R_curv (h se mantiene fija al valor inicial del otro slider)
for R_curv_step_val in R_curv_slider_values:
    h_for_calc = initial_h_sphere # Usar h inicial para este slider
    
    # El rango de h permitido depende de R_curv_step_val. Ajustar h_for_calc si es necesario.
    h_adjusted_for_R = min(h_for_calc, R_curv_step_val * 0.98)
    if h_adjusted_for_R < 0.01 * R_curv_step_val : h_adjusted_for_R = 0.01 * R_curv_step_val

    r_step, v_step, tan_phi_step, phi_rad_step = calculate_spherical_bowl_physics(h_adjusted_for_R, R_curv_step_val, g)
    
    X_bowl_step, Y_bowl_step, Z_bowl_step = get_spherical_bowl_geometry(R_curv_step_val, R_curv_step_val)
    
    x_path_step = r_step * np.cos(path_angles)
    y_path_step = r_step * np.sin(path_angles)
    z_path_step = np.full_like(x_path_step, h_adjusted_for_R)

    normal_line_x_step = [0, r_step]
    normal_line_z_step = [R_curv_step_val, h_adjusted_for_R]

    arc_disp_rad_step = min(r_step * 0.5, (R_curv_step_val - h_adjusted_for_R) * 0.5) if r_step > 0 and R_curv_step_val > h_adjusted_for_R else 0.1
    arc_disp_rad_step = max(arc_disp_rad_step, 0.1)
    phi_arc_line_step_pts = np.linspace(0, phi_rad_step, 20)
    x_phi_arc_step = r_step - arc_disp_rad_step * np.sin(phi_arc_line_step_pts)
    z_phi_arc_step = h_adjusted_for_R + arc_disp_rad_step * np.cos(phi_arc_line_step_pts)

    # Ajustar H_visualization para el título si R cambia mucho
    current_H_scene_vis = max(R_curv_step_val * 1.25, 5.0)

    annotations_step = [
        dict(text=f"h = {h_adjusted_for_R:.2f} m", x=0.05, y=0, z=h_adjusted_for_R/2, showarrow=False, xanchor="left", yanchor="middle", font=dict(color="cyan", size=14)),
        dict(text=f"r = {r_step:.2f} m", x=r_step/2 if r_step >0 else 0.01, y=0.05, z=h_adjusted_for_R, showarrow=False, xanchor="center", yanchor="bottom", font=dict(color="lime", size=14)),
        dict(text=f"R = {R_curv_step_val:.1f}m", x=0.05, y=0, z=current_H_scene_vis*0.85, showarrow=False, xanchor="left", font=dict(color="gray", size=12)),
        dict(text=f"φ = {np.degrees(phi_rad_step):.1f}°",
             x=(r_step - arc_disp_rad_step * np.sin(phi_rad_step/2) * 1.2),
             y=0,
             z=(h_adjusted_for_R + arc_disp_rad_step * np.cos(phi_rad_step/2) * 1.2),
             showarrow=False, xanchor="center", font=dict(color="red", size=14)),
        dict(text=f"<b>Velocidad: {v_step:.2f} m/s</b><br>tan(φ)={tan_phi_step:.2f}",
             x=0, y=0, z=current_H_scene_vis*0.95,
             showarrow=False, font=dict(size=16, color="black"), bgcolor="rgba(255,255,255,0.7)", align="left")
    ]
    
    sliders_list[1]['steps'].append(dict(
        method="update",
        args=[
            { # Actualizaciones de datos de traces (todos pueden cambiar con R_curv)
                'x': [X_bowl_step, x_path_step, [r_step], [0,0], [0,r_step], normal_line_x_step, x_phi_arc_step],
                'y': [Y_bowl_step, y_path_step, [0],      [0,0], [0,0],      [0,0],               np.zeros_like(x_phi_arc_step)],
                'z': [Z_bowl_step, z_path_step, [h_adjusted_for_R], [0,h_adjusted_for_R], [h_adjusted_for_R,h_adjusted_for_R], normal_line_z_step, z_phi_arc_step]
            },
            { # Actualizaciones de layout
                "title.text": f"Tazón Esférico: h={h_adjusted_for_R:.2f}m, R={R_curv_step_val:.1f}m, v={v_step:.2f}m/s",
                "scene.annotations": annotations_step,
                "scene.zaxis.range": [0, current_H_scene_vis] # Ajustar rango Z si R cambia
            }
        ],
        label=f"{R_curv_step_val:.1f}"
    ))

# Aplicar configuración inicial de título y anotaciones
initial_annotations = [
    dict(text=f"h = {h_current:.2f} m", x=0.05, y=0, z=h_current/2, showarrow=False, xanchor="left", yanchor="middle", font=dict(color="cyan", size=14)),
    dict(text=f"r = {r_current:.2f} m", x=r_current/2 if r_current>0 else 0.01, y=0.05, z=h_current, showarrow=False, xanchor="center", yanchor="bottom", font=dict(color="lime", size=14)),
    dict(text=f"R = {R_curv_current:.1f}m", x=0.05, y=0, z=H_visualization*0.85, showarrow=False, xanchor="left", font=dict(color="gray", size=12)),
    dict(text=f"φ = {np.degrees(phi_rad_current):.1f}°",
         x=(r_current - arc_display_radius * np.sin(phi_rad_current/2) * 1.2),
         y=0,
         z=(h_current + arc_display_radius * np.cos(phi_rad_current/2) * 1.2),
         showarrow=False, xanchor="center", font=dict(color="red", size=14)),
    dict(text=f"<b>Velocidad: {v_current:.2f} m/s</b><br>tan(φ)={tan_phi_current:.2f}",
         x=0, y=0, z=H_visualization*0.95,
         showarrow=False, font=dict(size=16, color="black"), bgcolor="rgba(255,255,255,0.7)", align="left")
]

fig.update_layout(
    title=f"Tazón Esférico: h={h_current:.2f}m, R={R_curv_current:.1f}m, v={v_current:.2f}m/s",
    scene=dict(
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        zaxis_title='Z (m) [Vértice en Z=0]',
        aspectmode='data',
        zaxis=dict(range=[0, H_visualization]), # Usar H_visualization actualizada
        camera=dict(eye=dict(x=1.8, y=1.8, z=max(h_current*1.5, 0.8))), # Ajustar cámara para ver mejor la esfera
        annotations=initial_annotations
    ),
    margin=dict(l=10, r=10, b=100, t=50),
    sliders=sliders_list,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

# --- Mostrar Figura ---
fig.show()