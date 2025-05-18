import plotly.graph_objects as go
import numpy as np

# --- Constantes y Parámetros Iniciales ---
g = 9.81  # Aceleración de la gravedad (m/s^2)
H_paraboloid_visualization = 8.0  # Altura máxima de la parábola para la visualización (m)

# Valores iniciales para los sliders
initial_k_paraboloid = 4.0  # Parámetro 'k' de la parábola (r^2 = kh)
initial_h_paraboloid = H_paraboloid_visualization / 2.0  # Altura del bloque desde el vértice (m)


# --- Funciones Auxiliares para el PARABOLOIDE ---

def get_paraboloid_geometry(k_val, H_max_vis):
    """Genera las coordenadas X, Y, Z para la superficie de un paraboloide."""
    # El paraboloide tiene el vértice en (0,0,0) y se abre hacia arriba (+Z).
    # r^2 = k_val * h  => h = r^2 / k_val
    # O bien, para un h dado, r = sqrt(k_val * h)
    
    if k_val <= 0: k_val = 1e-3 # Evitar k=0 o negativo

    z_coords = np.linspace(0, H_max_vis, 50)
    phi_angles = np.linspace(0, 2 * np.pi, 50)
    
    Z_grid, PHI_grid = np.meshgrid(z_coords, phi_angles)
    
    # r = sqrt(k_val * Z_grid)
    r_squared_values = k_val * Z_grid
    R_grid = np.sqrt(r_squared_values) # Z_grid es h
    
    X_grid = R_grid * np.cos(PHI_grid)
    Y_grid = R_grid * np.sin(PHI_grid)
    
    return X_grid, Y_grid, Z_grid

def calculate_paraboloid_physics(h_val, k_val, g_val):
    """Calcula r, v, y phi para el paraboloide."""
    if k_val <= 0 or h_val < 0:
        return 0, 0, 0, 0 # r, v, tan_phi, phi_rad

    h_clipped = max(h_val, 1e-6) # Evitar h=0 exacto para tan_phi si k > 0

    r_val = np.sqrt(k_val * h_clipped)
    
    v_val = np.sqrt(g_val * k_val / 2.0) if g_val * k_val >= 0 else 0
    
    # tan(phi) = k / (2*r)
    if r_val > 1e-5: # Evitar división por cero
        tan_phi_val = k_val / (2.0 * r_val)
        phi_rad_val = np.arctan(tan_phi_val)
    else: # En el vértice (h=0, r=0), la normal es vertical.
        tan_phi_val = float('inf') # dr/dh es infinito en el vértice para la parábola
        phi_rad_val = 0 # Ángulo de la normal con la vertical es 0

    return r_val, v_val, tan_phi_val, phi_rad_val


# --- Creación de la Figura Inicial ---
fig = go.Figure()

# Valores iniciales actuales
h_current = initial_h_paraboloid
k_current = initial_k_paraboloid

# Calcular r, v, y phi iniciales
r_current, v_current, tan_phi_current, phi_rad_current = calculate_paraboloid_physics(h_current, k_current, g)

# H_visualization es la altura de la escena.
H_visualization = H_paraboloid_visualization


# 1. Superficie del Paraboloide (Trace 0)
X_parab, Y_parab, Z_parab = get_paraboloid_geometry(k_current, H_visualization)
fig.add_trace(go.Surface(
    x=X_parab, y=Y_parab, z=Z_parab,
    opacity=0.5,
    colorscale='YlGnBu', # Un colorscale diferente
    showscale=False,
    name="Paraboloide",
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

# 3. Bloque (Trace 2)
fig.add_trace(go.Scatter3d(
    x=[r_current], y=[0], z=[h_current],
    mode='markers',
    marker=dict(size=10, color='saddlebrown', symbol='square'), # Manteniendo tu estilo de bloque
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

# 6. Línea de Referencia (Generatriz de la parábola) (Trace 5)
# Desde el vértice (0,0,0) hasta el bloque (r_current, 0, h_current)
fig.add_trace(go.Scatter3d(
    x=[0, r_current], y=[0, 0], z=[0, h_current],
    mode='lines',
    line=dict(color='purple', width=3), # Similar a tu generatriz del cono
    name="Generatriz Parábola"
))

# 7. Arco para el ángulo phi (Trace 6)
# phi es el ángulo entre la normal y la vertical.
# Pendiente de la tangente dr/dh = k/(2r).
# La normal tiene una pendiente perpendicular a la tangente.
# Si la tangente tiene vector (1, dr/dh) en el plano (h,r), la normal es (-dr/dh, 1).
# Ángulo de la normal con el eje h (vertical) es arctan(|-dr/dh / 1|) = arctan(dr/dh) = arctan(k/2r)
# Esto coincide con phi_rad_current.

# Longitud del arco para visualización
arc_display_radius = r_current * 0.3 if r_current > 0 else H_visualization * 0.03
arc_display_radius = max(arc_display_radius, H_visualization*0.03)

# Arco centrado en el bloque (r_current, 0, h_current)
# Similar al tazón esférico: el arco va desde la vertical hacia la normal.
phi_arc_line = np.linspace(0, phi_rad_current, 20)
# Si la normal va hacia "adentro" (componente radial negativa)
# dr/dh = k/(2r) > 0. Normal tiene componente radial opuesta a la tangente.
# La normal apunta hacia el eje Z, "cerrando" la parábola.
# Para visualizar phi (ángulo con la vertical):
x_phi_arc = r_current - arc_display_radius * np.sin(phi_arc_line)
z_phi_arc = h_current + arc_display_radius * np.cos(phi_arc_line)

fig.add_trace(go.Scatter3d(
    x=x_phi_arc, y=np.zeros_like(x_phi_arc), z=z_phi_arc,
    mode='lines',
    line=dict(color='red', width=3),
    name="Ángulo φ"
))


# --- Configuración del Layout y Sliders ---
h_slider_values = np.linspace(0.01 * H_visualization, H_visualization * 0.98, 15) # h > 0
k_slider_values = np.linspace(1.0, 10.0, 10) # Rango para k

sliders_list = [
    dict(
        active=np.argmin(np.abs(h_slider_values - initial_h_paraboloid)),
        currentvalue={"prefix": "Altura h (m): ", "suffix": " m", "font": {"size": 16}},
        pad={"t": 50, "b":10},
        x=0.05, xanchor="left",
        y=0.1, yanchor="top",
        len=0.4,
        steps=[]
    ),
    dict(
        active=np.argmin(np.abs(k_slider_values - initial_k_paraboloid)),
        currentvalue={"prefix": "Parámetro k: ", "suffix": "", "font": {"size": 16}}, # k no tiene unidad aquí
        pad={"t": 50, "b":10},
        x=0.55, xanchor="left",
        y=0.1, yanchor="top",
        len=0.4,
        steps=[]
    )
]

# --- Llenar los pasos de los sliders ---

# Slider para h (k se mantiene fijo al valor inicial del otro slider)
for h_step_val in h_slider_values:
    k_for_calc = initial_k_paraboloid # Usar k inicial para este slider
    
    r_step, v_step, tan_phi_step, phi_rad_step = calculate_paraboloid_physics(h_step_val, k_for_calc, g)
    
    x_path_step = r_step * np.cos(path_angles)
    y_path_step = r_step * np.sin(path_angles)
    z_path_step = np.full_like(x_path_step, h_step_val)
    
    arc_disp_rad_step = r_step * 0.3 if r_step > 0 else H_visualization * 0.03
    arc_disp_rad_step = max(arc_disp_rad_step, H_visualization*0.03)
    phi_arc_line_step_pts = np.linspace(0, phi_rad_step, 20)
    x_phi_arc_step = r_step - arc_disp_rad_step * np.sin(phi_arc_line_step_pts)
    z_phi_arc_step = h_step_val + arc_disp_rad_step * np.cos(phi_arc_line_step_pts)

    annotations_step = [
        dict(text=f"h = {h_step_val:.2f} m", x=0.05, y=0, z=h_step_val/2, showarrow=False, xanchor="left", yanchor="middle", font=dict(color="blue", size=14)),
        dict(text=f"r = {r_step:.2f} m", x=r_step/2 if r_step >0 else 0.01, y=0.05, z=h_step_val, showarrow=False, xanchor="center", yanchor="bottom", font=dict(color="green", size=14)),
        dict(text=f"k = {k_for_calc:.1f}", x=0.05, y=0, z=H_visualization*0.85, showarrow=False, xanchor="left", font=dict(color="gray", size=12)),
        dict(text=f"φ = {np.degrees(phi_rad_step):.1f}°",
             x=(r_step - arc_disp_rad_step * np.sin(phi_rad_step/2) * 1.2),
             y=0,
             z=(h_step_val + arc_disp_rad_step * np.cos(phi_rad_step/2) * 1.2),
             showarrow=False, xanchor="center", font=dict(color="red", size=14)),
        dict(text=f"<b>Velocidad: {v_step:.2f} m/s (constante)</b><br>tan(φ)={tan_phi_step if tan_phi_step != float('inf') else 'inf':.2f}",
             x=0, y=0, z=H_visualization*0.95,
             showarrow=False, font=dict(size=16, color="black"), bgcolor="rgba(255,255,255,0.7)", align="left")
    ]
    
    sliders_list[0]['steps'].append(dict(
        method="update",
        args=[
            { # Actualizaciones de datos de traces (Superficie (0) no cambia con h solo)
              # Path (1), Block (2), h_line (3), r_line (4), Generatrix (5), Phi_Arc (6)
                'x': [None, x_path_step, [r_step], [0,0], [0,r_step], [0,r_step], x_phi_arc_step],
                'y': [None, y_path_step, [0],      [0,0], [0,0],      [0,0],        np.zeros_like(x_phi_arc_step)],
                'z': [None, z_path_step, [h_step_val], [0,h_step_val], [h_step_val,h_step_val], [0,h_step_val], z_phi_arc_step]
            },
            { # Actualizaciones de layout
                "title.text": f"Paraboloide: h={h_step_val:.2f}m, k={k_for_calc:.1f}, v={v_step:.2f}m/s",
                "scene.annotations": annotations_step
            }
        ],
        label=f"{h_step_val:.2f}"
    ))

# Slider para k (h se mantiene fija al valor inicial del otro slider)
for k_step_val in k_slider_values:
    h_for_calc = initial_h_paraboloid # Usar h inicial para este slider
    
    r_step, v_step, tan_phi_step, phi_rad_step = calculate_paraboloid_physics(h_for_calc, k_step_val, g)
    
    X_parab_step, Y_parab_step, Z_parab_step = get_paraboloid_geometry(k_step_val, H_visualization)
    
    x_path_step = r_step * np.cos(path_angles)
    y_path_step = r_step * np.sin(path_angles)
    z_path_step = np.full_like(x_path_step, h_for_calc)
    
    arc_disp_rad_step = r_step * 0.3 if r_step > 0 else H_visualization * 0.03
    arc_disp_rad_step = max(arc_disp_rad_step, H_visualization*0.03)
    phi_arc_line_step_pts = np.linspace(0, phi_rad_step, 20)
    x_phi_arc_step = r_step - arc_disp_rad_step * np.sin(phi_arc_line_step_pts)
    z_phi_arc_step = h_for_calc + arc_disp_rad_step * np.cos(phi_arc_line_step_pts)

    annotations_step = [
        dict(text=f"h = {h_for_calc:.2f} m", x=0.05, y=0, z=h_for_calc/2, showarrow=False, xanchor="left", yanchor="middle", font=dict(color="blue", size=14)),
        dict(text=f"r = {r_step:.2f} m", x=r_step/2 if r_step>0 else 0.01, y=0.05, z=h_for_calc, showarrow=False, xanchor="center", yanchor="bottom", font=dict(color="green", size=14)),
        dict(text=f"k = {k_step_val:.1f}", x=0.05, y=0, z=H_visualization*0.85, showarrow=False, xanchor="left", font=dict(color="gray", size=12)),
        dict(text=f"φ = {np.degrees(phi_rad_step):.1f}°",
             x=(r_step - arc_disp_rad_step * np.sin(phi_rad_step/2) * 1.2),
             y=0,
             z=(h_for_calc + arc_disp_rad_step * np.cos(phi_rad_step/2) * 1.2),
             showarrow=False, xanchor="center", font=dict(color="red", size=14)),
        dict(text=f"<b>Velocidad: {v_step:.2f} m/s (constante)</b><br>tan(φ)={tan_phi_step if tan_phi_step != float('inf') else 'inf':.2f}",
             x=0, y=0, z=H_visualization*0.95,
             showarrow=False, font=dict(size=16, color="black"), bgcolor="rgba(255,255,255,0.7)", align="left")
    ]
    
    sliders_list[1]['steps'].append(dict(
        method="update",
        args=[
            { # Actualizaciones de datos de traces (todos pueden cambiar con k)
                'x': [X_parab_step, x_path_step, [r_step], [0,0], [0,r_step], [0,r_step], x_phi_arc_step],
                'y': [Y_parab_step, y_path_step, [0],      [0,0], [0,0],      [0,0],        np.zeros_like(x_phi_arc_step)],
                'z': [Z_parab_step, z_path_step, [h_for_calc], [0,h_for_calc], [h_for_calc,h_for_calc], [0,h_for_calc], z_phi_arc_step]
            },
            { # Actualizaciones de layout
                "title.text": f"Paraboloide: h={h_for_calc:.2f}m, k={k_step_val:.1f}, v={v_step:.2f}m/s",
                "scene.annotations": annotations_step
            }
        ],
        label=f"{k_step_val:.1f}"
    ))

# Aplicar configuración inicial de título y anotaciones
initial_annotations = [
    dict(text=f"h = {h_current:.2f} m", x=0.05, y=0, z=h_current/2, showarrow=False, xanchor="left", yanchor="middle", font=dict(color="blue", size=14)),
    dict(text=f"r = {r_current:.2f} m", x=r_current/2 if r_current>0 else 0.01, y=0.05, z=h_current, showarrow=False, xanchor="center", yanchor="bottom", font=dict(color="green", size=14)),
    dict(text=f"k = {k_current:.1f}", x=0.05, y=0, z=H_visualization*0.85, showarrow=False, xanchor="left", font=dict(color="gray", size=12)),
    dict(text=f"φ = {np.degrees(phi_rad_current):.1f}°",
         x=(r_current - arc_display_radius * np.sin(phi_rad_current/2) * 1.2),
         y=0,
         z=(h_current + arc_display_radius * np.cos(phi_rad_current/2) * 1.2),
         showarrow=False, xanchor="center", font=dict(color="red", size=14)),
    dict(text=f"<b>Velocidad: {v_current:.2f} m/s (constante)</b><br>tan(φ)={tan_phi_current if tan_phi_current != float('inf') else 'inf':.2f}",
         x=0, y=0, z=H_visualization*0.95,
         showarrow=False, font=dict(size=16, color="black"), bgcolor="rgba(255,255,255,0.7)", align="left")
]

fig.update_layout(
    title=f"Paraboloide: h={h_current:.2f}m, k={k_current:.1f}, v={v_current:.2f}m/s",
    scene=dict(
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        zaxis_title='Z (m) [Vértice en Z=0]',
        aspectmode='data',
        zaxis=dict(range=[0, H_visualization*1.05]),
        camera=dict(eye=dict(x=1.8, y=1.8, z=max(h_current*1.2, 0.7))),
        annotations=initial_annotations
    ),
    margin=dict(l=10, r=10, b=100, t=50),
    sliders=sliders_list,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

# --- Mostrar Figura ---
fig.show()