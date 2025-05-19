# --- BLOQUE COMÚN ---
import plotly.graph_objects as go
import numpy as np

g = 9.81  # Aceleración de la gravedad (m/s^2)
H_VISUALIZATION_DEFAULT = 8.0 # Altura/extensión de visualización por defecto
PATH_ANGLES = np.linspace(0, 2 * np.pi, 100) # Para la trayectoria circular

def create_visualization_figure(
    surface_traces_initial, # Lista de dicts para go.Surface (datos iniciales)
    initial_quantities, # Dict con h, r, v, tan_phi_geom, phi_rad, dr_dh_sign
    slider_configs, # Lista de dicts para configurar cada slider
    update_function_for_sliders, # Referencia a la función que genera datos para un step
    visualization_params, # Dict con H_max_vis_plot, plot_title_prefix, etc.
    surface_specific_annotations_func=None # Opcional
    ):

    fig = go.Figure()

    # Unpack cantidades iniciales
    h_init = initial_quantities['h']
    r_init = initial_quantities['r']
    v_init = initial_quantities['v']
    tan_phi_geom_init = initial_quantities['tan_phi_geom']
    phi_rad_init = initial_quantities['phi_rad']
    dr_dh_sign_init = initial_quantities.get('dr_dh_sign', 1) # Default a 1 si no se provee

    H_max_vis_plot = visualization_params.get('H_max_vis_plot', H_VISUALIZATION_DEFAULT)

    # --- TRACES INICIALES ---
    # 1. Superficie(s)
    trace_idx_offset = 0
    for i, trace_data in enumerate(surface_traces_initial):
        fig.add_trace(go.Surface(
            x=trace_data['x'], y=trace_data['y'], z=trace_data['z'],
            opacity=trace_data.get('opacity', 0.5),
            colorscale=trace_data.get('colorscale', 'Viridis'),
            showscale=False,
            name=trace_data.get('name', f"Superficie {i+1}"),
            hoverinfo='skip',
            lighting=dict(ambient=0.3, diffuse=0.8, specular=0.1, roughness=0.6, fresnel=0.1)
        ))
        trace_idx_offset += 1
    
    # Indices para actualizar los traces dinámicos
    idx_path = trace_idx_offset
    idx_block = idx_path + 1
    idx_h_line = idx_block + 1
    idx_r_line = idx_h_line + 1
    idx_slope_line = idx_r_line + 1 # Línea de pendiente/generatriz
    idx_normal_line = idx_slope_line + 1
    idx_phi_arc = idx_normal_line + 1
    
    initial_traces_dynamic_data = generate_dynamic_traces_data(
        h_init, r_init, phi_rad_init, dr_dh_sign_init, H_max_vis_plot
    )

    fig.add_trace(go.Scatter3d(x=initial_traces_dynamic_data['x_path'], y=initial_traces_dynamic_data['y_path'], z=initial_traces_dynamic_data['z_path'], mode='lines', line=dict(color='darkorange', width=6), name="Trayectoria"))
    fig.add_trace(go.Scatter3d(x=[r_init], y=[0], z=[h_init], mode='markers', marker=dict(size=10, color='gold', symbol='circle'), name="Bloque"))
    fig.add_trace(go.Scatter3d(x=[0,0], y=[0,0], z=[0,h_init], mode='lines', line=dict(color='cyan', width=4, dash='dash'), name="h"))
    fig.add_trace(go.Scatter3d(x=[0,r_init], y=[0,0], z=[h_init,h_init], mode='lines', line=dict(color='lime', width=4, dash='dash'), name="r"))
    fig.add_trace(go.Scatter3d(x=[0,r_init], y=[0,0], z=[0,h_init], mode='lines', line=dict(color='magenta', width=3), name="Generatriz/Ref")) # Línea de referencia
    fig.add_trace(go.Scatter3d(x=initial_traces_dynamic_data['x_normal_line'], y=initial_traces_dynamic_data['y_normal_line'], z=initial_traces_dynamic_data['z_normal_line'], mode='lines', line=dict(color='orangered', width=2, dash='dot'), name="Normal"))
    fig.add_trace(go.Scatter3d(x=initial_traces_dynamic_data['x_phi_arc'], y=initial_traces_dynamic_data['y_phi_arc'], z=initial_traces_dynamic_data['z_phi_arc'], mode='lines', line=dict(color='red', width=3), name="Ángulo φ"))

    # --- SLIDERS ---
    sliders_definitions = []
    # Mantener un diccionario de los valores activos de los sliders para pasarlos a la función de actualización
    active_slider_values_for_update = {
        s_conf['name']: s_conf['initial_value'] for s_conf in slider_configs
    }

    for i, s_conf in enumerate(slider_configs):
        steps = []
        for val_step in s_conf['values']:
            # Crear una copia para no modificar el diccionario base en cada iteración de step
            current_params_for_step = active_slider_values_for_update.copy()
            current_params_for_step[s_conf['name']] = val_step # Actualizar el valor del slider que se está construyendo

            # Llamar a la función de actualización específica de la superficie
            updated_data_dict = update_function_for_sliders(current_params_for_step, visualization_params)
            steps.append(updated_data_dict['slider_step_definition'])
        
        sliders_definitions.append(dict(
            active=np.argmin(np.abs(s_conf['values'] - s_conf['initial_value'])),
            currentvalue={"prefix": f"{s_conf.get('label', s_conf['name'])}: ", "suffix": s_conf.get('suffix',""), "font": {"size": 14}},
            pad={"t": 30, "b":10},
            x=0.05 + (i % 2) * 0.50, # Dos sliders por fila
            y=0.15 - (i // 2) * 0.08, # Filas de sliders
            len=0.40,
            steps=steps
        ))

    # --- LAYOUT INICIAL ---
    param_string_init = ", ".join([f"{s_conf.get('label_short', s_conf['name'])}={s_conf['initial_value']:.2f}" for s_conf in slider_configs if s_conf['name'] != 'h'])
    title_text = f"{visualization_params['plot_title_prefix']}: h={h_init:.2f}, v={v_init:.2f}m/s ({param_string_init})"
    
    initial_annotations = generate_annotations(
        initial_quantities,
        H_max_vis_plot,
        initial_traces_dynamic_data, # para pos del arco
        surface_specific_annotations_func,
        {s_conf['name']: s_conf['initial_value'] for s_conf in slider_configs} # initial params for specific annotations
    )
    
    max_r_extent = H_max_vis_plot * np.tan(np.deg2rad(75)) # Estimación
    if r_init > 0 : max_r_extent = max(max_r_extent, r_init * 1.8)

    fig.update_layout(
        title=title_text,
        scene=dict(
            xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='h (m) [Vértice en Z=0]',
            aspectmode='data',
            xaxis=dict(range=[-max_r_extent, max_r_extent], zeroline=False, gridcolor='rgba(128,128,128,0.3)'),
            yaxis=dict(range=[-max_r_extent, max_r_extent], zeroline=False, gridcolor='rgba(128,128,128,0.3)'),
            zaxis=dict(range=[0, H_max_vis_plot * 1.05], zeroline=False, gridcolor='rgba(128,128,128,0.3)'),
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.0), center=dict(x=0,y=0,z=h_init*0.35)),
            annotations=initial_annotations
        ),
        margin=dict(l=10, r=10, b=max(120, 60 + (len(slider_configs)//2 + len(slider_configs)%2)*50), t=50),
        sliders=sliders_definitions,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.show()

def generate_dynamic_traces_data(h, r, phi_rad, dr_dh_sign, H_max_vis):
    """Genera datos para los traces que dependen de h, r, phi."""
    x_path = r * np.cos(PATH_ANGLES)
    y_path = r * np.sin(PATH_ANGLES)
    z_path = np.full_like(x_path, h)

    arc_len = r * 0.25 if r > 0 else H_max_vis * 0.03
    arc_len = max(arc_len, H_max_vis * 0.02) # Mínima longitud para el arco

    # Normal line: desde (r,0,h). dr_dh_sign se usa para la dirección radial.
    # Si dr/dh > 0 (cono abre), normal interior tiene comp radial negativa.
    # Si dr/dh < 0 (esfera cierra), normal interior tiene comp radial positiva.
    # Usamos dr_dh_sign para controlar esto: x_norm_comp = -dr_dh_sign * sin(phi)
    # Pero phi ya es el ángulo con la vertical, así que sen(phi) es la componente horizontal de la normal.
    # El signo de la componente radial de la normal es opuesto al signo de dr/dh para una normal "interior".
    
    # Componente radial de la normal (hacia el eje z si la superficie se abre)
    norm_dx = -dr_dh_sign * arc_len * np.sin(phi_rad) 
    norm_dz = arc_len * np.cos(phi_rad)
    
    x_normal_line_end = r + norm_dx * 1.2
    z_normal_line_end = h + norm_dz * 1.2
    
    # Arco phi: desde la vertical en (r,0,h) hacia la normal
    # Ángulo inicial para el arco (vertical): pi/2
    # Ángulo final: pi/2 - dr_dh_sign * phi_rad
    phi_arc_angles_plot = np.linspace(np.pi/2, np.pi/2 - dr_dh_sign * phi_rad, 20)
    x_phi_arc = r + arc_len * np.cos(phi_arc_angles_plot)
    z_phi_arc = h + arc_len * np.sin(phi_arc_angles_plot)
    
    return {
        'x_path': x_path, 'y_path': y_path, 'z_path': z_path,
        'x_normal_line': [r, x_normal_line_end], 'y_normal_line': [0,0], 'z_normal_line': [h, z_normal_line_end],
        'x_phi_arc': x_phi_arc, 'y_phi_arc': np.zeros_like(x_phi_arc), 'z_phi_arc': z_phi_arc,
        'phi_arc_text_pos': { # Para la anotación del ángulo
            'x': r + arc_len * np.cos(np.pi/2 - dr_dh_sign * phi_rad/2) * 1.3,
            'y': 0,
            'z': h + arc_len * np.sin(np.pi/2 - dr_dh_sign * phi_rad/2) * 1.3
        }
    }

def generate_annotations(quantities, H_max_vis, dynamic_traces_data, surface_specific_func, current_params_dict):
    """Genera la lista de anotaciones para la escena."""
    h = quantities['h']
    r = quantities['r']
    v = quantities['v']
    tan_phi_geom = quantities['tan_phi_geom']
    phi_rad = quantities['phi_rad']
    
    phi_arc_text_pos = dynamic_traces_data['phi_arc_text_pos']

    base_annotations = [
        dict(text=f"h = {h:.2f}m", x=0, y=0.02*H_max_vis, z=h/2, showarrow=False, xanchor="left", font=dict(color="cyan", size=12)),
        dict(text=f"r = {r:.2f}m", x=r/2 if r>0 else 0.01*H_max_vis, y=0.02*H_max_vis, z=h, showarrow=False, xanchor="center", font=dict(color="lime", size=12)),
        dict(text=f"φ = {np.degrees(phi_rad):.1f}°",
             x=phi_arc_text_pos['x'], y=phi_arc_text_pos['y'], z=phi_arc_text_pos['z'],
             showarrow=False, xanchor="center", font=dict(color="red", size=12)),
        dict(text=f"<b>v = {v:.2f} m/s</b><br>tan(φ)<sub>geom</sub> = {tan_phi_geom if abs(tan_phi_geom) != float('inf') else 'inf':.2f}",
             x=0, y=0, z=H_max_vis*0.95, showarrow=False, font=dict(size=14, color="black"), bgcolor="rgba(255,255,255,0.7)",align="left")
    ]
    if surface_specific_func:
        base_annotations.extend(surface_specific_func(quantities, current_params_dict, H_max_vis))
    return base_annotations

def common_slider_update_logic(current_slider_values, vis_params,
                               calculate_geo_phys_func, get_surface_func,
                               surface_specific_annotations_func=None):
    """Lógica común para actualizar datos para un step de slider."""
    
    # 1. Calcular cantidades geométricas y físicas
    quantities = calculate_geo_phys_func(current_slider_values['h'], current_slider_values) # Pasar todos los params
    h, r, v = quantities['h'], quantities['r'], quantities['v']
    tan_phi_geom, phi_rad = quantities['tan_phi_geom'], quantities['phi_rad']
    dr_dh_sign = quantities.get('dr_dh_sign', 1)

    # 2. Obtener geometría de la superficie (puede depender de los parámetros del slider)
    # vis_params puede tener H_max_vis pero algunos params de superficie pueden influir también
    surface_traces_updated = get_surface_func(current_slider_values, vis_params)

    # 3. Generar datos para traces dinámicos (trayectoria, líneas, arco)
    dynamic_traces_updated_data = generate_dynamic_traces_data(h, r, phi_rad, dr_dh_sign, vis_params['H_max_vis_plot'])

    # 4. Preparar diccionario de actualización de traces para el slider
    # El orden debe coincidir con cómo se añadieron en create_visualization_figure
    # Primero las superficies, luego los dinámicos
    trace_updates_x = [st_upd['x'] for st_upd in surface_traces_updated] + \
                      [dynamic_traces_updated_data['x_path'], [r], [0,0], [0,r], [0,r],
                       dynamic_traces_updated_data['x_normal_line'], dynamic_traces_updated_data['x_phi_arc']]
    trace_updates_y = [st_upd['y'] for st_upd in surface_traces_updated] + \
                      [dynamic_traces_updated_data['y_path'], [0], [0,0], [0,0], [0,0],
                       dynamic_traces_updated_data['y_normal_line'], dynamic_traces_updated_data['y_phi_arc']]
    trace_updates_z = [st_upd['z'] for st_upd in surface_traces_updated] + \
                      [dynamic_traces_updated_data['z_path'], [h], [0,h], [h,h], [0,h],
                       dynamic_traces_updated_data['z_normal_line'], dynamic_traces_updated_data['z_phi_arc']]
    
    args_traces = {'x': trace_updates_x, 'y': trace_updates_y, 'z': trace_updates_z}

    # 5. Preparar actualizaciones del layout (título, anotaciones)
    param_string_title = ", ".join([f"{k.split('_')[0]}={val:.1f}" for k,val in current_slider_values.items() if k != 'h'])
    title_upd = f"{vis_params['plot_title_prefix']}: h={h:.2f}, v={v:.2f}m/s ({param_string_title})"
    
    annotations_upd = generate_annotations(quantities, vis_params['H_max_vis_plot'], dynamic_traces_updated_data,
                                           surface_specific_annotations_func, current_slider_values)
    
    # Dinámicamente ajustar el rango Z si la superficie cambia mucho de altura
    max_z_surface = 0
    for surf_data in surface_traces_updated:
        if surf_data['z'] is not None and len(surf_data['z']) > 0:
            max_z_surface = max(max_z_surface, np.max(surf_data['z']))
    
    z_axis_range_upd = [0, max(max_z_surface, h, vis_params['H_max_vis_plot']) * 1.05]


    args_layout = {
        "title.text": title_upd,
        "scene.annotations": annotations_upd,
        "scene.zaxis.range": z_axis_range_upd,
        # Opcional: ajustar centro de cámara
        # "scene.camera.center.z": h * 0.35 
    }

    # Determinar la etiqueta del step del slider
    # Si solo hay un parámetro en current_slider_values que no sea 'h', usar ese valor para la etiqueta.
    # O si es el slider 'h', usar h.
    label_val = h
    slider_param_name_for_label = vis_params.get('slider_param_name_for_label', 'h') # Slider que está siendo construido
    label_val = current_slider_values[slider_param_name_for_label]

    return {"slider_step_definition": dict(method="update", args=[args_traces, args_layout], label=f"{label_val:.2f}")}

# --- FIN BLOQUE COMÚN ---

# ------------------------------------------------------------------------------------
# CASO 0: CONO (Adaptado de tu código original y el .tex)
# tan(phi) = tan(theta)  => v = sqrt(gr * tan(theta)) = sqrt(g * h*tan(theta) * tan(theta)) = h * tan(theta)
# No, v = sqrt(gh) * tan(theta)
# ------------------------------------------------------------------------------------
def get_cone_surface_data(params, vis_params): # params={'theta_deg': val}, vis_params={'H_max_vis_plot': val}
    theta_rad = np.deg2rad(params['theta_deg'])
    H_max_vis = vis_params['H_max_vis_plot']
    if np.cos(theta_rad) < 1e-6: theta_rad = np.deg2rad(89.9) # Evita tan(90)

    z_coords = np.linspace(0, H_max_vis, 50)
    phi_surf_angles = np.linspace(0, 2 * np.pi, 50)
    Z_grid, PHI_grid = np.meshgrid(z_coords, phi_surf_angles)
    R_grid = Z_grid * np.tan(theta_rad)
    X_grid = R_grid * np.cos(PHI_grid)
    Y_grid = R_grid * np.sin(PHI_grid)
    return [{'x': X_grid, 'y': Y_grid, 'z': Z_grid, 'colorscale': 'Greys', 'name': 'Cono'}]

def calculate_cone_quantities(h_val, params): # params={'theta_deg': val}
    theta_deg = params['theta_deg']
    theta_rad = np.deg2rad(theta_deg)

    if h_val <= 0 or np.cos(theta_rad) < 1e-6:
        return {'h': h_val, 'r': 0, 'v': 0, 'tan_phi_geom': 0, 'phi_rad': 0, 'dr_dh_sign': 1}

    r_val = h_val * np.tan(theta_rad)
    # v = sqrt(g*h) * tan(theta)
    v_val = np.tan(theta_rad) * np.sqrt(g * h_val) if g * h_val >= 0 else 0
    
    # Para el cono, tan(phi_geom) = |dr/dh| = tan(theta) (asumiendo theta entre 0 y 90)
    tan_phi_geom_val = np.tan(theta_rad)
    phi_rad_val = theta_rad # Ángulo de la normal con la vertical es theta
    dr_dh_sign = 1 # dr/dh = tan(theta) > 0 para theta en (0, 90)
    
    return {'h': h_val, 'r': r_val, 'v': v_val, 
            'tan_phi_geom': tan_phi_geom_val, 'phi_rad': phi_rad_val, 'dr_dh_sign': dr_dh_sign}

def update_cone_slider_data(current_slider_vals, vis_params_cone):
    # vis_params_cone también necesita 'slider_param_name_for_label' para la lógica común
    # Se infiere del nombre del slider que se está construyendo.
    return common_slider_update_logic(current_slider_vals, vis_params_cone,
                                      calculate_cone_quantities,
                                      get_cone_surface_data)

def plot_cone_case():
    H_vis_cone = 8.0
    initial_h_cone = 3.0
    initial_theta_deg_cone = 30.0

    initial_quantities_cone = calculate_cone_quantities(initial_h_cone, {'theta_deg': initial_theta_deg_cone})
    initial_surface_cone = get_cone_surface_data({'theta_deg': initial_theta_deg_cone}, {'H_max_vis_plot': H_vis_cone})

    slider_configs_cone = [
        {'name': 'h', 'label': 'Altura h', 'values': np.linspace(0.1, H_vis_cone * 0.95, 15),
         'initial_value': initial_h_cone, 'suffix': ' m'},
        {'name': 'theta_deg', 'label': 'Ángulo θ', 'label_short': 'θ', 'values': np.linspace(5, 85, 15),
         'initial_value': initial_theta_deg_cone, 'suffix': '°'}
    ]
    
    vis_params_cone = {'H_max_vis_plot': H_vis_cone, 'plot_title_prefix': "Cono"}

    create_visualization_figure(
        surface_traces_initial=initial_surface_cone,
        initial_quantities=initial_quantities_cone,
        slider_configs=slider_configs_cone,
        # La función de actualización necesita saber cuál es el slider principal que está construyendo su etiqueta
        update_function_for_sliders=lambda csv, vp: update_cone_slider_data(csv, {**vp, 'slider_param_name_for_label': csv.get('_slider_being_built','h')}),
        visualization_params=vis_params_cone
    )

# ------------------------------------------------------------------------------------
# CASO A: TAZÓN ESFÉRICO
# r^2 = 2hR - h^2 => r = sqrt(2hR - h^2)
# tan(phi) = (R-h)/r => v = sqrt(g(R-h))
# ------------------------------------------------------------------------------------
def get_spherical_bowl_surface_data(params, vis_params): # params={'R_curv': val}
    R_curv = params['R_curv']
    # H_max_vis_plot es la altura máxima de la escena, pero la esfera solo va hasta R_curv
    z_limit_sphere = min(R_curv, vis_params['H_max_vis_plot'])
    if R_curv <=0: R_curv = 1e-3

    z_coords = np.linspace(0, z_limit_sphere, 70) # Más puntos para curvatura
    phi_surf_angles = np.linspace(0, 2 * np.pi, 70)
    Z_grid, PHI_grid = np.meshgrid(z_coords, phi_surf_angles)

    r_squared_values = 2 * Z_grid * R_curv - Z_grid**2
    r_squared_values[r_squared_values < 1e-9] = 0 # Evitar negativos pequeños
    R_grid = np.sqrt(r_squared_values)
    
    X_grid = R_grid * np.cos(PHI_grid)
    Y_grid = R_grid * np.sin(PHI_grid)
    return [{'x': X_grid, 'y': Y_grid, 'z': Z_grid, 'colorscale': 'Blues', 'name': 'Tazón Esférico'}]

def calculate_spherical_bowl_quantities(h_val, params): # params={'R_curv': val, 'h': h_val}
    R_curv = params['R_curv']

    # Restricciones: 0 <= h_val < R_curv (para la mitad inferior y evitar singularidad en h=R)
    h_eff = np.clip(h_val, 0, R_curv * 0.9999) # Evitar h=R exacto para tan_phi

    if R_curv <= 0:
        return {'h': h_eff, 'r': 0, 'v': 0, 'tan_phi_geom': 0, 'phi_rad': 0, 'dr_dh_sign': 1}

    r_sq_eff = 2 * h_eff * R_curv - h_eff**2
    r_val = np.sqrt(r_sq_eff) if r_sq_eff > 0 else 0
    
    v_val = np.sqrt(g * (R_curv - h_eff)) if g * (R_curv - h_eff) >= 0 else 0
    
    # tan(phi_geom) = |dr/dh| = |(R_curv - h_eff) / r_val|
    if r_val > 1e-5:
        tan_phi_geom_val = np.abs((R_curv - h_eff) / r_val)
        # dr/dh = (R-h)/r. Para h < R, (R-h)>0, r>0, so dr/dh > 0
        dr_dh_sign = 1 if (R_curv - h_eff) >= 0 else -1 # Debería ser siempre 1 para h < R
    elif h_eff < 1e-5 : # En el vértice h=0, r=0
        tan_phi_geom_val = float('inf') # dr/dh -> inf. Normal es vertical. Ángulo phi con vertical = 0.
                                      # Esta tan_phi_geom no es tan(0). Es tan(ángulo de tangente con h-eje).
        dr_dh_sign = 1
    else: # r es casi 0 pero h no es 0 (no debería ocurrir para la esfera si h<R)
        tan_phi_geom_val = float('inf')
        dr_dh_sign = 1

    # Ángulo phi de la normal con la vertical
    # Si dr/dh = (R-h)/r, la tangente tiene vector (1, dr/dh) en plano (h,r)
    # Normal vector (-dr/dh, 1). Ángulo con eje h (vertical) es arctan(|-dr/dh / 1|) = arctan(dr/dh)
    # No, phi es el ángulo tal que tan(phi_fisica) = v^2/gr.
    # Geométricamente, si la normal N hace un ángulo phi con la vertical,
    # y la tangente T hace un ángulo alpha con la horizontal (eje r), entonces phi = alpha.
    # Pendiente de la tangente dh/dr. tan(alpha) = dh/dr.
    # dh/dr = r / (R-h). Entonces tan(phi) = r / (R-h).
    # Esto es 1/tan_phi_geom_val (si tan_phi_geom = (R-h)/r)
    
    # Usemos la definición del problema: phi es el ángulo de N con la vertical,
    # Y se asume que tan(phi_geom) = |dr/dh| es el tan(phi) que va en v^2 = gr tan(phi).
    # Aunque esto lleva a v = sqrt(g(R-h)), que es correcto.
    # El phi_rad a visualizar debe ser tal que su tangente es tan_phi_geom_val.
    # Si tan_phi_geom_val es 'inf', phi_rad es pi/2.
    if tan_phi_geom_val == float('inf'):
        phi_rad_val = np.pi/2
    else:
        phi_rad_val = np.arctan(tan_phi_geom_val)

    # Ajuste para el vértice: normal vertical, phi=0
    if h_eff < 1e-5 and r_val < 1e-5:
        phi_rad_val = 0 
        tan_phi_geom_val = 0 # Para mostrar, aunque dr/dh es inf. Físicamente N es vertical.
    
    return {'h': h_eff, 'r': r_val, 'v': v_val,
            'tan_phi_geom': tan_phi_geom_val, 'phi_rad': phi_rad_val, 'dr_dh_sign': dr_dh_sign}


def update_spherical_bowl_slider_data(current_slider_vals, vis_params_sphere):
    return common_slider_update_logic(current_slider_vals, vis_params_sphere,
                                      calculate_spherical_bowl_quantities,
                                      get_spherical_bowl_surface_data)

def plot_spherical_bowl_case():
    initial_R_curv_sphere = 5.0
    initial_h_sphere = initial_R_curv_sphere / 3.0
    # Para la esfera, la altura máxima de visualización es R_curv
    H_vis_sphere = initial_R_curv_sphere 

    initial_quantities_sphere = calculate_spherical_bowl_quantities(initial_h_sphere, {'R_curv': initial_R_curv_sphere, 'h': initial_h_sphere})
    initial_surface_sphere = get_spherical_bowl_surface_data({'R_curv': initial_R_curv_sphere}, {'H_max_vis_plot': H_vis_sphere})

    slider_configs_sphere = [
        {'name': 'h', 'label': 'Altura h', 'values': np.linspace(0.01, initial_R_curv_sphere * 0.98, 15),
         'initial_value': initial_h_sphere, 'suffix': ' m'},
        {'name': 'R_curv', 'label': 'Radio Tazón R', 'label_short': 'R', 'values': np.linspace(1.0, 10.0, 10),
         'initial_value': initial_R_curv_sphere, 'suffix': ' m'}
    ]
    
    vis_params_sphere = {'H_max_vis_plot': H_vis_sphere, 'plot_title_prefix': "Tazón Esférico"}

    create_visualization_figure(
        surface_traces_initial=initial_surface_sphere,
        initial_quantities=initial_quantities_sphere,
        slider_configs=slider_configs_sphere,
        update_function_for_sliders=lambda csv, vp: update_spherical_bowl_slider_data(csv, {**vp, 'slider_param_name_for_label': csv.get('_slider_being_built','h')}),
        visualization_params=vis_params_sphere
    )


# --- Main execution ---
if __name__ == '__main__':
    print("Generando visualización para el Cono...")
    plot_cone_case()
    
    print("\nGenerando visualización para el Tazón Esférico...")
    plot_spherical_bowl_case()

    # Aquí puedes añadir llamadas para los otros casos cuando los implementes:
    # print("\nGenerando visualización para el Paraboloide...")
    # plot_paraboloid_case()
    # print("\nGenerando visualización para el Hiperboloide...")
    # plot_hyperboloid_case()
    # print("\nGenerando visualización para el Elipsoide...")
    # plot_ellipsoid_case()





























































    # --- BLOQUE COMÚN (AJUSTADO) ---
import plotly.graph_objects as go
import numpy as np

g = 9.81  # Aceleración de la gravedad (m/s^2)
H_VISUALIZATION_DEFAULT = 8.0
PATH_ANGLES = np.linspace(0, 2 * np.pi, 100)

def create_visualization_figure(
    surface_traces_initial,
    initial_quantities,
    slider_configs,
    update_function_for_sliders, # Función específica de la superficie para construir los steps de los sliders
    visualization_params,
    surface_specific_annotations_func=None,
    custom_traces_initial=None, # Para traces adicionales como el arco theta del cono original
    custom_traces_indices_map=None # Para mapear nombres de traces custom a sus índices
    ):

    fig = go.Figure()

    h_init = initial_quantities['h']
    r_init = initial_quantities['r']
    v_init = initial_quantities['v']
    # phi_rad_init = initial_quantities['phi_rad'] # El cono original no usa 'phi' explícitamente de esta manera

    H_max_vis_plot = visualization_params.get('H_max_vis_plot', H_VISUALIZATION_DEFAULT)

    # --- TRACES INICIALES ---
    # 1. Superficie(s)
    trace_idx_offset = 0
    for i, trace_data in enumerate(surface_traces_initial):
        fig.add_trace(go.Surface(
            x=trace_data['x'], y=trace_data['y'], z=trace_data['z'],
            opacity=trace_data.get('opacity', 0.4),
            colorscale=trace_data.get('colorscale', 'Viridis'), # Default si no se especifica
            showscale=False,
            name=trace_data.get('name', f"Superficie {i+1}"),
            hoverinfo='skip',
            lighting=dict(ambient=0.3, diffuse=0.8, specular=0.1, roughness=0.6, fresnel=0.1)
        ))
        trace_idx_offset += 1
    
    # Indices para traces estándar (pueden ser sobreescritos/ignorados por custom_traces_indices_map)
    std_trace_indices = {
        'path': trace_idx_offset,
        'block': trace_idx_offset + 1,
        'h_line': trace_idx_offset + 2,
        'r_line': trace_idx_offset + 3,
        'ref_line': trace_idx_offset + 4, # Puede ser generatriz o una línea de normal
        # 'phi_arc': trace_idx_offset + 5 # El arco de ángulo se manejará por custom_traces si es necesario
    }
    next_available_idx = trace_idx_offset + 5

    # Traces estándar (si no se proveen custom)
    if not custom_traces_initial:
        fig.add_trace(go.Scatter3d(x=r_init * np.cos(PATH_ANGLES), y=r_init * np.sin(PATH_ANGLES), z=np.full_like(PATH_ANGLES, h_init), mode='lines', line=dict(color='darkorange', width=6), name="Trayectoria"))
        fig.add_trace(go.Scatter3d(x=[r_init], y=[0], z=[h_init], mode='markers', marker=dict(size=10, color='gold', symbol='circle'), name="Bloque")) # Default: esfera dorada
        fig.add_trace(go.Scatter3d(x=[0,0], y=[0,0], z=[0,h_init], mode='lines', line=dict(color='cyan', width=4, dash='dash'), name="h"))
        fig.add_trace(go.Scatter3d(x=[0,r_init], y=[0,0], z=[h_init,h_init], mode='lines', line=dict(color='lime', width=4, dash='dash'), name="r"))
        fig.add_trace(go.Scatter3d(x=[0,r_init], y=[0,0], z=[0,h_init], mode='lines', line=dict(color='magenta', width=3), name="Generatriz/Ref"))
    else:
        # Añadir custom traces
        for i, trace_dict in enumerate(custom_traces_initial):
            fig.add_trace(go.Scatter3d(**trace_dict))
            if custom_traces_indices_map and trace_dict.get('name') in custom_traces_indices_map:
                 # No necesitamos actualizar el índice aquí, el mapeo lo da
                 pass
            next_available_idx = trace_idx_offset + len(custom_traces_initial)


    # --- SLIDERS ---
    sliders_definitions = []
    # Guardar los valores iniciales de los sliders para la lógica de actualización "fija"
    initial_slider_values_map = {
        s_conf['name']: s_conf['initial_value'] for s_conf in slider_configs
    }

    for i_slider_conf, s_conf in enumerate(slider_configs):
        steps = []
        for val_step in s_conf['values']:
            # Construir el diccionario de parámetros actuales para este step del slider:
            # El parámetro de este slider toma val_step, los otros toman sus valores iniciales.
            current_params_for_step = initial_slider_values_map.copy()
            current_params_for_step[s_conf['name']] = val_step

            # Llamar a la función de actualización específica de la superficie
            updated_data_dict = update_function_for_sliders(
                current_params_for_step,
                visualization_params,
                initial_slider_values_map # Pasar valores iniciales para referencia si es necesario
            )
            steps.append(updated_data_dict['slider_step_definition'])
        
        sliders_definitions.append(dict(
            active=np.argmin(np.abs(s_conf['values'] - s_conf['initial_value'])),
            currentvalue={"prefix": f"{s_conf.get('label', s_conf['name'])}: ", "suffix": s_conf.get('suffix',""), "font": {"size": 16}}, # Tamaño original
            pad={"t": 50, "b":10}, # Original
            x=0.05 + (i_slider_conf % 2) * 0.50,
            y=0.1 - (i_slider_conf // 2) * 0.1, # Original
            len=0.4, # Original
            steps=steps
        ))

    # --- LAYOUT INICIAL ---
    param_string_init_list = []
    for sc in slider_configs:
        if sc['name'] == 'h':
            h_val_str = f"h={initial_slider_values_map.get('h', h_init):.2f}m"
        elif sc['name'] == 'theta_deg' and visualization_params.get('plot_title_prefix') == "Cono":
            theta_val_str = f"θ={initial_slider_values_map.get('theta_deg',0):.1f}°"
            param_string_init_list.append(theta_val_str)
        else:
            param_string_init_list.append(f"{sc.get('label_short', sc['name'])}={initial_slider_values_map.get(sc['name'],0):.2f}")
    
    title_text = f"{visualization_params['plot_title_prefix']}: {h_val_str}"
    if param_string_init_list:
        title_text += ", " + ", ".join(p for p in param_string_init_list if 'θ=' in p or 'h=' not in p)
    title_text += f", v={v_init:.2f}m/s"

    # Anotaciones iniciales: deben ser generadas por la función específica si se quiere replicar
    # Aquí pasamos un dict de los valores iniciales a la función de anotaciones.
    initial_annotations = surface_specific_annotations_func(
        initial_quantities,
        initial_slider_values_map, # Pasar los valores iniciales de todos los sliders
        visualization_params
    )
    
    max_r_extent = H_max_vis_plot * np.tan(np.deg2rad(75))
    if r_init > 0 : max_r_extent = max(max_r_extent, r_init * 2.5)


    fig.update_layout(
        title=title_text,
        scene=dict(
            xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)', # Título Z original
            aspectmode='data', # Original
            xaxis=dict(range=[-max_r_extent, max_r_extent], zeroline=False, gridcolor='rgba(128,128,128,0.3)'),
            yaxis=dict(range=[-max_r_extent, max_r_extent], zeroline=False, gridcolor='rgba(128,128,128,0.3)'),
            zaxis=dict(range=[0, H_max_vis_plot * 1.05], zeroline=False, gridcolor='rgba(128,128,128,0.3)'),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)), # Cámara original
            annotations=initial_annotations
        ),
        margin=dict(l=10, r=10, b=100, t=50), # Original
        sliders=sliders_definitions,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) # Original
    )
    fig.show()


def common_slider_update_logic_cone_style(
    current_slider_params, # Parámetros para este step (un slider cambió, otros fijos a inicial)
    vis_params,
    calculate_geo_phys_func,
    get_surface_func,
    generate_specific_traces_data_func, # Nueva func para datos de traces como tu arco theta
    generate_specific_annotations_func,
    custom_traces_indices_map # Mapeo de nombres a índices para actualizar
    ):

    # 1. Calcular cantidades (r, v)
    # Para el cono, calculate_physics toma h_val, theta_rad_val, g
    # Necesitamos extraer theta_deg de current_slider_params
    h_current_step = current_slider_params['h']
    theta_deg_current_step = current_slider_params.get('theta_deg') # Puede ser None si no es un param

    # Si es el cono, llamamos a la función original
    if vis_params.get("plot_title_prefix") == "Cono":
        r_step, v_step = calculate_cone_physics_original(h_current_step, np.deg2rad(theta_deg_current_step), g)
        # Guardar las cantidades calculadas para las anotaciones
        quantities_for_annotations = {'h': h_current_step, 'r': r_step, 'v': v_step}
    else:
        # Lógica general (aún no la usamos para el cono exacto)
        quantities = calculate_geo_phys_func(current_slider_params['h'], current_slider_params)
        r_step, v_step = quantities['r'], quantities['v']
        quantities_for_annotations = quantities


    # 2. Obtener geometría de la superficie (solo si el parámetro de la superficie cambió)
    surface_traces_updated_data = [{'x': None, 'y': None, 'z': None}] # Default: no actualizar superficie
    # Si el slider que cambió afecta a la superficie (e.g. theta_deg para el cono)
    if vis_params.get("surface_param_name") and vis_params["surface_param_name"] in current_slider_params:
         surface_data_list = get_surface_func(current_slider_params, vis_params)
         surface_traces_updated_data = surface_data_list # Asumimos una superficie por ahora

    # 3. Generar datos para traces específicos (como tu arco, trayectoria, etc.)
    specific_traces_data = generate_specific_traces_data_func(
        h_current_step, r_step, current_slider_params, vis_params
    )

    # 4. Preparar diccionario de actualización de traces
    # El orden debe coincidir con cómo se añadieron
    # Surface (0), Path (1), Block (2), h_line (3), r_line (4), Generatrix (5), Theta_Arc (6)
    
    # Mapeo para el cono (debe coincidir con custom_traces_indices_map)
    idx_map = custom_traces_indices_map

    # Inicializar con Nones para todos los traces
    num_total_traces = 1 + len(idx_map) # 1 superficie + num custom traces
    
    trace_updates_x = [None] * num_total_traces
    trace_updates_y = [None] * num_total_traces
    trace_updates_z = [None] * num_total_traces

    # Actualizar superficie si es necesario
    if surface_traces_updated_data[0]['x'] is not None:
        trace_updates_x[0] = surface_traces_updated_data[0]['x']
        trace_updates_y[0] = surface_traces_updated_data[0]['y']
        trace_updates_z[0] = surface_traces_updated_data[0]['z']

    # Actualizar los traces específicos
    trace_updates_x[idx_map['Trayectoria']] = specific_traces_data['x_path']
    trace_updates_y[idx_map['Trayectoria']] = specific_traces_data['y_path']
    trace_updates_z[idx_map['Trayectoria']] = specific_traces_data['z_path']

    trace_updates_x[idx_map['Bloque']] = [r_step]
    trace_updates_y[idx_map['Bloque']] = [0]
    trace_updates_z[idx_map['Bloque']] = [h_current_step]
    
    trace_updates_x[idx_map['h']] = [0,0]
    trace_updates_y[idx_map['h']] = [0,0]
    trace_updates_z[idx_map['h']] = [0, h_current_step]

    trace_updates_x[idx_map['r']] = [0, r_step]
    trace_updates_y[idx_map['r']] = [0, 0]
    trace_updates_z[idx_map['r']] = [h_current_step, h_current_step]

    trace_updates_x[idx_map['Generatriz']] = [0, r_step]
    trace_updates_y[idx_map['Generatriz']] = [0, 0]
    trace_updates_z[idx_map['Generatriz']] = [0, h_current_step]

    trace_updates_x[idx_map['Ángulo θ']] = specific_traces_data['x_theta_arc']
    trace_updates_y[idx_map['Ángulo θ']] = specific_traces_data['y_theta_arc']
    trace_updates_z[idx_map['Ángulo θ']] = specific_traces_data['z_theta_arc']

    args_traces = {'x': trace_updates_x, 'y': trace_updates_y, 'z': trace_updates_z}

    # 5. Preparar actualizaciones del layout (título, anotaciones)
    # Título
    title_upd_list = []
    if 'h' in current_slider_params: title_upd_list.append(f"h={current_slider_params['h']:.2f}m")
    if 'theta_deg' in current_slider_params: title_upd_list.append(f"θ={current_slider_params['theta_deg']:.1f}°")
    title_upd = f"{vis_params['plot_title_prefix']}: {', '.join(title_upd_list)}, v={v_step:.2f}m/s"

    annotations_upd = generate_specific_annotations_func(quantities_for_annotations, current_slider_params, vis_params)
    
    args_layout = {"title.text": title_upd, "scene.annotations": annotations_upd}

    # Etiqueta del step del slider
    # La función update_function_for_sliders se llamará con el nombre del slider actual
    # como vis_params['slider_being_built']
    slider_being_built = vis_params.get('slider_being_built', 'h') # default a 'h'
    label_val = current_slider_params[slider_being_built]
    label_suffix = "°" if slider_being_built == "theta_deg" else ""

    return {"slider_step_definition": dict(method="update", args=[args_traces, args_layout], label=f"{label_val:.2f}{label_suffix}")}

# --- FIN BLOQUE COMÚN (AJUSTADO) ---

# ------------------------------------------------------------------------------------
# CASO 0: CONO (Adaptado EXACTAMENTE de tu código original)
# ------------------------------------------------------------------------------------

# Funciones auxiliares de TU CÓDIGO ORIGINAL DEL CONO
def get_cone_geometry_original(theta_rad, H_max_vis):
    z_coords = np.linspace(0, H_max_vis, 50)
    phi_angles = np.linspace(0, 2 * np.pi, 50)
    Z_grid, PHI_grid = np.meshgrid(z_coords, phi_angles)
    R_grid = np.abs(Z_grid) * np.tan(theta_rad)
    X_grid = R_grid * np.cos(PHI_grid)
    Y_grid = R_grid * np.sin(PHI_grid)
    return X_grid, Y_grid, Z_grid

def calculate_cone_physics_original(h_val, theta_rad_val, g_val):
    if np.cos(theta_rad_val) < 1e-6 or h_val <=0:
        return 0, 0
    r_val = h_val * np.tan(theta_rad_val)
    v_val = np.tan(theta_rad_val) * np.sqrt(g_val * h_val) if g_val * h_val >=0 else 0
    return r_val, v_val

# Adaptación para la nueva estructura
def get_cone_surface_data_for_structure(params, vis_params):
    # params tendrá 'theta_deg'
    # vis_params tendrá 'H_max_vis_plot'
    theta_rad = np.deg2rad(params['theta_deg'])
    H_max = vis_params['H_max_vis_plot']
    if np.cos(theta_rad) < 1e-6: theta_rad = np.deg2rad(89.9)

    X, Y, Z = get_cone_geometry_original(theta_rad, H_max)
    return [{'x': X, 'y': Y, 'z': Z, 'colorscale': 'sunsetdark', 'name': 'Cono', 'opacity': 0.4}]

def generate_cone_specific_traces_data(h_val, r_val, current_params, vis_params):
    """Genera datos para los traces del cono (path, arco theta)."""
    theta_rad = np.deg2rad(current_params['theta_deg'])
    H_cone_visualization = vis_params['H_max_vis_plot'] # Tu variable original

    x_path = r_val * np.cos(PATH_ANGLES)
    y_path = r_val * np.sin(PATH_ANGLES)
    z_path = np.full_like(PATH_ANGLES, h_val)

    arc_display_radius = min(h_val / 2.5, H_cone_visualization / 8) if h_val > 0 else 0.1
    theta_arc_line_points = np.linspace(0, theta_rad, 20)
    x_theta_arc = arc_display_radius * np.sin(theta_arc_line_points)
    # Tu código original para z_theta_arc en slider de h era: -arc_display_radius_step * np.cos(theta_arc_line_step)
    # Pero para el slider de theta era: arc_display_radius_step * np.cos(theta_arc_line_step)
    # Dado que el cono abre hacia +Z, cos debe ser positivo.
    z_theta_arc = arc_display_radius * np.cos(theta_arc_line_points)

    return {
        'x_path': x_path, 'y_path': y_path, 'z_path': z_path,
        'x_theta_arc': x_theta_arc, 'y_theta_arc': np.zeros_like(x_theta_arc), 'z_theta_arc': z_theta_arc,
        'arc_display_radius': arc_display_radius # Para anotaciones
    }

def generate_cone_specific_annotations(quantities, current_params, vis_params):
    """Genera las anotaciones EXACTAS de tu código original del cono."""
    h_val = quantities['h']
    r_val = quantities['r']
    # v_val = quantities['v'] # La anotación de velocidad está separada en tu código
    theta_deg = current_params['theta_deg']
    theta_rad = np.deg2rad(theta_deg)

    # Recrear arc_display_radius como en tu código original
    # Esto es un poco redundante si ya se calculó en generate_cone_specific_traces_data
    # pero para asegurar la exactitud, lo recalculamos o lo pasamos.
    # Para simplificar, asumimos que está disponible o recalculamos
    H_cone_visualization = vis_params['H_max_vis_plot']
    arc_display_radius = min(h_val / 2.5, H_cone_visualization / 8) if h_val > 0 else 0.1

    # La posición de la anotación de velocidad depende de si es el slider h o theta en tu código original
    # Por ahora, usamos la posición del texto de velocidad del layout inicial (z=8)
    # o z=0.5 del slider. Vamos a usar la z=H_cone_visualization para que esté arriba.
    vel_annotation_z = H_cone_visualization * 0.98 # Ajustar según necesites

    annotations = [
        dict(text=f"h = {h_val:.2f} m", x=0.05, y=0, z=h_val/2, showarrow=False, xanchor="left", yanchor="middle", font=dict(color="blue", size=14)),
        dict(text=f"r = {r_val:.2f} m", x=r_val/2 if r_val >0 else 0.01, y=0.05, z=h_val, showarrow=False, xanchor="center", yanchor="bottom", font=dict(color="green", size=14)),
        dict(text=f"θ = {theta_deg:.1f}°",
             x=arc_display_radius * np.sin(theta_rad/1.5) * 1.1,
             y=0,
             # z = arc_display_radius * np.cos(theta_rad/1.5) * 1.1, # Si arco z es positivo
             z=arc_display_radius * np.cos(theta_rad/1.5) * 1.1 if vis_params.get("slider_being_built") != "h" else arc_display_radius * np.cos(theta_rad/1.5) * 1.1, # Ajustar signo si es necesario
             showarrow=False, xanchor="center", font=dict(color="red", size=14)),
        dict(text=f"<b>Velocidad Requerida: {quantities['v']:.2f} m/s</b>", x=0, y=0, z=vel_annotation_z,
             showarrow=False, font=dict(size=16, color="black"), bgcolor="rgba(255,255,255,0.7)")
    ]
    return annotations


def update_cone_slider_data_exact(current_slider_params, vis_params, initial_slider_values_map_ref):
    # Esta función será llamada por create_visualization_figure.
    # current_slider_params tiene el valor actual del slider que se movió,
    # y los valores iniciales para los otros sliders (como en tu lógica original).
    # No necesitamos initial_slider_values_map_ref aquí porque current_slider_params ya está configurado así.
    
    # Para asegurar que los traces se actualicen correctamente,
    # necesitamos un mapeo de nombres a índices como se definieron en plot_cone_case_exact
    custom_indices = vis_params.get('custom_traces_indices_map', {})

    return common_slider_update_logic_cone_style(
        current_slider_params,
        vis_params,
        calculate_cone_physics_original, # No realmente usado por common_slider_update_logic_cone_style
        get_cone_surface_data_for_structure,
        generate_cone_specific_traces_data,
        generate_cone_specific_annotations,
        custom_indices
    )


def plot_cone_case_exact():
    H_vis_cone = 8.0
    initial_h_cone = 5.0
    initial_theta_deg_cone = 30.0

    # Calcular r y v iniciales usando tu función original
    r_init, v_init = calculate_cone_physics_original(initial_h_cone, np.deg2rad(initial_theta_deg_cone), g)
    initial_quantities_cone = {'h': initial_h_cone, 'r': r_init, 'v': v_init, 'theta_deg': initial_theta_deg_cone} # Para anotaciones

    # Geometría inicial de la superficie del cono
    initial_surface_cone = get_cone_surface_data_for_structure(
        {'theta_deg': initial_theta_deg_cone},
        {'H_max_vis_plot': H_vis_cone}
    )

    # Traces iniciales EXACTOS como en tu código
    theta_rad_init = np.deg2rad(initial_theta_deg_cone)
    path_angles_init = np.linspace(0, 2 * np.pi, 100)
    x_path_init = r_init * np.cos(path_angles_init)
    y_path_init = r_init * np.sin(path_angles_init)
    z_path_init = np.full_like(path_angles_init, initial_h_cone)

    arc_display_radius_init = min(initial_h_cone / 2.5, H_vis_cone / 8) if initial_h_cone > 0 else 0.1
    theta_arc_line_init = np.linspace(0, theta_rad_init, 20)
    x_theta_arc_init = arc_display_radius_init * np.sin(theta_arc_line_init)
    z_theta_arc_init = arc_display_radius_init * np.cos(theta_arc_line_init) # Cono abre hacia +Z

    custom_traces_cone = [
        dict(x=x_path_init, y=y_path_init, z=z_path_init, mode='lines', line=dict(color='darkorange', width=6), name="Trayectoria"),
        dict(x=[r_init], y=[0], z=[initial_h_cone], mode='markers', marker=dict(size=10, color='saddlebrown', symbol='square'), name="Bloque"),
        dict(x=[0,0], y=[0,0], z=[0,initial_h_cone], mode='lines', line=dict(color='blue', width=4, dash='dash'), name="h"),
        dict(x=[0,r_init], y=[0,0], z=[initial_h_cone,initial_h_cone], mode='lines', line=dict(color='green', width=4, dash='dash'), name="r"),
        dict(x=[0,r_init], y=[0,0], z=[0,initial_h_cone], mode='lines', line=dict(color='purple', width=3), name="Generatriz"),
        dict(x=x_theta_arc_init, y=np.zeros_like(x_theta_arc_init), z=z_theta_arc_init, mode='lines', line=dict(color='red', width=3), name="Ángulo θ")
    ]
    
    # Mapeo de nombres a índices después de la superficie (índice 0)
    custom_trace_indices_map_cone = {trace['name']: i+1 for i, trace in enumerate(custom_traces_cone)}


    slider_configs_cone = [
        {'name': 'h', 'label': 'Altura h (m)', 'values': np.linspace(0.5, H_vis_cone * 0.8, 20),
         'initial_value': initial_h_cone, 'suffix': ' m'},
        {'name': 'theta_deg', 'label': 'Ángulo θ (grados)', 'label_short': 'θ', 'values': np.linspace(5, 85, 17),
         'initial_value': initial_theta_deg_cone, 'suffix': '°'}
    ]
    
    vis_params_cone = {
        'H_max_vis_plot': H_vis_cone,
        'plot_title_prefix': "Cono",
        'surface_param_name': 'theta_deg', # El parámetro que afecta la geometría de la superficie
        'custom_traces_indices_map': custom_trace_indices_map_cone
    }

    create_visualization_figure(
        surface_traces_initial=initial_surface_cone,
        initial_quantities=initial_quantities_cone,
        slider_configs=slider_configs_cone,
        update_function_for_sliders=lambda csv, vp, initial_slider_vals: update_cone_slider_data_exact({**initial_slider_vals, **csv}, vp, initial_slider_vals), # csv contiene solo el slider cambiado
        visualization_params=vis_params_cone,
        surface_specific_annotations_func=generate_cone_specific_annotations,
        custom_traces_initial=custom_traces_cone,
        custom_traces_indices_map=custom_trace_indices_map_cone
    )

# --- Main execution ---
if __name__ == '__main__':
    print("Generando visualización para el Cono (Estilo Original)...")
    plot_cone_case_exact()