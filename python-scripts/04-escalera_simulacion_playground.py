# bola_en_escalera_sim_final_con_revelacion.py
# Fusión del script funcional con la mejora de mostrar el resultado al final.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os


# --- 1. Parámetros y Constantes ---
g = 9.81
NUM_ESCALONES = 40

# --- 2. Configuración de la Figura y los Ejes ---
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(left=0.1, bottom=0.35, top=0.9)

# --- Artistas Gráficos ---
# Cargar imagen PNG de la pelota
# Resolver ruta absoluta del PNG relativo a este script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BALL_PATH = os.path.join(SCRIPT_DIR, '04-ball.png')
print(f"[Info] Cargando imagen de pelota desde: {BALL_PATH}")

# Cargar imagen; si falla, crear un círculo rojo sintético como fallback
try:
    ball_img = plt.imread(BALL_PATH)
except Exception as e:
    print(f"[Aviso] No se pudo cargar la imagen en {BALL_PATH}. Usando imagen sintética. Detalle: {e}")
    size = 128
    ball_img = np.zeros((size, size, 4), dtype=float)
    yy, xx = np.ogrid[:size, :size]
    r = size / 2 - 1
    mask = (xx - size / 2) ** 2 + (yy - size / 2) ** 2 <= r ** 2
    ball_img[..., 3] = 0.0  # transparente por defecto
    ball_img[mask, 0] = 1.0  # rojo
    ball_img[mask, 1] = 0.0
    ball_img[mask, 2] = 0.0
    ball_img[mask, 3] = 1.0  # opacidad en el círculo

escalera_line, = ax.plot([], [], 'k-', lw=2)
# Pelota como imagen anclada bottom-center sobre el punto (x,y)
offset_img = OffsetImage(ball_img, zoom=0.025)
ball_artist = AnnotationBbox(offset_img, (0, 0), frameon=False, box_alignment=(0.5, 0.0), zorder=5)
ball_artist.set_transform(ax.transData)
ax.add_artist(ball_artist)
trace, = ax.plot([], [], ':', color='red', alpha=0.6)
resultado_text = ax.text(0.95, 0.95, "Ajusta los parámetros y presiona LANZAR", 
                         ha='right', va='top', transform=ax.transAxes, 
                         fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.7))

# --- Configuración del Gráfico ---
ax.set_title("Simulación: La Bola en la Escalera")
ax.set_xlabel("Distancia Horizontal (m)")
ax.set_ylabel("Altura (m)")
ax.grid(True, linestyle='--')

# --- 3. Funciones de la Interfaz ---

def dibujar_escalera(w, h):
    x_coords, y_coords = [0], [0]
    for n in range(1, NUM_ESCALONES + 1):
        x_prev_step = (n - 1) * w
        x_curr_step = n * w
        y_curr_step = -n * h
        x_coords.extend([x_prev_step, x_curr_step])
        y_coords.extend([y_curr_step, y_curr_step])
    
    escalera_line.set_data(x_coords, y_coords)
    ax.set_xlim(-w * 1.1, 25 * w)
    ax.set_ylim(-(25 * h) - 0.5 * h, h * 2 + 0.5 * h)
    ax.set_aspect('equal', 'box')

def altura_escalera_en_x(x, w, h):
    # Superficie de la escalera como función de x
    if x <= 0:
        return 0.0
    # k es el índice del escalón (1,2,3,...) cuyo tramo horizontal corresponde a x
    k = int(np.ceil(x / w))
    return -k * h

def launch(event):
    try:
        resultado_text.set_text("Simulando…")
        fig.canvas.draw_idle()
        # --- PREPARACIÓN ---
        v0 = v0_slider.val
        w = w_slider.val
        h = h_slider.val
        
        # Resetear el estado visual, incluyendo el texto de resultado.
        dibujar_escalera(w, h)
        ball_artist.xy = (0.0, 0.0)
        ball_artist.update_positions(fig.canvas.get_renderer())
        ball_artist.set_visible(True)
        trace.set_data([], [])
        resultado_text.set_text("") # Limpiar el texto anterior.
        
        if w < 1e-6 or h < 1e-6:
            resultado_text.set_text("Dimensiones inválidas.")
            return
            
        # --- CÁLCULO (SIN MOSTRAR AÚN) ---
        # El cálculo se hace al principio para definir la duración de la animación.
        n_aterrizaje = np.ceil((2 * h * v0**2) / (g * w**2)) if v0 > 0 and w > 0 else 1
        
        # --- ANIMACIÓN ---
        tiempo_de_vuelo = np.sqrt(2 * n_aterrizaje * h / g)
        t_points = np.linspace(0, tiempo_de_vuelo, 100)
        x_points = v0 * t_points
        y_points = -0.5 * g * t_points**2
        
        x_trace, y_trace = [], []
        # Bucle de animación
        for i in range(len(t_points)):
            x_i = x_points[i]
            y_i = y_points[i]
            # Detectar contacto realista: ignorar primeros frames y exigir avance en x
            y_superficie = altura_escalera_en_x(x_i, w, h)
            if i > 2 and x_i > 1e-3 and y_i <= y_superficie:
                y_i = y_superficie
                ball_artist.xy = (x_i, y_i)
                ball_artist.update_positions(fig.canvas.get_renderer())
                x_trace.append(x_i)
                y_trace.append(y_i)
                trace.set_data(x_trace, y_trace)
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.01)
                break
            ball_artist.xy = (x_i, y_i)
            ball_artist.update_positions(fig.canvas.get_renderer())
            x_trace.append(x_i)
            y_trace.append(y_i)
            trace.set_data(x_trace, y_trace)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)

        # --- REVELACIÓN DEL RESULTADO ---
        # Esta línea ahora se ejecuta DESPUÉS de que el bucle de animación ha terminado.
        resultado_text.set_text(f"Cálculo: n = {int(n_aterrizaje)}")
        fig.canvas.draw_idle() # Forzar un redibujado final para asegurar que el texto aparezca.
    except Exception as e:
        resultado_text.set_text(f"Error: {type(e).__name__}: {e}")
        fig.canvas.draw_idle()

# --- 4. Creación de los Widgets ---
ax_v0 = plt.axes([0.275, 0.20, 0.5, 0.03])
ax_w = plt.axes([0.275, 0.15, 0.5, 0.03])
ax_h = plt.axes([0.275, 0.1, 0.5, 0.03])
ax_launch = plt.axes([0.8, 0.025, 0.1, 0.04])

v0_slider = Slider(ax=ax_v0, label='Velocidad Inicial: v₀ (m/s)', valmin=0.5, valmax=10, valinit=3.0)
w_slider = Slider(ax=ax_w, label='Ancho Escalón: w (m)', valmin=0.05, valmax=0.50, valinit=0.28)
h_slider = Slider(ax=ax_h, label='Altura Escalón: h (m)', valmin=0.05, valmax=0.50, valinit=0.18)
launch_button = Button(ax_launch, 'LANZAR', hovercolor='limegreen')

# --- 5. Conectar los Widgets a las Funciones ---
launch_button.on_clicked(launch)

# --- 6. Estado Inicial y Ejecución ---
dibujar_escalera(w_slider.valinit, h_slider.valinit)
plt.show()