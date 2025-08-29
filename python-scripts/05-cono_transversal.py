# 05-cono_transversal.py
# Dibujo de corte transversal de un cono con bloque y normal

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch

# ---------- Parámetros editables ----------
theta_deg = 30.0   # semiángulo vertical del cono (grados)
h         = 3.0    # altura del bloque medida desde el vértice
H_top     = 5.0    # altura total mostrada del cono (>= h)
# -----------------------------------------

theta = np.deg2rad(theta_deg)
r     = h * np.tan(theta)          # radio a la altura h
Rtop  = H_top * np.tan(theta)      # "radio" en la boca visible
phi_normal_deg = 90 - theta_deg    # ángulo de la normal con la vertical

# Lienzo
fig, ax = plt.subplots(figsize=(7, 6))
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-1.25*Rtop, 1.25*Rtop)
ax.set_ylim(-0.3, 1.15*H_top)
ax.axis('off')
ax.set_title("Corte transversal del cono", pad=14)

# Superficie (triángulo isósceles)
cone_poly = Polygon([(-Rtop, H_top), (0, 0), (Rtop, H_top)],
                    closed=True, facecolor=str(0.95), edgecolor='none')
ax.add_patch(cone_poly)

# Generatrices
ax.plot([0,  Rtop], [0, H_top], lw=2, color='k')   # derecha (remarcada)
ax.plot([0, -Rtop], [0, H_top], lw=1.2, color='k') # izquierda

# Eje de simetría (vertical)
ax.plot([0, 0], [0, H_top], ls='--', lw=1.5, color='k')
ax.text(0, H_top+0.1, "eje de simetría", ha='center', va='bottom', fontsize=9)

# Vértice
ax.plot(0, 0, 'ko')
ax.text(0, -0.15, "vértice", ha='center', va='top')

# Altura h y radio r
ax.plot([-Rtop, Rtop], [h, h], ls=':', color='k') # línea de nivel y=h
ax.arrow(0, h, r, 0, width=0.0, head_width=0.12, head_length=0.15,
         length_includes_head=True, color='k', lw=2)
ax.text(r/2, h+0.18, r"$\it r$", ha='center', fontsize=12)

# Marca de h sobre el eje
ax.arrow(0, 0, 0, h, width=0.0, head_width=0.12, head_length=0.15,
         length_includes_head=True, color='k', lw=1.7)
ax.text(0.08*Rtop, h/2, r"$\it h$", fontsize=12, va='center')

# Bloque en la pared (pequeño rectángulo paralelo a la generatriz)
x_c, y_c = r, h
alpha = np.deg2rad(90 - theta_deg)  # dirección de la generatriz (desde horizontal)
w, t = 0.28, 0.14                   # ancho y espesor visuales del bloque
dx, dy = (w/2)*np.cos(alpha), (w/2)*np.sin(alpha)
# espesor perpendicular a la generatriz
nx, ny = -np.sin(alpha), np.cos(alpha)
px = [x_c-dx, x_c+dx, x_c+dx + t*nx, x_c-dx + t*nx]
py = [y_c-dy, y_c+dy, y_c+dy + t*ny, y_c-dy + t*ny]
block = Polygon(np.c_[px, py], closed=True, facecolor='tan', edgecolor='saddlebrown')
ax.add_patch(block)

# Normal a la superficie (apuntando hacia el interior del cono)
# En este corte 2D, la normal forma un ángulo phi_normal = 90° - theta con la vertical.
# La generatriz derecha tiene dirección beta = pi/2 - theta. Una normal hacia el interior
# tiene ángulo beta + pi/2 = pi - theta medido desde el eje x.
L = 0.95
ang_normal = np.pi - theta
nx_line = L*np.cos(ang_normal)
ny_line = L*np.sin(ang_normal)
ax.add_patch(FancyArrowPatch((x_c, y_c), (x_c+nx_line, y_c+ny_line),
                             arrowstyle='-|>', mutation_scale=12, lw=2, color='k'))
ax.text(x_c + nx_line - 0.25, y_c + ny_line + 0.05, "normal", fontsize=9)

# Arco para theta en el vértice
arc_r = 0.9
t = np.linspace(0, theta, 100)
ax.plot(arc_r*np.sin(t), arc_r*np.cos(t), lw=1.6, color='k')
ax.text(1.05*arc_r*np.sin(theta/2), 1.05*arc_r*np.cos(theta/2),
        r"$\theta$", fontsize=12)

# Arco para phi_normal en el punto de contacto (desde vertical hasta la normal a la superficie)
phi = np.deg2rad(phi_normal_deg)  # ángulo entre vertical y normal a la superficie
phi_t = np.linspace(0, phi, 80)   # 0 = vertical, phi = normal
arc_radius = 0.65
arc_x = x_c - arc_radius * np.sin(phi_t)
arc_y = y_c + arc_radius * np.cos(phi_t)
ax.plot(arc_x, arc_y, lw=1.4, color='k')
ax.text(x_c - arc_radius * np.sin(phi/2) - 0.12,
        y_c + arc_radius * np.cos(phi/2) + 0.05,
        r"$\phi_{\rm normal}$", fontsize=10)

# Línea segmentada vertical desde el centro del arco de phi_normal hasta el vértice
arc_center_x = x_c - arc_radius * np.sin(phi/2)
arc_center_y = y_c + arc_radius * np.cos(phi/2)
ax.plot([arc_center_x, arc_center_x], [arc_center_y, 0],
        ls='--', lw=1.3, color='gray')

# Etiquetas útiles
ax.text(Rtop*0.12, 0.17, r"semiángulo $\theta$", fontsize=9)
# ax.text(r+0.1, h-0.16, rf"$\it r=\it h\,\tan({theta_deg:.0f}^\circ)$", fontsize=9)

plt.show()