#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP4 - Inteligencia Artificial - Transformada de Hough (rectas y circunferencias)
Alumno: Lucas Adrián Fleury

Los acumuladores de Hough están programados desde cero con NumPy (sin cv2.HoughLines
ni cv2.HoughCircles). OpenCV se usa únicamente para leer/guardar imágenes, suavizar
(GaussianBlur), detectar bordes (Canny) y dibujar el resultado. Para VALIDAR la
implementación propia se compara, al final, contra las funciones de OpenCV.

Uso:
    python hough_tp4.py prueba                  # imagen sintética con verdad conocida
    python hough_tp4.py lineas   imagenes/motor_original.png
    python hough_tp4.py circulos imagenes/motor_original.png
    python hough_tp4.py todo     imagenes/motor_original.png
"""
import sys, os, json
import numpy as np
import cv2

OUT = "resultados"
# Región del block en la imagen del caso (x0, y0, x1, y1): excluye ampliación y rótulos
ROI_BLOCK = (15, 30, 445, 385)


# ----------------------------------------------------------------------------
# 1. PREPROCESAMIENTO
# ----------------------------------------------------------------------------
def limpiar_anotaciones(bgr):
    """La imagen del caso trae ejes y flechas dibujados (amarillo, azul, verde).
    Se detectan por color en HSV y se reconstruyen con inpainting para que no
    voten como bordes falsos."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    amarillo = cv2.inRange(hsv, (15, 120, 150), (35, 255, 255))
    azul = cv2.inRange(hsv, (85, 120, 120), (110, 255, 255))
    verde = cv2.inRange(hsv, (35, 80, 120), (85, 255, 255))
    m = cv2.bitwise_or(cv2.bitwise_or(amarillo, azul), verde)
    m = cv2.dilate(m, np.ones((5, 5), np.uint8))
    return cv2.inpaint(bgr, m, 5, cv2.INPAINT_TELEA)


def bordes(gris, sigma=1.6, t1=60, t2=140):
    """Escala de grises -> suavizado gaussiano -> Canny (mapa binario de bordes)."""
    suave = cv2.GaussianBlur(gris, (0, 0), sigma)
    return cv2.Canny(suave, t1, t2)


def cargar(path, usar_roi=True, limpiar=True):
    bgr = cv2.imread(path)
    if bgr is None:
        sys.exit("No se pudo leer la imagen: " + path)
    if limpiar:
        bgr = limpiar_anotaciones(bgr)
    if usar_roi and bgr.shape[1] > 500:       # solo para la imagen del caso
        x0, y0, x1, y1 = ROI_BLOCK
        bgr = bgr[y0:y1, x0:x1]
    return bgr


# ----------------------------------------------------------------------------
# 2. TRANSFORMADA DE HOUGH PARA RECTAS   rho = x*cos(theta) + y*sin(theta)
# ----------------------------------------------------------------------------
def hough_rectas(borde, paso_theta=1.0, paso_rho=1.0):
    """Devuelve (acumulador, thetas[rad], rhos). Cada píxel de borde vota en una
    curva sinusoidal del espacio (theta, rho)."""
    h, w = borde.shape
    thetas = np.deg2rad(np.arange(-90.0, 90.0, paso_theta))
    rho_max = np.hypot(h, w)
    rhos = np.arange(-rho_max, rho_max + paso_rho, paso_rho)
    acc = np.zeros((len(rhos), len(thetas)), dtype=np.int32)
    ys, xs = np.nonzero(borde)
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)
    for j in range(len(thetas)):                      # un voto por (píxel, theta)
        r = xs * cos_t[j] + ys * sin_t[j]
        idx = np.round((r + rho_max) / paso_rho).astype(int)
        acc[:, j] += np.bincount(idx, minlength=len(rhos))
    return acc, thetas, rhos


def picos_rectas(acc, thetas, rhos, n_max=10, umbral_rel=0.35, vecindad=(15, 7)):
    """Selección de picos: umbral relativo + supresión de no-máximos por vecindad.
    Se evita que una misma recta aparezca repetida (picos adyacentes)."""
    a = acc.copy().astype(float)
    umbral = umbral_rel * a.max()
    dr, dt = vecindad
    picos = []
    for _ in range(n_max):
        i, j = np.unravel_index(np.argmax(a), a.shape)
        if a[i, j] < umbral:
            break
        picos.append((rhos[i], thetas[j], int(acc[i, j])))
        a[max(0, i - dr):i + dr + 1, max(0, j - dt):j + dt + 1] = 0
    return picos


def dibujar_rectas(bgr, picos):
    out = bgr.copy()
    d = int(np.hypot(*out.shape[:2]))
    for rho, th, _ in picos:
        x0, y0 = rho * np.cos(th), rho * np.sin(th)
        p1 = (int(x0 - d * np.sin(th)), int(y0 + d * np.cos(th)))
        p2 = (int(x0 + d * np.sin(th)), int(y0 - d * np.cos(th)))
        cv2.line(out, p1, p2, (0, 0, 255), 1, cv2.LINE_AA)
    return out


# ----------------------------------------------------------------------------
# 3. TRANSFORMADA DE HOUGH PARA CIRCUNFERENCIAS DE RADIO CONOCIDO
#    (x-xo)^2 + (y-yo)^2 = R^2  ->  cada borde vota en una circunferencia de
#    radio R centrada en el propio píxel, en el espacio (xo, yo)
# ----------------------------------------------------------------------------
def hough_circulos(borde, radios, gris=None, tol_grad=np.deg2rad(18), k_grad=7):
    """Un acumulador 2D (xo, yo) por cada radio R.

    * Modo clásico (gris=None): cada borde emite un voto por cada píxel de perímetro
      de la circunferencia de votación (n_ang = 2*pi*R).
    * Modo guiado por gradiente (gris dado): cada borde vota solo en la dirección de
      su gradiente (+/- tol_grad, ambos sentidos, porque el aro puede ser más claro o
      más oscuro que el fondo). Reduce el ruido de votos y el costo de cómputo.

    En ambos casos el valor de un pico dividido por 2*pi*R es la FRACCIÓN de la
    circunferencia respaldada por bordes (comparable entre radios)."""
    h, w = borde.shape
    ys, xs = np.nonzero(borde)
    if gris is not None:
        suave = cv2.GaussianBlur(gris, (0, 0), 1.6).astype(np.float32)
        gx = cv2.Sobel(suave, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(suave, cv2.CV_32F, 0, 1, ksize=3)
        phi = np.arctan2(gy[ys, xs], gx[ys, xs])
        deltas = np.linspace(-tol_grad, tol_grad, k_grad)
    accs = {}
    for R in radios:
        a = np.zeros((h, w), dtype=np.int32)
        if gris is None:
            n_ang = int(round(2 * np.pi * R))
            ang = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
            xo = np.round(xs[:, None] - R * np.cos(ang)[None, :]).astype(int)
            yo = np.round(ys[:, None] - R * np.sin(ang)[None, :]).astype(int)
        else:
            ang = phi[:, None] + deltas[None, :]
            ang = np.concatenate([ang, ang + np.pi], axis=1)      # ambos sentidos
            xo = np.round(xs[:, None] - R * np.cos(ang)).astype(int)
            yo = np.round(ys[:, None] - R * np.sin(ang)).astype(int)
        ok = (xo >= 0) & (xo < w) & (yo >= 0) & (yo < h)
        np.add.at(a, (yo[ok], xo[ok]), 1)
        accs[R] = a
    return accs


def picos_circulos(accs, n_max=5, umbral_rel=0.55, vecindad=15):
    """Selección de picos en dos pasos:
    1) Mapa de centros = suma, sobre todos los radios, de los acumuladores
       suavizados y normalizados por 2*pi*R (proyección del acumulador 3D
       (xo, yo, R) sobre el plano (xo, yo)). Un aro con borde interior y exterior
       refuerza el mismo centro con dos radios distintos.
    2) El radio de cada centro es el R cuyo acumulador local es máximo."""
    sm = {R: cv2.GaussianBlur(a.astype(np.float32), (0, 0), 1.5) / (2 * np.pi * R)
          for R, a in accs.items()}
    centros = sum(sm.values())
    umbral = umbral_rel * centros.max()
    picos = []
    for _ in range(n_max):
        yo, xo = np.unravel_index(np.argmax(centros), centros.shape)
        if centros[yo, xo] < umbral:
            break
        y0, y1, x0, x1 = max(0, yo - 2), yo + 3, max(0, xo - 2), xo + 3
        R = max(sm, key=lambda r: sm[r][y0:y1, x0:x1].max())
        picos.append((int(xo), int(yo), int(R), float(centros[yo, xo])))
        centros[max(0, yo - vecindad):yo + vecindad + 1,
                max(0, xo - vecindad):xo + vecindad + 1] = 0
    return picos


def dibujar_circulos(bgr, picos):
    out = bgr.copy()
    for k, (xo, yo, R, _) in enumerate(picos):
        col = (0, 0, 255) if k == 0 else (0, 200, 0)
        cv2.circle(out, (xo, yo), R, col, 1, cv2.LINE_AA)
        cv2.drawMarker(out, (xo, yo), col, cv2.MARKER_CROSS, 9, 1)
    return out


def acumulador_a_imagen(acc):
    a = acc.astype(np.float32)
    a = (255 * a / max(a.max(), 1)).astype(np.uint8)
    return cv2.applyColorMap(a, cv2.COLORMAP_JET)


# ----------------------------------------------------------------------------
# 4. PROGRAMAS PRINCIPALES
# ----------------------------------------------------------------------------
def prueba_sintetica():
    """Imagen con 3 rectas y 2 circunferencias conocidas + ruido: sirve para validar
    el algoritmo contra una verdad de terreno."""
    rng = np.random.default_rng(7)
    img = np.zeros((300, 400), np.uint8)
    cv2.line(img, (20, 40), (380, 40), 255, 1)               # horizontal  rho=40 th=90
    cv2.line(img, (100, 10), (100, 290), 255, 1)             # vertical    rho=100 th=0
    cv2.line(img, (30, 280), (370, 120), 255, 1)             # oblicua
    cv2.circle(img, (260, 190), 35, 255, 1)                  # R = 35
    cv2.circle(img, (150, 190), 20, 255, 1)                  # R = 20
    ruido = rng.random(img.shape) < 0.01
    img[ruido] = 255
    cv2.imwrite(f"{OUT}/sintetica_entrada.png", img)
    res = {}
    acc, th, rh = hough_rectas(img)
    pr = picos_rectas(acc, th, rh, n_max=3, umbral_rel=0.3)
    res["rectas"] = [(round(float(r), 1), round(float(np.degrees(t)), 1), v) for r, t, v in pr]
    cv2.imwrite(f"{OUT}/sintetica_rectas.png",
                dibujar_rectas(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), pr))
    accs = hough_circulos(img, range(15, 41))
    pc = picos_circulos(accs, n_max=2, umbral_rel=0.3)
    res["circulos"] = [(x, y, R) for x, y, R, _ in pc]
    cv2.imwrite(f"{OUT}/sintetica_circulos.png",
                dibujar_circulos(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), pc))
    print(json.dumps(res, indent=1))
    return res


def run_lineas(path):
    bgr = cargar(path)
    gris = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    b = bordes(gris, sigma=1.6, t1=50, t2=130)
    acc, th, rh = hough_rectas(b)
    pr = picos_rectas(acc, th, rh, n_max=6, umbral_rel=0.45)
    cv2.imwrite(f"{OUT}/rectas_0_entrada.png", bgr)
    cv2.imwrite(f"{OUT}/rectas_1_bordes.png", b)
    cv2.imwrite(f"{OUT}/rectas_2_acumulador.png", acumulador_a_imagen(acc))
    cv2.imwrite(f"{OUT}/rectas_3_resultado.png", dibujar_rectas(bgr, pr))
    print("RECTAS detectadas (rho[px], theta[°], votos):")
    for r, t, v in pr:
        print(f"  rho={r:7.1f}  theta={np.degrees(t):6.1f}  votos={v}")
    print("píxeles de borde:", int((b > 0).sum()))
    return pr


def run_circulos(path, r_min=20, r_max=30, guiado=True):
    bgr = cargar(path)
    gris = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    b = bordes(gris, sigma=1.6, t1=50, t2=130)
    accs = hough_circulos(b, range(r_min, r_max + 1), gris=gris if guiado else None)
    pc = picos_circulos(accs, n_max=3, umbral_rel=0.70)
    cv2.imwrite(f"{OUT}/circ_0_entrada.png", bgr)
    cv2.imwrite(f"{OUT}/circ_1_bordes.png", b)
    mejor = pc[0][2] if pc else r_min
    cv2.imwrite(f"{OUT}/circ_2_acumulador.png", acumulador_a_imagen(accs[mejor]))
    cv2.imwrite(f"{OUT}/circ_3_resultado.png", dibujar_circulos(bgr, pc))
    print(f"CIRCUNFERENCIAS detectadas (R entre {r_min} y {r_max}) (xo, yo, R, votos_norm):")
    for x, y, R, v in pc:
        print(f"  xo={x:4d}  yo={y:4d}  R={R:3d}  v={v:.3f}")
    return pc


def validar_con_opencv(path):
    """Contraste de la implementación propia con cv2 (solo como verificación)."""
    bgr = cargar(path)
    gris = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    b = bordes(gris, sigma=1.6, t1=50, t2=130)
    L = cv2.HoughLines(b, 1, np.pi / 180, 90)
    print("cv2.HoughLines (umbral 90 votos): rectas:", 0 if L is None else len(L))
    C = cv2.HoughCircles(cv2.GaussianBlur(gris, (0, 0), 1.6), cv2.HOUGH_GRADIENT, 1, 20,
                         param1=130, param2=14, minRadius=20, maxRadius=30)
    print("cv2.HoughCircles:", None if C is None else np.round(C[0][:5]).astype(int).tolist())


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    modo = sys.argv[1] if len(sys.argv) > 1 else "todo"
    ruta = sys.argv[2] if len(sys.argv) > 2 else "imagenes/motor_original.png"
    if modo in ("prueba", "todo"):
        prueba_sintetica()
    if modo in ("lineas", "todo"):
        run_lineas(ruta)
    if modo in ("circulos", "todo"):
        run_circulos(ruta)
    if modo == "todo":
        validar_con_opencv(ruta)
