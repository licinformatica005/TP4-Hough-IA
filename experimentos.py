#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TP4 IA - Genera las figuras del informe y las métricas de validación.
Alumno: Lucas Adrián Fleury.  Requiere haber ejecutado antes hough_tp4.py (usa sus funciones)."""
import json, time
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import hough_tp4 as H

plt.rcParams.update({"font.family": "Carlito", "font.size": 12, "axes.titlesize": 11})
R = "resultados/"
RING = (286, 170, 25)   # referencia estimada a mano sobre la ampliación del aro (coords. del recorte)


def rgb(p):
    return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)


# ---------- Figura conceptual (consigna 1) ----------
def fig_conceptual():
    fig, ax = plt.subplots(1, 4, figsize=(11.5, 2.9))
    # rectas: puntos de una recta -> sinusoides
    m, b = 0.6, 1.0
    xs = np.array([0.5, 1.5, 2.5, 3.5, 4.5]); ys = m * xs + b
    ax[0].plot([0, 5], [b, m * 5 + b], "k-", lw=1)
    ax[0].scatter(xs, ys, c=["C0", "C1", "C2", "C3", "C4"], zorder=3)
    ax[0].set(title="(a) Plano imagen (x, y)", xlabel="x", ylabel="y")
    th = np.deg2rad(np.linspace(-90, 90, 361))
    for k, (x, y) in enumerate(zip(xs, ys)):
        ax[1].plot(np.degrees(th), x * np.cos(th) + y * np.sin(th), color=f"C{k}", lw=1.2)
    th0 = np.arctan2(1, -m)  # normal a la recta y = m x + b
    rho0 = b * np.sin(th0)
    ax[1].scatter([np.degrees(th0) - 180 if np.degrees(th0) > 90 else np.degrees(th0)],
                  [rho0 if np.degrees(th0) <= 90 else -rho0], c="r", zorder=4, s=25)
    ax[1].set(title="(b) Espacio (θ, ρ): las curvas\nconcurren en un punto", xlabel="θ [°]", ylabel="ρ")
    # circunferencias
    t = np.linspace(0, 2 * np.pi, 200)
    cx, cy, r = 3, 3, 2
    ax[2].plot(cx + r * np.cos(t), cy + r * np.sin(t), "k-", lw=1)
    pts = [0.3, 1.3, 3.0, 4.6]
    for k, a in enumerate(pts):
        ax[2].scatter(cx + r * np.cos(a), cy + r * np.sin(a), c=f"C{k}", zorder=3)
    ax[2].set(title="(c) Plano imagen: R conocido", xlabel="x", ylabel="y", aspect="equal")
    for k, a in enumerate(pts):
        px, py = cx + r * np.cos(a), cy + r * np.sin(a)
        ax[3].plot(px + r * np.cos(t), py + r * np.sin(t), color=f"C{k}", lw=1, ls="--")
    ax[3].scatter([cx], [cy], c="r", zorder=4, s=25)
    ax[3].set(title="(d) Espacio (xo, yo): las circ.\nde voto concurren en el centro", xlabel="xo", ylabel="yo", aspect="equal")
    fig.tight_layout(); fig.savefig(R + "fig_conceptual.png", dpi=200); plt.close(fig)


# ---------- Figuras de pipeline (consignas 2 y 3) ----------
def fig_pipeline_rectas(picos, acc, th, rh):
    fig, ax = plt.subplots(1, 4, figsize=(11.5, 3.1))
    ax[0].imshow(rgb(R + "rectas_0_entrada.png")); ax[0].set_title("(a) Entrada")
    ax[1].imshow(cv2.imread(R + "rectas_1_bordes.png", 0), cmap="gray"); ax[1].set_title("(b) Bordes (Canny)")
    ax[2].imshow(acc, aspect="auto", cmap="jet", extent=[-90, 90, rh[-1], rh[0]])
    for r, t, _ in picos:
        ax[2].scatter(np.degrees(t), r, facecolors="none", edgecolors="w", s=45)
    ax[2].set(title="(c) Acumulador (θ, ρ) y picos", xlabel="θ [°]", ylabel="ρ [px]")
    ax[3].imshow(rgb(R + "rectas_3_resultado.png")); ax[3].set_title("(d) Rectas detectadas")
    for a in (ax[0], ax[1], ax[3]):
        a.axis("off")
    fig.tight_layout(); fig.savefig(R + "fig_rectas_pipeline.png", dpi=200); plt.close(fig)


def fig_pipeline_circulos(accs, pc):
    sm = sum(cv2.GaussianBlur(a.astype(np.float32), (0, 0), 1.5) / (2 * np.pi * r) for r, a in accs.items())
    fig, ax = plt.subplots(1, 4, figsize=(11.5, 3.1))
    ax[0].imshow(rgb(R + "circ_0_entrada.png")); ax[0].set_title("(a) Entrada")
    ax[1].imshow(cv2.imread(R + "circ_1_bordes.png", 0), cmap="gray"); ax[1].set_title("(b) Bordes (Canny)")
    ax[2].imshow(sm, cmap="jet"); ax[2].set_title("(c) Acumulador de centros")
    ax[3].imshow(rgb(R + "circ_3_resultado.png")); ax[3].set_title("(d) Circunferencia detectada")
    for a in ax:
        a.axis("off")
    fig.tight_layout(); fig.savefig(R + "fig_circulos_pipeline.png", dpi=200); plt.close(fig)


def fig_sintetica():
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 2.9))
    ax[0].imshow(cv2.imread(R + "sintetica_entrada.png", 0), cmap="gray"); ax[0].set_title("(a) Imagen sintética con ruido")
    ax[1].imshow(rgb(R + "sintetica_rectas.png")); ax[1].set_title("(b) Rectas detectadas")
    ax[2].imshow(rgb(R + "sintetica_circulos.png")); ax[2].set_title("(c) Circunferencias detectadas")
    for a in ax:
        a.axis("off")
    fig.tight_layout(); fig.savefig(R + "fig_sintetica.png", dpi=200); plt.close(fig)


def main():
    m = {}
    # 1) prueba sintética (verdad conocida)
    ps = H.prueba_sintetica()
    verdad_r = [(-40.0, -90.0), (100.0, 0.0), (266.1, 64.8)]
    m["sintetica_rectas"] = [{"rho": a[0], "theta": a[1], "votos": a[2]} for a in ps["rectas"]]
    m["sintetica_circulos"] = ps["circulos"]
    # 2) imagen del motor: rectas
    bgr = H.cargar("imagenes/motor_original.png"); g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    b = H.bordes(g, 1.6, 50, 130)
    t0 = time.time(); acc, th, rh = H.hough_rectas(b); t_l = time.time() - t0
    pr = H.picos_rectas(acc, th, rh, n_max=6, umbral_rel=0.45)
    m["motor_rectas"] = {"n_bordes": int((b > 0).sum()), "tiempo_s": round(t_l, 2),
                         "picos": [(round(float(r), 1), round(float(np.degrees(t)), 1), v) for r, t, v in pr],
                         "acc_shape": list(acc.shape)}
    fig_pipeline_rectas(pr, acc, th, rh)
    # 3) imagen del motor: circunferencias (guiado vs clásico, 3 configuraciones de bordes)
    def ok(p): return p and np.hypot(p[0][0] - RING[0], p[0][1] - RING[1]) <= 8
    tabla = []
    for sig, t1, t2 in [(1.6, 50, 130), (1.2, 40, 110), (2.0, 40, 100)]:
        bb = H.bordes(g, sig, t1, t2)
        fila = {"cfg": f"σ={sig}, Canny {t1}/{t2}"}
        for nombre, gris in (("guiado", g), ("clasico", None)):
            t0 = time.time()
            ac = H.hough_circulos(bb, range(20, 31), gris=gris)
            dt = time.time() - t0
            p = H.picos_circulos(ac, n_max=3, umbral_rel=0.7)
            fila[nombre] = {"ok": bool(ok(p)), "top1": p[0][:3] if p else None, "t": round(dt, 1)}
        tabla.append(fila)
    m["motor_circulos_robustez"] = tabla
    accs = H.hough_circulos(b, range(20, 31), gris=g)
    pc = H.picos_circulos(accs, n_max=3, umbral_rel=0.7)
    m["motor_circulos"] = {"detectado": pc[0][:3], "referencia": RING,
                           "error_centro_px": round(float(np.hypot(pc[0][0] - RING[0], pc[0][1] - RING[1])), 1)}
    fig_pipeline_circulos(accs, pc)
    # 4) contraste con OpenCV
    L = cv2.HoughLines(b, 1, np.pi / 180, 90)
    C = cv2.HoughCircles(cv2.GaussianBlur(g, (0, 0), 1.6), cv2.HOUGH_GRADIENT, 1, 20, param1=130, param2=14,
                         minRadius=20, maxRadius=30)
    cv2c = np.round(C[0][0]).astype(int).tolist()
    m["opencv"] = {"HoughLines_umbral90": 0 if L is None else int(len(L)),
                   "HoughCircles_top1": cv2c,
                   "error_centro_px": round(float(np.hypot(cv2c[0] - RING[0], cv2c[1] - RING[1])), 1)}
    fig_sintetica(); fig_conceptual()
    json.dump(m, open(R + "metricas.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(json.dumps(m, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


if __name__ == "__main__":
    main()
