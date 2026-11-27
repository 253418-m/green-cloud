"""Generación y exportación de las gráficas científicas de Green Cloud."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from calculo_vectorial import direccion_tangente, gradiente_numerico, normalizar_vector

FuncionCosto: TypeAlias = Callable[[float, float], float]
Registro: TypeAlias = Mapping[str, object]

_PUNTOS_MALLA = 7
_PASO_GRADIENTE = 0.001


def _crear_malla(funcion_costo: FuncionCosto) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evalúa la función en una malla regular de 7 por 7 dentro del dominio."""
    coordenadas = np.linspace(0.0, 100.0, _PUNTOS_MALLA)
    malla_x, malla_y = np.meshgrid(coordenadas, coordenadas)
    try:
        costos = np.asarray([
            [funcion_costo(float(x), float(y)) for x in coordenadas]
            for y in coordenadas
        ], dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("La función de costo debe producir valores numéricos en la malla") from exc
    if malla_x.shape != malla_y.shape or costos.shape != malla_x.shape:
        raise ValueError("Las dimensiones de la malla y los costos no coinciden")
    if not np.all(np.isfinite(costos)):
        raise ValueError("La función de costo produjo valores NaN o infinitos en la malla")
    return malla_x, malla_y, costos


def _validar_historial(historial: Sequence[Registro], *, requiere_posicion: bool) -> tuple[np.ndarray, ...]:
    """Extrae y valida los costos y, si se necesitan, las coordenadas."""
    if isinstance(historial, (str, bytes)) or not isinstance(historial, Sequence):
        raise TypeError("historial debe ser una secuencia de registros")
    if not historial:
        vacio = np.asarray([], dtype=float)
        return (vacio, vacio.copy(), vacio.copy()) if requiere_posicion else (vacio,)

    xs: list[float] = []
    ys: list[float] = []
    costos: list[float] = []
    try:
        for registro in historial:
            costo = float(registro["costo"])
            costos.append(costo)
            if requiere_posicion:
                xs.append(float(registro["x"]))
                ys.append(float(registro["y"]))
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Cada registro del historial debe contener valores numéricos válidos") from exc
    arrays = [np.asarray(costos, dtype=float)]
    if requiere_posicion:
        arrays.extend((np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)))
    if any(not np.all(np.isfinite(valores)) for valores in arrays):
        raise ValueError("El historial no puede contener valores NaN o infinitos")
    if requiere_posicion and (
        np.any((arrays[1] < 0) | (arrays[1] > 100))
        or np.any((arrays[2] < 0) | (arrays[2] > 100))
    ):
        raise ValueError("Las coordenadas del historial deben estar en [0, 100]")
    return tuple(arrays)


def _validar_punto(punto: Sequence[float], nombre: str = "p0") -> tuple[float, float]:
    try:
        componentes = np.asarray(punto, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{nombre} debe contener dos coordenadas finitas") from exc
    if componentes.shape != (2,) or not np.all(np.isfinite(componentes)):
        raise ValueError(f"{nombre} debe contener dos coordenadas finitas")
    x, y = map(float, componentes)
    if not (0 <= x <= 100 and 0 <= y <= 100):
        raise ValueError(f"{nombre} debe estar en [0, 100]")
    return x, y


def _punto_inicial(
    p0: Sequence[float] | None, historial: Sequence[Registro], xs: np.ndarray, ys: np.ndarray
) -> tuple[float, float] | None:
    if p0 is not None:
        return _validar_punto(p0)
    if len(historial):
        return float(xs[0]), float(ys[0])
    return None


def graficar_superficie(
    funcion_costo: FuncionCosto,
    historial: Sequence[Registro],
    p0: Sequence[float] | None = None,
    titulo: str = "Superficie 3D de Costo y Trayectoria",
) -> Figure:
    """Dibuja la superficie de costo y la trayectoria registrada."""
    if not callable(funcion_costo):
        raise TypeError("funcion_costo debe ser invocable")
    malla_x, malla_y, costos = _crear_malla(funcion_costo)
    costos_hist, xs, ys = _validar_historial(historial, requiere_posicion=True)
    punto_inicial = _punto_inicial(p0, historial, xs, ys)
    costo_inicial = None
    if punto_inicial is not None:
        costo_inicial = float(funcion_costo(*punto_inicial))
        if not np.isfinite(costo_inicial):
            raise ValueError("La función de costo produjo un valor NaN o infinito en p0")

    figura = plt.figure(figsize=(10, 7))
    eje = figura.add_subplot(111, projection="3d")
    superficie = eje.plot_surface(
        malla_x, malla_y, costos, cmap="viridis", alpha=0.78,
        linewidth=0, antialiased=True,
    )
    figura.colorbar(superficie, ax=eje, shrink=0.65, pad=0.1, label="Costo C_beta(x,y)")
    if len(costos_hist):
        eje.plot(xs, ys, costos_hist, color="black", marker="o", linewidth=2.2,
                 markersize=4, label="Trayectoria")
        eje.scatter(xs[-1], ys[-1], costos_hist[-1], color="green", s=65,
                    label="Punto final", depthshade=False)
    if punto_inicial is not None and costo_inicial is not None:
        eje.scatter(*punto_inicial, costo_inicial, color="red", s=65,
                    label="P0", depthshade=False)
    eje.set_title(titulo)
    eje.set_xlabel("CPU (%)", labelpad=10)
    eje.set_ylabel("Ancho de Banda (%)", labelpad=10)
    eje.set_zlabel("Costo C_beta(x,y)", labelpad=8)
    if len(costos_hist) or punto_inicial is not None:
        eje.legend(loc="best")
    figura.tight_layout()
    return figura


def graficar_curvas_nivel(
    funcion_costo: FuncionCosto,
    historial: Sequence[Registro],
    p0: Sequence[float] | None = None,
    vector_v: Sequence[float] | None = None,
    titulo: str = "Curvas de Nivel y Campo Gradiente",
) -> Figure:
    """Dibuja curvas de nivel, gradiente, trayectoria y direcciones locales."""
    if not callable(funcion_costo):
        raise TypeError("funcion_costo debe ser invocable")
    malla_x, malla_y, costos = _crear_malla(funcion_costo)
    _, xs, ys = _validar_historial(historial, requiere_posicion=True)
    punto_inicial = _punto_inicial(p0, historial, xs, ys)
    gradientes_x = np.empty_like(malla_x)
    gradientes_y = np.empty_like(malla_y)
    try:
        for fila in range(malla_x.shape[0]):
            for columna in range(malla_x.shape[1]):
                gradiente = gradiente_numerico(
                    funcion_costo, float(malla_x[fila, columna]),
                    float(malla_y[fila, columna]), _PASO_GRADIENTE,
                )
                gradientes_x[fila, columna], gradientes_y[fila, columna] = gradiente
    except (TypeError, ValueError, ArithmeticError, OverflowError) as exc:
        raise ValueError("No se pudo calcular un campo de gradientes finito") from exc
    if (
        gradientes_x.shape != costos.shape
        or gradientes_y.shape != costos.shape
        or not np.all(np.isfinite(gradientes_x))
        or not np.all(np.isfinite(gradientes_y))
    ):
        raise ValueError("El campo de gradientes tiene dimensiones inválidas o valores no finitos")

    tangente = None
    if punto_inicial is not None:
        try:
            gradiente_inicial = gradiente_numerico(
                funcion_costo, *punto_inicial, _PASO_GRADIENTE
            )
        except (TypeError, ValueError, ArithmeticError, OverflowError) as exc:
            raise ValueError("No se pudo calcular el gradiente en p0") from exc
        norma_gradiente = float(np.linalg.norm(gradiente_inicial))
        if not np.isfinite(norma_gradiente):
            raise ValueError("El gradiente en p0 no es finito")
        if norma_gradiente > 0:
            tangente = direccion_tangente(gradiente_inicial)
    direccion = None
    if vector_v is not None:
        try:
            direccion = normalizar_vector(vector_v)
        except (TypeError, ValueError, ArithmeticError) as exc:
            raise ValueError("vector_v debe ser un vector bidimensional no nulo") from exc
        if punto_inicial is None:
            raise ValueError("Se necesita p0 o un historial no vacío para dibujar vector_v")

    figura, eje = plt.subplots(figsize=(9, 7))
    relleno = eje.contourf(malla_x, malla_y, costos, levels=20, cmap="viridis")
    eje.contour(malla_x, malla_y, costos, levels=10, colors="white",
                linewidths=0.65, alpha=0.75)
    figura.colorbar(relleno, ax=eje, label="Costo C_beta(x,y)")
    eje.quiver(malla_x, malla_y, gradientes_x, gradientes_y, color="white",
               alpha=0.8, pivot="mid", label="Gradiente")
    if len(xs):
        eje.plot(xs, ys, color="black", marker="o", markersize=4,
                 linewidth=2, label="Trayectoria")
    if punto_inicial is not None:
        eje.scatter(*punto_inicial, color="red", s=65, zorder=5, label="P0")
    if tangente is not None and punto_inicial is not None:
        eje.quiver(*punto_inicial, *(tangente * 12.0), angles="xy",
                   scale_units="xy", scale=1, color="orange", width=0.009,
                   label="Tangente local")
    if direccion is not None and punto_inicial is not None:
        eje.quiver(*punto_inicial, *(direccion * 12.0), angles="xy",
                   scale_units="xy", scale=1, color="cyan", width=0.009,
                   label="Dirección unitaria")
    eje.set_title(titulo)
    eje.set_xlabel("CPU (%)")
    eje.set_ylabel("Ancho de Banda (%)")
    eje.set_xlim(0, 100)
    eje.set_ylim(0, 100)
    eje.grid(alpha=0.2)
    eje.legend(loc="best")
    figura.tight_layout()
    return figura


def _datos_convergencia(historial: Sequence[Registro]) -> tuple[np.ndarray, np.ndarray, Registro]:
    if isinstance(historial, (str, bytes)) or not isinstance(historial, Sequence) or not historial:
        raise ValueError("Cada historial de convergencia debe contener al menos un registro")
    costos, = _validar_historial(historial, requiere_posicion=False)
    iteraciones: list[float] = []
    try:
        for indice, registro in enumerate(historial):
            iteraciones.append(float(registro.get("iteracion", indice)))
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Las iteraciones deben ser numéricas") from exc
    valores_iteracion = np.asarray(iteraciones, dtype=float)
    if not np.all(np.isfinite(valores_iteracion)):
        raise ValueError("Las iteraciones no pueden contener valores NaN o infinitos")
    return valores_iteracion, costos, historial[-1]


def _motivo_final(registro: Registro) -> tuple[bool, str]:
    motivo = str(registro.get("motivo_terminacion", ""))
    convergio = bool(registro.get("convergio", motivo == "convergencia"))
    return convergio, motivo or ("convergencia" if convergio else "sin motivo indicado")


def graficar_convergencia(
    historial_alpha1: Sequence[Registro],
    historial_alpha2: Sequence[Registro],
    label_alpha1: str = "alpha=0.2",
    label_alpha2: str = "alpha=0.5",
    titulo: str = "Comparación de Convergencia",
) -> Figure:
    """Compara el costo por iteración y marca el estado de terminación."""
    iteraciones_1, costos_1, final_1 = _datos_convergencia(historial_alpha1)
    iteraciones_2, costos_2, final_2 = _datos_convergencia(historial_alpha2)
    figura, eje = plt.subplots(figsize=(9, 6))
    eje.plot(iteraciones_1, costos_1, marker="o", markersize=3,
             linewidth=1.8, label=label_alpha1)
    eje.plot(iteraciones_2, costos_2, marker="s", markersize=3,
             linewidth=1.8, label=label_alpha2)
    for iteraciones, costos, registro, etiqueta, color in (
        (iteraciones_1, costos_1, final_1, label_alpha1, "tab:green"),
        (iteraciones_2, costos_2, final_2, label_alpha2, "tab:purple"),
    ):
        convergio, motivo = _motivo_final(registro)
        color = "tab:green" if convergio else "tab:red"
        eje.scatter(iteraciones[-1], costos[-1], marker="o" if convergio else "X",
                    s=90, color=color, edgecolors="white", linewidths=0.8,
                    zorder=5, label=f"Final {etiqueta}: {motivo}")
    eje.set_title(titulo)
    eje.set_xlabel("Iteración")
    eje.set_ylabel("Costo C_beta(x,y)")
    eje.grid(True, alpha=0.3)
    eje.legend()
    figura.tight_layout()
    return figura


def generar_todas_las_figuras(
    funcion_costo: FuncionCosto,
    historial_a1: Sequence[Registro],
    historial_a2: Sequence[Registro],
    p0: Sequence[float],
    vector_v: Sequence[float],
) -> dict[str, Figure]:
    """Genera las tres figuras con las claves requeridas por la interfaz."""
    return {
        "superficie": graficar_superficie(funcion_costo, historial_a1, p0),
        "curvas": graficar_curvas_nivel(funcion_costo, historial_a1, p0, vector_v),
        "convergencia": graficar_convergencia(historial_a1, historial_a2),
    }


def exportar_figura(figura: Figure, destino: Path | str, dpi: int = 300) -> Path:
    """Guarda una figura en PNG a la resolución indicada y devuelve su ruta."""
    if not isinstance(figura, Figure):
        raise TypeError("figura debe ser una instancia de matplotlib.figure.Figure")
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("dpi debe ser un entero positivo")
    ruta = Path(destino)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ruta, dpi=dpi, bbox_inches="tight", format="png")
    return ruta


def exportar_todas_figuras(
    figuras: dict[str, Figure], carpeta_destino: Path | str, dpi: int = 300
) -> dict[str, Path]:
    """Exporta las figuras conocidas a archivos PNG con nombres estables."""
    nombres = {
        "superficie": "superficie_3d.png",
        "curvas": "curvas_nivel.png",
        "convergencia": "convergencia.png",
    }
    faltantes = nombres.keys() - figuras.keys()
    if faltantes:
        raise ValueError(f"Faltan figuras requeridas: {', '.join(sorted(faltantes))}")
    carpeta = Path(carpeta_destino)
    return {
        clave: exportar_figura(figuras[clave], carpeta / nombre, dpi)
        for clave, nombre in nombres.items()
    }
