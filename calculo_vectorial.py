"""Modelo sintético y derivadas numéricas para Green Cloud.

Las coordenadas representan porcentajes y pertenecen a [0, 100]. En el
interior se usan diferencias centradas. Si el paso sale del dominio, se
usa una diferencia unilateral de segundo orden hacia el lado que admita
los puntos x+h y x+2h (o sus equivalentes negativos). Si ninguna fórmula
dispone de puntos válidos, se lanza ValueError; nunca se recorta el punto.
"""

from math import isfinite
from numbers import Real

import numpy as np


def _real_finito(valor: object, nombre: str) -> float:
    if isinstance(valor, (bool, np.bool_)) or not isinstance(
        valor, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{nombre} debe ser un número real finito")
    try:
        resultado = float(valor)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{nombre} debe ser finito") from exc
    if not isfinite(resultado):
        raise ValueError(f"{nombre} debe ser finito")
    return resultado


def _coordenadas(x: object, y: object) -> tuple[float, float]:
    x = _real_finito(x, "x")
    y = _real_finito(y, "y")
    if not (0 <= x <= 100 and 0 <= y <= 100):
        raise ValueError("x e y deben estar en [0, 100]")
    return x, y


def _beta(beta: object) -> float:
    beta = _real_finito(beta, "beta")
    if beta < 0:
        raise ValueError("beta debe ser no negativo")
    return beta


def _resultado(valor: object) -> float:
    return _real_finito(valor, "el resultado de la función")


def costo_operativo(x: float, y: float) -> float:
    """Costo operativo académico C_elec(x, y)."""
    x, y = _coordenadas(x, y)
    return _resultado(1000 + 0.3*x*x + 0.5*y*y + 0.1*x*y - 20*x - 32*y)


def indicador_ambiental(x: float, y: float) -> float:
    """Indicador ambiental sintético E(x, y)."""
    x, y = _coordenadas(x, y)
    return _resultado(0.2*x*x + 0.3*y*y + 0.1*x*y)


def costo_objetivo(x: float, y: float, beta: float = 1.0) -> float:
    """Costo ponderado C_beta = C_elec + beta E."""
    x, y = _coordenadas(x, y)
    beta = _beta(beta)
    return _resultado(costo_operativo(x, y) + beta*indicador_ambiental(x, y))


def derivada_parcial(funcion, x: float, y: float, variable: str, h: float) -> float:
    """Derivada parcial centrada o unilateral de segundo orden en [0, 100]."""
    if not callable(funcion):
        raise TypeError("funcion debe ser invocable")
    x, y = _coordenadas(x, y)
    if not isinstance(variable, str) or variable not in ("x", "y"):
        raise ValueError("variable debe ser 'x' o 'y'")
    h = _real_finito(h, "h")
    if h <= 0:
        raise ValueError("h debe ser positivo")

    punto = x if variable == "x" else y

    def evaluar(desplazamiento: float) -> float:
        coordenada = punto + desplazamiento
        if not 0 <= coordenada <= 100:
            raise ValueError("la evaluación excede el dominio")
        argumentos = (coordenada, y) if variable == "x" else (x, coordenada)
        return _resultado(funcion(*argumentos))

    if punto - h >= 0 and punto + h <= 100:
        if punto - h == punto or punto + h == punto:
            raise ValueError("h es demasiado pequeño para distinguir puntos")
        valor = (evaluar(h) - evaluar(-h)) / (2*h)
    elif punto + 2*h <= 100:
        if punto + h == punto or punto + 2*h == punto + h:
            raise ValueError("h es demasiado pequeño para distinguir puntos")
        valor = (-3*evaluar(0) + 4*evaluar(h) - evaluar(2*h)) / (2*h)
    elif punto - 2*h >= 0:
        if punto - h == punto or punto - 2*h == punto - h:
            raise ValueError("h es demasiado pequeño para distinguir puntos")
        valor = (3*evaluar(0) - 4*evaluar(-h) + evaluar(-2*h)) / (2*h)
    else:
        raise ValueError("h no permite una diferencia válida dentro del dominio")
    return _resultado(valor)


def gradiente_numerico(funcion, x: float, y: float, h: float) -> np.ndarray:
    """Gradiente numérico [df/dx, df/dy] a partir de derivadas parciales."""
    return np.array([
        derivada_parcial(funcion, x, y, "x", h),
        derivada_parcial(funcion, x, y, "y", h),
    ], dtype=float)


def gradiente_analitico(x: float, y: float, beta: float = 1.0) -> np.ndarray:
    """Referencia independiente del gradiente de C_beta."""
    x, y = _coordenadas(x, y)
    beta = _beta(beta)
    return np.array([
        _resultado((0.6 + 0.4*beta)*x + (0.1 + 0.1*beta)*y - 20),
        _resultado((0.1 + 0.1*beta)*x + (1.0 + 0.6*beta)*y - 32),
    ], dtype=float)
