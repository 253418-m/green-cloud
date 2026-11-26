"""Modelo sintético y derivadas numéricas para Green Cloud.

Las coordenadas representan porcentajes y pertenecen a [0, 100]. En el
interior se usan diferencias centradas. Si el paso sale del dominio, se
usa una diferencia unilateral de segundo orden hacia el lado que admita
los puntos x+h y x+2h (o sus equivalentes negativos). Si ninguna fórmula
dispone de puntos válidos, se lanza ValueError; nunca se recorta el punto.
"""

from math import isfinite
from numbers import Integral, Real

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


def _vector_bidimensional(vector: object, nombre: str) -> np.ndarray:
    """Copia validada de dos componentes reales finitas."""
    try:
        componentes = np.asarray(vector, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{nombre} debe tener dos componentes") from exc
    if componentes.shape != (2,):
        raise ValueError(f"{nombre} debe tener dos componentes")
    return np.array([
        _real_finito(componentes[0], f"{nombre}[0]"),
        _real_finito(componentes[1], f"{nombre}[1]"),
    ], dtype=float)


def _norma(vector: np.ndarray) -> float:
    # hypot evita el desbordamiento intermedio al sumar cuadrados.
    with np.errstate(over="ignore"):
        norma = np.hypot(vector[0], vector[1])
    return _resultado(norma)


def normalizar_vector(vector) -> np.ndarray:
    """Devuelve una copia unitaria de un vector bidimensional no nulo."""
    componentes = _vector_bidimensional(vector, "vector")
    norma = _norma(componentes)
    if norma == 0:
        raise ValueError("vector no debe ser nulo")
    return componentes / norma


def derivada_direccional(f, punto, vector_v, *, h: float = 0.001) -> float:
    """Calcula gradiente_numerico(f, punto, h) · dirección unitaria."""
    if not callable(f):
        raise TypeError("f debe ser invocable")
    coordenadas = _vector_bidimensional(punto, "punto")
    x, y = _coordenadas(*coordenadas)
    direccion = normalizar_vector(vector_v)
    gradiente = gradiente_numerico(f, x, y, h)
    return _resultado(np.dot(gradiente, direccion))


def verificar_acotacion(gradiente, direccion, *, tolerancia: float = 1e-12) -> bool:
    """Comprueba -||gradiente|| <= gradiente · u <= ||gradiente||."""
    g = _vector_bidimensional(gradiente, "gradiente")
    u = normalizar_vector(direccion)
    tolerancia = _real_finito(tolerancia, "tolerancia")
    if tolerancia < 0:
        raise ValueError("tolerancia debe ser no negativa")
    norma = _norma(g)
    derivada = _resultado(np.dot(g, u))
    margen = tolerancia * max(1.0, norma)
    return -norma - margen <= derivada <= norma + margen


def direccion_tangente(gradiente) -> np.ndarray:
    """Dirección unitaria perpendicular a un gradiente no nulo."""
    g = _vector_bidimensional(gradiente, "gradiente")
    return normalizar_vector((-g[1], g[0]))


def descenso_gradiente(
    funcion, inicio, alpha, h, max_iter, tolerancia
) -> list[dict]:
    """Desciende con el gradiente numérico y devuelve puntos válidos.

    El historial inicia en iteración 0. Cada registro contiene iteracion,
    x, y, costo y norma_gradiente. El último añade motivo_terminacion y
    convergio. Los motivos son convergencia, max_iteraciones,
    fuera_dominio y divergencia. Una salida de [0,100]² no se evalúa ni
    registra. Se clasifica divergencia tras tres aumentos consecutivos
    del costo mayores que 1e-12 veces su escala, para ignorar redondeo.
    Resultados no finitos o pasos sin resolución provocan una excepción.
    """
    if not callable(funcion):
        raise TypeError("funcion debe ser invocable")
    punto = _vector_bidimensional(inicio, "inicio")
    x, y = _coordenadas(*punto)
    alpha = _real_finito(alpha, "alpha")
    h = _real_finito(h, "h")
    tolerancia = _real_finito(tolerancia, "tolerancia")
    if alpha <= 0 or h <= 0 or tolerancia <= 0:
        raise ValueError("alpha, h y tolerancia deben ser positivos")
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(
        max_iter, (Integral, np.integer)
    ) or max_iter <= 0:
        raise ValueError("max_iter debe ser un entero positivo")

    def evaluar(actual: np.ndarray, iteracion: int) -> tuple[dict, np.ndarray]:
        costo = _resultado(funcion(float(actual[0]), float(actual[1])))
        gradiente = gradiente_numerico(funcion, float(actual[0]), float(actual[1]), h)
        norma = _norma(gradiente)
        return {
            "iteracion": iteracion,
            "x": float(actual[0]),
            "y": float(actual[1]),
            "costo": costo,
            "norma_gradiente": norma,
        }, gradiente

    punto = np.array([x, y], dtype=float)
    registro, gradiente = evaluar(punto, 0)
    historial = [registro]
    aumentos_consecutivos = 0

    for iteracion in range(1, max_iter + 1):
        if registro["norma_gradiente"] <= tolerancia:
            motivo = "convergencia"
            break
        with np.errstate(over="ignore", invalid="ignore"):
            nuevo = punto - alpha * gradiente
        if not np.all(np.isfinite(nuevo)):
            raise ArithmeticError("el paso produjo coordenadas no finitas")
        if not np.all((0 <= nuevo) & (nuevo <= 100)):
            motivo = "fuera_dominio"
            break
        if np.array_equal(nuevo, punto):
            raise ArithmeticError("el paso no cambia el punto con esta precisión")

        nuevo_registro, nuevo_gradiente = evaluar(nuevo, iteracion)
        margen = 1e-12 * max(1.0, abs(registro["costo"]))
        if nuevo_registro["costo"] > registro["costo"] + margen:
            aumentos_consecutivos += 1
        else:
            aumentos_consecutivos = 0
        historial.append(nuevo_registro)
        punto, registro, gradiente = nuevo, nuevo_registro, nuevo_gradiente
        if registro["norma_gradiente"] <= tolerancia:
            motivo = "convergencia"
            break
        if aumentos_consecutivos >= 3:
            motivo = "divergencia"
            break
    else:
        motivo = "max_iteraciones"

    historial[-1]["motivo_terminacion"] = motivo
    historial[-1]["convergio"] = motivo == "convergencia"
    return historial
