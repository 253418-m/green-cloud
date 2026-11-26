"""Comprobación inicial de disponibilidad del módulo matemático."""

import importlib
import unittest

import numpy as np

from calculo_vectorial import (
    costo_objetivo,
    costo_operativo,
    derivada_parcial,
    gradiente_analitico,
    gradiente_numerico,
    indicador_ambiental,
)


class TestEstructuraCalculoVectorial(unittest.TestCase):
    def test_modulo_importable(self) -> None:
        self.assertIsNotNone(importlib.import_module("calculo_vectorial"))


class TestModeloMatematico(unittest.TestCase):
    def test_costo_operativo(self):
        self.assertAlmostEqual(costo_operativo(80, 70), 2090.0, places=9)

    def test_indicador_ambiental(self):
        self.assertAlmostEqual(indicador_ambiental(80, 70), 3310.0, places=9)

    def test_objetivo_beta_cero_uno_dos(self):
        for beta, esperado in ((0, 2090), (1, 5400), (2, 8710)):
            with self.subTest(beta=beta):
                self.assertAlmostEqual(costo_objetivo(80, 70, beta), esperado, places=9)

    def test_objetivo_origen(self):
        self.assertAlmostEqual(costo_objetivo(0, 0), 1000.0, places=9)

    def test_gradiente_analitico_p0(self):
        np.testing.assert_allclose(gradiente_analitico(80, 70), [74, 96], atol=1e-12)

    def test_gradiente_analitico_beta_configurable(self):
        np.testing.assert_allclose(gradiente_analitico(80, 70, 0), [35, 46], atol=1e-12)

    def test_referencia_analitica_del_minimo(self):
        x, y = 640/39, 700/39
        self.assertAlmostEqual(costo_objetivo(x, y), 21400/39, delta=1e-10)
        np.testing.assert_allclose(gradiente_analitico(x, y), [0, 0], atol=1e-12)

    def test_derivadas_parciales_p0(self):
        for variable, esperado in (("x", 74), ("y", 96)):
            with self.subTest(variable=variable):
                self.assertAlmostEqual(
                    derivada_parcial(costo_objetivo, 80, 70, variable, 0.001),
                    esperado, delta=1e-6,
                )

    def test_gradiente_numerico_p0_y_forma(self):
        gradiente = gradiente_numerico(costo_objetivo, 80, 70, 0.001)
        self.assertIsInstance(gradiente, np.ndarray)
        self.assertEqual(gradiente.shape, (2,))
        np.testing.assert_allclose(gradiente, [74, 96], atol=1e-6, rtol=0)

    def test_comparacion_analitica_y_numerica_con_beta(self):
        funcion = lambda x, y: costo_objetivo(x, y, beta=2)
        np.testing.assert_allclose(
            gradiente_numerico(funcion, 80, 70, 0.001),
            gradiente_analitico(80, 70, beta=2), atol=1e-6, rtol=0,
        )

    def test_cuatro_pasos(self):
        exacto = gradiente_analitico(80, 70)
        for h in (0.1, 0.01, 0.001, 0.0001):
            with self.subTest(h=h):
                np.testing.assert_allclose(
                    gradiente_numerico(costo_objetivo, 80, 70, h),
                    exacto, atol=1e-6, rtol=0,
                )

    def test_cerca_de_fronteras(self):
        for x, y in ((0.0001, 99.9999), (99.9999, 0.0001)):
            with self.subTest(x=x, y=y):
                np.testing.assert_allclose(
                    gradiente_numerico(costo_objetivo, x, y, 0.01),
                    gradiente_analitico(x, y), atol=1e-6, rtol=0,
                )

    def test_extremos_del_dominio(self):
        for x, y in ((0, 0), (0, 100), (100, 0), (100, 100)):
            with self.subTest(x=x, y=y):
                np.testing.assert_allclose(
                    gradiente_numerico(costo_objetivo, x, y, 0.01),
                    gradiente_analitico(x, y), atol=1e-6, rtol=0,
                )

    def test_prioridad_centrada_cuando_cabe(self):
        puntos = []

        def funcion(x, y):
            puntos.append(x)
            return x*x + y

        self.assertAlmostEqual(derivada_parcial(funcion, 0.1, 2, "x", 0.1), 0.2)
        self.assertEqual(len(puntos), 2)
        self.assertTrue(all(0 <= p <= 100 for p in puntos))

    def test_rechaza_coordenadas_invalidas(self):
        for x, y in ((-1, 5), (5, 101), (float("nan"), 5),
                     (5, float("inf")), (True, 5), ([1, 2], 5)):
            with self.subTest(x=x, y=y), self.assertRaises((TypeError, ValueError)):
                costo_objetivo(x, y)

    def test_rechaza_beta_invalido(self):
        for beta in (-1, float("nan"), float("inf"), True, [1]):
            with self.subTest(beta=beta), self.assertRaises((TypeError, ValueError)):
                costo_objetivo(10, 20, beta)

    def test_rechaza_h_invalido(self):
        for h in (0, -0.1, float("nan"), float("inf"), True, [0.1]):
            with self.subTest(h=h), self.assertRaises((TypeError, ValueError)):
                derivada_parcial(costo_objetivo, 10, 20, "x", h)

    def test_rechaza_funcion_no_invocable_y_variable_invalida(self):
        with self.assertRaises(TypeError):
            gradiente_numerico(4, 10, 20, 0.01)
        with self.assertRaises(ValueError):
            derivada_parcial(costo_objetivo, 10, 20, "z", 0.01)

    def test_rechaza_resultado_no_finito_o_no_escalar(self):
        for funcion in (lambda x, y: float("nan"),
                        lambda x, y: float("inf"),
                        lambda x, y: [x, y]):
            with self.subTest(funcion=funcion), self.assertRaises((TypeError, ValueError)):
                derivada_parcial(funcion, 10, 20, "x", 0.01)
        with self.assertRaises(ValueError):
            costo_objetivo(80, 70, 1e308)
        with self.assertRaises(ValueError):
            gradiente_analitico(80, 70, 1e308)

    def test_rechaza_paso_incompatible_con_dominio(self):
        with self.assertRaises(ValueError):
            derivada_parcial(costo_objetivo, 50, 50, "x", 60)
        with self.assertRaises(ValueError):
            derivada_parcial(costo_objetivo, 0, 0, "x", 51)

    def test_rechaza_paso_sin_resolucion(self):
        with self.assertRaises(ValueError):
            derivada_parcial(costo_objetivo, 50, 50, "x", 1e-20)
