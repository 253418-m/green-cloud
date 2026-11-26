"""Comprobación inicial de disponibilidad del módulo matemático."""

import importlib
import unittest

import numpy as np

from calculo_vectorial import (
    costo_objetivo,
    costo_operativo,
    descenso_gradiente,
    derivada_parcial,
    derivada_direccional,
    direccion_tangente,
    gradiente_analitico,
    gradiente_numerico,
    indicador_ambiental,
    normalizar_vector,
    verificar_acotacion,
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


class TestGeometriaVectorial(unittest.TestCase):
    def test_normalizar_tres_cuatro_y_conservar_entrada(self):
        vector = np.array([3.0, 4.0])
        copia = vector.copy()
        unitario = normalizar_vector(vector)
        self.assertIsInstance(unitario, np.ndarray)
        self.assertEqual(unitario.shape, (2,))
        np.testing.assert_allclose(unitario, [0.6, 0.8], atol=1e-15, rtol=0)
        self.assertAlmostEqual(np.linalg.norm(unitario), 1.0, delta=1e-15)
        np.testing.assert_array_equal(vector, copia)

    def test_normalizar_negativos_y_magnitudes_distintas(self):
        for vector, esperado in (([-3, -4], [-0.6, -0.8]),
                                 ([30, 40], [0.6, 0.8]),
                                 ([1e-300, 0], [1, 0]),
                                 ([1e308, 0], [1, 0])):
            with self.subTest(vector=vector):
                np.testing.assert_allclose(normalizar_vector(vector), esperado,
                                           atol=1e-15, rtol=0)

    def test_normalizar_rechaza_vectores_invalidos(self):
        for vector in ((0, 0), (float("nan"), 1), (float("inf"), 1),
                       (1,), (1, 2, 3), [[1], [2]], (True, 2),
                       (1, "dos"), (1.5e308, 1.5e308)):
            with self.subTest(vector=vector), self.assertRaises((TypeError, ValueError)):
                normalizar_vector(vector)

    def test_derivada_direccional_principal(self):
        resultado = derivada_direccional(costo_objetivo, (80, 70), (3, 4))
        self.assertIsInstance(resultado, float)
        self.assertAlmostEqual(resultado, 121.2, delta=1e-6)

    def test_derivada_direccional_opuesta_y_escalada(self):
        for vector, esperado in (((-3, -4), -121.2), ((30, 40), 121.2)):
            with self.subTest(vector=vector):
                self.assertAlmostEqual(
                    derivada_direccional(costo_objetivo, (80, 70), vector),
                    esperado, delta=1e-6,
                )

    def test_derivada_direccional_beta_dos(self):
        funcion = lambda x, y: costo_objetivo(x, y, beta=2)
        esperado = np.dot(gradiente_analitico(80, 70, beta=2), [0.6, 0.8])
        self.assertAlmostEqual(
            derivada_direccional(funcion, (80, 70), (3, 4)),
            esperado, delta=1e-6,
        )

    def test_derivada_usa_gradiente_numerico_y_h_configurable(self):
        funcion = lambda x, y: x**3 + y
        for h in (0.1, 0.2):
            with self.subTest(h=h):
                # Para x**3, la diferencia centrada en x=2 es 12+h**2.
                self.assertAlmostEqual(
                    derivada_direccional(funcion, (2, 20), (1, 0), h=h),
                    12 + h*h, delta=1e-10,
                )

    def test_derivada_direccional_en_fronteras(self):
        for punto in ((0, 100), (100, 0)):
            with self.subTest(punto=punto):
                esperado = np.dot(gradiente_analitico(*punto), [0.6, 0.8])
                self.assertAlmostEqual(
                    derivada_direccional(costo_objetivo, punto, (3, 4)),
                    esperado, delta=1e-6,
                )

    def test_derivada_direccional_rechaza_entradas_invalidas(self):
        casos = ((1, (80, 70), (3, 4), 0.001),
                 (costo_objetivo, (80, 70), (0, 0), 0.001),
                 (costo_objetivo, (101, 70), (3, 4), 0.001),
                 (costo_objetivo, (80,), (3, 4), 0.001),
                 (costo_objetivo, (80, 70), (3, 4), 0))
        for f, punto, vector, h in casos:
            with self.subTest(punto=punto, vector=vector, h=h):
                with self.assertRaises((TypeError, ValueError)):
                    derivada_direccional(f, punto, vector, h=h)
        with self.assertRaises(TypeError):
            derivada_direccional(costo_objetivo, (80, 70), (3, 4), 0.01)

    def test_cota_superior_inferior_y_perpendicular(self):
        g = np.array([74.0, 96.0])
        for direccion, esperado in ((g, np.linalg.norm(g)),
                                    (-g, -np.linalg.norm(g)),
                                    ([-96, 74], 0)):
            with self.subTest(direccion=direccion):
                u = normalizar_vector(direccion)
                self.assertAlmostEqual(np.linalg.norm(u), 1.0, delta=1e-15)
                self.assertAlmostEqual(np.dot(g, u), esperado, delta=1e-12)
                self.assertTrue(verificar_acotacion(g, direccion))

    def test_cota_gradiente_nulo_y_entradas_invalidas(self):
        self.assertTrue(verificar_acotacion((0, 0), (3, 4)))
        for g, direccion in (((float("nan"), 1), (3, 4)),
                             ((1, float("inf")), (3, 4)),
                             ((1, 2), (0, 0)),
                             ((1,), (3, 4))):
            with self.subTest(g=g, direccion=direccion):
                with self.assertRaises((TypeError, ValueError)):
                    verificar_acotacion(g, direccion)
        with self.assertRaises(ValueError):
            verificar_acotacion((1, 2), (3, 4), tolerancia=-1)

    def test_tangente_referencia_y_ortogonalidad(self):
        g = gradiente_numerico(costo_objetivo, 80, 70, 0.001)
        tangente = direccion_tangente(g)
        esperado = np.array([-96, 74]) / np.hypot(74, 96)
        self.assertIsInstance(tangente, np.ndarray)
        self.assertEqual(tangente.shape, (2,))
        np.testing.assert_allclose(tangente, esperado, atol=1e-10, rtol=0)
        self.assertAlmostEqual(np.linalg.norm(tangente), 1.0, delta=1e-12)
        self.assertAlmostEqual(np.dot(g, tangente), 0.0, delta=1e-12)
        self.assertAlmostEqual(
            derivada_direccional(costo_objetivo, (80, 70), tangente),
            0.0, delta=1e-6,
        )

    def test_tangente_rechaza_gradiente_invalido(self):
        for g in ((0, 0), (float("nan"), 1), (1, float("inf")), (1,)):
            with self.subTest(g=g), self.assertRaises((TypeError, ValueError)):
                direccion_tangente(g)

    def test_direcciones_aleatorias_reproducibles(self):
        rng = np.random.default_rng(2026)
        g = gradiente_numerico(costo_objetivo, 80, 70, 0.001)
        norma_g = np.hypot(*g)
        for vector in rng.normal(size=(100, 2)):
            u = normalizar_vector(vector)
            derivada = np.dot(g, u)
            self.assertAlmostEqual(np.linalg.norm(u), 1.0, delta=1e-14)
            self.assertLessEqual(abs(derivada), norma_g + 1e-10)
            self.assertTrue(verificar_acotacion(g, vector))


class TestDescensoGradiente(unittest.TestCase):
    def test_parametros_invalidos(self):
        for funcion, inicio, alpha, h, max_iter, tolerancia in (
            (None, (80, 70), 0.2, 0.001, 100, 1e-6),
            (costo_objetivo, (101, 70), 0.2, 0.001, 100, 1e-6),
            (costo_objetivo, (80,), 0.2, 0.001, 100, 1e-6),
            (costo_objetivo, (True, 70), 0.2, 0.001, 100, 1e-6),
            (costo_objetivo, (80, 70), 0, 0.001, 100, 1e-6),
            (costo_objetivo, (80, 70), -0.2, 0.001, 100, 1e-6),
            (costo_objetivo, (80, 70), float("inf"), 0.001, 100, 1e-6),
            (costo_objetivo, (80, 70), True, 0.001, 100, 1e-6),
            (costo_objetivo, (80, 70), 0.2, 0, 100, 1e-6),
            (costo_objetivo, (80, 70), 0.2, float("nan"), 100, 1e-6),
            (costo_objetivo, (80, 70), 0.2, 0.001, 100, 0),
            (costo_objetivo, (80, 70), 0.2, 0.001, 100, float("inf")),
            (costo_objetivo, (80, 70), 0.2, 0.001, 0, 1e-6),
            (costo_objetivo, (80, 70), 0.2, 0.001, 1.5, 1e-6),
            (costo_objetivo, (80, 70), 0.2, 0.001, True, 1e-6),
        ):
            with self.subTest(inicio=inicio, alpha=alpha, h=h,
                              max_iter=max_iter, tolerancia=tolerancia):
                with self.assertRaises((TypeError, ValueError)):
                    descenso_gradiente(funcion, inicio, alpha, h, max_iter, tolerancia)

    def test_iteracion_cero_y_primera_actualizacion(self):
        inicio = np.array([80.0, 70.0])
        copia = inicio.copy()
        for alpha, esperado in ((0.2, (65.2, 50.8)), (0.5, (43, 22))):
            with self.subTest(alpha=alpha):
                historial = descenso_gradiente(costo_objetivo, inicio, alpha,
                                               0.001, 1, 1e-6)
                self.assertEqual(len(historial), 2)
                self.assertEqual(historial[0]["iteracion"], 0)
                self.assertEqual(historial[0]["costo"], 5400.0)
                self.assertAlmostEqual(historial[0]["norma_gradiente"],
                                       np.hypot(74, 96), delta=1e-6)
                np.testing.assert_allclose(
                    [historial[1]["x"], historial[1]["y"]], esperado,
                    atol=1e-8, rtol=0,
                )
                self.assertEqual(historial[-1]["motivo_terminacion"],
                                 "max_iteraciones")
                self.assertFalse(historial[-1]["convergio"])
        np.testing.assert_array_equal(inicio, copia)

    def test_historial_coherente_y_pasos_sucesivos(self):
        alpha = 0.2
        historial = descenso_gradiente(costo_objetivo, (80, 70), alpha,
                                       0.001, 1000, 1e-6)
        self.assertLessEqual(len(historial), 1001)
        for k, registro in enumerate(historial):
            self.assertEqual(registro["iteracion"], k)
            self.assertTrue(0 <= registro["x"] <= 100)
            self.assertTrue(0 <= registro["y"] <= 100)
            self.assertAlmostEqual(registro["costo"],
                                   costo_objetivo(registro["x"], registro["y"]),
                                   delta=1e-9)
            gradiente = gradiente_numerico(costo_objetivo, registro["x"],
                                          registro["y"], 0.001)
            self.assertAlmostEqual(registro["norma_gradiente"],
                                   np.hypot(*gradiente), delta=1e-10)
            if k:
                anterior = historial[k-1]
                gradiente_anterior = gradiente_numerico(
                    costo_objetivo, anterior["x"], anterior["y"], 0.001,
                )
                np.testing.assert_allclose(
                    [registro["x"], registro["y"]],
                    np.array([anterior["x"], anterior["y"]])
                    - alpha*gradiente_anterior, atol=1e-12, rtol=0,
                )
                self.assertNotEqual((registro["x"], registro["y"]),
                                    (anterior["x"], anterior["y"]))
        self.assertEqual(set(historial[-1]) - set(historial[-2]),
                         {"motivo_terminacion", "convergio"})

    def test_convergencia_dos_tasas_y_referencias(self):
        finales = []
        for alpha in (0.2, 0.5):
            with self.subTest(alpha=alpha):
                historial = descenso_gradiente(costo_objetivo, (80, 70), alpha,
                                               0.001, 1000, 1e-6)
                final = historial[-1]
                self.assertEqual(final["motivo_terminacion"], "convergencia")
                self.assertTrue(final["convergio"])
                self.assertLessEqual(final["norma_gradiente"], 1e-6)
                np.testing.assert_allclose(
                    [final["x"], final["y"]], [640/39, 700/39],
                    atol=2e-6, rtol=0,
                )
                self.assertAlmostEqual(final["costo"], 21400/39,
                                       delta=1e-8)
                finales.append(final)
        self.assertNotEqual(finales[0]["iteracion"], finales[1]["iteracion"])
        np.testing.assert_allclose(
            [finales[0]["x"], finales[0]["y"]],
            [finales[1]["x"], finales[1]["y"]], atol=2e-7, rtol=0,
        )

    def test_punto_inicial_ya_convergido(self):
        minimo = (640/39, 700/39)
        historial = descenso_gradiente(costo_objetivo, minimo, 0.2,
                                       0.001, 100, 1e-6)
        self.assertEqual(len(historial), 1)
        self.assertEqual(historial[0]["iteracion"], 0)
        self.assertEqual(historial[0]["motivo_terminacion"], "convergencia")
        self.assertTrue(historial[0]["convergio"])

    def test_salida_dominio_no_evalua_punto_propuesto(self):
        puntos = []

        def funcion(x, y):
            puntos.append((x, y))
            return costo_objetivo(x, y)

        historial = descenso_gradiente(funcion, (80, 70), 1.0,
                                       0.001, 100, 1e-6)
        self.assertEqual(len(historial), 1)
        self.assertEqual(historial[0]["motivo_terminacion"], "fuera_dominio")
        self.assertFalse(historial[0]["convergio"])
        self.assertTrue(all(0 <= x <= 100 and 0 <= y <= 100 for x, y in puntos))
        self.assertTrue(all(0 <= r["x"] <= 100 and 0 <= r["y"] <= 100
                            for r in historial))

    def test_divergencia_por_crecimiento_persistente(self):
        funcion = lambda x, y: (x-50)**2
        historial = descenso_gradiente(funcion, (51, 50), 1.5,
                                       0.001, 100, 1e-6)
        self.assertEqual(historial[-1]["motivo_terminacion"], "divergencia")
        self.assertFalse(historial[-1]["convergio"])
        self.assertEqual(historial[-1]["iteracion"], 3)
        self.assertTrue(all(0 <= r["x"] <= 100 for r in historial))
        self.assertTrue(all(historial[k]["costo"] > historial[k-1]["costo"]
                            for k in range(1, len(historial))))

    def test_errores_numericos_explicitos(self):
        with self.assertRaises(ValueError):
            descenso_gradiente(lambda x, y: float("nan"), (80, 70),
                               0.2, 0.001, 100, 1e-6)
        with self.assertRaises(ValueError):
            descenso_gradiente(lambda x, y: float("inf") if x < 70 else x*x+y,
                               (80, 70), 0.2, 0.001, 100, 1e-6)
        with self.assertRaises(ArithmeticError):
            descenso_gradiente(costo_objetivo, (80, 70), 1e308,
                               0.001, 100, 1e-6)

    def test_gradiente_numerico_se_usa_para_el_paso(self):
        funcion = lambda x, y: x**3 + y
        historial = descenso_gradiente(funcion, (2, 20), 0.1,
                                       0.1, 1, 1e-6)
        self.assertAlmostEqual(historial[1]["x"], 2 - 0.1*12.01,
                               delta=1e-10)
        self.assertAlmostEqual(historial[1]["y"], 19.9, delta=1e-10)

    def test_beta_parametrizado_con_referencia_propia(self):
        funcion = lambda x, y: costo_objetivo(x, y, beta=2)
        historial = descenso_gradiente(funcion, (80, 70), 0.2,
                                       0.001, 1000, 1e-6)
        final = historial[-1]
        esperado = np.linalg.solve([[1.4, 0.3], [0.3, 2.2]], [20, 32])
        self.assertEqual(final["motivo_terminacion"], "convergencia")
        np.testing.assert_allclose([final["x"], final["y"]], esperado,
                                   atol=2e-6, rtol=0)
        self.assertAlmostEqual(final["costo"], funcion(*esperado), delta=1e-8)

    def test_reproducibilidad(self):
        argumentos = (costo_objetivo, (80, 70), 0.5, 0.001, 1000, 1e-6)
        self.assertEqual(descenso_gradiente(*argumentos),
                         descenso_gradiente(*argumentos))
