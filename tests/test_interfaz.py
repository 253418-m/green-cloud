"""Validaciones de la GUI que funcionan también sin pantalla."""

import importlib
import unittest

from interfaz import Parametros, textos_predeterminados, validar_parametros


class TestValidacionInterfaz(unittest.TestCase):
    def setUp(self) -> None:
        self.entradas = textos_predeterminados()

    def test_importacion_y_predeterminados(self) -> None:
        self.assertIsNotNone(importlib.import_module("interfaz"))
        self.assertEqual(validar_parametros(self.entradas), Parametros())
        self.assertEqual((self.entradas["x0"], self.entradas["y0"]), ("80.0", "70.0"))

    def test_limites_inclusivos_del_dominio(self) -> None:
        for campo in ("x0", "y0", "umbral_critico"):
            for valor in ("0", "100"):
                with self.subTest(campo=campo, valor=valor):
                    entradas = dict(self.entradas, **{campo: valor})
                    self.assertEqual(getattr(validar_parametros(entradas), campo), float(valor))

    def test_numeros_invalidos_y_no_finitos(self) -> None:
        etiquetas = {"tolerancia": "Tolerancia", "max_iteraciones": "Máximo de iteraciones",
                     "umbral_critico": "Umbral crítico"}
        for campo in self.entradas:
            for valor in ("", "abc", "NaN", "inf", "-inf"):
                with self.subTest(campo=campo, valor=valor):
                    entradas = dict(self.entradas, **{campo: valor})
                    with self.assertRaises(ValueError) as error:
                        validar_parametros(entradas)
                    self.assertIn(etiquetas.get(campo, campo), str(error.exception))

    def test_rangos_y_signos(self) -> None:
        casos = {
            "x0": ("-0.1", "100.1"), "y0": ("-1", "101"),
            "umbral_critico": ("-1", "101"), "h": ("0", "-1"),
            "alpha_1": ("0", "-0.1"), "alpha_2": ("0", "-1"),
            "tolerancia": ("0", "-1"), "beta": ("-0.1",),
            "max_iteraciones": ("0", "-1", "1.5"),
        }
        for campo, valores in casos.items():
            for valor in valores:
                with self.subTest(campo=campo, valor=valor):
                    with self.assertRaises(ValueError):
                        validar_parametros(dict(self.entradas, **{campo: valor}))

    def test_vector_nulo(self) -> None:
        with self.assertRaisesRegex(ValueError, "vector"):
            validar_parametros(dict(self.entradas, vx="0", vy="0"))

    def test_iteraciones_grandes_no_provocan_error_no_controlado(self) -> None:
        entradas = dict(self.entradas, max_iteraciones="9" * 400)
        self.assertEqual(validar_parametros(entradas).max_iteraciones, int("9" * 400))

    def test_copia_para_restablecimiento(self) -> None:
        copia = textos_predeterminados()
        copia["x0"] = "4"
        self.assertEqual(textos_predeterminados()["x0"], "80.0")


class TestVentanaSiHayPantalla(unittest.TestCase):
    def test_formulario_y_restablecimiento(self) -> None:
        try:
            import tkinter as tk
        except ImportError as error:
            self.skipTest(f"Tkinter no disponible: {error}")
        try:
            raiz = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Entorno gráfico no disponible: {error}")
        try:
            from interfaz import InterfazGreenCloud

            raiz.withdraw()
            app = InterfazGreenCloud(raiz)
            raiz.update_idletasks()
            self.assertEqual(len(app.variables), 11)
            self.assertEqual(len(app.valores_resultado), 12)
            self.assertEqual(len(app.marcos_figura), 3)
            app.variables["x0"].set("15")
            app.restablecer()
            self.assertEqual(app.variables["x0"].get(), "80.0")
            app.ejecutar()
            self.assertIn("pendiente", app.estado.get())
        finally:
            raiz.destroy()
