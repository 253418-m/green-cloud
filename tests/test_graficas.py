"""Pruebas de generación y exportación de las gráficas científicas."""

import struct
import tempfile
import unittest
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from calculo_vectorial import costo_objetivo
from graficas import (
    exportar_figura,
    exportar_todas_figuras,
    generar_todas_las_figuras,
    graficar_convergencia,
    graficar_curvas_nivel,
    graficar_superficie,
)


def _historial(xy: tuple[tuple[float, float], ...], motivo: str = "convergencia") -> list[dict]:
    historial = [
        {"iteracion": indice, "x": x, "y": y, "costo": costo_objetivo(x, y)}
        for indice, (x, y) in enumerate(xy)
    ]
    historial[-1]["motivo_terminacion"] = motivo
    historial[-1]["convergio"] = motivo == "convergencia"
    return historial


def _dpi_png(ruta: Path) -> tuple[float, float]:
    """Lee la resolución física del chunk pHYs estándar de PNG."""
    datos = ruta.read_bytes()
    posicion = 8
    while posicion < len(datos):
        longitud = struct.unpack(">I", datos[posicion:posicion + 4])[0]
        tipo = datos[posicion + 4:posicion + 8]
        contenido = datos[posicion + 8:posicion + 8 + longitud]
        if tipo == b"pHYs":
            pixeles_x, pixeles_y, unidad = struct.unpack(">IIB", contenido)
            if unidad == 1:
                return pixeles_x * 0.0254, pixeles_y * 0.0254
        posicion += longitud + 12
    raise AssertionError("El PNG no contiene metadatos de resolución pHYs")


class TestGraficas(unittest.TestCase):
    def setUp(self) -> None:
        self.historial_1 = _historial(((80.0, 70.0), (65.0, 55.0), (50.0, 45.0)))
        self.historial_2 = _historial(
            ((80.0, 70.0), (60.0, 50.0), (45.0, 40.0)), "max_iteraciones"
        )
        self.p0 = (80.0, 70.0)
        self.vector_v = (3.0, 4.0)

    def _assert_figure(self, figura: Figure) -> None:
        self.assertIsInstance(figura, Figure)
        self.assertTrue(figura.axes)

    def _assert_300_dpi(self, ruta: Path) -> None:
        for dpi in _dpi_png(ruta):
            self.assertAlmostEqual(dpi, 300.0, delta=0.001)

    def test_graficar_superficie_crea_figura_con_ejes_etiquetados(self) -> None:
        figura = graficar_superficie(costo_objetivo, self.historial_1, self.p0)
        try:
            self._assert_figure(figura)
            eje = figura.axes[0]
            self.assertEqual(eje.get_xlabel(), "CPU (%)")
            self.assertEqual(eje.get_ylabel(), "Ancho de Banda (%)")
            self.assertEqual(eje.get_zlabel(), "Costo C_beta(x,y)")
            self.assertEqual(len(eje.lines[0].get_data_3d()[2]), len(self.historial_1))
            self.assertEqual(
                list(eje.lines[0].get_data_3d()[2]),
                [registro["costo"] for registro in self.historial_1],
            )
        finally:
            plt.close(figura)

    def test_graficar_curvas_nivel_crea_figura(self) -> None:
        figura = graficar_curvas_nivel(
            costo_objetivo, self.historial_1, self.p0, self.vector_v
        )
        try:
            self._assert_figure(figura)
            self.assertEqual(figura.axes[0].get_xlabel(), "CPU (%)")
            self.assertEqual(figura.axes[0].get_ylabel(), "Ancho de Banda (%)")
            self.assertEqual(len(figura.axes), 2)  # Eje principal y barra de color.
            etiquetas = [texto.get_text() for texto in figura.axes[0].get_legend().get_texts()]
            self.assertIn("Gradiente", etiquetas)
            self.assertIn("Dirección unitaria", etiquetas)
            self.assertIn("Tangente local", etiquetas)
        finally:
            plt.close(figura)

    def test_graficar_convergencia_crea_figura_y_marca_terminacion(self) -> None:
        figura = graficar_convergencia(self.historial_1, self.historial_2)
        try:
            self._assert_figure(figura)
            eje = figura.axes[0]
            self.assertEqual(eje.get_xlabel(), "Iteración")
            self.assertEqual(eje.get_ylabel(), "Costo C_beta(x,y)")
            etiquetas = [texto.get_text() for texto in eje.get_legend().get_texts()]
            self.assertTrue(any("convergencia" in etiqueta for etiqueta in etiquetas))
            self.assertTrue(any("max_iteraciones" in etiqueta for etiqueta in etiquetas))
        finally:
            plt.close(figura)

    def test_generar_todas_las_figuras_devuelve_claves_de_interfaz(self) -> None:
        figuras = generar_todas_las_figuras(
            costo_objetivo, self.historial_1, self.historial_2, self.p0, self.vector_v
        )
        try:
            self.assertEqual(set(figuras), {"superficie", "curvas", "convergencia"})
            for figura in figuras.values():
                self._assert_figure(figura)
        finally:
            for figura in figuras.values():
                plt.close(figura)

    def test_figuras_usan_dimensiones_compactas_para_pantalla(self) -> None:
        figuras = generar_todas_las_figuras(
            costo_objetivo, self.historial_1, self.historial_2, self.p0, self.vector_v
        )
        try:
            for figura in figuras.values():
                self.assertEqual(tuple(figura.get_size_inches()), (6.0, 3.6))
                self.assertEqual(figura.dpi, 85)
        finally:
            for figura in figuras.values():
                plt.close(figura)

    def test_exportar_figura_crea_directorios_y_png_a_300_dpi(self) -> None:
        figura = graficar_convergencia(self.historial_1, self.historial_2)
        try:
            with tempfile.TemporaryDirectory() as temporal:
                destino = Path(temporal) / "subcarpeta" / "prueba.png"
                resultado = exportar_figura(figura, destino)
                self.assertEqual(resultado, destino)
                self.assertTrue(destino.is_file())
                self._assert_300_dpi(destino)
        finally:
            plt.close(figura)

    def test_exportar_todas_figuras_usa_nombres_estables_y_300_dpi(self) -> None:
        figuras = generar_todas_las_figuras(
            costo_objetivo, self.historial_1, self.historial_2, self.p0, self.vector_v
        )
        try:
            with tempfile.TemporaryDirectory() as temporal:
                rutas = exportar_todas_figuras(figuras, temporal)
                self.assertEqual(
                    {clave: ruta.name for clave, ruta in rutas.items()},
                    {
                        "superficie": "superficie_3d.png",
                        "curvas": "curvas_nivel.png",
                        "convergencia": "convergencia.png",
                    },
                )
                for ruta in rutas.values():
                    self.assertTrue(ruta.is_file())
                    self._assert_300_dpi(ruta)
        finally:
            for figura in figuras.values():
                plt.close(figura)

    def test_rechaza_nan_en_costo_y_historial(self) -> None:
        with self.assertRaisesRegex(ValueError, "NaN"):
            graficar_superficie(lambda x, y: float("nan"), self.historial_1)
        historial_invalido = [dict(self.historial_1[0], costo=float("nan"))]
        with self.assertRaisesRegex(ValueError, "NaN"):
            graficar_curvas_nivel(costo_objetivo, historial_invalido)


if __name__ == "__main__":
    unittest.main()
