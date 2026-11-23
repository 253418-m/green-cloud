"""Comprobación inicial de disponibilidad del módulo de gráficas."""

import importlib
import unittest


class TestEstructuraGraficas(unittest.TestCase):
    def test_modulo_importable(self) -> None:
        self.assertIsNotNone(importlib.import_module("graficas"))
