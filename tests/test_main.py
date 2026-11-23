"""Comprobación inicial de disponibilidad del punto de entrada."""

import importlib
import unittest


class TestEstructuraMain(unittest.TestCase):
    def test_modulo_importable(self) -> None:
        modulo = importlib.import_module("main")
        self.assertTrue(callable(modulo.main))
