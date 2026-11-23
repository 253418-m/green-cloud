"""Comprobación inicial de disponibilidad del módulo matemático."""

import importlib
import unittest


class TestEstructuraCalculoVectorial(unittest.TestCase):
    def test_modulo_importable(self) -> None:
        self.assertIsNotNone(importlib.import_module("calculo_vectorial"))
