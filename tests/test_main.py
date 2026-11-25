"""Comprobación inicial de disponibilidad del punto de entrada."""

import importlib
import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch


class TestEstructuraMain(unittest.TestCase):
    def test_modulo_importable(self) -> None:
        modulo = importlib.import_module("main")
        self.assertTrue(callable(modulo.main))

    def test_cli_inicial_se_conserva(self) -> None:
        from main import main

        salida = io.StringIO()
        with redirect_stdout(salida):
            self.assertEqual(main([]), 0)
        self.assertIn("pendiente", salida.getvalue())

    def test_gui_se_inicia_solo_con_argumento(self) -> None:
        from main import main

        with patch("interfaz.iniciar_interfaz", return_value=0) as iniciar:
            self.assertEqual(main(["--gui"]), 0)
        iniciar.assert_called_once_with()
