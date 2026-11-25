"""Punto de entrada CLI y selector de la interfaz Green Cloud."""

import argparse


def main(argv: list[str] | None = None) -> int:
    """Mantiene la CLI inicial y abre la GUI solo con --gui."""
    parser = argparse.ArgumentParser(
        description="Green Cloud: optimización académica de recursos Cloud."
    )
    parser.add_argument(
        "--gui", action="store_true", help="Abrir la interfaz gráfica de escritorio."
    )
    argumentos = parser.parse_args(argv)
    if argumentos.gui:
        from interfaz import iniciar_interfaz

        return iniciar_interfaz()
    print("Green Cloud: simulador pendiente de implementación.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
