"""Punto de entrada del proyecto Green Cloud.

La interfaz y la simulación se incorporarán en fases posteriores.
"""

import argparse


def main(argv: list[str] | None = None) -> int:
    """Muestra el estado inicial y ofrece la ayuda de la interfaz."""
    parser = argparse.ArgumentParser(
        description="Green Cloud: optimización académica de recursos Cloud."
    )
    parser.parse_args(argv)
    print("Green Cloud: simulador pendiente de implementación.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
