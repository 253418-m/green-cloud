# Green Cloud — Optimización de Costo y Huella de Carbono

Proyecto final de Cálculo Vectorial de Ingeniería en Tecnologías de la Información e Innovación Digital, Universidad Politécnica de Chiapas (cuarto cuatrimestre). Docente: Dr. Alonso Guadalupe Hernández Mendoza. **Entrega: jueves 3 de diciembre de 2026.**

## Caso y objetivo

Opción C: optimización de recursos en servidores Cloud con enfoque de computación verde. El simulador científico estudiará una función **académica y sintética** de costo operativo en función de la utilización de CPU (`x`) y ancho de banda (`y`), ambas entre 0 y 100 %. En fases posteriores permitirá analizar derivadas, gradientes, direcciones y trayectorias, ejecutar descenso del gradiente, comparar tasas de aprendizaje y exportar figuras para un artículo en LaTeX. El modelo de referencia facilitado para planear la arquitectura todavía **no está implementado** y no representa costos, emisiones ni impuestos reales.

Los requisitos del documento oficial *Proyecto final-Calculo de varias variabels.pdf* y del *Contexto integral del proyecto Green Cloud* deben contrastarse con esta base antes del desarrollo funcional. Esos documentos no estaban en el directorio durante la preparación inicial.

## Tecnologías y límites

Python, NumPy y Matplotlib; `unittest` y `argparse` de la biblioteca estándar; LaTeX, BibTeX y Git. SymPy se considerará solo si hace falta verificar resultados analíticos. Los algoritmos numéricos se implementarán manualmente: sin `scipy.optimize`, diferenciación automática, autograd, PyTorch, TensorFlow ni JAX para derivar.

## Estructura inicial

```text
green-cloud/
├── main.py                    # entrada mínima; simulador pendiente
├── calculo_vectorial.py       # reserva del motor matemático
├── graficas.py                # reserva de visualización
├── requirements.txt
├── README.md
├── tests/
│   ├── test_calculo_vectorial.py
│   ├── test_graficas.py
│   └── test_main.py
└── latex/
    ├── referencias.bib        # vacío hasta verificar referencias
    └── figuras/
        └── .gitkeep           # permite conservar la carpeta en Git
```

`latex/reporte.tex` requiere la plantilla oficial del docente. `reporte_final.pdf` será un entregable futuro. Ninguno forma parte de esta base.

## Organización del equipo y Git

Las siguientes ramas son **propuestas**; todavía no se han creado. Cada integrante debe aportar trabajo real, revisar los cambios de los demás y comprender el simulador completo. Los commits deben reflejar avances verificables, sin atribuir trabajo no realizado. Coordinar cualquier modificación de un archivo asignado a otra persona.

| Integrante | Rama propuesta | Responsabilidad principal | Archivos principales |
| --- | --- | --- | --- |
| 1 | `feat/calculo-vectorial` | Motor matemático, validación numérica y analítica | `calculo_vectorial.py`, `tests/test_calculo_vectorial.py` |
| 2 | `feat/graficas-cientificas` | Superficie, curvas de nivel, campo, trayectoria, convergencia y exportación | `graficas.py`, `tests/test_graficas.py` |
| 3 | `feat/simulador-cli` | CLI, validación de entradas, orquestación, integración y documentación de uso | `main.py`, `tests/test_main.py`, `requirements.txt`, `README.md` |
| 4 | `docs/reporte-latex` | Desarrollo analítico, primeras iteraciones manuales, tablas, interpretación, bibliografía y artículo | `latex/reporte.tex` (futuro), `latex/referencias.bib`, `reporte_final.pdf` (futuro) |

Todos comparten la revisión matemática, la comparación entre resultados analíticos y numéricos, las pruebas de integración, los requisitos oficiales, la revisión de pull requests y la preparación de la defensa.

## Contratos internos propuestos

**Estos nombres, parámetros y retornos son acuerdos para integrar trabajo futuro; ninguna de las operaciones de las tablas está implementada.** Se confirmarán contra los documentos oficiales antes de codificarlas. `x` e `y` son porcentajes en `[0, 100]`, `beta` pondera el componente ambiental, `h` es el paso de diferencias finitas, `alpha` la tasa de aprendizaje y `u` un vector de dos componentes. El módulo matemático devolverá datos y no dependerá de la CLI ni de la visualización. Los arreglos de dos componentes mantendrán el orden `[x, y]`.

| Operación propuesta | Parámetros previstos | Retorno previsto | Responsabilidad y módulo propietario |
| --- | --- | --- | --- |
| `costo_objetivo` | `x: float, y: float, beta: float` | `float` | Evaluar el modelo acordado; `calculo_vectorial.py` |
| `derivada_parcial` | `funcion: callable, x: float, y: float, variable: str, h: float` | `float` | Aproximar una parcial con diferencias finitas; `calculo_vectorial.py` |
| `gradiente_numerico` | `funcion: callable, x: float, y: float, h: float` | `numpy.ndarray` de forma `(2,)` | Reunir las dos parciales; `calculo_vectorial.py` |
| `normalizar_vector` | `vector: numpy.ndarray` de forma `(2,)` | `numpy.ndarray` de forma `(2,)` | Obtener dirección unitaria y definir el manejo del vector cero; `calculo_vectorial.py` |
| `derivada_direccional` | `gradiente: numpy.ndarray, direccion: numpy.ndarray` | `float` | Evaluar el producto escalar con dirección unitaria; `calculo_vectorial.py` |
| `verificar_acotacion` | `gradiente: numpy.ndarray, direccion: numpy.ndarray` | `bool` | Comprobar la cota de la derivada direccional con tolerancia acordada; `calculo_vectorial.py` |
| `direccion_tangente` | `gradiente: numpy.ndarray` | `numpy.ndarray` de forma `(2,)` | Hallar una dirección perpendicular para costo local constante; `calculo_vectorial.py` |
| `descenso_gradiente` | `funcion: callable, inicio: tuple[float, float], alpha: float, h: float, max_iter: int, tolerancia: float` | `list[dict]` | Iterar desde el punto inicial y registrar historial; `calculo_vectorial.py` |

El historial propuesto contendrá, por iteración, `iteracion: int`, `x: float`, `y: float`, `costo: float` y `norma_gradiente: float`. La política de límites del dominio, parada y errores debe acordarse antes de implementar descenso. El punto inicial exigido para el estudio será `(0, 0)`; el contrato admite un inicio explícito para facilitar pruebas. No se duplicarán cálculos en `graficas.py`.

| Operación propuesta | Parámetros previstos | Retorno previsto | Responsabilidad y módulo propietario |
| --- | --- | --- | --- |
| `graficar_superficie` | `x_malla, y_malla, costos: numpy.ndarray` | `matplotlib.figure.Figure` | Superficie 3D; `graficas.py` |
| `graficar_curvas_nivel` | `x_malla, y_malla, costos: numpy.ndarray` | `matplotlib.figure.Figure` | Niveles de costo 2D; `graficas.py` |
| `graficar_campo` | `x_malla, y_malla, gx, gy: numpy.ndarray` | `matplotlib.figure.Figure` | Campo vectorial de gradientes recibidos; `graficas.py` |
| `graficar_trayectoria` | `x_malla, y_malla, costos: numpy.ndarray; historial: list[dict]` | `matplotlib.figure.Figure` | Superponer iteraciones en niveles de costo; `graficas.py` |
| `graficar_convergencia` | `historiales: dict[float, list[dict]]` | `matplotlib.figure.Figure` | Comparar costos por iteración entre tasas `alpha`; `graficas.py` |
| `exportar_figura` | `figura: matplotlib.figure.Figure, destino: pathlib.Path` | `pathlib.Path` | Guardar la figura a 300 dpi; `graficas.py` |
| `main` | `argv: list[str] \| None` | `int` | Analizar argumentos y orquestar cálculos, comparación y figuras; `main.py` |

`graficas.py` podrá consumir resultados del motor matemático, sin recalcularlos. `main.py` será el único punto de entrada; las pruebas importarán el módulo que corresponda. No habrá importaciones de `graficas.py` o `main.py` desde `calculo_vectorial.py`, ni dependencias circulares o estado global mutable innecesario. La CLI recibirá parámetros y presentará resultados, sin contener algoritmos numéricos. Las pruebas matemáticas, de entradas, convergencia, figuras e integración se incorporarán cuando existan esas funciones.

## Instalación y validación inicial

Desde la raíz del proyecto, con Python disponible:

```console
python -m pip install -r requirements.txt
python -m compileall -q main.py calculo_vectorial.py graficas.py
python -m unittest discover -s tests -v
python main.py --help
```

En Windows, `py` puede sustituir a `python`. La instalación aún no es necesaria para importar los módulos de esta base. Las tres pruebas iniciales comprueban solo importación y disponibilidad del punto de entrada; **no validan matemáticas, figuras ni integración funcional**. Ejecutar `python main.py` muestra que el simulador sigue pendiente.

## Estado y requisitos pendientes

La estructura, documentación de responsabilidades y contratos, CLI mínima y pruebas de importación están preparados. Permanecen pendientes: revisar ambos documentos oficiales; obtener la plantilla LaTeX; cerrar los contratos y la política del dominio; implementar y validar el modelo y los algoritmos manuales; generar y verificar las tres figuras obligatorias; comparar resultados con el desarrollo analítico; documentar referencias reales; redactar y compilar el artículo; realizar pruebas de integración y preparar la defensa. La preparación de archivos no demuestra todavía ningún resultado matemático ni el cumplimiento final de la rúbrica.
