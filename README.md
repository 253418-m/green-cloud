# Green Cloud — Optimización de Costo y Huella de Carbono

Proyecto final de Cálculo Vectorial, Universidad Politécnica de Chiapas. Caso de estudio C: optimización de recursos en servidores Cloud. **Entrega: 3 de diciembre de 2026.** `x` representa la utilización de CPU y `y` la utilización de ancho de banda; ambas se consideran en el dominio `[0, 100]`.

El modelo de costo y emisiones es académico y sintético. Sus algoritmos numéricos aún no están implementados y no se deben interpretar los parámetros propuestos como restricciones impuestas por el docente. El documento oficial *Proyecto final-Calculo de varias variabels.pdf* tiene prioridad; su copia local está en `docs/` y queda fuera de este commit. El documento pide un simulador modular en Python, cálculos y tres figuras científicas. El pedido de interfaz gráfica proviene del equipo; aquí no se atribuye al PDF.

## Estado del proyecto

La CLI mínima, la interfaz de escritorio y las pruebas de validación de formularios están disponibles. `calculo_vectorial.py` y `graficas.py` son todavía módulos reservados: no calculan resultados ni crean figuras. Por ello, al ejecutar la GUI se pueden editar, validar y restablecer parámetros, pero la simulación informa **Motor matemático pendiente**. El área de resultados permanece vacía y el botón de exportación deshabilitado hasta recibir figuras y un exportador reales. No hay todavía evidencia numérica de optimización.

La estructura actual es:

```text
green-cloud/
├── main.py                  # CLI y selector --gui
├── interfaz.py              # ventana, formulario, validación y presentación
├── calculo_vectorial.py     # motor pendiente
├── graficas.py              # visualización científica pendiente
├── requirements.txt         # NumPy y Matplotlib
├── README.md
├── tests/
│   ├── test_calculo_vectorial.py
│   ├── test_graficas.py
│   ├── test_interfaz.py
│   └── test_main.py
└── latex/
    ├── referencias.bib
    └── figuras/
```

## Instalación y ejecución

Se requiere Python 3.13 con Tkinter y las dependencias de `requirements.txt`. Tkinter forma parte de la instalación de Python para Windows, pero su disponibilidad debe comprobarse en el entorno de ejecución. Desde la raíz del repositorio:

```console
py -m pip install -r requirements.txt
py main.py
py main.py --help
py main.py --gui
py interfaz.py
py -m unittest discover -s tests -v
```

`py main.py` conserva la salida de la CLI inicial. `--help` muestra las opciones sin importar Tkinter ni abrir una ventana. `--gui` y `py interfaz.py` abren la ventana; si Tkinter o la pantalla no están disponibles, se informa el error y el proceso termina sin crear resultados. En sistemas donde no exista el lanzador `py`, se puede usar `python`.

## Interfaz de escritorio

La ventana **Green Cloud | Simulador de Optimización** agrupa las entradas en carga inicial, dirección, configuración numérica y modelo. Incluye botones **Ejecutar simulación**, **Restablecer parámetros**, **Exportar gráficas** y **Salir**; una barra de estado, doce campos de resultados y pestañas **Superficie 3D**, **Curvas de nivel** y **Convergencia**. La configuración se puede desplazar verticalmente y las gráficas se ajustan a su contenedor.

Los valores iniciales propuestos son `x0=80`, `y0=70`, `vx=3`, `vy=4`, `h=0.001`, `alpha_1=0.2`, `alpha_2=0.5`, `beta=1`, `tolerancia=0.000001`, `max_iteraciones=1000` y `umbral_critico=75`. Así, el punto inicial propuesto es **P0 = (80, 70)**. El docente no exige un punto inicial fijo según el contexto compartido; este valor se puede editar durante la defensa.

La validación rechaza entradas vacías, valores no numéricos, NaN, infinito, direcciones nulas, `x0`, `y0` y umbral fuera de `[0,100]`, `beta < 0`, pasos, tasas y tolerancia no positivos, y máximo de iteraciones que no sea un entero positivo. La validación no limita el dominio para acomodar diferencias finitas cerca de las fronteras: esa política se acordará con el motor matemático.

`interfaz.validar_parametros()` devuelve un `Parametros` inmutable y puede probarse sin Tk. `main.py` solo analiza argumentos y selecciona el modo. `interfaz.py` contiene widgets, validación de formulario y presentación; no duplica el modelo matemático. La futura orquestación entregará a `InterfazGreenCloud` un `ejecutor(Parametros) -> ResultadoSimulacion`. Este objeto contendrá un mapa de resultados reales y un mapa de objetos `matplotlib.figure.Figure` reales con claves `superficie`, `curvas` y `convergencia`. `mostrar_figura(clave, figura)` usa `FigureCanvasTkAgg`, admite `NavigationToolbar2Tk` y sustituye la figura anterior. Un `exportador(figura, destino: pathlib.Path) -> pathlib.Path` externo habilitará la exportación; se espera que proceda de `graficas.py` cuando exista. La GUI no crea datos, ejecutores ni figuras de sustitución.

Las claves previstas del mapa de resultados son `costo_inicial`, `derivada_x`, `derivada_y`, `gradiente`, `norma_gradiente`, `vector_unitario`, `derivada_direccional`, `estado_critico`, `punto_minimo`, `costo_minimo`, `iteraciones` y `motivo_terminacion`. Cualquier clave sin valor recibido aparece como no disponible.

## Contratos matemáticos propuestos

Estas firmas y retornos son acuerdos para integrar trabajo futuro. **Ninguna operación está implementada aún.** El motor no debe depender de la GUI, la CLI ni de `graficas.py`. Los pares de coordenadas y arreglos usan el orden `[x, y]`.

| Operación | Parámetros previstos | Retorno previsto |
| --- | --- | --- |
| `costo_objetivo(x, y, beta)` | Tres números reales finitos; `x,y` en el dominio | `float` |
| `derivada_parcial(funcion, x, y, variable, h)` | Función escalar de dos variables; `variable` identifica `x` o `y`; `h>0` | `float` |
| `gradiente_numerico(funcion, x, y, h)` | Función, punto y paso | `numpy.ndarray` de forma `(2,)` |
| `normalizar_vector(vector)` | Vector de dos componentes no nulo | `numpy.ndarray` de forma `(2,)` |
| **`derivada_direccional(f, punto, vector_v)`** | `f` función escalar de dos variables, `punto=(x,y)`, `vector_v` dirección de dos componentes no nula | `float`; derivada en la dirección unitaria correspondiente |
| `verificar_acotacion(gradiente, direccion)` | Dos vectores de dos componentes | `bool` |
| `direccion_tangente(gradiente)` | Gradiente de dos componentes | Vector de dos componentes perpendicular al gradiente |
| `descenso_gradiente(funcion, inicio, alpha, h, max_iter, tolerancia)` | Función, punto inicial, tasa y paso positivos, iteraciones enteras positivas y tolerancia positiva | `list[dict]` |

La firma obligatoria de `derivada_direccional` recibe la función, el punto y el vector; no recibe un gradiente precalculado. El método numérico interno y la selección del paso deben acordarse con el responsable del motor antes de codificarla. Para el descenso se propone registrar por elemento `iteracion: int`, `x: float`, `y: float`, `costo: float` y `norma_gradiente: float`, incluidos los datos necesarios para comparar trayectorias. Aún falta acordar si el historial incluye el punto inicial como iteración cero, la condición exacta de parada y el manejo de errores.

**Fronteras del dominio pendientes:** las diferencias finitas centrales pueden evaluar fuera de `[0,100]` cerca del borde. El equipo matemático debe acordar diferencias unilaterales, proyección u otra política antes de implementar el motor. La GUI acepta puntos válidos en las fronteras y no presume esa decisión.

## Contratos de figuras propuestos

`graficas.py` será propietario de las figuras científicas y su exportación. Se prevén `graficar_superficie(x_malla, y_malla, costos)`, `graficar_curvas_nivel(x_malla, y_malla, costos)`, `graficar_campo(x_malla, y_malla, gx, gy)`, `graficar_trayectoria(x_malla, y_malla, costos, historial)` y `graficar_convergencia(historiales)`, todos con retorno `matplotlib.figure.Figure`. `exportar_figura(figura, destino)` deberá devolver `pathlib.Path`. Las mallas, costos, gradientes e historiales procederán del cálculo real. Estas firmas continúan sujetas a coordinación con el integrante responsable de gráficas.

## Equipo y ramas

Al revisar este checkout, las cuatro ramas locales existían y seguían ramas `origin/*` en el mismo commit inicial `b45a2bf`; no se hizo `fetch` para confirmar el estado remoto más reciente. Este trabajo se realiza en `feat/simulador-cli` y no modifica la rama `main`.

| Integrante | Rama | Responsabilidad |
| --- | --- | --- |
| 1 | `feat/calculo-vectorial` | Motor y pruebas matemáticas |
| 2 | `feat/graficas-cientificas` | Figuras y exportación científica |
| 3 | `feat/simulador-cli` | CLI, GUI, validación e integración |
| 4 | `docs/reporte-latex` | Desarrollo analítico y artículo |

La integración final requiere resultados verificados de los integrantes 1 y 2, acuerdo sobre fronteras e historial, y revisión final de cumplimiento del documento oficial. Las pruebas actuales verifican estructura, CLI, validación y controles básicos de la ventana, pero aún no prueban exactitud matemática ni calidad de figuras científicas.
