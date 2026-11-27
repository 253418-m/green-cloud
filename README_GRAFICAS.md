# Green Cloud — Documentación técnica del motor de gráficas

## OVERVIEW & DEPENDENCY_AUDIT

`graficas.py` implementa la generación de las visualizaciones científicas 2D/3D y la exportación PNG del simulador Green Cloud (Cálculo Vectorial). Recibe una función de costo y los historiales del motor matemático; devuelve objetos Matplotlib `Figure`. No ejecuta ni delega la optimización.

Metadatos de responsabilidad proporcionados por el equipo:

```json
{
  "module": "graficas.py",
  "developer": "imnichan",
  "role": "Integrante 2 — Responsable de Gráficas Científicas",
  "branch": "feat/graficas-cientificas"
}
```

### Auditoría de imports

Imports inspeccionados directamente en `graficas.py` y `tests/test_graficas.py`:

| Import detectado | Archivo | Clasificación/uso | Estado según rúbrica |
|---|---|---|---|
| `matplotlib.pyplot` (`plt`) | Ambos | Librería permitida; construcción, figuras, guardado y cierre | Permitido |
| `matplotlib.figure.Figure` | Ambos | Clase pública de Matplotlib para tipos y comprobaciones | Permitido |
| `numpy` (`np`) | `graficas.py` | Mallas, arrays, validación numérica y gradientes | Permitido |
| `matplotlib` y `matplotlib.use("Agg")` | `tests/test_graficas.py` | Backend headless, configurado antes de importar `pyplot` | Permitido |
| `collections.abc` (`Callable`, `Mapping`, `Sequence`) | `graficas.py` | Tipado de funciones y registros | Biblioteca estándar |
| `pathlib.Path` | Ambos | Rutas de salida y temporales | Biblioteca estándar permitida |
| `typing.TypeAlias` | `graficas.py` | Alias de tipos | Biblioteca estándar |
| `unittest` | `tests/test_graficas.py` | Framework de pruebas | Biblioteca estándar permitida |
| `tempfile` | `tests/test_graficas.py` | Directorio temporal para pruebas de exportación | Biblioteca estándar permitida |
| `struct` | `tests/test_graficas.py` | Lectura de metadatos de resolución del PNG | Biblioteca estándar de Python |
| `zlib` | `tests/test_graficas.py` | Importado, pero no utilizado por las pruebas actuales | Biblioteca estándar de Python |
| `calculo_vectorial` | Ambos | Módulo interno del proyecto: costo y funciones vectoriales | Dependencia interna, no paquete externo |

No hay imports de `tkinter` en estos dos archivos; la incrustación Tk pertenece a `interfaz.py`. Tampoco se importa `mplot3d` explícitamente: la proyección 3D se solicita a Matplotlib mediante `projection="3d"`.

### Comparación con la rúbrica

| Categoría | Paquetes/módulos | Resultado |
|---|---|---|
| Científicos autorizados | `numpy`, `matplotlib`, `matplotlib.pyplot`, `matplotlib.figure` | Permitidos |
| Backend de pruebas autorizado | `Agg`, a través de `matplotlib.use("Agg")` | Permitido |
| Biblioteca estándar | `collections.abc`, `pathlib`, `typing`, `unittest`, `tempfile`, `struct`, `zlib` | Permitidos |
| Módulos internos | `calculo_vectorial` | Permitido; código del proyecto |
| Optimizadores externos prohibidos | `scipy.optimize`, `autograd`, PyTorch, TensorFlow u otras cajas negras de optimización | No importados |

**Cumplimiento académico: 100% Permitido.** La inspección de imports no detectó dependencias prohibidas ni optimizadores externos. El método de descenso pertenece a `calculo_vectorial.py`; este módulo únicamente calcula datos visuales y dibuja figuras.

Dependencias externas del proyecto requeridas: `numpy` y `matplotlib` (ver [requirements.txt](./requirements.txt)). Para pruebas sin servidor gráfico, `tests/test_graficas.py` fija `Agg` antes de importar `pyplot`.

## API_CONTRACTS & SPECIFICATIONS

### Tipos compartidos

```python
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from matplotlib.figure import Figure
from typing import TypeAlias

FuncionCosto: TypeAlias = Callable[[float, float], float]
Registro: TypeAlias = Mapping[str, object]
Historial: TypeAlias = Sequence[Registro]
```

### Firmas públicas implementadas

```python
def graficar_superficie(
    funcion_costo: FuncionCosto,
    historial: Historial,
    p0: Sequence[float] | None = None,
    titulo: str = "Superficie 3D de Costo y Trayectoria",
    figsize: tuple[float, float] = (6.0, 3.6),
    dpi: int = 85,
) -> Figure: ...

def graficar_curvas_nivel(
    funcion_costo: FuncionCosto,
    historial: Historial,
    p0: Sequence[float] | None = None,
    vector_v: Sequence[float] | None = None,
    titulo: str = "Curvas de Nivel y Campo Gradiente",
    figsize: tuple[float, float] = (6.0, 3.6),
    dpi: int = 85,
) -> Figure: ...

def graficar_convergencia(
    historial_alpha1: Historial,
    historial_alpha2: Historial,
    label_alpha1: str = "alpha=0.2",
    label_alpha2: str = "alpha=0.5",
    titulo: str = "Comparación de Convergencia",
    figsize: tuple[float, float] = (6.0, 3.6),
    dpi: int = 85,
) -> Figure: ...

def generar_todas_las_figuras(
    funcion_costo: FuncionCosto,
    historial_a1: Historial,
    historial_a2: Historial,
    p0: Sequence[float],
    vector_v: Sequence[float],
    *,
    label_alpha1: str = "alpha=0.2",
    label_alpha2: str = "alpha=0.5",
    figsize: tuple[float, float] = (6.0, 3.6),
    dpi: int = 85,
) -> dict[str, Figure]: ...

def exportar_figura(
    figura: Figure,
    destino: Path | str,
    dpi: int = 300,
) -> Path: ...

def exportar_todas_figuras(
    figuras: dict[str, Figure],
    carpeta_destino: Path | str,
    dpi: int = 300,
) -> dict[str, Path]: ...
```

`figsize` y `dpi` de las funciones generadoras determinan las dimensiones/resolución nominal de la figura en memoria. Los exportadores proporcionan su propio `dpi=300` predeterminado y no dependen del DPI de pantalla.

### Contratos por función

#### `graficar_superficie`

- Evalúa `funcion_costo(x, y)` en una malla uniforme `7 × 7` sobre `[0, 100]²`.
- Devuelve una `Figure` con superficie 3D, colorbar, trayectoria (negra), punto final (verde) y `P0` (rojo).
- Los historiales de posición usan las claves `x`, `y`, `costo`.
- Si `p0` es `None`, infiere el inicio de la primera entrada del historial. Sin `p0` ni trayectoria, muestra solo la superficie.

#### `graficar_curvas_nivel`

- Devuelve una `Figure` con contornos de costo, líneas de nivel y campo de gradiente (`quiver`).
- Calcula gradientes numéricos con `h=0.001` en cada punto de la malla.
- Puede añadir trayectoria, `P0`, tangente local y dirección unitaria correspondiente a `vector_v`.
- Si se proporciona `vector_v`, debe ser no nulo y se necesita `p0` o un historial no vacío.

#### `graficar_convergencia`

- Compara costo contra iteración para dos historiales.
- Usa `registro["iteracion"]`, o el índice del elemento si esa clave no existe.
- Cada historial debe contener al menos un registro y un costo numérico finito.
- Marca el estado final según `convergio` o `motivo_terminacion`, e incluye el motivo en la leyenda.

#### `generar_todas_las_figuras`

Devuelve exactamente las claves siguientes:

```json
{
  "superficie": "Figure de graficar_superficie",
  "curvas": "Figure de graficar_curvas_nivel",
  "convergencia": "Figure de graficar_convergencia"
}
```

Usa `historial_a1` para superficie y curvas, y compara `historial_a1` con `historial_a2` para convergencia. `label_alpha1` y `label_alpha2` permiten mostrar las tasas realmente usadas. El llamador es responsable de liberar las figuras al terminar.

#### Exportación

- `exportar_figura` valida `Figure` y DPI entero positivo, crea los directorios padre, guarda PNG con `bbox_inches="tight"` y devuelve la ruta como `Path`.
- `exportar_todas_figuras` exige las tres claves públicas y crea:

```json
{
  "superficie": "superficie_3d.png",
  "curvas": "curvas_nivel.png",
  "convergencia": "convergencia.png"
}
```

- Ambos exportadores usan `dpi=300` por defecto, con independencia de que la figura se generara a 85 dpi.

## GUI_LAYOUT_RESPONSIVENESS_FIX

- Las figuras para pantalla usan `figsize=(6.0, 3.6)` y `dpi=85`. Esto reduce el tamaño nominal de los canvases `FigureCanvasTkAgg` embebidos y ayuda a mantener visible el área **Detalle de resultados** junto con las pestañas de gráficas en `interfaz.py`.
- El panel principal de la GUI tiene scroll vertical y el frame de detalle conserva un mínimo de 140 px. La interfaz empaqueta el widget Matplotlib con `padx=5, pady=5` sin estirar la figura a dimensiones de exportación.
- El tamaño reducido afecta solo a la presentación en pantalla. El `savefig(..., dpi=300)` del exportador conserva la resolución de impresión a 300 dpi.
- Al reemplazar gráficas, destruir los widgets/canvas previos y cerrar las referencias Matplotlib antes de construir las nuevas. La interfaz actual cierra las figuras almacenadas y llama `plt.close("all")` durante la limpieza si había figuras previas; no debe dejar widgets Tk antiguos montados.

## TESTING & VERIFICATION

Comando de pruebas:

```bash
python3 -m unittest tests/test_graficas.py -v
```

Verificación relacionada:

- `matplotlib.use("Agg")` se ejecuta en `tests/test_graficas.py` antes de importar `matplotlib.pyplot`; no se requiere pantalla.
- Las pruebas verifican objetos `Figure`, ejes/leyendas, claves de la API, dimensiones de pantalla `6.0 × 3.6` a 85 dpi, archivos PNG y metadatos de exportación a 300 dpi.
- Las figuras de prueba se liberan con `plt.close(figura)` en bloques `finally`. La aplicación debe hacer la misma limpieza al invalidar o reemplazar figuras para evitar acumulación de memoria.
- Las pruebas usan directorios temporales para I/O y validan el rechazo de valores NaN.
