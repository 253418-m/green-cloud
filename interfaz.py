"""Interfaz de escritorio de Green Cloud; no contiene cálculos científicos."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Mapping

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class Parametros:
    x0: float = 80.0
    y0: float = 70.0
    vx: float = 3.0
    vy: float = 4.0
    h: float = 0.001
    alpha_1: float = 0.2
    alpha_2: float = 0.5
    beta: float = 1.0
    tolerancia: float = 0.000001
    max_iteraciones: int = 1000
    umbral_critico: float = 75.0


PREDETERMINADOS = Parametros()


def textos_predeterminados() -> dict[str, str]:
    """Devuelve una copia editable de los valores iniciales propuestos."""
    return {nombre: str(valor) for nombre, valor in vars(PREDETERMINADOS).items()}


def validar_parametros(entradas: Mapping[str, str]) -> Parametros:
    """Convierte y valida el formulario sin necesitar una ventana Tk."""
    etiquetas = {
        "x0": "Utilización CPU x0", "y0": "Utilización ancho de banda y0",
        "vx": "Componente vx", "vy": "Componente vy", "h": "Paso h",
        "alpha_1": "Tasa alpha_1", "alpha_2": "Tasa alpha_2",
        "beta": "Ponderación ambiental beta", "tolerancia": "Tolerancia",
        "max_iteraciones": "Máximo de iteraciones", "umbral_critico": "Umbral crítico",
    }
    valores: dict[str, float | int] = {}
    for nombre, etiqueta in etiquetas.items():
        texto = entradas.get(nombre, "")
        if not isinstance(texto, str) or not texto.strip():
            raise ValueError(f"{etiqueta}: introduce un valor.")
        try:
            if nombre == "max_iteraciones":
                if not texto.strip().isdigit():
                    raise ValueError
                valor: float | int = int(texto.strip())
            else:
                valor = float(texto.strip())
        except (ValueError, OverflowError):
            tipo = "un entero positivo" if nombre == "max_iteraciones" else "un número válido"
            raise ValueError(f"{etiqueta}: introduce {tipo}.") from None
        if nombre != "max_iteraciones" and not math.isfinite(valor):
            raise ValueError(f"{etiqueta}: el valor debe ser finito.")
        valores[nombre] = valor
    for nombre in ("x0", "y0", "umbral_critico"):
        if not 0 <= valores[nombre] <= 100:
            raise ValueError(f"{etiquetas[nombre]}: debe estar entre 0 y 100.")
    for nombre in ("h", "alpha_1", "alpha_2", "tolerancia", "max_iteraciones"):
        if valores[nombre] <= 0:
            raise ValueError(f"{etiquetas[nombre]}: debe ser mayor que cero.")
    if valores["beta"] < 0:
        raise ValueError(f"{etiquetas['beta']}: debe ser mayor o igual a cero.")
    if valores["vx"] == 0 and valores["vy"] == 0:
        raise ValueError("Dirección: el vector (vx, vy) no puede ser (0, 0).")
    return Parametros(**valores)


@dataclass(frozen=True)
class ResultadoSimulacion:
    """Datos reales entregados por la futura orquestación."""
    resultados: Mapping[str, object]
    figuras: Mapping[str, Figure]


ResultadoEjecutor = Callable[[Parametros], ResultadoSimulacion]
Exportador = Callable[["Figure", Path], Path]


class InterfazGreenCloud:
    """Construye la ventana y presenta datos recibidos del motor externo."""

    GRUPOS = (
        ("Carga inicial", (("x0", "Utilización CPU x0 (%)"), ("y0", "Ancho de banda y0 (%)"))),
        ("Dirección", (("vx", "Componente vx"), ("vy", "Componente vy"))),
        ("Configuración numérica", (("h", "Paso h"), ("alpha_1", "Tasa alpha_1"),
                                    ("alpha_2", "Tasa alpha_2"), ("tolerancia", "Tolerancia"),
                                    ("max_iteraciones", "Máximo de iteraciones"))),
        ("Modelo", (("beta", "Ponderación ambiental beta"),
                    ("umbral_critico", "Umbral crítico (%)"))),
    )
    RESULTADOS = (
        ("costo_inicial", "Costo inicial"),
        ("derivada_x", "Derivada parcial respecto a x"),
        ("derivada_y", "Derivada parcial respecto a y"),
        ("gradiente", "Vector gradiente"),
        ("norma_gradiente", "Norma del gradiente"),
        ("vector_unitario", "Vector unitario"),
        ("derivada_direccional", "Derivada direccional"),
        ("estado_critico", "Estado crítico"),
        ("punto_minimo", "Punto mínimo encontrado"),
        ("costo_minimo", "Costo mínimo"),
        ("iteraciones", "Iteraciones ejecutadas"),
        ("motivo_terminacion", "Motivo de terminación"),
    )
    PESTANAS = (("superficie", "Superficie 3D"),
                ("curvas", "Curvas de nivel"), ("convergencia", "Convergencia"))

    def __init__(self, raiz, ejecutor: ResultadoEjecutor | None = None,
                 exportador: Exportador | None = None) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.raiz = raiz
        self.ejecutor = ejecutor
        self.exportador = exportador
        self.variables = {nombre: tk.StringVar(raiz, value=valor)
                          for nombre, valor in textos_predeterminados().items()}
        self.valores_resultado: dict[str, tk.StringVar] = {}
        self.marcos_figura: dict[str, ttk.Frame] = {}
        self.figuras: dict[str, Figure] = {}
        self.estado = tk.StringVar(raiz, value="Aplicación iniciada. Motor matemático pendiente.")
        raiz.title("Green Cloud | Simulador de Optimización")
        raiz.geometry("1160x760")
        raiz.minsize(920, 640)
        self._estilos(ttk)
        self._construir(tk, ttk)
        for variable in self.variables.values():
            variable.trace_add("write", self._entradas_cambiaron)
        raiz.protocol("WM_DELETE_WINDOW", self.cerrar)

    def _estilos(self, ttk) -> None:
        estilo = ttk.Style(self.raiz)
        estilo.configure("Fondo.TFrame", background="#f3f6f4")
        estilo.configure("Cabecera.TFrame", background="#153a30")
        estilo.configure("Marca.TLabel", background="#153a30", foreground="white",
                         font=("Segoe UI", 19, "bold"))
        estilo.configure("Subtitulo.TLabel", background="#153a30", foreground="#d6efe3",
                         font=("Segoe UI", 10))
        estilo.configure("Titulo.TLabel", background="#f3f6f4", foreground="#153a30",
                         font=("Segoe UI", 11, "bold"))
        estilo.configure("Accion.TButton", font=("Segoe UI", 10, "bold"))
        estilo.configure("Estado.TLabel", background="#dcebe2", foreground="#153a30",
                         padding=(12, 7))

    def _construir(self, tk, ttk) -> None:
        raiz = self.raiz
        raiz.configure(background="#f3f6f4")
        raiz.rowconfigure(1, weight=1)
        raiz.columnconfigure(0, weight=1)
        cabecera = ttk.Frame(raiz, style="Cabecera.TFrame", padding=(20, 12))
        cabecera.grid(row=0, column=0, sticky="ew")
        ttk.Label(cabecera, text="Green Cloud", style="Marca.TLabel").pack(anchor="w")
        ttk.Label(cabecera, text="Optimización de Costo y Huella de Carbono · Aplicación académica",
                  style="Subtitulo.TLabel").pack(anchor="w")
        cuerpo = ttk.PanedWindow(raiz, orient="horizontal")
        cuerpo.grid(row=1, column=0, sticky="nsew", padx=12, pady=10)
        configuracion = ttk.Frame(cuerpo, style="Fondo.TFrame")
        presentacion = ttk.Frame(cuerpo, style="Fondo.TFrame")
        cuerpo.add(configuracion, weight=0)
        cuerpo.add(presentacion, weight=1)
        self._panel_configuracion(configuracion, tk, ttk)
        self._panel_presentacion(presentacion, tk, ttk)
        ttk.Label(raiz, textvariable=self.estado, style="Estado.TLabel", anchor="w").grid(
            row=2, column=0, sticky="ew")

    def _panel_configuracion(self, padre, tk, ttk) -> None:
        padre.configure(width=340)
        padre.grid_propagate(False)
        padre.rowconfigure(1, weight=1)
        padre.columnconfigure(0, weight=1)
        ttk.Label(padre, text="Configuración", style="Titulo.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8))
        lienzo = tk.Canvas(padre, background="#f3f6f4", highlightthickness=0)
        barra = ttk.Scrollbar(padre, orient="vertical", command=lienzo.yview)
        lienzo.configure(yscrollcommand=barra.set)
        lienzo.grid(row=1, column=0, sticky="nsew")
        barra.grid(row=1, column=1, sticky="ns")
        interior = ttk.Frame(lienzo, style="Fondo.TFrame")
        ventana = lienzo.create_window((0, 0), window=interior, anchor="nw")
        interior.bind("<Configure>", lambda _e: lienzo.configure(scrollregion=lienzo.bbox("all")))
        lienzo.bind("<Configure>", lambda e: lienzo.itemconfigure(ventana, width=e.width))
        for indice, (titulo, campos) in enumerate(self.GRUPOS):
            grupo = ttk.LabelFrame(interior, text=titulo, padding=10)
            grupo.grid(row=indice, column=0, sticky="ew", padx=(0, 6), pady=(0, 9))
            grupo.columnconfigure(1, weight=1)
            for fila, (nombre, etiqueta) in enumerate(campos):
                ttk.Label(grupo, text=etiqueta).grid(row=fila, column=0, sticky="w", pady=3)
                ttk.Entry(grupo, textvariable=self.variables[nombre], width=13).grid(
                    row=fila, column=1, sticky="ew", padx=(10, 0), pady=3)
        interior.columnconfigure(0, weight=1)
        botones = ttk.Frame(padre, style="Fondo.TFrame")
        botones.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        botones.columnconfigure((0, 1), weight=1)
        ttk.Button(botones, text="Ejecutar simulación", style="Accion.TButton",
                   command=self.ejecutar).grid(row=0, column=0, columnspan=2, sticky="ew", pady=3)
        ttk.Button(botones, text="Restablecer parámetros", command=self.restablecer).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=3)
        self.boton_exportar = ttk.Button(botones, text="Exportar gráficas", command=self.exportar)
        self.boton_exportar.grid(row=2, column=0, sticky="ew", padx=(0, 4), pady=3)
        self.boton_exportar.state(["disabled"])
        ttk.Button(botones, text="Salir", command=self.cerrar).grid(
            row=2, column=1, sticky="ew", padx=(4, 0), pady=3)

    def _panel_presentacion(self, padre, tk, ttk) -> None:
        padre.rowconfigure(1, weight=1)
        padre.columnconfigure(0, weight=1)
        resultados = ttk.LabelFrame(padre, text="Resultados", padding=10)
        resultados.grid(row=0, column=0, sticky="ew", padx=(10, 0))
        for columna in (0, 2):
            resultados.columnconfigure(columna, weight=1)
        for indice, (clave, etiqueta) in enumerate(self.RESULTADOS):
            fila, bloque = divmod(indice, 2)
            columna = bloque * 2
            ttk.Label(resultados, text=etiqueta + ":", wraplength=160).grid(
                row=fila, column=columna, sticky="w", padx=(0, 7), pady=3)
            variable = tk.StringVar(self.raiz, value="—")
            self.valores_resultado[clave] = variable
            ttk.Label(resultados, textvariable=variable, foreground="#176b4e", wraplength=105).grid(
                row=fila, column=columna + 1, sticky="w", padx=(0, 12), pady=3)
        pestanas = ttk.Notebook(padre)
        pestanas.grid(row=1, column=0, sticky="nsew", padx=(10, 0), pady=(10, 0))
        for clave, titulo in self.PESTANAS:
            marco = ttk.Frame(pestanas, padding=6)
            marco.pack_propagate(False)
            pestanas.add(marco, text=titulo)
            ttk.Label(marco, text="No hay resultados disponibles.", anchor="center").pack(
                fill="both", expand=True)
            self.marcos_figura[clave] = marco

    def restablecer(self) -> None:
        for nombre, texto in textos_predeterminados().items():
            self.variables[nombre].set(texto)
        self.estado.set("Parámetros restablecidos.")

    def _entradas_cambiaron(self, *_args) -> None:
        self.mostrar_resultados({})
        if self.figuras:
            from matplotlib import pyplot as plt

            for figura in self.figuras.values():
                plt.close(figura)
            self.figuras.clear()
            for marco in self.marcos_figura.values():
                for hijo in marco.winfo_children():
                    hijo.destroy()
                from tkinter import ttk

                ttk.Label(marco, text="No hay resultados disponibles.", anchor="center").pack(
                    fill="both", expand=True)
            self.boton_exportar.state(["disabled"])
        self.estado.set("Parámetros modificados. Ejecute para actualizar resultados.")

    def cerrar(self) -> None:
        if self.figuras:
            from matplotlib import pyplot as plt

            for figura in self.figuras.values():
                plt.close(figura)
        self.raiz.destroy()

    def ejecutar(self) -> None:
        try:
            parametros = validar_parametros(
                {nombre: variable.get() for nombre, variable in self.variables.items()}
            )
        except ValueError as error:
            self.estado.set(f"Parámetros inválidos: {error}")
            return
        self.estado.set("Parámetros válidos.")
        if self.ejecutor is None:
            self.estado.set("Motor matemático pendiente. No se ejecutó la simulación.")
            return
        try:
            resultado = self.ejecutor(parametros)
            if not isinstance(resultado, ResultadoSimulacion):
                raise TypeError("El motor debe devolver ResultadoSimulacion.")
            self.mostrar_resultados(resultado.resultados)
            for clave, figura in resultado.figuras.items():
                self.mostrar_figura(clave, figura)
        except Exception as error:
            self.estado.set(f"Error de ejecución: {error}")
            return
        self.estado.set("Simulación finalizada.")

    def mostrar_resultados(self, resultados: Mapping[str, object]) -> None:
        """Muestra únicamente datos recibidos de una ejecución real."""
        for clave, variable in self.valores_resultado.items():
            dato = resultados.get(clave)
            variable.set("—" if dato is None else str(dato))

    def mostrar_figura(self, pestana: str, figura: Figure, *, barra: bool = True) -> None:
        """Incrusta una figura existente, reemplazando el lienzo anterior."""
        from matplotlib.figure import Figure as FigureReal
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib import pyplot as plt

        if pestana not in self.marcos_figura:
            raise ValueError(f"Pestaña desconocida: {pestana}.")
        if not isinstance(figura, FigureReal):
            raise TypeError("Se requiere una figura Matplotlib real.")
        marco = self.marcos_figura[pestana]
        anterior = self.figuras.get(pestana)
        for hijo in marco.winfo_children():
            hijo.destroy()
        if anterior is not None and anterior is not figura:
            plt.close(anterior)
        lienzo = FigureCanvasTkAgg(figura, master=marco)
        lienzo.get_tk_widget().pack(fill="both", expand=True)
        lienzo.draw()
        if barra:
            herramientas = NavigationToolbar2Tk(lienzo, marco, pack_toolbar=False)
            herramientas.update()
            herramientas.pack(fill="x")
        self.figuras[pestana] = figura
        if self.exportador is not None:
            self.boton_exportar.state(["!disabled"])

    def exportar(self) -> None:
        if not self.figuras or self.exportador is None:
            self.estado.set("Exportación pendiente: no hay figuras o exportador disponible.")
            return
        from tkinter import filedialog

        destino = filedialog.askdirectory(parent=self.raiz, title="Seleccionar carpeta de exportación")
        if not destino:
            return
        try:
            for clave, figura in self.figuras.items():
                self.exportador(figura, Path(destino) / f"green_cloud_{clave}.png")
        except Exception as error:
            self.estado.set(f"Error al exportar gráficas: {error}")
            return
        self.estado.set("Gráficas exportadas.")


def iniciar_interfaz(ejecutor: ResultadoEjecutor | None = None,
                     exportador: Exportador | None = None) -> int:
    """Abre la aplicación solo al llamarse explícitamente."""
    try:
        import tkinter as tk
    except ImportError:
        print("No se pudo iniciar Green Cloud: Tkinter no está disponible.")
        return 1
    try:
        raiz = tk.Tk()
    except tk.TclError as error:
        print(f"No se pudo abrir la interfaz gráfica: {error}")
        return 1
    try:
        InterfazGreenCloud(raiz, ejecutor=ejecutor, exportador=exportador)
    except Exception as error:
        raiz.destroy()
        print(f"No se pudo iniciar la interfaz: {error}")
        return 1
    raiz.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(iniciar_interfaz())
