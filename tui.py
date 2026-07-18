#!/usr/bin/env python3
"""Interfaz TUI (Textual) para el buscador de servidores Ollama.

Permite, desde una sola pantalla:
  - Cargar / actualizar la lista de servidores (Shodan)      -> tecla u
  - Probar los modelos y ver su velocidad                    -> tecla t
  - Elegir modelo y servidor de forma ágil (listas laterales)
  - Chatear con el modelo elegido en una interfaz cuidada

Uso:
    python tui.py

Reutiliza la lógica de find_ollama.py (caché, Shodan, pruebas, streaming).
"""
import concurrent.futures
import threading

from rich.markdown import Markdown
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Footer, Header, Input, Label, ListItem, ListView, RichLog, Static,
)
from textual.worker import get_current_worker

import find_ollama as fo


def looks_like_markdown(text: str) -> bool:
    """Heurística ligera: ¿el texto contiene marcas típicas de Markdown?
    Si no, lo mostramos como texto plano (más fiel y sin reformatear)."""
    import re
    patterns = (
        r"(?m)^\s{0,3}#{1,6}\s",       # encabezados  # ...
        r"(?m)^\s*[-*+]\s+\S",          # listas con viñetas
        r"(?m)^\s*\d+\.\s+\S",          # listas numeradas
        r"```",                          # bloques de código
        r"`[^`]+`",                      # código en línea
        r"\*\*[^*]+\*\*",               # negrita
        r"(?m)^\s*>\s",                  # citas
        r"\[[^\]]+\]\([^)]+\)",         # enlaces [txt](url)
        r"(?m)^\s*\|.+\|\s*$",          # tablas
    )
    return any(re.search(p, text) for p in patterns)

CHAT_TIMEOUT = 120.0

# Ciclo del filtro por proveedor: None (todos) + cada proveedor conocido.
PROVIDER_CYCLE = [None] + list(fo.PROVIDERS.keys())


class ModelItem(ListItem):
    """Elemento de la lista de modelos. Se pinta en verde si al menos un
    servidor que lo ofrece ha pasado la prueba (--test) con ok:true."""
    def __init__(self, model: str, count: int, working: bool = False):
        name = f"[green]{model}[/green]" if working else model
        super().__init__(Label(f"{name}  [dim]({count} srv)[/dim]"))
        self.model = model


class ServerItem(ListItem):
    """Elemento de la lista de servidores. Muestra, si hay datos de --test
    para el modelo seleccionado, el estado (rojo si falla) y la velocidad."""
    def __init__(self, host: dict, model: str | None = None):
        self.host = host
        prov = host.get("provider", "ollama")
        base = f"{host['ip']}:{host['port']} [dim]({prov})[/dim]"
        loc = f" · {host['country']}" if host.get("country") else ""
        t = fo.get_test(host, model) if model else None
        if t is None:
            n = len(host.get("live_models") or [])
            label = f"{base}  [dim]({n} mod{loc})[/dim]"
        elif not t.get("ok"):
            label = (f"[red]{base}  ✗ FALLO ({t.get('error', '?')})[/red]"
                     f"[dim]{loc}[/dim]")
        else:
            tps = t.get("tokens_per_sec")
            speed = f"{tps:.1f} tok/s" if tps else "? tok/s"
            label = (f"[green]{base}[/green]  [b]{speed}[/b] · "
                     f"{t.get('latency', 0):.1f}s[dim]{loc}[/dim]")
        super().__init__(Label(label))


class OllamaTUI(App):
    CSS = """
    #sidebar { width: 46; border-right: solid $panel; }
    #sidebar .title { padding: 0 1; background: $boost; color: $text; text-style: bold; }
    #main { width: 1fr; }
    #status { padding: 0 1; background: $panel; color: $text-muted; }
    #chatlog { height: 1fr; padding: 0 1; }
    #live { padding: 0 1; color: $success; }
    ListView { height: 1fr; }
    #filter, #prompt { margin: 0; }
    /* Resaltar el panel con el foco para ver dónde estamos con el teclado. */
    ListView:focus { border: tall $accent; }
    Input:focus { border: tall $accent; }
    /* Selector en gris neutro (con barra lateral): contrasta con el verde de
       los modelos/servidores activos y con el rojo de los que fallan, cosa
       que el azul por defecto no hacía. La clase real es -highlight. */
    ListView > ListItem.-highlight {
        background: $surface-lighten-1;
        border-left: wide $panel;
    }
    ListView:focus > ListItem.-highlight {
        background: $surface-lighten-3;
        border-left: wide $warning;
        text-style: bold;
    }
    """

    BINDINGS = [
        # Acciones (funcionan cuando el foco NO está en una caja de texto).
        Binding("u", "update", "Actualizar"),
        Binding("t", "test", "Probar"),
        Binding("o", "toggle_working", "Solo disponibles"),
        Binding("p", "cycle_provider", "Proveedor"),
        Binding("r", "reset", "Reset chat"),
        Binding("ctrl+l", "clear", "Limpiar log"),
        Binding("q", "quit", "Salir"),
        # Copiar al portapapeles (funcionan también mientras escribes).
        Binding("ctrl+y", "copy_reply", "Copiar respuesta"),
        Binding("ctrl+b", "copy_target", "Copiar modelo/servidor"),
        # Navegación por teclado entre paneles (Tab/Shift+Tab ya rota el foco).
        Binding("f2", "focus_filter", "Filtro", show=False),
        Binding("f3", "focus_prompt", "Escribir", show=False),
        # Escape: salir de la caja de texto hacia la navegación.
        Binding("escape", "focus_models", "Navegar", show=True),
        # Izquierda/derecha saltan entre listas (solo con una lista enfocada).
        Binding("left", "focus_models", "◀ Modelos", show=False),
        Binding("right", "focus_servers", "Servidores ▶", show=False),
    ]

    def __init__(self):
        super().__init__()
        self.hosts: list = []
        self.model_hosts: dict = {}
        self.models: list = []
        self.model: str | None = None
        self.host: dict | None = None
        self.messages: list = []
        self.last_reply: str = ""
        self.busy = False
        # Operación de fondo en curso ("update" | "test" | None). Evita que
        # actualizar y probar se solapen y se pisen los datos entre sí.
        self.bg_busy: str | None = None
        # Si True, oculta modelos/servidores que no han pasado la prueba.
        self.only_working = False
        # Proveedor por el que filtrar (None = todos). Afecta a las listas y a
        # qué se busca al actualizar.
        self.provider_filter: str | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="sidebar"):
                yield Static("MODELOS", classes="title")
                yield Input(placeholder="filtrar modelos…", id="filter")
                yield ListView(id="models")
                yield Static("SERVIDORES", classes="title")
                yield ListView(id="servers")
            with Vertical(id="main"):
                yield Static("Cargando…", id="status")
                yield RichLog(id="chatlog", wrap=True, markup=True, highlight=False)
                yield Static("", id="live")
                yield Input(placeholder="Escribe tu mensaje y pulsa Enter…", id="prompt")
        yield Footer()

    # ------------------------------------------------------------------ setup
    def on_mount(self):
        self.title = "Ollama Scanner TUI"
        cached = fo.load_cache()
        if cached:
            self.hosts = cached
            self.rebuild_index()
            self.populate_models()
            self.set_status(f"Caché: {len(self.hosts)} servidores, "
                            f"{len(self.models)} modelos. "
                            f"[u] actualizar · [t] probar")
        else:
            self.set_status("Sin caché. Pulsa [u] para buscar en Shodan.")
        self.write_log(
            "[b]Teclado:[/b] [cyan]Esc[/cyan] navegar · [cyan]↑↓[/cyan] moverse · "
            "[cyan]Enter[/cyan] elegir · [cyan]←/→[/cyan] entre listas · "
            "[cyan]F2[/cyan] filtro · [cyan]F3[/cyan] escribir · "
            "[cyan]Tab[/cyan] cambiar panel"
        )
        self.write_log(
            "[b]Copiar:[/b] [cyan]Ctrl+Y[/cyan] respuesta del chat · "
            "[cyan]Ctrl+B[/cyan] modelo/servidor resaltado"
        )
        self.write_log(
            "[b]Filtrar:[/b] [cyan]o[/cyan] solo disponibles (verde) · "
            "[cyan]p[/cyan] proveedor (Ollama, vLLM, llama.cpp…)"
        )
        self.query_one("#prompt", Input).focus()

    def rebuild_index(self):
        """Reconstruye modelo -> servidores que lo ofrecen."""
        self.model_hosts = {}
        for h in self.hosts:
            for m in (h.get("live_models") or []):
                self.model_hosts.setdefault(m, []).append(h)
        self.models = sorted(self.model_hosts.keys(),
                             key=lambda m: (-len(self.model_hosts[m]), m.lower()))

    def _hosts_for(self, model: str) -> list:
        """Servidores que ofrecen el modelo, aplicando el filtro de proveedor."""
        hs = self.model_hosts.get(model, [])
        if self.provider_filter:
            hs = [h for h in hs if h.get("provider", "ollama") == self.provider_filter]
        return hs

    def populate_models(self, filter_text: str = ""):
        lv = self.query_one("#models", ListView)
        lv.clear()
        ft = filter_text.lower().strip()
        for m in self.models:
            if ft and ft not in m.lower():
                continue
            hs = self._hosts_for(m)
            if not hs:  # ningún servidor de ese proveedor ofrece este modelo
                continue
            working = any((fo.get_test(h, m) or {}).get("ok") for h in hs)
            # Con "solo disponibles" activo, ocultar los que no funcionan.
            if self.only_working and not working:
                continue
            lv.append(ModelItem(m, len(hs), working))

    def populate_servers(self, model: str):
        lv = self.query_one("#servers", ListView)
        lv.clear()
        # Ordenados por velocidad: más rápido primero, fallos/no probados al final.
        candidates = sorted(self._hosts_for(model),
                            key=lambda h: fo.server_speed_key(h, model))
        for h in candidates:
            # Con "solo disponibles" activo, mostrar solo los que pasan la prueba.
            if self.only_working and not (fo.get_test(h, model) or {}).get("ok"):
                continue
            lv.append(ServerItem(h, model))

    def set_status(self, text: str):
        self.query_one("#status", Static).update(text)

    def write_log(self, renderable):
        self.query_one("#chatlog", RichLog).write(renderable)

    # --------------------------------------------------------------- eventos
    @on(Input.Changed, "#filter")
    def on_filter(self, event: Input.Changed):
        self.populate_models(event.value)

    @on(ListView.Selected, "#models")
    def on_model_selected(self, event: ListView.Selected):
        item = event.item
        if isinstance(item, ModelItem):
            self.model = item.model
            self.host = None
            self.populate_servers(item.model)
            n = len(self.model_hosts[item.model])
            self.set_status(f"Modelo: [b]{item.model}[/b] — elige uno de {n} servidores.")
            self.query_one("#servers", ListView).focus()

    @on(ListView.Selected, "#servers")
    def on_server_selected(self, event: ListView.Selected):
        item = event.item
        if isinstance(item, ServerItem):
            self.host = item.host
            self.messages = []
            self.set_status(
                f"Listo: [b]{self.model}[/b] @ [b]{item.host['ip']}:{item.host['port']}[/b] "
                f"· contexto reiniciado. Escribe abajo ↓"
            )
            self.write_log(f"[dim]── Conectado a {self.model} @ "
                     f"{item.host['ip']}:{item.host['port']} ──[/dim]")
            self.query_one("#prompt", Input).focus()

    @on(Input.Submitted, "#prompt")
    def on_prompt(self, event: Input.Submitted):
        text = event.value.strip()
        self.query_one("#prompt", Input).value = ""
        if not text:
            return
        if not (self.model and self.host):
            self.notify("Elige primero un modelo y un servidor.", severity="warning")
            return
        if self.busy:
            self.notify("Espera a que termine la respuesta actual.", severity="warning")
            return
        self.messages.append({"role": "user", "content": text})
        # El texto del usuario va como Text para no interpretar corchetes.
        self.write_log(Text.assemble(("Tú  ", "bold cyan"), text))
        self.busy = True
        self.query_one("#live", Static).update("[dim]…generando…[/dim]")
        self.stream_reply()

    # ------------------------------------------------------------- workers
    @work(thread=True, exclusive=True, group="chat")
    def stream_reply(self):
        worker = get_current_worker()
        host_url = f"http://{self.host['ip']}:{self.host['port']}"
        protocol = self.host.get("protocol", "ollama")
        parts = []
        for piece in fo.chat_stream_iter(host_url, self.model, self.messages,
                                         CHAT_TIMEOUT, protocol):
            if worker.is_cancelled:
                break
            if isinstance(piece, tuple):  # ("__error__", msg)
                self.call_from_thread(self.finish_error, piece[1])
                return
            parts.append(piece)
            self.call_from_thread(self.update_live, "".join(parts))
        self.call_from_thread(self.finish_reply, "".join(parts))

    def update_live(self, text: str):
        # Durante el streaming mostramos el texto tal cual (sin interpretar
        # markup); el render final de Markdown se hace en finish_reply.
        self.query_one("#live", Static).update(Text(text))

    def finish_reply(self, text: str):
        self.query_one("#live", Static).update("")
        self.busy = False
        if text:
            self.messages.append({"role": "assistant", "content": text})
            self.last_reply = text
            # Cabecera con estilo propio (segura, es texto nuestro).
            self.write_log(f"[bold green]{self.model}[/bold green]  [dim](Ctrl+Y copia)[/dim]")
            # Cuerpo: si parece Markdown lo renderizamos; si no, texto plano
            # (con Text para no interpretar corchetes como markup de Rich).
            if looks_like_markdown(text):
                self.write_log(Markdown(text))
            else:
                self.write_log(Text(text))
            self.write_log("")

    def finish_error(self, msg: str):
        self.query_one("#live", Static).update("")
        self.busy = False
        # Deshacer el último mensaje del usuario para poder reintentar.
        if self.messages and self.messages[-1]["role"] == "user":
            self.messages.pop()
        self.write_log(f"[bold red]Error:[/bold red] {msg}")
        self.notify(msg, severity="error")

    @work(thread=True, exclusive=True, group="update")
    def do_update(self):
        # Siempre busca todos los proveedores; el filtro de proveedor ([p]) es
        # solo para la vista, así una actualización nunca pierde los demás.
        try:
            hosts = fo.fetch_from_shodan(limit=100, no_probe=False, timeout=5.0)
        except SystemExit as e:
            self.call_from_thread(self.notify, str(e), severity="error")
            self.call_from_thread(self._bg_clear)
            return
        fo.save_cache(hosts)
        self.call_from_thread(self.after_update, hosts)

    def _bg_clear(self):
        self.bg_busy = None

    def after_update(self, hosts: list):
        self.bg_busy = None
        self.hosts = hosts
        self.rebuild_index()
        self.populate_models(self.query_one("#filter", Input).value)
        self.set_status(f"Actualizado: {len(self.hosts)} servidores, "
                        f"{len(self.models)} modelos. Pulsa [t] para probar "
                        f"y marcar en verde los que funcionan.")
        self.notify("Lista actualizada desde Shodan y guardada en caché.")

    @work(thread=True, exclusive=True, group="test")
    def do_test(self):
        worker = get_current_worker()
        usables = [h for h in self.hosts if h.get("live_models")]
        pairs = [(h, m) for h in usables for m in h["live_models"]]
        total = len(pairs)
        workers = fo.env_test_threads()
        self.call_from_thread(
            self.write_log,
            f"[bold]── Probando {total} modelos ({workers} hilos) ──[/bold]")

        # Pre-crear el dict "tests" de cada host para que los hilos solo asignen
        # claves distintas (seguro con el GIL) sin competir por crearlo.
        for h in usables:
            h.setdefault("tests", {})

        results = []
        counter = {"done": 0}
        lock = threading.Lock()

        def do_one(pair):
            h, m = pair
            if worker.is_cancelled:
                return None
            res = fo.test_model(h["ip"], h["port"], m, 30.0, 12,
                                h.get("protocol", "ollama"))
            h["tests"][m] = res
            with lock:
                counter["done"] += 1
                i = counter["done"]
            if res["ok"]:
                tps = res["tokens_per_sec"]
                line = (f"[green]OK[/green] {h['ip']}:{h['port']}  {m}  "
                        f"[b]{tps:.1f}[/b] tok/s · {res['latency']:.1f}s"
                        if tps else
                        f"[green]OK[/green] {h['ip']}:{h['port']}  {m}  {res['latency']:.1f}s")
            else:
                line = f"[red]FALLO[/red] {h['ip']}:{h['port']}  {m}  ({res['error']})"
            self.call_from_thread(self.write_log, f"  [{i}/{total}] {line}")
            return (h, m, res)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(do_one, p) for p in pairs]
            for fut in concurrent.futures.as_completed(futures):
                r = fut.result()
                if r is not None:
                    results.append(r)
        self.call_from_thread(self.after_test, results)

    def after_test(self, results):
        self.bg_busy = None
        fo.save_cache(self.hosts)
        ok = [(h, m, r) for (h, m, r) in results if r["ok"]]
        ok.sort(key=lambda x: (x[2]["tokens_per_sec"] is None,
                               -(x[2]["tokens_per_sec"] or 0)))
        self.write_log(f"[bold]── Resultado: {len(ok)}/{len(results)} funcionan ──[/bold]")
        for h, m, r in ok[:10]:
            tps = f"{r['tokens_per_sec']:.1f}" if r["tokens_per_sec"] else "?"
            self.write_log(f"  [green]★[/green] {tps} tok/s  {h['ip']}:{h['port']}  {m}")
        # Refrescar las listas para reflejar los nuevos datos (modelos que
        # funcionan en verde, servidores ordenados por velocidad).
        self.populate_models(self.query_one("#filter", Input).value)
        if self.model:
            self.populate_servers(self.model)
        self.notify(f"Pruebas terminadas: {len(ok)}/{len(results)} OK.")

    # -------------------------------------------------------------- acciones
    def _bg_reject(self) -> bool:
        """Devuelve True (y avisa) si ya hay una operación de fondo en curso."""
        if self.bg_busy:
            nombre = "actualización" if self.bg_busy == "update" else "prueba"
            self.notify(f"Espera a que termine la {nombre} en curso.",
                        severity="warning")
            return True
        return False

    def action_update(self):
        if self._bg_reject():
            return
        self.bg_busy = "update"
        self.set_status("Consultando Shodan… (puede tardar)")
        self.notify("Buscando servidores en Shodan…")
        self.do_update()

    def action_test(self):
        if self._bg_reject():
            return
        if not any(h.get("live_models") for h in self.hosts):
            self.notify("No hay modelos que probar. Actualiza con [u].", severity="warning")
            return
        self.bg_busy = "test"
        self.notify("Probando modelos… mira el registro.")
        self.do_test()

    def action_toggle_working(self):
        self.only_working = not self.only_working
        # Recordar el modelo resaltado para no perder la posición.
        self.populate_models(self.query_one("#filter", Input).value)
        if self.model:
            self.populate_servers(self.model)
        if self.only_working:
            n = len(self.query_one("#models", ListView).children)
            self.set_status("Filtro: [b]solo disponibles[/b] (verde). "
                            "Pulsa [o] para ver todos.")
            if n == 0:
                self.notify("Nada disponible aún. Prueba con [t] primero.",
                            severity="warning")
            else:
                self.notify(f"Mostrando solo disponibles: {n} modelos.")
        else:
            self.set_status("Filtro: mostrando [b]todos[/b] los modelos.")
            self.notify("Mostrando todos los modelos y servidores.")

    def action_cycle_provider(self):
        # Avanza al siguiente proveedor del ciclo (None = todos).
        i = PROVIDER_CYCLE.index(self.provider_filter)
        self.provider_filter = PROVIDER_CYCLE[(i + 1) % len(PROVIDER_CYCLE)]
        self.model = None
        self.host = None
        self.query_one("#servers", ListView).clear()
        self.populate_models(self.query_one("#filter", Input).value)
        nombre = self.provider_filter or "todos"
        n = len(self.query_one("#models", ListView).children)
        # ¿Cuántos servidores hay de ese proveedor?
        if self.provider_filter:
            srv = sum(1 for h in self.hosts
                      if h.get("provider", "ollama") == self.provider_filter)
        else:
            srv = len(self.hosts)
        self.set_status(f"Proveedor: [b]{nombre}[/b] · {n} modelos, {srv} servidores. "
                        f"Pulsa [p] para cambiar.")
        self.notify(f"Proveedor: {nombre} ({srv} servidores).")

    def action_reset(self):
        self.messages = []
        self.write_log("[dim]── contexto de chat reiniciado ──[/dim]")
        self.notify("Contexto reiniciado.")

    def action_clear(self):
        self.query_one("#chatlog", RichLog).clear()

    # ----------------------------------------------------- navegación teclado
    def action_focus_models(self):
        self.query_one("#models", ListView).focus()

    def action_focus_servers(self):
        self.query_one("#servers", ListView).focus()

    def action_focus_filter(self):
        self.query_one("#filter", Input).focus()

    def action_focus_prompt(self):
        self.query_one("#prompt", Input).focus()

    # ----------------------------------------------------- copiar portapapeles
    def action_copy_reply(self):
        if not self.last_reply:
            self.notify("Aún no hay respuesta que copiar.", severity="warning")
            return
        self.copy_to_clipboard(self.last_reply)
        self.notify("Última respuesta copiada al portapapeles.")

    def action_copy_target(self):
        """Copia según el contexto: el modelo o servidor resaltado en su lista,
        o si no, el par activo 'modelo @ http://ip:port'."""
        focused = self.focused
        if isinstance(focused, ListView):
            item = focused.highlighted_child
            if isinstance(item, ModelItem):
                self.copy_to_clipboard(item.model)
                self.notify(f"Copiado modelo: {item.model}")
                return
            if isinstance(item, ServerItem):
                addr = f"{item.host['ip']}:{item.host['port']}"
                self.copy_to_clipboard(addr)
                self.notify(f"Copiado servidor: {addr}")
                return
        if self.model and self.host:
            text = f"{self.model} @ http://{self.host['ip']}:{self.host['port']}"
            self.copy_to_clipboard(text)
            self.notify(f"Copiado: {text}")
        else:
            self.notify("Sitúate en un modelo o servidor para copiarlo.",
                        severity="warning")


if __name__ == "__main__":
    OllamaTUI().run()
