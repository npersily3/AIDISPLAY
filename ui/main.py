"""MainWindow — the only place that knows about the Engine contract.

It connects every BE->FE signal to a slot and translates terminal commands into FE->BE
method calls. All engine calls go through _safe(): the contract methods raise
NotImplementedError until the backend fills them in, so _safe() turns "not wired up yet"
into a terminal line instead of a crash. That's what lets the app run with only the early
stages (list/load/run) implemented while breakpoints/inspect are still stubs.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QMainWindow, QStackedWidget, QDockWidget, QListWidget,
                               QWidget, QVBoxLayout, QLabel)

import re

from ui.startup import Startup
from ui.display import Display
from ui.terminal import Terminal


def _cfg_text(cfg):
    """Multi-line model config summary for the misc dock, using all AutoConfig fields."""
    lines = []
    if cfg.get("model_type"):
        lines.append(f"type:    {cfg['model_type']}")
    lines.append(f"layers:  {cfg.get('layers', '?')}")
    lines.append(f"hidden:  {cfg.get('hidden', '?')}")
    heads = cfg.get("heads", "?")
    kv = cfg.get("kv_heads")
    if kv and kv != heads:
        lines.append(f"heads:   {heads}  (KV: {kv} — GQA)")
    else:
        lines.append(f"heads:   {heads}")
    if cfg.get("mlp_size"):
        lines.append(f"MLP:     {cfg['mlp_size']}")
    if cfg.get("vocab_size"):
        lines.append(f"vocab:   {cfg['vocab_size']:,}")
    if cfg.get("max_seq"):
        lines.append(f"max seq: {cfg['max_seq']:,}")
    return "\n".join(lines)


def _layer_index(location):
    """Pull a layer number out of a location string like 'block.5' / 'layer 5.attn'."""
    m = re.search(r"\d+", str(location))
    return int(m.group()) if m else None


class MainWindow(QMainWindow):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.setWindowTitle("LLM Visualizer")
        self.resize(1200, 800)

        # central: startup page <-> debugger page
        self.startup = Startup()
        self.display = Display()
        self.stack = QStackedWidget()
        self.stack.addWidget(self.startup)
        self.stack.addWidget(self.display)
        self.setCentralWidget(self.stack)

        # docks (hidden until a model is ready)
        self.terminal = Terminal()
        tdock = QDockWidget("Terminal", self)
        tdock.setWidget(self.terminal)
        self.addDockWidget(Qt.RightDockWidgetArea, tdock)

        self.bp_list = QListWidget()
        self.cfg = QLabel("no model loaded")
        self.location = QLabel("location: —")
        misc = QWidget()
        ml = QVBoxLayout(misc)
        ml.addWidget(QLabel("Breakpoints"))
        ml.addWidget(self.bp_list)
        ml.addWidget(self.cfg)
        ml.addWidget(self.location)
        mdock = QDockWidget("Breakpoints & misc", self)
        mdock.setWidget(misc)
        self.splitDockWidget(tdock, mdock, Qt.Vertical)
        self._docks = (tdock, mdock)
        for d in self._docks:
            d.hide()

        self._wire()

    # ----- wiring -----
    def _wire(self):
        e = self.engine
        # FE -> BE
        self.startup.scan_requested.connect(lambda d: self._safe(e.list_models, d))
        self.startup.model_chosen.connect(lambda p: self._safe(e.load_model, p))
        self.terminal.command.connect(self._on_command)
        self.display.arch.breakpoint_toggled.connect(self._on_bp_toggled)
        # BE -> FE
        e.models_listed.connect(self.startup.set_models)  # list[{"name","path"}]
        e.load_progress.connect(self.startup.set_progress)
        e.ready.connect(self._on_ready)
        e.status.connect(self.display.set_status)
        e.tokenized.connect(self.display.set_tokens)
        e.paused.connect(self._on_paused)
        e.activations.connect(lambda d: self.display.update_layer(d.get("layer", -1), d.get("summary", d)))
        e.output.connect(self.display.append_output)
        e.finished.connect(lambda: self.terminal.log("[done]"))
        e.value.connect(lambda ref, data: self.terminal.log(f"{ref} = {data!r}"))
        e.error.connect(lambda m: self.terminal.log("[error] " + m))

    def _safe(self, fn, *args):
        try:
            fn(*args)
        except NotImplementedError:
            self.terminal.log(f"[{getattr(fn, '__name__', fn)}] not wired up yet")
        except Exception as ex:
            self.terminal.log(f"[error] {ex}")

    # ----- BE -> FE slots -----
    def _on_ready(self, cfg):
        self.stack.setCurrentWidget(self.display)
        for d in self._docks:
            d.show()
        self.cfg.setText(_cfg_text(cfg))
        self.display.set_architecture(cfg)
        self.terminal.log("[model ready] type 'run <prompt>' to start")

    def _on_paused(self, payload):
        loc = payload.get("location", "?")
        self.location.setText(f"location: {loc}")
        self.terminal.log(f"[breakpoint] paused at {loc} — type 'g' to continue")
        idx = _layer_index(loc)
        if idx is not None:
            self.display.focus_layer(idx)
            self.display.update_layer(idx, payload.get("snapshot"))

    # ----- terminal command parsing (FE -> BE) -----
    def _on_command(self, text):
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""
        e = self.engine

        if cmd in ("run", "r") and arg:
            self.display.reset()
            self._safe(e.run, arg)
        elif cmd in ("g", "go"):
            self._safe(e.continue_)
        elif cmd in ("p", "step", "t"):
            self._safe(e.step)
        elif cmd == "bp" and arg:
            self._safe(e.set_breakpoint, arg)
            self.bp_list.addItem(arg)
            self._sync_rail_bp(arg, True)
        elif cmd == "bc" and arg:
            self._safe(e.clear_breakpoint, arg)
            self._remove_bp(arg)
            self._sync_rail_bp(arg, False)
        elif cmd == "inspect" and arg:
            self._safe(e.inspect, arg)
        elif cmd == "break":
            self._safe(e.break_in)
        else:
            self.terminal.log(f"[?] unknown command: {text}")

    def _remove_bp(self, name):
        for i in range(self.bp_list.count()):
            if self.bp_list.item(i).text() == name:
                self.bp_list.takeItem(i)
                return

    def _sync_rail_bp(self, target, on):
        """Reflect a terminal bp/bc command on the rail's gutter dot."""
        idx = _layer_index(target)
        if idx is not None:
            self.display.arch.set_breakpoint(idx, on)

    def _on_bp_toggled(self, idx, on):
        """A breakpoint dot was clicked on the rail -> tell the engine + the misc list."""
        target = f"block.{idx}"
        if on:
            self._safe(self.engine.set_breakpoint, target)
            self.bp_list.addItem(target)
        else:
            self._safe(self.engine.clear_breakpoint, target)
            self._remove_bp(target)
