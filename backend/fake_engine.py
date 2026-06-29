"""Canned-signal engine so the frontend runs with no model loaded.

Implements only the early stages — list_models / load_model / run (with fake output and
fake "neurons firing") — and deliberately leaves the debugger stages (continue_, step,
set_breakpoint, inspect, break_in) inherited from Engine, where they raise
NotImplementedError. That proves the UI degrades gracefully: typing `bp` or `g` logs
"not wired up yet" instead of crashing.

Swap `FakeEngine` for the real `Engine` in app.py once the backend is implemented.
Note: this fakes everything on the GUI thread with QTimers — the real engine must run the
forward pass on a worker QThread instead (see core.py).
"""

import os
import numpy as np
from PySide6.QtCore import QTimer

from backend.core import Engine

# Two fake model profiles: one standard MHA (GPT-2 style), one GQA (Llama style).
# The ready payload now mirrors what AutoConfig.from_pretrained() would return so the UI
# can be built and tested against realistic field names before the real engine exists.
_FAKE_MODELS = [
    {
        "name": "gpt2-fake",
        "model_type": "gpt2",
        "layers": 12, "hidden": 768,
        "heads": 12,  "kv_heads": 12,   # standard MHA: kv_heads == heads
        "mlp_size": 3072,                # 4 × hidden (GPT-2 style)
        "vocab_size": 50257, "max_seq": 1024,
    },
    {
        "name": "llama-7b-fake",
        "model_type": "llama",
        "layers": 32, "hidden": 4096,
        "heads": 32,  "kv_heads": 32,   # Llama-1/2 7B uses MHA; 70B would have kv_heads=8
        "mlp_size": 11008,               # SwiGLU: ~8/3 × hidden
        "vocab_size": 32000, "max_seq": 4096,
    },
]


class FakeEngine(Engine):
    def __init__(self):
        super().__init__()
        self._gen = QTimer(self)
        self._gen.timeout.connect(self._tick)
        self._loader = QTimer(self)
        self._loader.timeout.connect(self._load_tick)
        self._load_frac = 0.0
        self._words = []
        self._i = 0
        self._profile = _FAKE_MODELS[0]

    def list_models(self, directory: str):
        # Real engine: walk directory, find subdirs with config.json, return full paths.
        # Fake: ignore directory, return our two canned profiles so the UI works offline.
        entries = [{"name": m["name"], "path": os.path.join(directory, m["name"])}
                   for m in _FAKE_MODELS]
        self.models_listed.emit(entries)

    def load_model(self, path: str):
        # Real engine: AutoConfig.from_pretrained(path) then load weights on worker thread.
        # Fake: pick profile by matching name substring in path.
        self._profile = next(
            (m for m in _FAKE_MODELS if m["name"] in path), _FAKE_MODELS[0])
        self._load_frac = 0.0
        self._loader.start(120)

    def _load_tick(self):
        self._load_frac += 0.15
        if self._load_frac >= 1.0:
            self._loader.stop()
            self.load_progress.emit(1.0)
            self.ready.emit(dict(self._profile))   # full AutoConfig-shaped dict
        else:
            self.load_progress.emit(self._load_frac)

    def run(self, prompt):
        self.tokenized.emit(prompt.split())
        self.status.emit("running (fake)…")
        self._words = "the quick brown fox jumps over the lazy dog".split()
        self._i = 0
        self._gen.start(250)

    def _tick(self):
        layers = self._profile["layers"]
        hidden = self._profile["hidden"]
        if self._i >= len(self._words):
            self._gen.stop()
            self.status.emit("idle")
            self.finished.emit()
            return
        self.status.emit(f"generating token {self._i + 1}/{len(self._words)}")
        self.output.emit(self._words[self._i] + " ")
        # one activation frame per layer — the real engine emits these from its hooks.
        for layer in range(layers):
            vals = np.abs(np.random.randn(hidden)) * (0.4 + layer / layers)
            self.activations.emit({"layer": layer, "summary": {"values": vals.tolist()}})
        self._i += 1
