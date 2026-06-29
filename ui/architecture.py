"""Architecture pane — a layer rail on the left, a cause→effect slice on the right.

LEFT RAIL: every layer as a compact row — a clickable breakpoint dot (WinDBG-style
gutter), the layer label, and a 'firing' dot. Scales to hundreds of layers; this is the
full-architecture spine you compare across models. Click a row to select it; click the
gutter dot to toggle a breakpoint.

RIGHT DETAIL: the selected layer drawn with its previous and next neighbours as three
columns of nodes with edges between them — Neural-Network-Zoo style, on a small slice so
it actually renders. Columns are colored by role (cause / current / effect). Only this
3-layer slice is ever drawn heavy, so cost never grows with model size.

Built from the `ready` config; live activations fill the firing dots and the slice via
update_layer(). Breakpoint toggles bubble up as breakpoint_toggled(idx, on).
"""

import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import QPainter, QColor, QPen
from PySide6.QtWidgets import (QWidget, QScrollArea, QVBoxLayout, QHBoxLayout, QLabel)

SLICE_NODES = 16   # nodes drawn per column in the detail view


# ---------- helpers ----------
def _node_color(hue, v):
    return QColor.fromHsvF(hue, 0.82, 0.28 + 0.67 * max(0.0, min(1.0, v)))


def _to_array(summary):
    if summary is None:
        return None
    if isinstance(summary, dict):
        for key in ("values", "neurons", "activations", "heatmap", "summary"):
            if summary.get(key) is not None:
                summary = summary[key]
                break
        else:
            return None
    a = np.asarray(summary, dtype=float).ravel()
    return a if a.size else None


def _norm(a):
    a = a - a.min()
    m = a.max()
    return a / m if m > 0 else a


def _sample(arr, k):
    """Evenly sample k values from a 1D array and normalize to 0..1."""
    if arr is None:
        return None
    a = np.asarray(arr, dtype=float).ravel()
    if a.size == 0:
        return None
    idx = np.linspace(0, a.size - 1, min(a.size, k)).round().astype(int)
    return _norm(a[idx])


def _spread(lo, hi, n):
    if n <= 1:
        return [(lo + hi) / 2]
    return list(np.linspace(lo, hi, n))


# ---------- pieces ----------
class ActivationDot(QWidget):
    def __init__(self):
        super().__init__()
        self._v = 0.0
        self.setFixedSize(14, 14)

    def set_value(self, v):
        self._v = float(v)
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(_node_color(0.34, self._v))
        p.drawEllipse(2, 2, 10, 10)


class BreakpointDot(QWidget):
    """The gutter breakpoint marker: hollow ring = off, red disc = set."""
    toggled = Signal(bool)

    def __init__(self):
        super().__init__()
        self._on = False
        self.setFixedSize(18, 18)
        self.setCursor(Qt.PointingHandCursor)

    def set_on(self, on):
        self._on = bool(on)
        self.update()

    def mousePressEvent(self, e):
        self._on = not self._on
        self.update()
        self.toggled.emit(self._on)
        e.accept()   # don't let the row treat this as a 'select' click

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        if self._on:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#e2463c"))
            p.drawEllipse(4, 4, 10, 10)
        else:
            p.setPen(QColor("#555"))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(4, 4, 10, 10)


class RailItem(QWidget):
    clicked = Signal(int)
    bp_toggled = Signal(int, bool)

    def __init__(self, idx, cfg=None):
        super().__init__()
        self._idx = idx
        self._selected = False
        self.setObjectName("railitem")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setCursor(Qt.PointingHandCursor)

        self.bp = BreakpointDot()
        self.bp.toggled.connect(lambda on: self.bp_toggled.emit(self._idx, on))
        self.label = QLabel(_rail_label(idx, cfg))
        self.label.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        self.fire = ActivationDot()

        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 1, 4, 1)
        lay.setSpacing(4)
        lay.addWidget(self.bp)
        lay.addWidget(self.label, 1)
        lay.addWidget(self.fire)
        self._restyle()

    def mousePressEvent(self, _e):
        self.clicked.emit(self._idx)

    def set_selected(self, on):
        self._selected = on
        self._restyle()

    def set_firing(self, v):
        self.fire.set_value(v)

    def set_bp(self, on):
        self.bp.set_on(on)

    def _restyle(self):
        bg = "#1d3a1d" if self._selected else "transparent"
        self.setStyleSheet(f"#railitem{{background:{bg};}} QLabel{{color:#cfc;}}")


def _rail_label(idx, cfg):
    if not cfg:
        return f"Layer {idx}"
    heads = cfg.get("heads", "")
    kv = cfg.get("kv_heads")
    mlp = cfg.get("mlp_size")
    head_str = f"{heads}H" + (f"/{kv}KV" if kv and kv != heads else "")
    mlp_str = f" · MLP {mlp}" if mlp else ""
    return f"L{idx}  {head_str}{mlp_str}"


class LayerRail(QScrollArea):
    selected = Signal(int)
    breakpoint = Signal(int, bool)

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        host = QWidget()
        self._lay = QVBoxLayout(host)
        self._lay.setContentsMargins(2, 2, 2, 2)
        self._lay.setSpacing(0)
        self._lay.addStretch()
        self.setWidget(host)
        self.items = []

    def build(self, n, cfg=None):
        for it in self.items:
            it.setParent(None)
        self.items.clear()
        for i in range(n):
            it = RailItem(i, cfg)
            it.clicked.connect(self.select)
            it.bp_toggled.connect(self.breakpoint)
            self.items.append(it)
            self._lay.insertWidget(self._lay.count() - 1, it)

    def select(self, idx):
        for it in self.items:
            it.set_selected(it._idx == idx)
        if 0 <= idx < len(self.items):
            self.ensureWidgetVisible(self.items[idx])
        self.selected.emit(idx)

    def set_firing(self, idx, v):
        if 0 <= idx < len(self.items):
            self.items[idx].set_firing(v)

    def set_bp(self, idx, on):
        if 0 <= idx < len(self.items):
            self.items[idx].set_bp(on)

    def clear_firing(self):
        for it in self.items:
            it.set_firing(0.0)


class SliceView(QWidget):
    """Draws prev → current → next as three columns of nodes with edges."""

    ROLES = [(0.55, "cause"), (0.34, "current"), (0.08, "effect")]  # hues: cyan/green/orange

    def __init__(self):
        super().__init__()
        self._cols = None
        self._layers = 0
        self.setMinimumHeight(240)

    def clear(self):
        self._cols = None
        self.update()

    def show_slice(self, cur_idx, prev, cur, nxt, layers):
        self._layers = layers
        self._cols = [(cur_idx - 1, prev), (cur_idx, cur), (cur_idx + 1, nxt)]
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#0c0c0c"))
        if not self._cols:
            p.setPen(QColor("#2f6f2f"))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "select a layer on the left to inspect it and its neighbours")
            return

        w, h = self.width(), self.height()
        xs = [w * 0.2, w * 0.5, w * 0.8]
        ys = _spread(48, h - 24, SLICE_NODES)
        vals = [_sample(arr, SLICE_NODES) for _idx, arr in self._cols]

        # edges between adjacent columns (brighter where both ends fire)
        for c in range(2):
            a, b = vals[c], vals[c + 1]
            if a is None or b is None:
                continue
            for i in range(len(a)):
                for j in range(len(b)):
                    s = float(a[i] * b[j])
                    if s < 0.06:
                        continue
                    p.setPen(QPen(QColor(120, 220, 160, int(60 * s)), 1))
                    p.drawLine(QPointF(xs[c], ys[i]), QPointF(xs[c + 1], ys[j]))

        # column headers + nodes
        for c, (idx, _arr) in enumerate(self._cols):
            hue, role = self.ROLES[c]
            if 0 <= idx < self._layers:
                head = f"Layer {idx}  ·  {role}"
            else:
                head = "(input)" if idx < 0 else "(output)"
            p.setPen(QColor("#9c9"))
            p.drawText(QRectF(xs[c] - 75, 8, 150, 20), Qt.AlignCenter, head)

            col = vals[c]
            if col is None:
                continue
            emph = (c == 1)
            p.setPen(Qt.NoPen)
            for i, v in enumerate(col):
                v = float(v)
                cx, cy = xs[c], ys[i]
                base = 5 if emph else 3
                r = base + base * v
                if emph:                       # glow on the current layer
                    p.setBrush(QColor(60, 255, 130, int(70 * v)))
                    p.drawEllipse(QPointF(cx, cy), r * 2.2, r * 2.2)
                p.setBrush(_node_color(hue, v))
                p.drawEllipse(QPointF(cx, cy), r, r)


class ArchitectureView(QWidget):
    breakpoint_toggled = Signal(int, bool)

    def __init__(self):
        super().__init__()
        self.rail = LayerRail()
        self.rail.setFixedWidth(160)
        self.slice = SliceView()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(self.rail)
        lay.addWidget(self.slice, 1)

        self.rail.selected.connect(self._show)
        self.rail.breakpoint.connect(self.breakpoint_toggled)
        self._acts = []
        self._sel = None
        self._layers = 0

    def set_architecture(self, cfg):
        self._layers = int(cfg.get("layers", 0))
        self._acts = [None] * self._layers
        self._sel = None
        self.rail.build(self._layers, cfg)
        self.slice.clear()

    def update_layer(self, idx, summary):
        if not (0 <= idx < self._layers):
            return
        arr = _to_array(summary)
        self._acts[idx] = arr
        if arr is not None:
            self.rail.set_firing(idx, float(_norm(arr).mean()))
        if self._sel is not None and abs(idx - self._sel) <= 1:
            self._refresh()

    def set_breakpoint(self, idx, on):
        self.rail.set_bp(idx, on)

    def focus_layer(self, idx):
        if 0 <= idx < self._layers:
            self.rail.select(idx)

    def reset(self):
        self._acts = [None] * self._layers
        self.rail.clear_firing()
        self.slice.clear()

    def _show(self, idx):
        self._sel = idx
        self._refresh()

    def _refresh(self):
        s = self._sel
        if s is None:
            return
        prev = self._acts[s - 1] if s - 1 >= 0 else None
        nxt = self._acts[s + 1] if s + 1 < self._layers else None
        self.slice.show_slice(s, prev, self._acts[s], nxt, self._layers)
