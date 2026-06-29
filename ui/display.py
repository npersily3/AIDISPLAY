"""Central pane: status strip on top, token stream + architecture accordion below.

The token stream shows the prompt then the generated output appended live. The
architecture view lists every layer (collapsed) and lets you expand any of them to see
all of that layer's nodes — see ui/architecture.py.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSplitter, QPlainTextEdit

from ui.architecture import ArchitectureView


class Display(QWidget):
    def __init__(self):
        super().__init__()
        self.status = QLabel("idle")
        self.status.setStyleSheet("padding:4px; background:#111; color:#6f6;")

        self.tokens = QPlainTextEdit()
        self.tokens.setReadOnly(True)
        self.tokens.setStyleSheet("font-family: Consolas, monospace;")

        self.arch = ArchitectureView()

        split = QSplitter(Qt.Vertical)
        split.addWidget(self.tokens)
        split.addWidget(self.arch)
        split.setSizes([240, 460])

        lay = QVBoxLayout(self)
        lay.addWidget(self.status)
        lay.addWidget(split)

    def set_status(self, text):
        self.status.setText(text)

    def set_tokens(self, toks):
        self.tokens.setPlainText("PROMPT: " + " ".join(str(t) for t in toks) + "\n\nOUTPUT: ")

    def append_output(self, tok):
        self.tokens.moveCursor(QTextCursor.End)
        self.tokens.insertPlainText(tok)

    def set_architecture(self, cfg):
        self.arch.set_architecture(cfg)

    def update_layer(self, idx, summary):
        self.arch.update_layer(idx, summary)

    def focus_layer(self, idx):
        self.arch.focus_layer(idx)

    def reset(self):
        self.tokens.clear()
        self.status.setText("idle")
        self.arch.reset()
