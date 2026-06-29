"""The WinDBG-style command window: scrollback + a single command line.

Dumb on purpose — it only emits the raw command string. main.py parses it and decides
which Engine method to call, so all the FE->BE wiring lives in one place.
"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPlainTextEdit, QLineEdit


class Terminal(QWidget):
    command = Signal(str)

    def __init__(self):
        super().__init__()
        self.out = QPlainTextEdit()
        self.out.setReadOnly(True)
        self.out.setStyleSheet("font-family: Consolas, monospace;")
        self.inp = QLineEdit()
        self.inp.setPlaceholderText("run <prompt> · bp <layer> · g · p · inspect <ref> · break")
        self.inp.returnPressed.connect(self._submit)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.out)
        lay.addWidget(self.inp)

    def _submit(self):
        text = self.inp.text().strip()
        if not text:
            return
        self.log("> " + text)
        self.inp.clear()
        self.command.emit(text)

    def log(self, text):
        self.out.appendPlainText(text)
