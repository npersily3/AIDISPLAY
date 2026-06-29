"""Model-select page.

Row 1: directory text field + Browse button + Scan button.
Row 2: model combo (filled by models_listed) + Load button.
Row 3: progress bar + status label.

The directory defaults to the standard HuggingFace local cache. Clicking Scan calls
engine.list_models(directory); model_chosen emits the FULL PATH to the chosen model so
the engine can pass it straight to AutoConfig.from_pretrained(path).
"""

import os
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QComboBox, QPushButton, QProgressBar, QLineEdit,
                               QFileDialog)

_DEFAULT_HF_DIR = os.path.expanduser("~/.cache/huggingface/hub")


class Startup(QWidget):
    scan_requested  = Signal(str)   # directory path -> engine.list_models(dir)
    model_chosen    = Signal(str)   # full model path -> engine.load_model(path)

    def __init__(self):
        super().__init__()
        self._directory = _DEFAULT_HF_DIR if os.path.isdir(_DEFAULT_HF_DIR) else os.path.expanduser("~")
        self._names = []   # display names parallel to full paths

        title = QLabel("LLM Visualizer — WinDBG for LLM inference")
        title.setStyleSheet("font-size:20px;")
        title.setAlignment(Qt.AlignCenter)

        # --- directory row ---
        dir_label = QLabel("Model dir:")
        self.dir_edit = QLineEdit(self._directory)
        self.dir_edit.textChanged.connect(self._dir_changed)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse)
        scan_btn = QPushButton("Scan")
        scan_btn.clicked.connect(self._scan)

        dir_row = QHBoxLayout()
        dir_row.addWidget(dir_label)
        dir_row.addWidget(self.dir_edit, 1)
        dir_row.addWidget(browse_btn)
        dir_row.addWidget(scan_btn)

        # --- model row ---
        self.combo = QComboBox()
        self.load_btn = QPushButton("Load model")
        self.load_btn.clicked.connect(self._choose)

        model_row = QHBoxLayout()
        model_row.addWidget(self.combo, 1)
        model_row.addWidget(self.load_btn)

        # --- status row ---
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.info = QLabel("Set a model directory and click Scan")
        self.info.setAlignment(Qt.AlignCenter)

        lay = QVBoxLayout(self)
        lay.addStretch()
        lay.addWidget(title)
        lay.addSpacing(16)
        lay.addLayout(dir_row)
        lay.addLayout(model_row)
        lay.addWidget(self.bar)
        lay.addWidget(self.info)
        lay.addStretch()

    def _dir_changed(self, text):
        self._directory = text.strip()

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Select model directory", self._directory)
        if d:
            self._directory = d
            self.dir_edit.setText(d)

    def _scan(self):
        self.info.setText(f"Scanning {self._directory}…")
        self.scan_requested.emit(self._directory)

    def set_models(self, entries):
        """entries: list of {"name": display_str, "path": full_path}"""
        self.combo.clear()
        self._names = entries
        self.combo.addItems([e["name"] if isinstance(e, dict) else str(e) for e in entries])
        n = len(entries)
        self.info.setText(f"{n} model(s) found" if n else "No models found in that directory")

    def set_progress(self, frac):
        self.bar.setValue(int(frac * 100))
        self.info.setText(f"Loading… {int(frac * 100)}%")

    def _choose(self):
        i = self.combo.currentIndex()
        if i < 0:
            return
        entry = self._names[i] if i < len(self._names) else None
        if entry is None:
            return
        path = entry["path"] if isinstance(entry, dict) else os.path.join(self._directory, entry)
        self.info.setText(f"Loading {self.combo.currentText()}…")
        self.model_chosen.emit(path)
