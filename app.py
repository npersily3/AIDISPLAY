"""Entry point. Run:  python app.py

Uses FakeEngine so the UI is alive today. To run against the real backend, change the
import below to `from backend.core import Engine` (or wherever the implemented engine
lives) — nothing else in the frontend changes, because the contract is identical.
"""

import sys
from PySide6.QtWidgets import QApplication

from backend.fake_engine import FakeEngine as Engine
from ui.main import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet("QWidget { background:#1b1b1b; color:#ddd; }")
    win = MainWindow(Engine())
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
