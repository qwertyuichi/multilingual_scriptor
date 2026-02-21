"""エントリポイント。

python main.py でアプリケーションを起動する。
"""
import sys

from PySide6.QtCore import QLoggingCategory
from PySide6.QtWidgets import QApplication

from core.logging_config import setup_logging
from ui.app import VideoTranscriptionApp

QLoggingCategory.setFilterRules("qt.multimedia.ffmpeg=false")


def main() -> None:
    setup_logging()
    app = QApplication(sys.argv)
    window = VideoTranscriptionApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
