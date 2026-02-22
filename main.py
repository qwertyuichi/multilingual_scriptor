"""エントリポイント。

python main.py でアプリケーションを起動する。
"""
import os
import sys

# OpenMP競合を回避（CPUモード実行時のクラッシュ防止）
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

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
