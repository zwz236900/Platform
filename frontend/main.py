"""PySide6 前端入口."""

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QIcon

from app import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Platform")
    app.setOrganizationName("Platform")

    # 全局字体
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # 加载样式
    style_path = Path(__file__).parent / "styles" / "theme.qss"
    if style_path.exists():
        app.setStyleSheet(style_path.read_text(encoding="utf-8"))

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
