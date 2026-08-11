"""页面基类."""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt


class BasePage(QWidget):
    """所有页面的基类，提供统一的标题 + 内容布局."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(16)

        # 页面标题
        self.title_label = QLabel(title)
        self.title_label.setObjectName("pageTitle")
        layout.addWidget(self.title_label)

        # 子类填充内容到这里
        self.content_layout = QVBoxLayout()
        self.content_layout.setSpacing(12)
        layout.addLayout(self.content_layout)

        layout.addStretch()
