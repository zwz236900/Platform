"""项目管理页面."""

from PySide6.QtWidgets import (
    QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QWidget, QScrollArea,
    QFrame,
)
from PySide6.QtCore import Qt

from .base_page import BasePage


class ProjectCard(QFrame):
    """项目卡片."""

    def __init__(self, name: str, status: str, num_images: int, parent=None):
        super().__init__(parent)
        self.setObjectName("projectCard")
        self.setMinimumHeight(80)
        self.setCursor(Qt.PointingHandCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 12, 20, 12)

        # 左侧信息
        info = QVBoxLayout()
        title = QLabel(name)
        title.setObjectName("cardTitle")
        info.addWidget(title)
        detail = QLabel(f"状态: {status}  ·  {num_images} 张图片")
        detail.setObjectName("cardDetail")
        info.addWidget(detail)
        layout.addLayout(info, 1)

        # 右侧状态标签
        badge = QLabel(status)
        badge.setObjectName(f"badge_{status}")
        badge.setAlignment(Qt.AlignCenter)
        badge.setFixedWidth(72)
        badge.setFixedHeight(28)
        layout.addWidget(badge)


class ProjectsPage(BasePage):
    def __init__(self):
        super().__init__("项目管理")

        # 操作栏
        toolbar = QHBoxLayout()
        self.refresh_btn = QPushButton("🔄 刷新")
        self.refresh_btn.setObjectName("primaryBtn")
        self.create_btn = QPushButton("＋ 新建项目")
        self.create_btn.setObjectName("primaryBtn")
        toolbar.addStretch()
        toolbar.addWidget(self.refresh_btn)
        toolbar.addWidget(self.create_btn)
        self.content_layout.addLayout(toolbar)

        # 提示
        hint = QLabel("连接后端 API 后，项目列表将在此显示。")
        hint.setObjectName("hint")
        hint.setAlignment(Qt.AlignCenter)
        self.content_layout.addWidget(hint)

        # 项目卡片容器（滚动区域）
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setObjectName("cardScroll")
        self.cards_widget = QWidget()
        self.cards_layout = QVBoxLayout(self.cards_widget)
        self.cards_layout.setSpacing(8)
        self.cards_layout.addStretch()
        scroll.setWidget(self.cards_widget)
        self.content_layout.addWidget(scroll, 1)
