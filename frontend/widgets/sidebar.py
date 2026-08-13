"""左侧导航侧边栏."""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QSizePolicy
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont, QIcon


class SidebarButton(QPushButton):
    """单个菜单按钮."""

    def __init__(self, key: str, label: str, icon_char: str = ""):
        text = f"  {icon_char}  {label}" if icon_char else f"  {label}"
        super().__init__(text)
        self._key = key
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setObjectName("sidebarBtn")
        self.setMinimumHeight(44)

    @property
    def page_key(self) -> str:
        return self._key


class Sidebar(QWidget):
    """侧边栏容器."""

    page_changed = Signal(str)

    ICONS = {
        "projects":    "📁",
        "preprocess":  "🖼",
        "sfm":         "🔍",
        "gs":          "✨",
        "sam":         "✂",
        "filter":      "🧹",
        "mask_obj":    "🖌",
    }

    def __init__(self, items: list[tuple[str, str]], parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self._buttons: dict[str, SidebarButton] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Logo / 标题
        title = QLabel("  Platform")
        title.setObjectName("sidebarTitle")
        title.setMinimumHeight(64)
        title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(title)

        # 分隔线
        sep = QWidget()
        sep.setObjectName("sidebarSep")
        sep.setFixedHeight(1)
        layout.addWidget(sep)

        # 菜单按钮
        layout.addSpacing(8)
        for key, label in items:
            icon = self.ICONS.get(key, "")
            btn = SidebarButton(key, label, icon)
            btn.clicked.connect(self._on_click)
            self._buttons[key] = btn
            layout.addWidget(btn)

        layout.addStretch()

        # 底部版本号
        ver = QLabel("  v1.0.0")
        ver.setObjectName("sidebarVersion")
        ver.setMinimumHeight(36)
        ver.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(ver)

    def set_active(self, key: str):
        for k, btn in self._buttons.items():
            btn.setChecked(k == key)
        self.page_changed.emit(key)

    def _on_click(self):
        btn = self.sender()
        if isinstance(btn, SidebarButton):
            self.set_active(btn.page_key)
