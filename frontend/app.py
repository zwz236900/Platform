"""主窗口 — 左侧边栏 + 右侧内容区."""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QStackedWidget, QLabel, QVBoxLayout,
)
from PySide6.QtCore import Qt, QSize

from widgets.sidebar import Sidebar
from pages.projects import ProjectsPage
from pages.preprocess import PreprocessPage
from pages.sfm import SfMPage
from pages.mvs import MvsPage
from pages.sam import SAMPage
from pages.filter_page import FilterPage
from pages.mask import MaskPage
from pages.gs import GsPage

PAGES = [
    ("projects",     "项目管理",     ProjectsPage),
    ("preprocess",   "预处理",       PreprocessPage),
    ("sfm",          "SfM 重建",     SfMPage),
    ("mvs",          "MVS 重建",     MvsPage),
    ("gs",           "3DGS 重建",    GsPage),
    ("sam",          "SAM 分割",     SAMPage),
    ("filter",       "点云过滤",     FilterPage),
    ("mask",         "图像掩码",     MaskPage),
]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Platform — 3D Reconstruction Pipeline")
        self.resize(1200, 800)
        self.setMinimumSize(900, 600)

        # --- 中央容器 ---
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # --- 左侧边栏 ---
        self.sidebar = Sidebar([(key, label) for key, label, _ in PAGES])
        self.sidebar.setFixedWidth(220)
        self.sidebar.page_changed.connect(self._switch_page)

        # --- 右侧页面栈 ---
        self.stack = QStackedWidget()
        self.pages: dict[str, QWidget] = {}
        for key, label, PageCls in PAGES:
            page = PageCls()
            self.pages[key] = page
            self.stack.addWidget(page)

        root_layout.addWidget(self.sidebar)
        root_layout.addWidget(self.stack, 1)

        # 首页
        self.sidebar.set_active("projects")

    def _switch_page(self, page_key: str):
        page = self.pages.get(page_key)
        if page:
            self.stack.setCurrentWidget(page)
