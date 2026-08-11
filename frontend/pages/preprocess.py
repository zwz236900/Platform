"""预处理页面."""

from PySide6.QtWidgets import QLabel, QPushButton, QHBoxLayout, QComboBox
from PySide6.QtCore import Qt

from .base_page import BasePage


class PreprocessPage(BasePage):
    def __init__(self):
        super().__init__("预处理")

        # 项目选择
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("目标项目:"))
        self.project_combo = QComboBox()
        self.project_combo.setMinimumWidth(200)
        self.project_combo.setEditable(True)
        row1.addWidget(self.project_combo)
        row1.addStretch()
        self.content_layout.addLayout(row1)

        # 操作按钮
        row2 = QHBoxLayout()
        self.strip_btn = QPushButton("🧹 Strip Metadata")
        self.strip_btn.setObjectName("actionBtn")
        self.undistort_btn = QPushButton("📐 去畸变 (Undistort)")
        self.undistort_btn.setObjectName("actionBtn")
        row2.addWidget(self.strip_btn)
        row2.addWidget(self.undistort_btn)
        row2.addStretch()
        self.content_layout.addLayout(row2)

        # 输出日志区
        self.content_layout.addWidget(QLabel("输出日志"))
        self.log_label = QLabel("就绪，等待操作...")
        self.log_label.setObjectName("logArea")
        self.log_label.setMinimumHeight(200)
        self.log_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.log_label.setWordWrap(True)
        self.content_layout.addWidget(self.log_label, 1)
