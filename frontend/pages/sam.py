"""SAM 分割页面."""

from PySide6.QtWidgets import (
    QLabel, QPushButton, QHBoxLayout, QComboBox, QDoubleSpinBox, QGroupBox,
    QFormLayout, QProgressBar, QLineEdit,
)
from PySide6.QtCore import Qt

from .base_page import BasePage


class SAMPage(BasePage):
    def __init__(self):
        super().__init__("SAM 分割")

        # 项目选择
        row = QHBoxLayout()
        row.addWidget(QLabel("目标项目:"))
        self.project_combo = QComboBox()
        self.project_combo.setEditable(True)
        self.project_combo.setMinimumWidth(200)
        row.addWidget(self.project_combo)
        row.addStretch()
        self.content_layout.addLayout(row)

        # 参数
        group = QGroupBox("推理参数")
        form = QFormLayout(group)

        self.prompt_edit = QLineEdit()
        self.prompt_edit.setPlaceholderText("输入文字提示，如: steel plate")
        form.addRow("文字提示:", self.prompt_edit)

        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.1, 1.0)
        self.confidence_spin.setValue(0.5)
        self.confidence_spin.setSingleStep(0.05)
        form.addRow("置信度:", self.confidence_spin)

        self.content_layout.addWidget(group)

        # 操作
        row2 = QHBoxLayout()
        self.run_btn = QPushButton("▶ 运行 SAM")
        self.run_btn.setObjectName("primaryBtn")
        row2.addStretch()
        row2.addWidget(self.run_btn)
        self.content_layout.addLayout(row2)

        # 进度
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.content_layout.addWidget(self.progress)

        # 日志
        self.log_label = QLabel("就绪")
        self.log_label.setObjectName("logArea")
        self.log_label.setMinimumHeight(150)
        self.log_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.log_label.setWordWrap(True)
        self.content_layout.addWidget(self.log_label, 1)
