"""点云过滤页面（方案 A）."""

from PySide6.QtWidgets import (
    QLabel, QPushButton, QHBoxLayout, QComboBox, QDoubleSpinBox,
    QLineEdit, QGroupBox, QFormLayout,
)
from PySide6.QtCore import Qt

from .base_page import BasePage


class FilterPage(BasePage):
    def __init__(self):
        super().__init__("点云过滤（方案 A）")

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
        group = QGroupBox("过滤参数")
        form = QFormLayout(group)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setValue(0.3)
        self.threshold_spin.setSingleStep(0.05)
        form.addRow("分数阈值:", self.threshold_spin)

        self.prompt_edit = QLineEdit()
        self.prompt_edit.setPlaceholderText("SAM mask 的 prompt slug，如: center_object")
        self.prompt_edit.setText("center_object")
        form.addRow("Prompt Slug:", self.prompt_edit)

        self.content_layout.addWidget(group)

        # 操作按钮
        row2 = QHBoxLayout()
        self.run_btn = QPushButton("▶ 运行点云过滤")
        self.run_btn.setObjectName("primaryBtn")
        row2.addStretch()
        row2.addWidget(self.run_btn)
        self.content_layout.addLayout(row2)

        # 结果
        self.result_label = QLabel("等待运行...")
        self.result_label.setObjectName("logArea")
        self.result_label.setMinimumHeight(200)
        self.result_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.result_label.setWordWrap(True)
        self.content_layout.addWidget(self.result_label, 1)
