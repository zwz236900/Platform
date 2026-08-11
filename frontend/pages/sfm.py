"""SfM 重建页面."""

from PySide6.QtWidgets import (
    QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QComboBox, QCheckBox,
    QSpinBox, QGroupBox, QFormLayout, QProgressBar,
)
from PySide6.QtCore import Qt

from .base_page import BasePage


class SfMPage(BasePage):
    def __init__(self):
        super().__init__("SfM 重建")

        # 项目选择
        row = QHBoxLayout()
        row.addWidget(QLabel("目标项目:"))
        self.project_combo = QComboBox()
        self.project_combo.setEditable(True)
        self.project_combo.setMinimumWidth(200)
        row.addWidget(self.project_combo)
        row.addStretch()
        self.content_layout.addLayout(row)

        # 参数设置
        group = QGroupBox("重建参数")
        form = QFormLayout(group)

        self.matcher_combo = QComboBox()
        self.matcher_combo.addItems(["exhaustive", "sequential", "vocab_tree", "spatial"])
        form.addRow("匹配策略:", self.matcher_combo)

        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["SIMPLE_RADIAL", "PINHOLE", "SIMPLE_PINHOLE", "RADIAL", "OPENCV"])
        form.addRow("相机模型:", self.camera_combo)

        self.gpu_check = QCheckBox("使用 GPU")
        self.gpu_check.setChecked(True)
        form.addRow("", self.gpu_check)

        self.single_cam_check = QCheckBox("单相机模式")
        self.single_cam_check.setChecked(True)
        form.addRow("", self.single_cam_check)

        self.features_spin = QSpinBox()
        self.features_spin.setRange(1000, 100000)
        self.features_spin.setValue(8192)
        self.features_spin.setSingleStep(1000)
        form.addRow("特征点数:", self.features_spin)

        self.content_layout.addWidget(group)

        # 操作按钮
        row2 = QHBoxLayout()
        self.run_btn = QPushButton("▶ 开始重建")
        self.run_btn.setObjectName("primaryBtn")
        row2.addStretch()
        row2.addWidget(self.run_btn)
        self.content_layout.addLayout(row2)

        # 进度条
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
