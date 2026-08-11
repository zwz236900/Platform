"""SfM 重建页面 — 参数配置与 pipeline 启动."""

from __future__ import annotations

from datetime import datetime

from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QComboBox,
    QCheckBox,
    QSpinBox,
    QGroupBox,
    QFormLayout,
    QProgressBar,
    QTextEdit,
    QMessageBox,
    QSizePolicy,
    QWidget,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont

from .base_page import BasePage
from .projects import ApiClient

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

MATCHERS = ["exhaustive", "sequential", "vocab_tree", "spatial"]
CAMERA_MODELS = ["SIMPLE_RADIAL", "PINHOLE", "SIMPLE_PINHOLE", "RADIAL", "OPENCV"]

STEP_LABELS = {
    "extract": "特征提取",
    "match": "特征匹配",
    "sfm": "稀疏重建",
    "undistort": "去畸变",
}


# ---------------------------------------------------------------------------
# SfM 重建页面
# ---------------------------------------------------------------------------

class SfMPage(BasePage):
    """SfM 重建 — 参数配置与 pipeline 启动。"""

    def __init__(self):
        super().__init__("SfM 重建")

        self._api = ApiClient.instance()
        self._api.finished.connect(self._on_response)
        self._running = False
        self._current_project = ""
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_log)
        self._poll_timer.setInterval(3000)
        self._poll_count = 0

        # ── 项目选择 ──
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("目标项目:"))
        self.project_combo = QComboBox()
        self.project_combo.setMinimumWidth(220)
        self.project_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row1.addWidget(self.project_combo)

        self._refresh_btn = QPushButton("🔄 刷新")
        self._refresh_btn.setObjectName("actionBtn")
        self._refresh_btn.clicked.connect(self._load_projects)
        row1.addWidget(self._refresh_btn)
        row1.addStretch()
        self.content_layout.addLayout(row1)

        # ── Pipeline 步骤 ──
        step_group = QGroupBox("重建步骤")
        step_layout = QHBoxLayout(step_group)
        self._step_checks: dict[str, QCheckBox] = {}
        for key, label in STEP_LABELS.items():
            cb = QCheckBox(label)
            cb.setChecked(True)
            self._step_checks[key] = cb
            step_layout.addWidget(cb)
        step_layout.addStretch()
        self.content_layout.addWidget(step_group)

        # ── 参数设置 ──
        params = QHBoxLayout()
        params.setSpacing(12)

        # 左列
        left_group = QGroupBox("匹配与相机")
        lform = QFormLayout(left_group)

        self.matcher_combo = QComboBox()
        self.matcher_combo.addItems(MATCHERS)
        lform.addRow("匹配策略:", self.matcher_combo)

        self.camera_combo = QComboBox()
        self.camera_combo.addItems(CAMERA_MODELS)
        lform.addRow("相机模型:", self.camera_combo)

        self.features_spin = QSpinBox()
        self.features_spin.setRange(1000, 100000)
        self.features_spin.setValue(8192)
        self.features_spin.setSingleStep(1000)
        lform.addRow("最大特征点数:", self.features_spin)
        params.addWidget(left_group)

        # 右列
        right_group = QGroupBox("选项")
        rform = QFormLayout(right_group)

        self.gpu_check = QCheckBox("使用 GPU")
        self.gpu_check.setChecked(True)
        rform.addRow("", self.gpu_check)

        self.single_cam_check = QCheckBox("单相机模式（共享内参）")
        self.single_cam_check.setChecked(True)
        rform.addRow("", self.single_cam_check)

        self.geo_check = QCheckBox("禁用几何一致性检查")
        self.geo_check.setChecked(False)
        rform.addRow("", self.geo_check)

        self.gpu_index_spin = QSpinBox()
        self.gpu_index_spin.setRange(0, 7)
        self.gpu_index_spin.setValue(0)
        rform.addRow("GPU 编号:", self.gpu_index_spin)
        params.addWidget(right_group)

        self.content_layout.addLayout(params)

        # ── 操作按钮 ──
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self._status_label = QLabel("就绪，请选择项目")
        self._status_label.setStyleSheet("color:#8b949e; font-size:13px;")
        btn_row.addWidget(self._status_label)

        btn_row.addStretch()

        self.run_btn = QPushButton("▶ 开始重建")
        self.run_btn.setObjectName("primaryBtn")
        self.run_btn.setMinimumWidth(120)
        self.run_btn.clicked.connect(self._on_run)
        btn_row.addWidget(self.run_btn)

        self.stop_btn = QPushButton("⏹ 停止")
        self.stop_btn.setObjectName("actionBtn")
        self.stop_btn.setMinimumWidth(100)
        self.stop_btn.setStyleSheet(
            "#actionBtn { color:#da3633; border-color:#da363333; }"
            "#actionBtn:hover { background-color:#da363322; }"
        )
        self.stop_btn.clicked.connect(self._on_stop)
        self.stop_btn.setVisible(False)
        btn_row.addWidget(self.stop_btn)
        self.content_layout.addLayout(btn_row)

        # ── 进度条 ──
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate
        self.progress.setVisible(False)
        self.content_layout.addWidget(self.progress)

        # ── 日志 ──
        log_header = QHBoxLayout()
        log_header.addWidget(QLabel("运行日志"))
        log_header.addStretch()
        clear_btn = QPushButton("清空")
        clear_btn.setObjectName("actionBtn")
        clear_btn.setFixedSize(50, 24)
        clear_btn.clicked.connect(lambda: self._log.clear())
        log_header.addWidget(clear_btn)
        self.content_layout.addLayout(log_header)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setObjectName("logArea")
        self._log.setMinimumHeight(150)
        self._log.setFont(QFont("Consolas", 10))
        self.content_layout.addWidget(self._log, 1)

        # ── 启动 ──
        self._load_projects()

    # ------------------------------------------------------------------
    # 项目列表
    # ------------------------------------------------------------------

    def _load_projects(self):
        self._refresh_btn.setEnabled(False)
        self._api.get("/api/projects", tag="list_sfm_projects")

    # ------------------------------------------------------------------
    # 启动 Pipeline
    # ------------------------------------------------------------------

    def _on_run(self):
        project = self.project_combo.currentText().strip()
        if not project:
            QMessageBox.warning(self, "提示", "请先选择一个项目。")
            return

        # 收集参数
        steps = [k for k, cb in self._step_checks.items() if cb.isChecked()]
        if not steps:
            QMessageBox.warning(self, "提示", "请至少选择一个重建步骤。")
            return

        config = {
            "steps": ",".join(steps),
            "matcher": self.matcher_combo.currentText(),
            "camera_model": self.camera_combo.currentText(),
            "single_camera": str(self.single_cam_check.isChecked()).lower(),
            "use_gpu": str(self.gpu_check.isChecked()).lower(),
            "gpu_index": str(self.gpu_index_spin.value()),
            "max_num_features": str(self.features_spin.value()),
            "no_geom_consistency": str(self.geo_check.isChecked()).lower(),
        }

        self._current_project = project
        self._set_running(True)
        self._log_msg(f"▶ 启动 Pipeline — 项目: {project}")
        self._log_msg(f"  步骤: {' → '.join(steps)}")
        self._log_msg(f"  匹配: {config['matcher']}, 相机: {config['camera_model']}")
        self._log_msg(f"  特征点: {config['max_num_features']}, GPU: {config['use_gpu']}")
        self._status_label.setText("Pipeline 启动中...")

        self._api.post(
            f"/api/projects/{project}/run",
            config,
            tag="run_pipeline",
        )

    def _on_stop(self):
        project = self._current_project
        if not project:
            return
        reply = QMessageBox.question(
            self,
            "确认停止",
            f"确定要停止项目「{project}」的 Pipeline 吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._log_msg("⏹ 正在停止 Pipeline...")
        self._status_label.setText("正在停止...")
        self._api.post(f"/api/projects/{project}/stop", tag="stop_pipeline")

    # ------------------------------------------------------------------
    # 轮询日志
    # ------------------------------------------------------------------

    def _poll_log(self):
        if not self._current_project:
            self._poll_timer.stop()
            return
        self._poll_count += 1
        # 每 5 次（15s）检查一次项目状态
        if self._poll_count % 5 == 0:
            self._api.get(
                f"/api/projects/{self._current_project}",
                tag="check_status",
            )
        else:
            self._api.get(
                f"/api/projects/{self._current_project}/log?lines=40",
                tag="poll_log",
            )

    # ------------------------------------------------------------------
    # 网络响应
    # ------------------------------------------------------------------

    def _on_response(self, method: str, path: str, payload: object):
        if not isinstance(payload, dict):
            return

        tag = payload.get("_tag", "")
        error = payload.get("_error", "")

        # ── 项目列表 ──
        if tag == "list_sfm_projects":
            self._refresh_btn.setEnabled(True)
            if error:
                return
            projects = payload.get("projects", [])
            current = self.project_combo.currentText()
            self.project_combo.clear()
            selected_idx = -1
            for i, p in enumerate(projects):
                pn = p.get("project_name", "")
                self.project_combo.addItem(pn)
                if pn == current:
                    selected_idx = i
            if selected_idx >= 0:
                self.project_combo.setCurrentIndex(selected_idx)

        # ── Pipeline 启动响应 ──
        elif tag == "run_pipeline":
            if error:
                self._set_running(False)
                self._log_msg(f"✗ 启动失败: {error}")
                self._status_label.setText("启动失败")
                QMessageBox.warning(self, "Pipeline 启动失败", str(error))
            else:
                self._log_msg("✓ Pipeline 已启动，等待完成...")
                self._status_label.setText("Pipeline 运行中...")
                self._poll_log()
                self._poll_timer.start()

        # ── 停止响应 ──
        elif tag == "stop_pipeline":
            if error:
                self._log_msg(f"✗ 停止失败: {error}")
                QMessageBox.warning(self, "停止失败", str(error))
            else:
                self._set_running(False)
                self._log_msg("⏹ Pipeline 已停止")
                self._status_label.setText("Pipeline 已停止")
                self._poll_timer.stop()

        # ── 日志轮询 ──
        elif tag == "poll_log":
            log_text = payload.get("log", "")
            if log_text and log_text != "(no log yet)":
                # 只追加新增的日志
                self._log.setPlainText(log_text)
                # 滚动到底部
                scrollbar = self._log.verticalScrollBar()
                if scrollbar:
                    scrollbar.setValue(scrollbar.maximum())

            # 检查是否已完成（通过检查项目状态）
            # 如果日志最后一行包含错误或完成信息
            if "FAILED" in log_text or "Pipeline exited" in log_text:
                self._set_running(False)
                self._log_msg("✗ Pipeline 失败，请检查日志。")
                self._status_label.setText("Pipeline 失败")

        # ── 项目详情（用于检查运行状态）──
        elif tag == "check_status":
            status = payload.get("status", "")
            if status in ("completed",):
                self._set_running(False)
                elapsed = (payload.get("pipeline") or {}).get("elapsed_seconds", 0)
                self._log_msg(f"✓ Pipeline 完成 — 耗时 {elapsed:.0f}s")
                self._status_label.setText("✓ Pipeline 已完成")
                self._poll_timer.stop()
            elif status in ("failed",):
                self._set_running(False)
                err = payload.get("error_message", "未知错误")
                self._log_msg(f"✗ Pipeline 失败: {err}")
                self._status_label.setText("Pipeline 失败")
                self._poll_timer.stop()

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def _log_msg(self, text: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self._log.append(f"[{ts}] {text}")

    def _set_running(self, running: bool):
        self._running = running
        self.run_btn.setEnabled(not running)
        self.stop_btn.setVisible(running)
        self.project_combo.setEnabled(not running)
        self.progress.setVisible(running)
        if running:
            self.run_btn.setText("⏳ 运行中...")
            self.progress.setRange(0, 0)
        else:
            self.run_btn.setText("▶ 开始重建")
            self.progress.setVisible(False)

    def showEvent(self, event):
        super().showEvent(event)
        self._load_projects()
        if not self._poll_timer.isActive():
            # 如果正在运行，恢复轮询
            pass

    def hideEvent(self, event):
        super().hideEvent(event)
        # 切走页面时也保持轮询（pipeline 在后台跑）
