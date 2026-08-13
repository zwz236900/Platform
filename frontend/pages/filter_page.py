"""点云过滤页面 — 通过 SAM 掩码多视角一致性过滤稠密点云.

功能：参数配置 → 运行过滤 → 展示结果统计 → 下载过滤后的 PLY 点云。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QHBoxLayout,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QGroupBox,
    QFormLayout,
    QTextEdit,
    QMessageBox,
    QFileDialog,
    QProgressBar,
    QSizePolicy,
)
from PySide6.QtCore import Qt, QUrl
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply
from PySide6.QtGui import QFont

from .base_page import BasePage
from .projects import ApiClient
from state import ProjectState, fill_project_combo

API_BASE = "http://localhost:8000"

FILTERED_FILENAME = "filtered_scene.ply"

# 输入点云来源选项（value → 相对项目目录的路径；auto/custom 走特判）
SOURCE_OPTIONS = [
    ("auto",        "自动检测（按顺序找可用点云）"),
    ("colmap",      "COLMAP 稠密点云（fused.ply）"),
    ("openmvs",     "OpenMVS 稠密点云（scene_dense.ply）"),
    ("openmvs_mesh","OpenMVS 网格点云（scene_dense_mesh.ply）"),
    ("custom",      "自定义路径…"),
]
SOURCE_PATHS = {
    "colmap":       "MVS_output/colmap/fused.ply",
    "openmvs":      "MVS_output/openmvs/scene_dense.ply",
    "openmvs_mesh": "MVS_output/openmvs/scene_dense_mesh.ply",
}


def _format_size(n: int) -> str:
    """人类可读的文件大小。"""
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.2f} MB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.1f} KB"
    return f"{n} B"


class FilterPage(BasePage):
    """点云过滤 — 参数配置、运行与下载。"""

    def __init__(self):
        super().__init__("点云过滤")

        self._api = ApiClient.instance()
        self._api.finished.connect(self._on_response)
        self._state = ProjectState.instance()
        self._state.current_changed.connect(self._on_shared_project_changed)

        # 独立下载管理器（用于下载二进制 PLY，避免干扰 JSON 解析）
        self._dl_mgr = QNetworkAccessManager(self)
        self._dl_mgr.finished.connect(self._on_download_finished)

        self._running = False
        self._current_project = ""

        # ── 项目选择 ──
        row = QHBoxLayout()
        row.addWidget(QLabel("目标项目:"))
        self.project_combo = QComboBox()
        self.project_combo.setMinimumWidth(220)
        self.project_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.project_combo.currentTextChanged.connect(self._on_project_changed)
        row.addWidget(self.project_combo)

        self._refresh_btn = QPushButton("🔄 刷新")
        self._refresh_btn.setObjectName("actionBtn")
        self._refresh_btn.clicked.connect(self._load_projects)
        row.addWidget(self._refresh_btn)
        row.addStretch()
        self.content_layout.addLayout(row)

        # ── 参数 ──
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

        # 输入点云来源：选择过滤哪一份 MVS 点云（COLMAP / OpenMVS / 自定义）
        self.source_combo = QComboBox()
        for value, label in SOURCE_OPTIONS:
            self.source_combo.addItem(label, value)
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        form.addRow("输入点云:", self.source_combo)

        self.ply_path_edit = QLineEdit()
        self.ply_path_edit.setPlaceholderText(
            "服务器上的点云路径（绝对路径，或相对项目目录，如 MVS_output/xxx.ply）")
        self.ply_path_edit.setVisible(False)
        form.addRow("自定义路径:", self.ply_path_edit)

        self.content_layout.addWidget(group)

        # ── 说明 ──
        hint = QLabel(
            "将每个三维点投影到所有相机视图，统计其落入 SAM 掩码内的比例，"
            "比例 ≥ 阈值的点被保留。\n"
            "输入点云可选 COLMAP 或 OpenMVS 的稠密点云（默认自动检测）。\n"
            "需要：MVS 点云 + COLMAP 稀疏模型 + SAM 掩码（输出 algorithm_output/filtered_scene.ply）。"
        )
        hint.setStyleSheet("color:#8b949e; font-size:12px;")
        hint.setWordWrap(True)
        self.content_layout.addWidget(hint)

        # ── 操作按钮 ──
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self._status_label = QLabel("就绪，请选择项目")
        self._status_label.setStyleSheet("color:#8b949e; font-size:13px;")
        btn_row.addWidget(self._status_label)

        btn_row.addStretch()

        self.run_btn = QPushButton("▶ 运行点云过滤")
        self.run_btn.setObjectName("primaryBtn")
        self.run_btn.setMinimumWidth(140)
        self.run_btn.clicked.connect(self._on_run)
        btn_row.addWidget(self.run_btn)

        self.download_btn = QPushButton("⬇ 下载过滤后点云")
        self.download_btn.setObjectName("actionBtn")
        self.download_btn.setMinimumWidth(150)
        self.download_btn.setToolTip("下载过滤结果 filtered_scene.ply 到本地")
        self.download_btn.clicked.connect(self._on_download)
        self.download_btn.setEnabled(False)
        btn_row.addWidget(self.download_btn)

        self.content_layout.addLayout(btn_row)

        # ── 进度条 ──
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate
        self.progress.setVisible(False)
        self.content_layout.addWidget(self.progress)

        # ── 结果统计 ──
        self.result_label = QLabel("等待运行...")
        self.result_label.setObjectName("logArea")
        self.result_label.setMinimumHeight(100)
        self.result_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.result_label.setWordWrap(True)
        self.content_layout.addWidget(self.result_label, 1)

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
        self._log.setMaximumHeight(140)
        self._log.setFont(QFont("Consolas", 9))
        self.content_layout.addWidget(self._log)

    # ------------------------------------------------------------------
    # 项目列表
    # ------------------------------------------------------------------

    def _load_projects(self):
        self._refresh_btn.setEnabled(False)
        self._api.get("/api/projects", tag="filter_list_projects")

    def _on_project_changed(self, project: str):
        """切换项目时检查是否已有过滤结果。"""
        self._state.set_current(project)
        self._current_project = project
        self.download_btn.setEnabled(False)
        if project:
            self._api.get(
                f"/api/projects/{project}/filter/result",
                tag="filter_check_result",
            )

    def _on_shared_project_changed(self, project: str):
        """其余页面选中了项目 → 同步本页下拉框。

        运行中不跟随：过滤在后台跑时，_current_project 决定下载目标，
        不能被切页选择改掉。
        """
        if self._running:
            return
        if self.project_combo.currentText() == project:
            return
        if project:
            idx = self.project_combo.findText(project)
            if idx >= 0:
                self.project_combo.setCurrentIndex(idx)
        else:
            self.project_combo.setCurrentIndex(-1)

    # ------------------------------------------------------------------
    # 运行过滤
    # ------------------------------------------------------------------

    def _on_source_changed(self):
        """选中「自定义路径」时显示路径输入框，其余隐藏。"""
        custom = self.source_combo.currentData() == "custom"
        self.ply_path_edit.setVisible(custom)

    def _resolve_ply_path(self) -> str:
        """根据输入点云来源返回 ply_path（空串 = 让后端自动检测）。"""
        source = self.source_combo.currentData() or "auto"
        if source == "custom":
            return self.ply_path_edit.text().strip()
        return SOURCE_PATHS.get(source, "")

    def _on_run(self):
        project = self.project_combo.currentText().strip()
        if not project:
            QMessageBox.warning(self, "提示", "请先选择一个项目。")
            return

        prompt = self.prompt_edit.text().strip()
        if not prompt:
            QMessageBox.warning(self, "提示", "请填写 Prompt Slug。")
            return

        ply_path = self._resolve_ply_path()
        if self.source_combo.currentData() == "custom" and not ply_path:
            QMessageBox.warning(self, "提示", "请填写自定义点云路径。")
            return

        self._current_project = project
        self._set_running(True)
        self._status_label.setText("过滤运行中...")
        self.result_label.setText("⏳ 正在过滤，请稍候（密集点云可能需要几分钟）...")
        self._log_msg(f"▶ 开始点云过滤 — 项目: {project}")
        self._log_msg(f"  阈值: {self.threshold_spin.value():.2f}, prompt: {prompt}")
        self._log_msg(f"  输入点云: {self.source_combo.currentText()}"
                      + (f" → {ply_path}" if ply_path else ""))

        self._api.post(
            f"/api/projects/{project}/preprocess/filter-pointcloud",
            {
                "threshold": str(self.threshold_spin.value()),
                "prompt_slug": prompt,
                "ply_path": ply_path,
            },
            tag="filter_run",
        )

    # ------------------------------------------------------------------
    # 下载
    # ------------------------------------------------------------------

    def _on_download(self):
        if not self._current_project:
            return
        default_name = f"{self._current_project}_{FILTERED_FILENAME}"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存过滤后的点云",
            default_name,
            "PLY 点云 (*.ply);;所有文件 (*)",
        )
        if not path:
            return
        self.download_btn.setEnabled(False)
        self.download_btn.setText("⏳ 下载中...")
        self._log_msg(f"⬇ 正在下载过滤后点云 → {path}")

        url = QUrl(f"{API_BASE}/api/projects/{self._current_project}/filter/download/{FILTERED_FILENAME}")
        req = QNetworkRequest(url)
        reply = self._dl_mgr.get(req)
        reply.setProperty("_save_path", path)
        reply.setProperty("_project", self._current_project)

    # ------------------------------------------------------------------
    # 网络响应
    # ------------------------------------------------------------------

    def _on_response(self, method: str, path: str, payload: object):
        if not isinstance(payload, dict):
            return
        tag = payload.get("_tag", "")
        error = payload.get("_error", "")

        # ── 项目列表 ──
        if tag == "filter_list_projects":
            self._refresh_btn.setEnabled(True)
            if error:
                self._status_label.setText(f"加载项目失败: {error}")
                return
            projects = payload.get("projects", [])
            previous = self.project_combo.currentText()
            selected = fill_project_combo(self.project_combo, projects)
            self._log_msg(f"✓ 加载了 {len(projects)} 个项目")
            # fill 期间屏蔽了信号：若全局选中项与下拉框现值不同，
            # 需手动触发 _on_project_changed 以检查结果。
            if selected != previous:
                self._on_project_changed(selected)

        # ── 检查已有过滤结果 ──
        elif tag == "filter_check_result":
            if error:
                self._status_label.setText(f"检查结果失败: {error}")
                return
            exists = payload.get("exists", False)
            if exists:
                n = payload.get("num_points", 0)
                size = _format_size(payload.get("size_bytes", 0))
                self.download_btn.setEnabled(True)
                self.result_label.setText(
                    f"✅ 已有过滤结果\n"
                    f"  输出: {payload.get('output_path', '')}\n"
                    f"  点数: {n:,}  |  大小: {size}"
                )
                self._status_label.setText("已有过滤结果，可下载")
            else:
                self.result_label.setText("暂无过滤结果，请先运行过滤。")
                self._status_label.setText("暂无过滤结果")

        # ── 运行过滤 ──
        elif tag == "filter_run":
            self._set_running(False)
            if error:
                self._log_msg(f"✗ 过滤失败: {error}")
                self.result_label.setText(f"❌ 过滤失败\n{error}")
                self._status_label.setText("过滤失败")
                QMessageBox.warning(self, "过滤失败", str(error))
            else:
                data = payload.get("data", {})
                before = data.get("num_points_before", 0)
                after = data.get("num_points_after", 0)
                output_path = data.get("output_path", "")
                input_path = data.get("input_path", "")
                self._log_msg(
                    f"✓ 过滤完成: {after:,} / {before:,} 点 → {output_path}"
                )
                self.result_label.setText(
                    f"✅ 过滤完成\n"
                    f"  输入: {input_path or '（自动检测）'}\n"
                    f"  过滤前: {before:,} 点\n"
                    f"  过滤后: {after:,} 点\n"
                    f"  保留率: {100 * after / before:.1f}%\n"
                    f"  输出: {output_path}"
                )
                self._status_label.setText("✓ 过滤完成，可下载")
                self.download_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # 下载完成
    # ------------------------------------------------------------------

    def _on_download_finished(self, reply: QNetworkReply):
        save_path = reply.property("_save_path")
        project = reply.property("_project")
        self.download_btn.setEnabled(True)
        self.download_btn.setText("⬇ 下载过滤后点云")
        try:
            if reply.error() != QNetworkReply.NoError:
                err_text = bytes(reply.readAll()).decode("utf-8", errors="replace")
                self._log_msg(f"✗ 下载失败: {err_text or reply.errorString()}")
                self._status_label.setText("下载失败")
                QMessageBox.warning(self, "下载失败", str(err_text or reply.errorString()))
                return
            data = bytes(reply.readAll())
            with open(save_path, "wb") as f:
                f.write(data)
            self._log_msg(
                f"✅ 下载完成: {save_path} ({_format_size(len(data))})"
            )
            self._status_label.setText("✓ 下载完成")
        except OSError as e:
            self._log_msg(f"✗ 写入文件失败: {e}")
            self._status_label.setText("下载失败")
            QMessageBox.warning(self, "下载失败", str(e))
        finally:
            reply.deleteLater()

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def _set_running(self, running: bool):
        self._running = running
        self.run_btn.setEnabled(not running)
        self.project_combo.setEnabled(not running)
        self.threshold_spin.setEnabled(not running)
        self.prompt_edit.setEnabled(not running)
        self.progress.setVisible(running)
        if running:
            self.run_btn.setText("⏳ 过滤中...")
        else:
            self.run_btn.setText("▶ 运行点云过滤")

    def showEvent(self, event):
        super().showEvent(event)
        # 进入页面时重新拉取列表，让「别处选中的项目」也能同步到下拉框
        self._load_projects()

    def _log_msg(self, text: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self._log.append(f"[{ts}] {text}")
