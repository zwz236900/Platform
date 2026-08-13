"""掩码目标重建页面 — 按 SAM 掩码裁剪去畸变图 → 复用现有 MVS 做稠密重建。

功能：选择掩码来源与裁剪外扩 → 启动（裁剪建模 + MVS）→ 轮询日志/结果 → 下载产物。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QComboBox,
    QSpinBox,
    QGroupBox,
    QFormLayout,
    QProgressBar,
    QTextEdit,
    QMessageBox,
    QFileDialog,
    QSizePolicy,
    QWidget,
)
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply
from PySide6.QtGui import QFont

from .base_page import BasePage
from .projects import ApiClient
from state import ProjectState, fill_project_combo

API_BASE = "http://localhost:8000"

# 掩码来源（value 与后端 mask_source 一致）
MASK_SOURCES = [
    ("point", "点选掩码 point_results"),
    ("box", "框选掩码 box_results"),
]


def _format_size(n: int) -> str:
    """人类可读的文件大小。"""
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.2f} MB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.1f} KB"
    return f"{n} B"


class MaskObjPage(BasePage):
    """掩码目标重建 — 参数配置、运行、日志、结果与下载。"""

    def __init__(self):
        super().__init__("掩码目标重建")

        self._api = ApiClient.instance()
        self._api.finished.connect(self._on_response)
        self._state = ProjectState.instance()
        self._state.current_changed.connect(self._on_shared_project_changed)

        # 独立下载管理器（二进制下载，避免干扰 JSON 解析）
        self._dl_mgr = QNetworkAccessManager(self)
        self._dl_mgr.finished.connect(self._on_download_finished)

        self._running = False
        self._current_project = ""
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_status)
        self._poll_timer.setInterval(3000)

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
        group = QGroupBox("掩码目标参数")
        form = QFormLayout(group)

        self.mask_source_combo = QComboBox()
        for value, label in MASK_SOURCES:
            self.mask_source_combo.addItem(label, value)
        self.mask_source_combo.setToolTip("SAM 掩码来源：点选(point_results)或框选(box_results)")
        form.addRow("掩码来源:", self.mask_source_combo)

        self.padding_spin = QSpinBox()
        self.padding_spin.setRange(0, 50)
        self.padding_spin.setValue(5)
        self.padding_spin.setSuffix(" %")
        self.padding_spin.setToolTip("掩码 bbox 外扩百分比，避免目标贴边")
        form.addRow("裁剪外扩:", self.padding_spin)

        self.content_layout.addWidget(group)

        # ── 操作按钮 ──
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self._status_label = QLabel("就绪，请选择项目")
        self._status_label.setStyleSheet("color:#8b949e; font-size:13px;")
        btn_row.addWidget(self._status_label)

        btn_row.addStretch()

        self.run_btn = QPushButton("▶ 运行掩码目标重建")
        self.run_btn.setObjectName("primaryBtn")
        self.run_btn.setMinimumWidth(150)
        self.run_btn.clicked.connect(self._on_run)
        btn_row.addWidget(self.run_btn)

        self.content_layout.addLayout(btn_row)

        # ── 进度条 ──
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate
        self.progress.setVisible(False)
        self.content_layout.addWidget(self.progress)

        # ── 结果面板 ──
        self.result_label = QLabel("等待运行...")
        self.result_label.setObjectName("logArea")
        self.result_label.setMinimumHeight(110)
        self.result_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.result_label.setWordWrap(True)
        self.content_layout.addWidget(self.result_label, 1)

        # ── 下载区 ──
        dl_group = QGroupBox("下载产物")
        dl_form = QFormLayout(dl_group)

        dl_row = QHBoxLayout()
        self.dl_file_combo = QComboBox()
        self.dl_file_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        dl_row.addWidget(self.dl_file_combo, 1)

        self.download_btn = QPushButton("⬇ 下载")
        self.download_btn.setObjectName("actionBtn")
        self.download_btn.setToolTip("从 algorithm_output/mask_obj 下载选中的产物")
        self.download_btn.clicked.connect(self._on_download)
        dl_row.addWidget(self.download_btn)

        self._dl_status = QLabel("")
        self._dl_status.setStyleSheet("color:#8b949e; font-size:12px;")
        dl_row.addWidget(self._dl_status)
        dl_form.addRow(dl_row)

        self.content_layout.addWidget(dl_group)
        self._update_download_files()

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
        self._log.setMaximumHeight(150)
        self._log.setFont(QFont("Consolas", 9))
        self.content_layout.addWidget(self._log)

    # ------------------------------------------------------------------
    # 项目
    # ------------------------------------------------------------------

    def _load_projects(self):
        self._refresh_btn.setEnabled(False)
        self._api.get("/api/projects", tag="mask_obj_list_projects")

    def _on_project_changed(self, project: str):
        self._state.set_current(project)
        self._current_project = project
        if project:
            self._check_result()
        self._update_download_files()

    def _on_shared_project_changed(self, project: str):
        """其余页面选中了项目 → 同步本页下拉框。运行中不跟随。"""
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
    # 运行
    # ------------------------------------------------------------------

    def _on_run(self):
        project = self.project_combo.currentText().strip()
        if not project:
            QMessageBox.warning(self, "提示", "请先选择一个项目。")
            return

        mask_source = self.mask_source_combo.currentData() or "point"
        padding = self.padding_spin.value()

        self._current_project = project
        self._set_running(True)
        self._status_label.setText("掩码目标重建启动中...")
        self.result_label.setText("⏳ 正在裁剪图像并启动 MVS...")
        self._log_msg(f"▶ 启动掩码目标重建 — 项目: {project} | 掩码: {mask_source} | 外扩: {padding}%")

        self._api.post(
            f"/api/projects/{project}/mask-obj/run",
            {"mask_source": mask_source, "padding": str(padding)},
            tag="mask_obj_run",
        )

    # ------------------------------------------------------------------
    # 轮询状态
    # ------------------------------------------------------------------

    def _poll_status(self):
        if not self._current_project:
            self._poll_timer.stop()
            return
        self._api.get(
            f"/api/projects/{self._current_project}/mask-obj/log?lines=60",
            tag="mask_obj_poll_log",
        )
        self._check_result()

    def _check_result(self):
        if self._current_project:
            self._api.get(
                f"/api/projects/{self._current_project}/mask-obj/result",
                tag="mask_obj_result",
            )

    # ------------------------------------------------------------------
    # 下载
    # ------------------------------------------------------------------

    def _update_download_files(self):
        """从后端读取 algorithm_output/mask_obj 里实际存在的产物，填充下载下拉框。"""
        project = self._current_project or self.project_combo.currentText().strip()
        if not project:
            self.dl_file_combo.clear()
            return
        self._api.get(
            f"/api/projects/{project}/mask-obj/files",
            tag="mask_obj_files",
        )

    def _on_download(self):
        if not self._current_project:
            QMessageBox.warning(self, "提示", "请先选择一个项目。")
            return
        filename = self.dl_file_combo.currentData() \
            or self.dl_file_combo.currentText().strip()
        if not filename:
            QMessageBox.warning(self, "提示", "请选择要下载的产物。")
            return

        default_name = f"{self._current_project}_{Path(filename).name}"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存掩码目标重建产物",
            default_name,
            "产物文件 (*.ply *.zip *.log);;所有文件 (*)",
        )
        if not path:
            return

        self.download_btn.setEnabled(False)
        self.download_btn.setText("⏳ 下载中...")
        self._dl_status.setText(f"下载 {filename} ...")
        self._log_msg(f"⬇ 正在下载产物 → {path}")

        url = QUrl(
            f"{API_BASE}/api/projects/{self._current_project}/mask-obj/download/{filename}"
        )
        req = QNetworkRequest(url)
        reply = self._dl_mgr.get(req)
        reply.setProperty("_save_path", path)
        reply.setProperty("_filename", filename)

    # ------------------------------------------------------------------
    # 网络响应
    # ------------------------------------------------------------------

    def _on_response(self, method: str, path: str, payload: object):
        if not isinstance(payload, dict):
            return
        tag = payload.get("_tag", "")
        error = payload.get("_error", "")

        # ── 项目列表 ──
        if tag == "mask_obj_list_projects":
            self._refresh_btn.setEnabled(True)
            if error:
                self._status_label.setText(f"加载项目失败: {error}")
                return
            projects = payload.get("projects", [])
            previous = self.project_combo.currentText()
            selected = fill_project_combo(self.project_combo, projects)
            self._log_msg(f"✓ 加载了 {len(projects)} 个项目")
            if selected != previous:
                self._on_project_changed(selected)

        # ── 启动响应 ──
        elif tag == "mask_obj_run":
            if error:
                self._set_running(False)
                self._log_msg(f"✗ 启动失败: {error}")
                self._status_label.setText("启动失败")
                QMessageBox.warning(self, "启动失败", str(error))
            else:
                data = payload.get("data", {})
                self._log_msg(
                    f"✓ 裁剪完成: {data.get('images_cropped', 0)} 张，"
                    f"跳过 {data.get('images_skipped', 0)} 张 — MVS 已启动")
                self._status_label.setText("掩码目标重建运行中...")
                self._poll_status()
                self._poll_timer.start()

        # ── 日志轮询（仅展示；完成判定以后端 /mask-obj/result 的 run_status 为准）──
        elif tag == "mask_obj_poll_log":
            log_text = payload.get("log", "")
            if log_text and log_text != "(no log yet)":
                self._log.setPlainText(log_text)
                scrollbar = self._log.verticalScrollBar()
                if scrollbar:
                    scrollbar.setValue(scrollbar.maximum())

        # ── 下载产物列表 ──
        elif tag == "mask_obj_files":
            if error:
                self._dl_status.setText(f"加载产物列表失败: {error}")
                return
            products = payload.get("products", [])
            self.dl_file_combo.blockSignals(True)
            self.dl_file_combo.clear()
            for item in products:
                self.dl_file_combo.addItem(
                    f"{item['step']} · {item['label']}", item["path"])
            self.dl_file_combo.blockSignals(False)
            self._dl_status.setText("暂无产物可下载" if not products else "")

        # ── 结果查询（含后端运行状态，据此判定完成）──
        elif tag == "mask_obj_result":
            if error:
                self.result_label.setText(f"⚠ 查询结果失败: {error}")
                return
            run_status = payload.get("run_status")
            exit_code = payload.get("exit_code")

            if self._running:
                if run_status == "running":
                    self._status_label.setText("掩码目标重建运行中...")
                    return
                self._set_running(False)
                self._poll_timer.stop()
                if run_status == "ok":
                    self._log_msg("✓ MVS 进程正常退出")
                    self._status_label.setText("重建已结束")
                elif run_status == "failed":
                    self._log_msg(f"✗ MVS 进程异常退出 (exit_code: {exit_code})")
                    self._status_label.setText("重建失败")
                else:
                    # idle：状态缺失（后端重启等）→ 保守停止，避免无限轮询
                    self._log_msg("⚠ 运行状态丢失，已停止轮询")
                    self._status_label.setText("状态未知")
                    return

            if run_status == "failed":
                msg = "❌ 掩码目标重建失败"
                if exit_code is not None:
                    msg += f" (exit_code: {exit_code})"
                self.result_label.setText(msg)
                return

            ready = payload.get("ready", False)
            if not ready:
                msg = payload.get("message", "暂无结果。")
                self.result_label.setText(f"ℹ {msg}")
                if not self._running:
                    self._status_label.setText("暂无结果")
                return

            lines = []
            status = payload.get("status", "")
            if status == "ok":
                lines.append("✅ 掩码目标重建成功")
            else:
                lines.append(f"⚠ 状态: {status}")
            for key, label in (("images_cropped", "裁剪图像"),
                               ("images_skipped_no_mask", "跳过(无掩码)"),
                               ("num_cameras", "相机数"),
                               ("num_points3D", "3D 点"),
                               ("fused_size_mb", "稠密点云大小")):
                val = payload.get(key)
                if val is not None:
                    unit = " MB" if key == "fused_size_mb" else ""
                    lines.append(f"   {label}: {val}{unit}")
            self.result_label.setText("\n".join(lines))
            if not self._running:
                self._status_label.setText("✓ 已获取结果")
            self._update_download_files()

    # ------------------------------------------------------------------
    # 下载完成
    # ------------------------------------------------------------------

    def _on_download_finished(self, reply: QNetworkReply):
        save_path = reply.property("_save_path")
        filename = reply.property("_filename")
        self.download_btn.setEnabled(True)
        self.download_btn.setText("⬇ 下载")
        self._dl_status.setText("")
        try:
            if reply.error() != QNetworkReply.NoError:
                err_text = bytes(reply.readAll()).decode("utf-8", errors="replace")
                self._log_msg(f"✗ 下载失败: {err_text or reply.errorString()}")
                QMessageBox.warning(self, "下载失败", str(err_text or reply.errorString()))
                return
            data = bytes(reply.readAll())
            with open(save_path, "wb") as f:
                f.write(data)
            self._dl_status.setText(f"✓ {_format_size(len(data))}")
            self._log_msg(f"✅ 下载完成: {filename} ({_format_size(len(data))}) → {save_path}")
        except OSError as e:
            self._log_msg(f"✗ 写入文件失败: {e}")
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
        self.mask_source_combo.setEnabled(not running)
        self.padding_spin.setEnabled(not running)
        self.progress.setVisible(running)
        if running:
            self.run_btn.setText("⏳ 运行中...")
        else:
            self.run_btn.setText("▶ 运行掩码目标重建")

    def showEvent(self, event):
        super().showEvent(event)
        # 进入页面时重新拉取列表，让「别处选中的项目」也能同步到下拉框
        self._load_projects()

    def _log_msg(self, text: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self._log.append(f"[{ts}] {text}")
