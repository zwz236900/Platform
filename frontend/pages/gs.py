"""3DGS 重建页面 — 三维高斯泼溅训练与渲染.

输入采用 SfM 去畸变阶段的输出：SFM_output/dense/images（去畸变图像）+
SFM_output/dense/sparse（PINHOLE 模型）。功能：命令选择 → 参数配置 → 启动 →
轮询日志 → 查看渲染图 / 点云 → 下载产物。
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
    QCheckBox,
    QSpinBox,
    QLineEdit,
    QGroupBox,
    QFormLayout,
    QProgressBar,
    QTextEdit,
    QMessageBox,
    QFileDialog,
    QSizePolicy,
    QScrollArea,
    QFrame,
    QWidget,
    QSlider,
)
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply
from PySide6.QtGui import QFont, QPixmap

from .base_page import BasePage
from .projects import ApiClient
from state import ProjectState, fill_project_combo

API_BASE = "http://localhost:8000"

# 命令
GS_COMMANDS = [
    ("full",    "完整流程（prepare → train → render）"),
    ("prepare", "准备输入（构建 images/ + sparse/0）"),
    ("train",   "训练高斯模型"),
    ("render",  "渲染新视角"),
]

# 各命令可用的参数（其余置灰）
COMMAND_PARAMS = {
    "full": [
        "iterations", "resolution", "white_background", "quiet",
        "checkpoint_interval", "start_checkpoint",
    ],
    "prepare": [],
    "train": [
        "iterations", "resolution", "white_background", "quiet",
        "checkpoint_interval", "start_checkpoint",
    ],
    "render": ["quiet"],
}

# 常见下载产物（位于 3DGS_output/ 下）
COMMON_FILES = [
    "point_cloud/iteration_30000/point_cloud.ply",
    "point_cloud.ply",
    "train/ours_30000/renders/xxx.png",
]

STEP_LABELS = {"prepare": "准备", "train": "训练", "render": "渲染"}


def _format_size(n: int) -> str:
    """人类可读的文件大小。"""
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.2f} MB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.1f} KB"
    return f"{n} B"


class GsPage(BasePage):
    """3DGS 重建 — 参数配置、运行、日志、结果预览与下载。"""

    def __init__(self):
        super().__init__("3DGS 重建")

        # ── 页面内容较多，改为上下滚动的布局，避免小窗口下排版重叠 ──
        base_layout = self.layout()
        title = self.title_label
        while base_layout.count():
            item = base_layout.takeAt(0)
            if item.widget() is title:
                continue  # 标题稍后加回
        base_layout.addWidget(title)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        base_layout.addWidget(self._scroll, 1)

        self._scroll_container = QWidget()
        self._scroll_container.setObjectName("gsScrollContainer")
        self.content_layout = QVBoxLayout(self._scroll_container)
        self.content_layout.setContentsMargins(0, 0, 8, 0)
        self.content_layout.setSpacing(12)
        self._scroll.setWidget(self._scroll_container)

        self._api = ApiClient.instance()
        self._api.finished.connect(self._on_response)
        self._state = ProjectState.instance()
        self._state.current_changed.connect(self._on_shared_project_changed)

        # 独立下载 / 图片加载管理器（二进制，避免干扰 JSON 解析）
        self._dl_mgr = QNetworkAccessManager(self)
        self._dl_mgr.finished.connect(self._on_download_finished)
        self._img_mgr = QNetworkAccessManager(self)
        self._img_mgr.finished.connect(self._on_image_finished)

        self._running = False
        self._current_project = ""
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_log)
        self._poll_timer.setInterval(3000)
        self._renders: list[str] = []
        self._current_render = 0

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

        # ── 输入说明 ──
        input_hint = QLabel(
            "输入：SfM 去畸变阶段的输出（SFM_output/dense）——去畸变图像 + PINHOLE 相机模型。"
            "请先运行 SfM 管线（含去畸变）。"
        )
        input_hint.setStyleSheet("color:#8b949e; font-size:12px;")
        input_hint.setWordWrap(True)
        self.content_layout.addWidget(input_hint)

        # ── 命令 ──
        cmd_group = QGroupBox("运行命令")
        cmd_form = QFormLayout(cmd_group)

        self.command_combo = QComboBox()
        for value, label in GS_COMMANDS:
            self.command_combo.addItem(label, value)
        self.command_combo.currentIndexChanged.connect(self._on_command_changed)
        cmd_form.addRow("命令:", self.command_combo)
        self.content_layout.addWidget(cmd_group)

        # ── 参数设置 ──
        params = QHBoxLayout()
        params.setSpacing(12)

        left_group = QGroupBox("训练参数")
        lform = QFormLayout(left_group)

        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(1000, 500_000)
        self.iterations_spin.setValue(30_000)
        self.iterations_spin.setSingleStep(1000)
        self.iterations_spin.setToolTip("iterations：训练迭代次数")
        lform.addRow("迭代次数:", self.iterations_spin)

        self.resolution_spin = QSpinBox()
        self.resolution_spin.setRange(1, 8)
        self.resolution_spin.setValue(1)
        self.resolution_spin.setToolTip("resolution：图像降采样因子（1=全分辨率）")
        lform.addRow("分辨率:", self.resolution_spin)

        self.ckpt_spin = QSpinBox()
        self.ckpt_spin.setRange(0, 1_000_000)
        self.ckpt_spin.setValue(0)
        self.ckpt_spin.setSingleStep(1000)
        self.ckpt_spin.setToolTip("checkpoint-interval：每 N 次迭代保存一次检查点（0=仅在默认迭代）")
        lform.addRow("检查点间隔:", self.ckpt_spin)

        self.start_ckpt_edit = QLineEdit()
        self.start_ckpt_edit.setPlaceholderText("如 /data/xxx/3DGS_output/.../point_cloud.pth")
        self.start_ckpt_edit.setToolTip("start_checkpoint：从已有检查点续训")
        lform.addRow("起始检查点:", self.start_ckpt_edit)

        params.addWidget(left_group)

        right_group = QGroupBox("选项")
        rform = QFormLayout(right_group)

        self.white_check = QCheckBox("白底背景（默认黑底）")
        self.white_check.setChecked(False)
        rform.addRow("", self.white_check)

        self.quiet_check = QCheckBox("安静模式（减少输出）")
        self.quiet_check.setChecked(False)
        rform.addRow("", self.quiet_check)

        params.addWidget(right_group)

        self.content_layout.addLayout(params)

        # ── 操作按钮 ──
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self._status_label = QLabel("就绪，请选择项目")
        self._status_label.setStyleSheet("color:#8b949e; font-size:13px;")
        btn_row.addWidget(self._status_label)

        btn_row.addStretch()

        self.run_btn = QPushButton("▶ 运行 3DGS")
        self.run_btn.setObjectName("primaryBtn")
        self.run_btn.setMinimumWidth(120)
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
        self.result_label.setMinimumHeight(80)
        self.result_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.result_label.setWordWrap(True)
        self.content_layout.addWidget(self.result_label)

        # ── 渲染图预览 ──
        render_group = QGroupBox("渲染图预览")
        rl = QVBoxLayout(render_group)

        nav = QHBoxLayout()
        nav.addWidget(QLabel("视角:"))
        self.render_combo = QComboBox()
        self.render_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.render_combo.currentIndexChanged.connect(self._on_render_selected)
        nav.addWidget(self.render_combo, 1)
        self.prev_btn = QPushButton("◀")
        self.prev_btn.setObjectName("actionBtn")
        self.prev_btn.setFixedSize(36, 28)
        self.prev_btn.clicked.connect(lambda: self._step_render(-1))
        nav.addWidget(self.prev_btn)
        self.next_btn = QPushButton("▶")
        self.next_btn.setObjectName("actionBtn")
        self.next_btn.setFixedSize(36, 28)
        self.next_btn.clicked.connect(lambda: self._step_render(1))
        nav.addWidget(self.next_btn)
        rl.addLayout(nav)

        self.render_canvas = QLabel("（运行后展示训练集渲染图）")
        self.render_canvas.setObjectName("logArea")
        self.render_canvas.setMinimumHeight(200)
        self.render_canvas.setAlignment(Qt.AlignCenter)
        self.render_canvas.setStyleSheet("color:#8b949e; font-size:12px;")
        rl.addWidget(self.render_canvas)

        self.content_layout.addWidget(render_group)

        # ── 下载区 ──
        dl_group = QGroupBox("下载产物")
        dl_form = QFormLayout(dl_group)

        dl_row = QHBoxLayout()
        self.dl_file_combo = QComboBox()
        self.dl_file_combo.setEditable(True)
        self.dl_file_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        dl_row.addWidget(self.dl_file_combo, 1)
        dl_form.addRow("文件:", dl_row)
        self._update_download_files()

        dl_btn_row = QHBoxLayout()
        self.download_btn = QPushButton("⬇ 下载")
        self.download_btn.setObjectName("actionBtn")
        self.download_btn.setToolTip("从 3DGS_output 目录下载选中的文件")
        self.download_btn.clicked.connect(self._on_download)
        dl_btn_row.addWidget(self.download_btn)
        dl_btn_row.addStretch()
        self._dl_status = QLabel("")
        self._dl_status.setStyleSheet("color:#8b949e; font-size:12px;")
        dl_btn_row.addWidget(self._dl_status)
        dl_form.addRow("", dl_btn_row)

        self.content_layout.addWidget(dl_group)

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

        # 命令相关的参数启用/禁用
        self._param_widgets = {
            "iterations": self.iterations_spin,
            "resolution": self.resolution_spin,
            "white_background": self.white_check,
            "quiet": self.quiet_check,
            "checkpoint_interval": self.ckpt_spin,
            "start_checkpoint": self.start_ckpt_edit,
        }
        self._on_command_changed()

    # ------------------------------------------------------------------
    # 项目 / 命令
    # ------------------------------------------------------------------

    def _load_projects(self):
        self._refresh_btn.setEnabled(False)
        self._api.get("/api/projects", tag="gs_list_projects")

    def _on_project_changed(self, project: str):
        self._state.set_current(project)
        self._current_project = project
        if project:
            self._check_result()

    def _on_shared_project_changed(self, project: str):
        """其余页面选中了项目 → 同步本页下拉框。

        运行中不跟随：3DGS 在后台跑时，_current_project 决定轮询目标，
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

    def _on_command_changed(self):
        cmd = self.command_combo.currentData() or "full"
        enabled = set(COMMAND_PARAMS.get(cmd, []))
        for key, w in self._param_widgets.items():
            w.setEnabled(key in enabled)

    def _update_download_files(self):
        files = list(COMMON_FILES)
        # 若有实际渲染图 / 点云路径，追加
        if self._renders:
            files = list(self._renders) + files
        self.dl_file_combo.blockSignals(True)
        self.dl_file_combo.clear()
        self.dl_file_combo.addItems(files)
        self.dl_file_combo.blockSignals(False)

    # ------------------------------------------------------------------
    # 运行
    # ------------------------------------------------------------------

    def _on_run(self):
        project = self.project_combo.currentText().strip()
        if not project:
            QMessageBox.warning(self, "提示", "请先选择一个项目。")
            return

        command = self.command_combo.currentData()

        config = {
            "command": command,
            "iterations": str(self.iterations_spin.value()),
            "resolution": str(self.resolution_spin.value()),
            "white_background": str(self.white_check.isChecked()).lower(),
            "quiet": str(self.quiet_check.isChecked()).lower(),
            "checkpoint_interval": str(self.ckpt_spin.value()),
            "start_checkpoint": self.start_ckpt_edit.text().strip(),
        }

        self._current_project = project
        self._set_running(True)
        self._status_label.setText("3DGS 启动中...")
        self.result_label.setText("⏳ 正在启动 3DGS 任务...")
        self._log_msg(f"▶ 启动 3DGS — 项目: {project} | 命令: {command}")
        if command in ("full", "train"):
            self._log_msg(
                f"  迭代: {config['iterations']}, 分辨率: {config['resolution']}, "
                f"白底: {config['white_background']}"
            )

        self._api.post(
            f"/api/projects/{project}/3dgs/run",
            config,
            tag="gs_run",
        )

    # ------------------------------------------------------------------
    # 轮询日志与结果
    # ------------------------------------------------------------------

    def _poll_log(self):
        if not self._current_project:
            self._poll_timer.stop()
            return
        self._api.get(
            f"/api/projects/{self._current_project}/3dgs/log?lines=60",
            tag="gs_poll_log",
        )
        self._check_result()

    def _check_result(self):
        self._api.get(
            f"/api/projects/{self._current_project}/3dgs/result",
            tag="gs_result",
        )

    # ------------------------------------------------------------------
    # 渲染图预览
    # ------------------------------------------------------------------

    def _on_render_selected(self):
        if 0 <= self.render_combo.currentIndex() < len(self._renders):
            self._current_render = self.render_combo.currentIndex()
            self._load_render_image(self._current_render)

    def _step_render(self, delta: int):
        if not self._renders:
            return
        idx = (self._current_render + delta) % len(self._renders)
        self.render_combo.setCurrentIndex(idx)

    def _load_render_image(self, idx: int):
        if not self._renders or not (0 <= idx < len(self._renders)):
            return
        rel = self._renders[idx]
        self.render_canvas.setText(f"加载 {Path(rel).name} ...")
        req = QNetworkRequest(QUrl(f"{API_BASE}/api/projects/{self._current_project}/3dgs/download/{rel}"))
        reply = self._img_mgr.get(req)
        reply.setProperty("_idx", idx)

    # ------------------------------------------------------------------
    # 下载
    # ------------------------------------------------------------------

    def _on_download(self):
        if not self._current_project:
            QMessageBox.warning(self, "提示", "请先选择一个项目。")
            return
        filename = self.dl_file_combo.currentText().strip()
        if not filename:
            QMessageBox.warning(self, "提示", "请填写要下载的文件名。")
            return

        default_name = f"{self._current_project}_{Path(filename).name}"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存 3DGS 产物",
            default_name,
            "3DGS 产物 (*.ply *.png *.pth);;所有文件 (*)",
        )
        if not path:
            return

        self.download_btn.setEnabled(False)
        self.download_btn.setText("⏳ 下载中...")
        self._dl_status.setText(f"下载 {filename} ...")
        self._log_msg(f"⬇ 正在下载 3DGS 产物 → {path}")

        url = QUrl(f"{API_BASE}/api/projects/{self._current_project}/3dgs/download/{filename}")
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
        if tag == "gs_list_projects":
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
        elif tag == "gs_run":
            if error:
                self._set_running(False)
                self._log_msg(f"✗ 启动失败: {error}")
                self._status_label.setText("启动失败")
                QMessageBox.warning(self, "3DGS 启动失败", str(error))
            else:
                self._log_msg("✓ 3DGS 任务已启动，开始轮询日志...")
                self._status_label.setText("3DGS 运行中...")
                self._poll_log()
                self._poll_timer.start()

        # ── 日志轮询 ──
        elif tag == "gs_poll_log":
            log_text = payload.get("log", "")
            if log_text and log_text != "(no log yet)":
                self._log.setPlainText(log_text)
                scrollbar = self._log.verticalScrollBar()
                if scrollbar:
                    scrollbar.setValue(scrollbar.maximum())

        # ── 结果查询 ──
        elif tag == "gs_result":
            if error:
                self.result_label.setText(f"⚠ 查询结果失败: {error}")
                return
            ready = payload.get("ready", False)
            if not ready:
                msg = payload.get("message", "暂无 3DGS 结果。")
                self.result_label.setText(f"ℹ {msg}")
                return

            renders = payload.get("renders", [])
            iteration_name = payload.get("iteration_name", "")
            renders_count = payload.get("renders_count", 0)
            has_pc = payload.get("has_point_cloud", False)
            pc_ply = payload.get("point_cloud_ply", "")

            lines = [f"✅ 3DGS 结果就绪（迭代 {iteration_name or '?'}）"]
            lines.append(f"   渲染图: {renders_count} 张")
            lines.append(f"   点云: {'✓ 存在' if has_pc else '✗ 无'}")
            if pc_ply:
                lines.append(f"   点云文件: {pc_ply}")
            self.result_label.setText("\n".join(lines))

            # 更新渲染图列表
            self._renders = renders
            self._current_render = 0
            self.render_combo.blockSignals(True)
            self.render_combo.clear()
            for rel in renders:
                self.render_combo.addItem(Path(rel).name, rel)
            self.render_combo.blockSignals(False)
            if renders:
                self._load_render_image(0)

            # 刷新下载列表
            self._update_download_files()
            if pc_ply:
                idx = self.dl_file_combo.findText(pc_ply)
                if idx >= 0:
                    self.dl_file_combo.setCurrentIndex(idx)

    # ------------------------------------------------------------------
    # 图片加载完成
    # ------------------------------------------------------------------

    def _on_image_finished(self, reply: QNetworkReply):
        try:
            if reply.error() != QNetworkReply.NoError:
                self.render_canvas.setText(f"加载失败: {reply.errorString()}")
                return
            data = bytes(reply.readAll())
            pix = QPixmap()
            if not pix.loadFromData(data):
                self.render_canvas.setText("无法解析渲染图")
                return
            # 缩放到画布宽度（最大 560px），保持宽高比
            self.render_canvas.setPixmap(
                pix.scaledToWidth(
                    min(560, self.render_canvas.width() or 560),
                    Qt.SmoothTransformation,
                )
            )
        finally:
            reply.deleteLater()

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
        self.command_combo.setEnabled(not running)
        self.progress.setVisible(running)
        if running:
            self.run_btn.setText("⏳ 运行中...")
        else:
            self.run_btn.setText("▶ 运行 3DGS")

    def showEvent(self, event):
        super().showEvent(event)
        # 进入页面时重新拉取列表，让「别处选中的项目」也能同步到下拉框
        self._load_projects()

    def _log_msg(self, text: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self._log.append(f"[{ts}] {text}")
