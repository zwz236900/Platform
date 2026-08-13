"""MVS 重建页面 — COLMAP / OpenMVS 稠密重建、网格化与纹理映射.

功能：选择引擎与命令 → 配置参数 → 启动 → 轮询日志 → 查看结果 → 下载产物。
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
    QDoubleSpinBox,
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
)
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply
from PySide6.QtGui import QFont

from .base_page import BasePage
from .projects import ApiClient
from state import ProjectState, fill_project_combo

API_BASE = "http://localhost:8000"

# 引擎
MVS_TYPES = [
    ("openmvs", "OpenMVS"),
    ("colmap",  "COLMAP"),
]

# 命令（value 与后端 MVSCommand 枚举一致）
MVS_COMMANDS = [
    ("pipeline",    "完整流程 pipeline"),
    ("interface",   "导入 interface"),
    ("densify",     "稠密重建 densify"),
    ("reconstruct", "表面重建 reconstruct"),
    ("refine",      "网格细化 refine"),
    ("texture",     "纹理映射 texture"),
    ("check",       "检查 check"),
]

# COLMAP 引擎不支持单步（后端恒跑 mvs，复用 SfM 去畸变输出；无去畸变时自动补 undistort）
MVS_COMMANDS_COLMAP = [
    ("pipeline", "完整流程（稠密重建）"),
]

# 各命令可用的参数（其余置灰）
COMMAND_PARAMS = {
    "pipeline": [
        "model", "resolution_level", "max_resolution", "min_resolution",
        "number_views", "no_cuda", "refine_resolution", "scales",
        "gradient_step", "texture_resolution", "export_type",
    ],
    "interface": ["model"],
    "densify":   ["resolution_level", "max_resolution", "number_views", "no_cuda"],
    "reconstruct": [],
    "refine":    ["refine_resolution", "scales"],
    "texture":   ["texture_resolution", "min_resolution", "export_type"],
    "check":     [],
}

# 常见输出产物（用于下载）— 按引擎区分，各自位于 MVS_output/<type>/ 下
COMMON_FILES = {
    "openmvs": [
        "scene.mvs",
        "scene_dense.mvs",
        "scene_dense.ply",
        "scene_dense_mesh.mvs",
        "scene_dense_mesh.ply",
        "scene_dense_mesh_refine.mvs",
        "scene_dense_mesh_refine.ply",
        "scene_dense_mesh_refine_texture.mvs",
        "scene_dense_mesh_refine_texture.ply",
        "scene_dense_mesh_refine_texture.png",
    ],
    "colmap": [
        "fused.ply",
        "fused_del.ply",
    ],
}

STEP_LABELS = {
    "interface": "导入",
    "densify": "稠密",
    "reconstruct": "重建",
    "refine": "细化",
    "texture": "纹理",
    # COLMAP 管线的步骤名
    "extract": "特征提取",
    "match": "特征匹配",
    "sfm": "稀疏重建",
    "undistort": "去畸变",
    "mvs": "稠密重建",
}

STATUS_LABELS = {"ok": "✓", "failed": "✗", "running": "⏳"}


def _format_size(n: int) -> str:
    """人类可读的文件大小。"""
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.2f} MB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.1f} KB"
    return f"{n} B"


class MvsPage(BasePage):
    """MVS 重建 — 参数配置、运行、日志、结果与下载。"""

    def __init__(self):
        super().__init__("MVS 重建")

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
        self._scroll_container.setObjectName("mvsScrollContainer")
        self.content_layout = QVBoxLayout(self._scroll_container)
        self.content_layout.setContentsMargins(0, 0, 8, 0)
        self.content_layout.setSpacing(12)
        self._scroll.setWidget(self._scroll_container)

        self._api = ApiClient.instance()
        self._api.finished.connect(self._on_response)
        self._state = ProjectState.instance()
        self._state.current_changed.connect(self._on_shared_project_changed)

        # 独立下载管理器（二进制下载，避免干扰 JSON 解析）
        self._dl_mgr = QNetworkAccessManager(self)
        self._dl_mgr.finished.connect(self._on_download_finished)

        self._running = False
        self._current_project = ""
        self._current_mvs_type = "openmvs"
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_log)
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

        # ── 引擎与命令 ──
        engine_group = QGroupBox("引擎与命令")
        engine_form = QFormLayout(engine_group)

        self.mvs_type_combo = QComboBox()
        for value, label in MVS_TYPES:
            self.mvs_type_combo.addItem(label, value)
        self.mvs_type_combo.currentIndexChanged.connect(self._on_mvs_type_changed)
        engine_form.addRow("MVS 引擎:", self.mvs_type_combo)

        self.command_combo = QComboBox()
        for value, label in MVS_COMMANDS:
            self.command_combo.addItem(label, value)
        self.command_combo.currentIndexChanged.connect(self._on_command_changed)
        engine_form.addRow("运行命令:", self.command_combo)

        self._engine_hint = QLabel(
            "OpenMVS: 基于 SfM 结果运行稠密重建流程；COLMAP: 复用 SfM 去畸变输出 + patch_match 稠密重建。"
        )
        self._engine_hint.setStyleSheet("color:#8b949e; font-size:12px;")
        self._engine_hint.setWordWrap(True)
        engine_group.layout().addWidget(self._engine_hint)

        self.content_layout.addWidget(engine_group)

        # ── 参数设置 ──
        params = QHBoxLayout()
        params.setSpacing(12)

        # 左列：重建分辨率
        left_group = QGroupBox("分辨率")
        lform = QFormLayout(left_group)

        self.resolution_spin = QSpinBox()
        self.resolution_spin.setRange(0, 8)
        self.resolution_spin.setValue(1)
        self.resolution_spin.setToolTip("resolution-level：图像降采样级别")
        lform.addRow("分辨率级别:", self.resolution_spin)

        self.max_res_spin = QSpinBox()
        self.max_res_spin.setRange(100, 20000)
        self.max_res_spin.setValue(3200)
        self.max_res_spin.setSingleStep(100)
        self.max_res_spin.setToolTip("max-resolution：最大图像尺寸")
        lform.addRow("最大分辨率:", self.max_res_spin)

        self.min_res_spin = QSpinBox()
        self.min_res_spin.setRange(100, 20000)
        self.min_res_spin.setValue(640)
        self.min_res_spin.setSingleStep(100)
        self.min_res_spin.setToolTip("min-resolution：最小图像尺寸")
        lform.addRow("最小分辨率:", self.min_res_spin)

        self.refine_res_spin = QSpinBox()
        self.refine_res_spin.setRange(0, 8)
        self.refine_res_spin.setValue(1)
        self.refine_res_spin.setToolTip("refine-resolution：细化分辨率级别")
        lform.addRow("细化级别:", self.refine_res_spin)

        self.texture_res_spin = QSpinBox()
        self.texture_res_spin.setRange(0, 8)
        self.texture_res_spin.setValue(1)
        self.texture_res_spin.setToolTip("texture-resolution：纹理分辨率级别")
        lform.addRow("纹理级别:", self.texture_res_spin)

        params.addWidget(left_group)

        # 右列：深度与选项
        right_group = QGroupBox("深度与选项")
        rform = QFormLayout(right_group)

        self.views_spin = QSpinBox()
        self.views_spin.setRange(1, 20)
        self.views_spin.setValue(5)
        self.views_spin.setToolTip("number-views：每张图像使用的相邻视图数")
        rform.addRow("相邻视图数:", self.views_spin)

        self.scales_spin = QSpinBox()
        self.scales_spin.setRange(1, 4)
        self.scales_spin.setValue(2)
        self.scales_spin.setToolTip("scales：细化金字塔尺度")
        rform.addRow("细化尺度:", self.scales_spin)

        self.gradient_spin = QDoubleSpinBox()
        self.gradient_spin.setRange(0.0, 180.0)
        self.gradient_spin.setValue(45.0)
        self.gradient_spin.setDecimals(1)
        self.gradient_spin.setToolTip("gradient-step：纹理梯度步长")
        rform.addRow("梯度步长:", self.gradient_spin)

        self.no_cuda_check = QCheckBox("CPU 模式（当前构建无 GPU 加速）")
        self.no_cuda_check.setChecked(True)
        self.no_cuda_check.setToolTip(
            "openmvs 容器内为 2019 年 CPU-only 构建（未编译 CUDA），\n"
            "此开关仅透传 --no-cuda 标记，暂不影响实际运行结果。"
        )
        rform.addRow("", self.no_cuda_check)

        self.model_edit = QLineEdit()
        self.model_edit.setText("0")
        self.model_edit.setPlaceholderText("模型编号，如 0")
        self.model_edit.setToolTip("model：使用 SfM 结果中的第几个模型")
        rform.addRow("模型编号:", self.model_edit)

        self.export_combo = QComboBox()
        self.export_combo.addItems(["ply", "obj"])
        self.export_combo.setToolTip("export-type：导出格式")
        rform.addRow("导出格式:", self.export_combo)

        params.addWidget(right_group)

        self.content_layout.addLayout(params)

        # ── 操作按钮 ──
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self._status_label = QLabel("就绪，请选择项目")
        self._status_label.setStyleSheet("color:#8b949e; font-size:13px;")
        btn_row.addWidget(self._status_label)

        btn_row.addStretch()

        self.run_btn = QPushButton("▶ 运行 MVS")
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
        self.result_label.setMinimumHeight(110)
        self.result_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.result_label.setWordWrap(True)
        self.content_layout.addWidget(self.result_label, 1)

        # ── 下载区 ──
        dl_group = QGroupBox("下载产物")
        dl_form = QFormLayout(dl_group)

        dl_type_row = QHBoxLayout()
        self.dl_type_combo = QComboBox()
        for value, label in [("openmvs", "OpenMVS"), ("colmap", "COLMAP")]:
            self.dl_type_combo.addItem(label, value)
        self.dl_type_combo.currentIndexChanged.connect(self._on_dl_type_changed)
        dl_type_row.addWidget(self.dl_type_combo)

        self.dl_file_combo = QComboBox()
        self.dl_file_combo.setEditable(True)
        self.dl_file_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        dl_type_row.addWidget(self.dl_file_combo, 1)
        dl_form.addRow("文件:", dl_type_row)
        self._update_download_files()

        dl_btn_row = QHBoxLayout()
        self.download_btn = QPushButton("⬇ 下载")
        self.download_btn.setObjectName("actionBtn")
        self.download_btn.setToolTip("从 MVS_output 目录下载选中的文件")
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
            "model": self.model_edit,
            "resolution_level": self.resolution_spin,
            "max_resolution": self.max_res_spin,
            "min_resolution": self.min_res_spin,
            "number_views": self.views_spin,
            "no_cuda": self.no_cuda_check,
            "refine_resolution": self.refine_res_spin,
            "scales": self.scales_spin,
            "gradient_step": self.gradient_spin,
            "texture_resolution": self.texture_res_spin,
            "export_type": self.export_combo,
        }
        self._on_command_changed()

    # ------------------------------------------------------------------
    # 项目 / 引擎 / 命令
    # ------------------------------------------------------------------

    def _load_projects(self):
        self._refresh_btn.setEnabled(False)
        self._api.get("/api/projects", tag="mvs_list_projects")

    def _on_project_changed(self, project: str):
        self._state.set_current(project)
        self._current_project = project
        if project:
            self._check_result()

    def _on_shared_project_changed(self, project: str):
        """其余页面选中了项目 → 同步本页下拉框。

        运行中不跟随：MVS 在后台跑时，_current_project 决定轮询目标，
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

    def _repopulate_commands(self):
        """按当前引擎重填命令下拉框：colmap 只保留完整流程。"""
        commands = MVS_COMMANDS_COLMAP if self._current_mvs_type == "colmap" else MVS_COMMANDS
        previous = self.command_combo.currentData()
        self.command_combo.blockSignals(True)
        self.command_combo.clear()
        for value, label in commands:
            self.command_combo.addItem(label, value)
        idx = self.command_combo.findData(previous)
        if idx >= 0:
            self.command_combo.setCurrentIndex(idx)
        self.command_combo.blockSignals(False)
        self._on_command_changed()

    def _on_mvs_type_changed(self):
        self._current_mvs_type = self.mvs_type_combo.currentData()
        # 命令按引擎过滤：colmap 不支持单步，只保留完整流程
        self._repopulate_commands()
        if self._current_mvs_type == "openmvs":
            self._engine_hint.setText(
                "OpenMVS: 基于 SfM 结果运行稠密重建流程；COLMAP: 复用 SfM 去畸变输出 + patch_match 稠密重建。"
            )
            # openmvs 容器内为 2019 年 CPU-only 构建，GPU 开关无意义 → 默认 CPU 模式
            self.no_cuda_check.setText("CPU 模式（当前构建无 GPU 加速）")
            self.no_cuda_check.setToolTip(
                "openmvs 容器内为 2019 年 CPU-only 构建（未编译 CUDA），\n"
                "此开关仅透传 --no-cuda 标记，暂不影响实际运行结果。"
            )
            self.no_cuda_check.setChecked(True)
        else:
            self._engine_hint.setText(
                "COLMAP MVS: 复用 SfM 去畸变输出（SFM_output/dense），只跑稠密重建；"
                "若无去畸变输出则自动补 undistort。结果输出到 MVS_output/colmap。"
            )
            # colmap 4.2 带 CUDA → 默认启用 GPU
            self.no_cuda_check.setText("禁用 GPU（CPU 模式）")
            self.no_cuda_check.setToolTip(
                "colmap 为带 CUDA 构建，勾选后 patch_match 等步骤回退 CPU。"
            )
            self.no_cuda_check.setChecked(False)
        # 同步下载区的引擎与文件列表
        self._sync_download_engine()
        if self._current_project:
            self._check_result()

    def _sync_download_engine(self):
        idx = self.dl_type_combo.findData(self._current_mvs_type)
        if idx >= 0:
            self.dl_type_combo.blockSignals(True)
            self.dl_type_combo.setCurrentIndex(idx)
            self.dl_type_combo.blockSignals(False)
        self._update_download_files()

    def _on_dl_type_changed(self):
        self._update_download_files()

    def _update_download_files(self):
        mvs_type = self.dl_type_combo.currentData() or "openmvs"
        files = COMMON_FILES.get(mvs_type, [])
        self.dl_file_combo.blockSignals(True)
        self.dl_file_combo.clear()
        self.dl_file_combo.addItems(files)
        self.dl_file_combo.blockSignals(False)

    def _on_command_changed(self):
        cmd = self.command_combo.currentData() or "pipeline"
        enabled = set(COMMAND_PARAMS.get(cmd, []))
        # colmap 引擎只认 no_cuda（GPU 开关），其余参数后端不接收 → 置灰
        if self._current_mvs_type == "colmap":
            enabled &= {"no_cuda"}
        for key, w in self._param_widgets.items():
            w.setEnabled(key in enabled)

    # ------------------------------------------------------------------
    # 运行
    # ------------------------------------------------------------------

    def _on_run(self):
        project = self.project_combo.currentText().strip()
        if not project:
            QMessageBox.warning(self, "提示", "请先选择一个项目。")
            return

        mvs_type = self.mvs_type_combo.currentData()
        command = self.command_combo.currentData()

        config = {
            "mvs_type": mvs_type,
            "command": command,
            "model": self.model_edit.text().strip() or "0",
            "resolution_level": str(self.resolution_spin.value()),
            "max_resolution": str(self.max_res_spin.value()),
            "min_resolution": str(self.min_res_spin.value()),
            "number_views": str(self.views_spin.value()),
            "no_cuda": str(self.no_cuda_check.isChecked()).lower(),
            "refine_resolution": str(self.refine_res_spin.value()),
            "scales": str(self.scales_spin.value()),
            "gradient_step": str(self.gradient_spin.value()),
            "texture_resolution": str(self.texture_res_spin.value()),
            "export_type": self.export_combo.currentText(),
        }

        self._current_project = project
        self._set_running(True)
        self._status_label.setText("MVS 启动中...")
        self.result_label.setText("⏳ 正在启动 MVS 任务...")
        self._log_msg(f"▶ 启动 MVS — 项目: {project} | 引擎: {mvs_type} | 命令: {command}")
        self._log_msg(f"  导出格式: {config['export_type']}, 计算: {'CPU' if config['no_cuda'] == 'true' else 'GPU'}")

        self._api.post(
            f"/api/projects/{project}/mvs/run",
            config,
            tag="mvs_run",
        )

    # ------------------------------------------------------------------
    # 轮询日志与结果
    # ------------------------------------------------------------------

    def _poll_log(self):
        if not self._current_project:
            self._poll_timer.stop()
            return
        self._api.get(
            f"/api/projects/{self._current_project}/mvs/log?lines=60&mvs_type={self._current_mvs_type}",
            tag="mvs_poll_log",
        )
        # 每 tick 都查后端运行状态：进程退出即落定，据此判定完成
        self._check_result()

    def _check_result(self):
        self._api.get(
            f"/api/projects/{self._current_project}/mvs/result?mvs_type={self._current_mvs_type}",
            tag="mvs_result",
        )

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
        mvs_type = self.dl_type_combo.currentData() or ""

        default_name = f"{self._current_project}_{Path(filename).name}"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存 MVS 产物",
            default_name,
            "MVS 产物 (*.ply *.mvs *.obj);;所有文件 (*)",
        )
        if not path:
            return

        self.download_btn.setEnabled(False)
        self.download_btn.setText("⏳ 下载中...")
        self._dl_status.setText(f"下载 {filename} ...")
        self._log_msg(f"⬇ 正在下载 MVS 产物 → {path}")

        url = QUrl(
            f"{API_BASE}/api/projects/{self._current_project}/mvs/download/{filename}"
        )
        if mvs_type:
            url.setQuery(f"mvs_type={mvs_type}")
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
        if tag == "mvs_list_projects":
            self._refresh_btn.setEnabled(True)
            if error:
                self._status_label.setText(f"加载项目失败: {error}")
                return
            projects = payload.get("projects", [])
            previous = self.project_combo.currentText()
            selected = fill_project_combo(self.project_combo, projects)
            self._log_msg(f"✓ 加载了 {len(projects)} 个项目")
            # fill 期间屏蔽了信号：若全局选中项与下拉框现值不同，
            # 需手动触发 _on_project_changed 以刷新结果面板。
            if selected != previous:
                self._on_project_changed(selected)

        # ── 启动响应 ──
        elif tag == "mvs_run":
            if error:
                self._set_running(False)
                self._log_msg(f"✗ 启动失败: {error}")
                self._status_label.setText("启动失败")
                QMessageBox.warning(self, "MVS 启动失败", str(error))
            else:
                self._log_msg("✓ MVS 任务已启动，开始轮询日志...")
                self._status_label.setText("MVS 运行中...")
                self._poll_log()
                self._poll_timer.start()

        # ── 日志轮询（仅展示；完成判定以后端 /mvs/result 的 run_status 为准）──
        elif tag == "mvs_poll_log":
            log_text = payload.get("log", "")
            if log_text and log_text != "(no log yet)":
                self._log.setPlainText(log_text)
                scrollbar = self._log.verticalScrollBar()
                if scrollbar:
                    scrollbar.setValue(scrollbar.maximum())

        # ── 结果查询（含后端运行状态，据此判定完成）──
        elif tag == "mvs_result":
            if error:
                self.result_label.setText(f"⚠ 查询结果失败: {error}")
                return
            run_status = payload.get("run_status")
            exit_code = payload.get("exit_code")

            # 运行中：以后端 run_status 判定完成，替代解析日志关键字
            if self._running:
                if run_status == "running":
                    self._status_label.setText("MVS 运行中...")
                    return
                self._set_running(False)
                self._poll_timer.stop()
                if run_status == "ok":
                    self._log_msg("✓ MVS 进程正常退出")
                    self._status_label.setText("MVS 已结束")
                elif run_status == "failed":
                    self._log_msg(f"✗ MVS 进程异常退出 (exit_code: {exit_code})")
                    self._status_label.setText("MVS 失败")
                else:
                    # idle：状态缺失（后端重启等）→ 保守停止，避免无限轮询
                    self._log_msg("⚠ MVS 运行状态丢失，已停止轮询")
                    self._status_label.setText("MVS 状态未知")
                    return

            # 失败优先展示失败信息（即使结果文件缺失或残留旧步骤）
            if run_status == "failed":
                msg = "❌ MVS 进程异常退出"
                if exit_code is not None:
                    msg += f" (exit_code: {exit_code})"
                self.result_label.setText(msg)
                return

            ready = payload.get("ready", False)
            if not ready:
                msg = payload.get("message", "暂无 MVS 结果。")
                self.result_label.setText(f"ℹ {msg}")
                if not self._running:
                    self._status_label.setText("暂无 MVS 结果")
                return

            status = payload.get("status", "")
            steps = payload.get("steps", {})
            total = payload.get("total_elapsed")

            lines = []
            if status == "ok":
                lines.append("✅ MVS 重建成功")
            else:
                lines.append(f"⚠ MVS 状态: {status}")
            if total is not None:
                lines.append(f"   总耗时: {total:.1f}s")
            if steps:
                lines.append("   步骤:")
                for key, st in steps.items():
                    st_status = st.get("status", "?")
                    mark = STATUS_LABELS.get(st_status, "?")
                    label = STEP_LABELS.get(key, key)
                    elapsed = st.get("elapsed")
                    out = st.get("output", "")
                    seg = f"   {mark} {label}"
                    if elapsed is not None:
                        seg += f"  ({elapsed:.1f}s)"
                    if out:
                        seg += f"\n      → {out}"
                    lines.append(seg)
            self.result_label.setText("\n".join(lines))
            if not self._running:
                self._status_label.setText("✓ 已获取 MVS 结果")

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
        self.mvs_type_combo.setEnabled(not running)
        self.command_combo.setEnabled(not running)
        self.progress.setVisible(running)
        if running:
            self.run_btn.setText("⏳ 运行中...")
        else:
            self.run_btn.setText("▶ 运行 MVS")

    def showEvent(self, event):
        super().showEvent(event)
        # 进入页面时重新拉取列表，让「别处选中的项目」也能同步到下拉框
        self._load_projects()

    def _log_msg(self, text: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self._log.append(f"[{ts}] {text}")
