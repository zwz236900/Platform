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
from widgets.colmap_param_control import (
    build_param_control,
    collect_group_values,
    set_param_value,
)
from widgets.sfm_advanced_dialog import sfm_param_label

API_BASE = "http://localhost:8000"

# 引擎
MVS_TYPES = [
    ("openmvs", "OpenMVS"),
    ("colmap",  "COLMAP"),
]

# 命令（value 与后端 MVSCommand 枚举一致）
# 按实际运行步骤排序：interface 等单步命令可直接使用，无需先跑 pipeline；
# 完整流程 pipeline 放在最后作为一键方案。
MVS_COMMANDS = [
    ("interface",   "导入 interface"),
    ("densify",     "稠密重建 densify"),
    ("reconstruct", "表面重建 reconstruct"),
    ("refine",      "网格细化 refine"),
    ("texture",     "纹理映射 texture"),
    ("check",       "检查 check"),
    ("pipeline",    "完整流程 pipeline"),
]

# COLMAP 引擎不支持单步（后端恒跑 mvs，复用 SfM 去畸变输出；无去畸变时自动补 undistort）
MVS_COMMANDS_COLMAP = [
    ("pipeline", "完整流程（稠密重建）"),
]

# 各命令可用的参数
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

# 参数 → 所在 GroupBox 的归属（用于整组显隐）
_PARAM_GROUPS = {
    "left": ["resolution_level", "max_resolution", "min_resolution",
             "refine_resolution", "texture_resolution"],
    "right": ["number_views", "scales", "gradient_step", "model", "export_type"],
    "run": ["no_cuda", "no_geom_consistency"],
    "patchmatch": [],  # 由 COLMAP_PATCHMATCH_PARAMS 填
    "fusion": [],      # 由 COLMAP_FUSION_PARAMS 填
}

# ── COLMAP 稠密重建超参数 ────────────────────────────────────────────────
# 分两张表，对应界面上两个可勾选的分组：勾上组标题才把该组参数发给后端，
# 没勾就一个 flag 都不发 → COLMAP 用自己的默认值，也就是改动之前的行为。
# 每个控件的初值即 COLMAP 自己的默认值，所以「只改一项」时同组其余项虽也被
# 显式发送，数值仍与默认相同，结果不变。
#
# kind: int/float → QSpinBox/QDoubleSpinBox（lo..hi，初值 default）；
#       choice    → QComboBox，发送 currentData()。
# 范围与后端 ColmapMVSParams 的校验一致，避免界面能选出后端会拒的值。
COLMAP_PATCHMATCH_PARAMS = [
    {"key": "pms_max_image_size", "label": "最大边长", "kind": "int",
     "lo": -1, "hi": 20000, "step": 50, "default": -1,
     "hint": "深度图计算所用的最大图像边长，越小越快；-1 = 不限制（COLMAP 默认 -1）"},
    {"key": "pms_window_radius", "label": "窗口半径", "kind": "int",
     "lo": 1, "hi": 32, "step": 1, "default": 5,
     "hint": "匹配窗口半径，弱纹理场景可加大；COLMAP 默认 5，上限 32"},
    {"key": "pms_window_step", "label": "窗口步长", "kind": "int",
     "lo": 1, "hi": 2, "step": 1, "default": 1,
     "hint": "窗口滑动步长，2 更快但精度略降；COLMAP 默认 1"},
    {"key": "pms_num_samples", "label": "采样数", "kind": "int",
     "lo": 1, "hi": 200, "step": 1, "default": 15,
     "hint": "每次迭代的深度采样数，越大越准也越慢；COLMAP 默认 15"},
    {"key": "pms_num_iterations", "label": "迭代次数", "kind": "int",
     "lo": 1, "hi": 100, "step": 1, "default": 5,
     "hint": "PatchMatch 迭代次数；COLMAP 默认 5"},
    {"key": "pms_cache_size", "label": "缓存 (GB)", "kind": "int",
     "lo": 1, "hi": 512, "step": 1, "default": 32,
     "hint": "图像/深度图缓存大小（GB）；COLMAP 默认 32"},
    {"key": "pms_geom_consistency_max_cost", "label": "几何一致性阈值", "kind": "float",
     "lo": 0.0, "hi": 100.0, "step": 0.1, "decimals": 2, "default": 3.0,
     "hint": "几何一致性检查的最大代价，越小越严；COLMAP 默认 3"},
    {"key": "pms_filter", "label": "深度图滤波", "kind": "choice",
     "choices": [("1", "开启 (1)"), ("0", "关闭 (0)")], "default": "1",
     "hint": "是否对深度图做一致性滤波；COLMAP 默认开启"},
    {"key": "pms_filter_min_ncc", "label": "滤波最小 NCC", "kind": "float",
     "lo": -1.0, "hi": 1.0, "step": 0.01, "decimals": 3, "default": 0.1,
     "hint": "保留像素所需的最小 NCC 相似度，越大越严；COLMAP 默认 0.1"},
    {"key": "pms_filter_min_triangulation_angle", "label": "滤波最小三角化角 (°)", "kind": "float",
     "lo": 0.0, "hi": 179.9, "step": 0.5, "decimals": 1, "default": 3.0,
     "hint": "保留像素所需的最小三角化角（度）；COLMAP 默认 3"},
    {"key": "pms_filter_min_num_consistent", "label": "滤波最小一致视图数", "kind": "int",
     "lo": 0, "hi": 32, "step": 1, "default": 2,
     "hint": "至少多少张视图一致才保留该像素；COLMAP 默认 2"},
    {"key": "pms_filter_geom_consistency_max_cost", "label": "滤波几何一致性阈值", "kind": "float",
     "lo": 0.0, "hi": 100.0, "step": 0.1, "decimals": 2, "default": 1.0,
     "hint": "滤波阶段的几何一致性最大代价；COLMAP 默认 1"},
]

COLMAP_FUSION_PARAMS = [
    {"key": "fusion_input_type", "label": "融合输入类型", "kind": "choice",
     "choices": [("auto", "auto（跟随几何一致性）"),
                 ("geometric", "geometric（几何一致性结果）"),
                 ("photometric", "photometric（光度一致性结果）")],
     "default": "auto",
     "hint": "读哪种深度图。auto 跟随几何一致性开关，是安全默认；\n"
             "关了几何一致性却选 geometric 会融合不出点（后端会直接报错）"},
    {"key": "fusion_max_image_size", "label": "最大边长", "kind": "int",
     "lo": -1, "hi": 20000, "step": 50, "default": -1,
     "hint": "融合所用的最大图像边长，越小越快；-1 = 不限制（COLMAP 默认 -1）"},
    {"key": "fusion_min_num_pixels", "label": "最少像素数", "kind": "int",
     "lo": 0, "hi": 1000, "step": 1, "default": 5,
     "hint": "一个三维点至少要被多少像素观测到；COLMAP 默认 5"},
    {"key": "fusion_max_num_pixels", "label": "最多像素数", "kind": "int",
     "lo": 1, "hi": 100000, "step": 100, "default": 10000,
     "hint": "一个三维点最多被多少像素观测到（滤掉重复纹理）；COLMAP 默认 10000"},
    {"key": "fusion_max_reproj_error", "label": "最大重投影误差 (px)", "kind": "float",
     "lo": 0.0, "hi": 50.0, "step": 0.1, "decimals": 2, "default": 2.0,
     "hint": "融合允许的最大重投影误差（像素）；COLMAP 默认 2"},
    {"key": "fusion_max_depth_error", "label": "最大深度误差", "kind": "float",
     "lo": 0.0, "hi": 1.0, "step": 0.001, "decimals": 4, "default": 0.01,
     "hint": "融合允许的最大相对深度误差；COLMAP 默认 0.01"},
    {"key": "fusion_max_normal_error", "label": "最大法向误差 (°)", "kind": "float",
     "lo": 0.0, "hi": 180.0, "step": 1.0, "decimals": 1, "default": 10.0,
     "hint": "融合允许的最大法向夹角（度）；COLMAP 默认 10"},
    {"key": "fusion_check_num_images", "label": "检查图像数", "kind": "int",
     "lo": 1, "hi": 500, "step": 1, "default": 50,
     "hint": "每个点做一致性检查时抽看的图像数；COLMAP 默认 50"},
]

# 预设按钮：勾选哪些分组 + 组内要改成什么值（未列出的项保持 COLMAP 默认）。
# 「均衡」只关掉覆盖开关、不动控件里的数字；「恢复默认」额外把数字也改回默认值。
COLMAP_PRESETS = {
    "快速预览": {"geom_off": True, "override": ("patchmatch",),
                 "values": {"pms_window_step": 2, "pms_num_samples": 10, "pms_num_iterations": 3},
                 "reset_values": False},
    "均衡":     {"geom_off": False, "override": (), "values": {}, "reset_values": False},
    "高精度":   {"geom_off": False, "override": ("patchmatch",),
                 "values": {"pms_max_image_size": -1, "pms_window_step": 1,
                            "pms_num_samples": 20, "pms_num_iterations": 8},
                 "reset_values": False},
    "恢复默认": {"geom_off": False, "override": (), "values": {}, "reset_values": True},
}

COLMAP_PARAM_KEYS = frozenset(
    p["key"] for p in COLMAP_PATCHMATCH_PARAMS + COLMAP_FUSION_PARAMS)
_PARAM_GROUPS["patchmatch"] = [p["key"] for p in COLMAP_PATCHMATCH_PARAMS]
_PARAM_GROUPS["fusion"] = [p["key"] for p in COLMAP_FUSION_PARAMS]

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
        self.left_group = left_group  # OpenMVS 专属：分辨率参数

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
        self.right_group = right_group  # OpenMVS 专属：深度与选项

        # 运行开关：两个引擎共用。colmap 模式下隐藏上面两组 OpenMVS 参数，
        # 只保留此组（colmap 带 CUDA，no_cuda = 回退 CPU；几何一致性是 colmap 专有）。
        gpu_group = QGroupBox("运行选项")
        gform = QFormLayout(gpu_group)
        self.no_cuda_check = QCheckBox("CPU 模式（当前构建无 GPU 加速）")
        self.no_cuda_check.setChecked(True)
        self.no_cuda_check.setToolTip(
            "openmvs 容器内为 2019 年 CPU-only 构建（未编译 CUDA），\n"
            "此开关仅透传 --no-cuda 标记，暂不影响实际运行结果。"
        )
        gform.addRow(self.no_cuda_check)

        # COLMAP 专属：关掉几何一致性只跑一趟，快很多，但融合要跟着改用
        # photometric（否则读不到深度图、产出空点云——后端已自动处理并拦错）。
        self.geom_check = QCheckBox("关闭几何一致性（单趟，更快）")
        self.geom_check.setToolTip(
            "COLMAP 专属。勾选后 patch_match 只跑一趟（不生成 .geometric.bin），\n"
            "速度快很多但深度图未经几何一致性校验；融合输入类型会自动改用\n"
            "photometric。注意：这种布局不支持断点续算，中断后需整段重跑。"
        )
        gform.addRow(self.geom_check)
        params.addWidget(gpu_group)
        self.gpu_group = gpu_group

        self.content_layout.addLayout(params)

        # ── COLMAP 稠密重建超参数（两组可勾选：勾上才覆盖，不勾即 COLMAP 默认）──
        self._colmap_groups: dict[str, QGroupBox] = {}
        self._colmap_forms: dict[str, QFormLayout] = {}
        self._colmap_widgets: dict[str, QWidget] = {}
        self._colmap_group_widgets: dict[str, list] = {}
        colmap_params = QHBoxLayout()
        colmap_params.setSpacing(12)
        for gname, title, specs in (
            ("patchmatch", "深度图 PatchMatch（勾选 = 覆盖 COLMAP 默认）", COLMAP_PATCHMATCH_PARAMS),
            ("fusion", "深度融合 StereoFusion（勾选 = 覆盖 COLMAP 默认）", COLMAP_FUSION_PARAMS),
        ):
            group = QGroupBox(title)
            group.setCheckable(True)
            group.setChecked(False)  # 默认不覆盖 → 一个 flag 都不发
            form = QFormLayout(group)
            widgets = []
            for spec in specs:
                w = build_param_control(spec)
                form.addRow(spec["label"] + ":", w)
                self._colmap_widgets[spec["key"]] = w
                widgets.append(w)
            self._colmap_groups[gname] = group
            self._colmap_forms[gname] = form
            self._colmap_group_widgets[gname] = widgets
            group.toggled.connect(lambda on, g=gname: self._set_group_enabled(g, on))
            self._set_group_enabled(gname, False)
            colmap_params.addWidget(group)
        self.content_layout.addLayout(colmap_params)

        # 预设按钮排（只对 COLMAP 有意义）
        self._preset_bar = QWidget()
        preset_row = QHBoxLayout(self._preset_bar)
        preset_row.setContentsMargins(0, 0, 0, 0)
        preset_row.addWidget(QLabel("预设:"))
        for name in COLMAP_PRESETS:
            btn = QPushButton(name)
            btn.setObjectName("actionBtn")
            btn.setToolTip("一键填入一组常用参数组合（可再手动调整）")
            btn.clicked.connect(lambda _=False, n=name: self._apply_preset(n))
            preset_row.addWidget(btn)
        preset_row.addStretch()
        self._preset_bar.setVisible(False)
        self.content_layout.addWidget(self._preset_bar)

        # ── 操作按钮 ──
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self._status_label = QLabel("就绪，请选择项目")
        self._status_label.setStyleSheet("color:#8b949e; font-size:13px;")
        btn_row.addWidget(self._status_label)

        btn_row.addStretch()

        self.resume_btn = QPushButton("↻ 断点续算")
        self.resume_btn.setObjectName("actionBtn")
        self.resume_btn.setMinimumWidth(110)
        self.resume_btn.setToolTip(
            "COLMAP MVS 中断/被杀后断点续算：基于 MVS_output/colmap 已有的深度图，"
            "跳过已算好的影像只补缺（不经 reuse，不清历史）。仅对 COLMAP 引擎有效。"
        )
        self.resume_btn.clicked.connect(self._on_resume)
        btn_row.addWidget(self.resume_btn)

        self.run_btn = QPushButton("▶ 运行 MVS")
        self.run_btn.setObjectName("primaryBtn")
        self.run_btn.setMinimumWidth(120)
        self.run_btn.clicked.connect(self._on_run)
        btn_row.addWidget(self.run_btn)

        self.content_layout.addLayout(btn_row)

        # ── 进度区：步骤文案 + 详情 + 进度条 ──
        self._step_label = QLabel("")
        self._step_label.setStyleSheet("color:#e6edf3; font-size:13px; font-weight:600;")
        self._step_label.setVisible(False)
        self._detail_label = QLabel("")
        self._detail_label.setStyleSheet("color:#8b949e; font-size:12px;")
        self._detail_label.setVisible(False)
        self.content_layout.addWidget(self._step_label)
        self.content_layout.addWidget(self._detail_label)

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
            "no_geom_consistency": self.geom_check,
        }
        # 记录每个参数在所属 QFormLayout 中的行号，用于整行显隐——
        # 只 setVisible(False) 字段会让 label 文字残留，需 setRowVisible 连 label 一起隐藏。
        # COLMAP 两组参数的表必须一并列入，否则它们的行永远不会被隐藏/显示。
        self._param_widgets.update(self._colmap_widgets)
        all_forms = (lform, rform, gform,
                     self._colmap_forms["patchmatch"], self._colmap_forms["fusion"])
        self._param_rows: dict[str, tuple] = {}
        for key, w in self._param_widgets.items():
            for form in all_forms:
                row, _role = form.getWidgetPosition(w)
                if row >= 0:
                    self._param_rows[key] = (form, row)
                    break
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
            self._probe_active_run()
        self._update_download_files()

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
        """按当前引擎重填命令下拉框：colmap 只保留完整流程。

        首次加载或原选中项不在当前引擎列表时，默认选「完整流程 pipeline」；
        其余情况保留之前选中的单步命令。
        """
        commands = MVS_COMMANDS_COLMAP if self._current_mvs_type == "colmap" else MVS_COMMANDS
        previous = self.command_combo.currentData()
        self.command_combo.blockSignals(True)
        self.command_combo.clear()
        for value, label in commands:
            self.command_combo.addItem(label, value)
        idx = self.command_combo.findData(previous)
        if idx < 0:
            idx = self.command_combo.findData("pipeline")
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
                "OpenMVS: 基于 SfM 结果运行稠密重建，单步命令（interface→densify→…）可直接执行，无需先跑完整流程；"
                "COLMAP: 复用 SfM 去畸变输出 + patch_match 稠密重建。"
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
        """从后端读取 MVS_output/<type>/ 里实际存在的文件，填充下载下拉框。"""
        mvs_type = self.dl_type_combo.currentData() or "openmvs"
        project = self._current_project or self.project_combo.currentText().strip()
        if not project:
            self.dl_file_combo.clear()
            return
        self._api.get(
            f"/api/projects/{project}/mvs/files?mvs_type={mvs_type}",
            tag="mvs_files",
        )

    def _set_group_enabled(self, group_name: str, on: bool):
        """整组置灰/启用（组标题的勾选框控制是否覆盖 COLMAP 默认值）。"""
        for w in self._colmap_group_widgets[group_name]:
            w.setEnabled(on)

    def _on_command_changed(self):
        cmd = self.command_combo.currentData() or "pipeline"
        enabled = set(COMMAND_PARAMS.get(cmd, []))
        is_colmap = self._current_mvs_type == "colmap"

        # 只展示当前命令实际用到的参数（openmvs 按命令显隐，colmap 显示稠密重建超参数），
        # 不再把全部参数摊开再置灰。整行隐藏（含 label），避免「输入框没了、文字还在」。
        visible = ({"no_cuda", "no_geom_consistency"} | COLMAP_PARAM_KEYS
                   if is_colmap else enabled)

        for key, (form, row) in self._param_rows.items():
            form.setRowVisible(row, key in visible)

        # 分组外壳：组内没有任何可见参数时整组隐藏（如 interface 只留模型、reconstruct 无参数）
        for gname, group in (("left", self.left_group),
                             ("right", self.right_group),
                             ("run", self.gpu_group),
                             ("patchmatch", self._colmap_groups["patchmatch"]),
                             ("fusion", self._colmap_groups["fusion"])):
            group.setVisible(any(k in visible for k in _PARAM_GROUPS[gname]))

        # 预设按钮只对 COLMAP 有意义
        self._preset_bar.setVisible(is_colmap)

    # ------------------------------------------------------------------
    # COLMAP 稠密重建超参数
    # ------------------------------------------------------------------

    def _apply_preset(self, name: str):
        """一键填入一组常用参数（见 COLMAP_PRESETS），之后仍可手动改。"""
        preset = COLMAP_PRESETS[name]
        self.geom_check.setChecked(preset["geom_off"])
        if preset["reset_values"]:
            for specs in (COLMAP_PATCHMATCH_PARAMS, COLMAP_FUSION_PARAMS):
                for spec in specs:
                    self._set_colmap_value(spec["key"], spec["default"])
        for key, value in preset["values"].items():
            self._set_colmap_value(key, value)
        # 勾选状态最后设：控件值先就位，避免 toggled 期间瞬时启用半套参数
        for gname, group in self._colmap_groups.items():
            group.setChecked(gname in preset["override"])
        self._log_msg(f"  已应用预设「{name}」")

    def _set_colmap_value(self, key: str, value):
        set_param_value(self._colmap_widgets[key], value)

    def _collect_colmap_params(self) -> dict:
        """勾选了覆盖的分组才发送；没勾的组一个字段都不发 → COLMAP 用自己的默认值。"""
        out: dict[str, str] = {}
        for gname, specs in (("patchmatch", COLMAP_PATCHMATCH_PARAMS),
                             ("fusion", COLMAP_FUSION_PARAMS)):
            if not self._colmap_groups[gname].isChecked():
                continue
            out.update(collect_group_values(specs, self._colmap_widgets))
        return out

    # ------------------------------------------------------------------
    # SfM 工作区陈旧（409）
    # ------------------------------------------------------------------

    def _show_stale_workspace(self, body: dict):
        """把 409 的结构化 detail 摊成人话：变了的参数 + 下一步怎么做。

        后端刻意不提供「强制复用」后门——旧工作区里的稀疏模型与冻结的
        stereo/patch-match.cfg 是用另一组参数产出的，复用只会得到看起来正常、
        实际错配的点云。所以要给用户一条明确的出路（回 SfM 页重跑），
        而不是一个能绕过检查的按钮。
        """
        changed = body.get("changed") or []
        lines = [body.get("message", "SfM 工作区与当前参数不一致。")]
        if changed:
            lines += ["", "变化的参数："]
            for item in changed:
                lines.append(
                    f"  • {sfm_param_label(str(item.get('name', '')))}："
                    f"{self._fmt_param(item.get('produced'))} → {self._fmt_param(item.get('current'))}")
        guidance = body.get("guidance")
        if guidance:
            lines += ["", str(guidance)]
        self._status_label.setText("SfM 工作区已过期，需重建")
        QMessageBox.warning(self, "SfM 工作区与当前参数不一致", "\n".join(lines))

    @staticmethod
    def _fmt_param(value) -> str:
        """None = 该项没有覆盖，走 COLMAP 自己的默认值。"""
        return "（用默认值）" if value is None else str(value)

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
        if mvs_type == "colmap":
            # 稠密重建超参数（未勾选覆盖的组不发送 → COLMAP 默认）
            config.update(self._collect_colmap_params())
            config["no_geom_consistency"] = str(self.geom_check.isChecked()).lower()

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

    def _on_resume(self):
        """断点续算 COLMAP MVS（不清历史深度图，跳过已算好的只补缺）。"""
        project = self.project_combo.currentText().strip()
        if not project:
            QMessageBox.warning(self, "提示", "请先选择一个项目。")
            return

        # 续算只对 COLMAP MVS 有意义：自动把引擎切到 colmap（与 _probe_active_run 的
        # 切换一致，保证 /mvs/result 按 colmap 判定完成）。
        idx = self.mvs_type_combo.findData("colmap")
        if idx < 0:
            QMessageBox.warning(self, "提示", "无 COLMAP 引擎项，无法续算。")
            return
        if self.mvs_type_combo.currentData() != "colmap":
            self.mvs_type_combo.blockSignals(True)
            self.mvs_type_combo.setCurrentIndex(idx)
            self.mvs_type_combo.blockSignals(False)
            self._current_mvs_type = "colmap"
            self._repopulate_commands()
            self._sync_download_engine()

        self._current_project = project
        self._set_running(True)
        self._status_label.setText("续算启动中...")
        self.result_label.setText("⏳ 正在请求断点续算...")
        self._log_msg(f"↻ 请求 COLMAP MVS 断点续算 — 项目: {project}")

        self._api.post(
            f"/api/projects/{project}/mvs/resume",
            {},
            tag="mvs_resume",
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
        # 每 tick 都查确定性进度（步骤 + MVS 深度图计数），驱动进度条
        self._poll_progress()

    def _poll_progress(self):
        if not self._current_project:
            return
        self._api.get(
            f"/api/projects/{self._current_project}/progress",
            tag="mvs_progress",
        )

    def _probe_active_run(self):
        """进入页面/切换项目时探测：该项目是否已有管线在跑（含后端重启后
        仍在容器里跑的孤儿 COLMAP）。有则自动进入运行态并恢复轮询，否则停在
        就绪态 —— 否则已运行的 MVS 永远不显示进度。
        """
        if self._running or not self._current_project:
            return
        self._api.get(
            f"/api/projects/{self._current_project}/progress",
            tag="mvs_probe",
        )

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
        filename = self.dl_file_combo.currentData() \
            or self.dl_file_combo.currentText().strip()
        if not filename:
            QMessageBox.warning(self, "提示", "请选择要下载的产物。")
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
                self._status_label.setText("启动失败")
                # 工作区陈旧（409）：把变化的参数列成人话，并说清下一步怎么做。
                # 这条路径后端不会启动容器，所以这里只是提示，不是失败。
                body = payload.get("_error_body")
                if isinstance(body, dict) and body.get("code") == "sfm_workspace_stale":
                    self._log_msg(f"⚠ SfM 工作区已过期，未启动 MVS：{body.get('message', '')}")
                    self._show_stale_workspace(body)
                else:
                    self._log_msg(f"✗ 启动失败: {error}")
                    QMessageBox.warning(self, "MVS 启动失败", str(error))
            else:
                self._log_msg("✓ MVS 任务已启动，开始轮询日志...")
                self._status_label.setText("MVS 运行中...")
                self._poll_log()
                self._poll_timer.start()

        # ── 断点续算启动响应 ──
        elif tag == "mvs_resume":
            if error:
                self._set_running(False)
                self._log_msg(f"✗ 续算启动失败: {error}")
                self._status_label.setText("续算未启动")
                QMessageBox.warning(self, "续算启动失败", str(error))
            else:
                message = payload.get("message", "COLMAP MVS 续算已启动")
                self._log_msg(f"✓ {message}")
                data = payload.get("data") or {}
                if data.get("refs"):
                    self._log_msg(
                        f"  已有 photometric {data.get('photo_done')}/{data.get('refs')}、"
                        f"geometric {data.get('geo_done')}/{data.get('refs')}，续算中…")
                self._status_label.setText("MVS 续算运行中...")
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

        # ── 下载产物列表（按步骤标注，展示真实产物而非整目录）──
        elif tag == "mvs_files":
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

        # ── 恢复已存在运行：探测到运行中 → 进入运行态并开始轮询 ──
        elif tag == "mvs_probe":
            if self._running or error:
                return
            if not payload.get("running"):
                return
            kind = payload.get("kind")
            if kind in ("colmap", "openmvs"):
                # 把引擎下拉框对齐到实际在跑的引擎，保证 /mvs/result 判完成正确
                idx = self.mvs_type_combo.findData(kind)
                if idx >= 0 and self.mvs_type_combo.currentData() != kind:
                    self.mvs_type_combo.blockSignals(True)
                    self.mvs_type_combo.setCurrentIndex(idx)
                    self.mvs_type_combo.blockSignals(False)
                    self._current_mvs_type = kind
                    self._repopulate_commands()
            self._log_msg("↻ 检测到该项目已有 MVS 在运行，恢复进度监控…")
            self._status_label.setText("MVS 运行中...")
            self._set_running(True)
            self._poll_log()
            self._poll_timer.start()

        # ── 确定性进度（步骤标签 + MVS 深度图计数百分比）──
        elif tag == "mvs_progress":
            # 仅在运行期间渲染；结束/空闲的迟到响应一律忽略
            if not self._running or error:
                return
            if not payload.get("running"):
                return
            label = payload.get("label") or ""
            detail = payload.get("detail") or ""
            percent = payload.get("percent")

            if label:
                self._step_label.setText(label)
                self._step_label.setVisible(True)
            if isinstance(percent, (int, float)):
                # 确定值：两趟深度图计数驱动（COLAMP PatchMatch）
                self.progress.setRange(0, 100)
                self.progress.setValue(int(round(percent)))
                if detail:
                    self._detail_label.setText(detail)
                    self._detail_label.setVisible(True)
            else:
                # 无法计数（特征/匹配/SfM 或 OpenMVS）：保持滚动条 + 步骤文案
                self.progress.setRange(0, 0)

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
        self.resume_btn.setEnabled(not running)
        self.project_combo.setEnabled(not running)
        self.mvs_type_combo.setEnabled(not running)
        self.command_combo.setEnabled(not running)
        self.progress.setVisible(running)
        self.progress.setRange(0, 0)  # 每次重开先回到滚动态，等首个 /progress 结果
        # 步骤/详情文案只在运行期间显示，先隐藏避免残留上一轮内容
        self._step_label.setVisible(False)
        self._detail_label.setVisible(False)
        if running:
            self.run_btn.setText("⏳ 运行中...")
        else:
            self.run_btn.setText("▶ 运行 MVS")

    def showEvent(self, event):
        super().showEvent(event)
        # 进入页面时重新拉取列表，让「别处选中的项目」也能同步到下拉框
        self._load_projects()
        # 探测是否有已存在的运行：恢复监控（页面打开前就已在跑的任务）
        if self._current_project:
            self._probe_active_run()

    def _log_msg(self, text: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self._log.append(f"[{ts}] {text}")
