"""SfM（稀疏重建）高级参数对话框。

SfM 主页只放最常用的几项（匹配器、相机模型、特征点数…），其余 24 个
COLMAP 稀疏重建超参数收进这个独立对话框，避免主页过长。

**覆盖语义**：每个参数组是一个可勾选的 ``QGroupBox``。勾上 = 该组字段随请求发送；
不勾 = 一个字段都不发 → 后端 ``ColmapSFMParams`` 全为 ``None`` → CLI 不追加该 flag
→ COLMAP 用自己的默认值。控件里显示的初值就是 COLMAP 的真实默认值，所以「不勾选」
时看到的数字也是实际生效值，不会误导。这与 MVS 页的 PatchMatch / 深度融合两组完全一致。

``default`` 与 ``hint`` 里的默认值取自容器内 ``colmap <cmd> -h``（COLMAP 4.2.0.dev0）
的实测输出，不是文档抄来的——文档里的若干默认值（三角化参数名、峰值阈值、
``ba_local_max_num_iterations`` 等）与实际二进制不符。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .colmap_param_control import build_param_control, set_param_value

_ON_OFF = [("1", "开启 (1)"), ("0", "关闭 (0)")]

SFM_SIFT_PARAMS = [
    {"key": "sift_max_image_size", "label": "最大边长", "kind": "int",
     "lo": -1, "hi": 20000, "step": 100, "default": -1,
     "hint": "特征提取前先缩放图像，越小越快越省内存；-1 = 不限制（COLMAP 默认 -1）"},
    {"key": "sift_first_octave", "label": "起始金字塔层", "kind": "int",
     "lo": -1, "hi": 10, "step": 1, "default": -1,
     "hint": "SIFT 起始 octave；-1 = 先上采样一倍再提取（COLMAP 默认 -1），\n"
             "低纹理/小图场景可设为 0 或 -1 换取更多特征"},
    {"key": "sift_num_octaves", "label": "金字塔组数", "kind": "int",
     "lo": 1, "hi": 16, "step": 1, "default": 4,
     "hint": "高斯金字塔组数，越多尺度越丰富也越慢（COLMAP 默认 4）"},
    {"key": "sift_peak_threshold", "label": "峰值阈值", "kind": "float",
     "lo": 0.0, "hi": 0.1, "step": 0.0005, "decimals": 5, "default": 0.00667,
     "hint": "特征点检测阈值，越小特征越密（COLMAP 默认 0.00667）；\n"
             "弱纹理场景可降到 0.004 左右"},
    {"key": "sift_edge_threshold", "label": "边缘响应阈值", "kind": "float",
     "lo": 0.0, "hi": 1000.0, "step": 1.0, "decimals": 1, "default": 10,
     "hint": "过滤边缘上的不稳定特征点（COLMAP 默认 10）；\n"
             "建筑/工业件等强边缘多的场景可减小以保留更多点"},
]

SFM_MATCH_PARAMS = [
    {"key": "match_max_ratio", "label": "最近邻距离比", "kind": "float",
     "lo": 0.0, "hi": 1.0, "step": 0.01, "decimals": 2, "default": 0.8,
     "hint": "Lowe 比值检验，越小越严格、误匹配越少（COLMAP 默认 0.8）"},
    {"key": "match_cross_check", "label": "双向交叉验证", "kind": "choice",
     "choices": _ON_OFF, "default": "1",
     "hint": "互相为最近邻才保留，更稳但更慢（COLMAP 默认开启）"},
    {"key": "tv_max_error", "label": "几何校验最大误差 (px)", "kind": "float",
     "lo": 0.0, "hi": 100.0, "step": 0.1, "decimals": 2, "default": 4,
     "hint": "两视图几何校验（RANSAC）的最大重投影误差（COLMAP 默认 4）"},
    {"key": "tv_min_num_inliers", "label": "几何校验最少内点", "kind": "int",
     "lo": 0, "hi": 10000, "step": 1, "default": 15,
     "hint": "一对图像至少多少内点才算有效匹配（COLMAP 默认 15）"},
]

SFM_SEQ_PARAMS = [
    {"key": "seq_overlap", "label": "相邻图像数", "kind": "int",
     "lo": 0, "hi": 100, "step": 1, "default": 10,
     "hint": "每张图与前后各多少张匹配（COLMAP 默认 10）；仅顺序匹配时生效"},
    {"key": "seq_loop_detection", "label": "回环检测", "kind": "choice",
     "choices": _ON_OFF, "default": "0",
     "hint": "额外做词汇树回环检测，适合绕圈拍摄（COLMAP 默认关闭）；\n"
             "仅顺序匹配时生效，需要 vocab_tree_path"},
]

SFM_MAPPER_PARAMS = [
    {"key": "mapper_min_num_matches", "label": "最少匹配数", "kind": "int",
     "lo": 0, "hi": 10000, "step": 5, "default": 15,
     "hint": "一对图像要进入重建所需的最少匹配数（COLMAP 默认 15）"},
    {"key": "mapper_init_min_num_inliers", "label": "初始化最少内点", "kind": "int",
     "lo": 0, "hi": 10000, "step": 10, "default": 100,
     "hint": "选初始图像对所需的最少内点数（COLMAP 默认 100）"},
    {"key": "mapper_abs_pose_min_num_inliers", "label": "注册最少内点", "kind": "int",
     "lo": 0, "hi": 10000, "step": 5, "default": 30,
     "hint": "新图像 PnP 注册所需的最少内点数（COLMAP 默认 30）；\n"
             "调小可提高注册率，但容易注册错位"},
    {"key": "mapper_max_reg_trials", "label": "最大注册尝试次数", "kind": "int",
     "lo": 0, "hi": 100, "step": 1, "default": 3,
     "hint": "每张图像注册失败后重试次数（COLMAP 默认 3）"},
    {"key": "mapper_ignore_two_view_tracks", "label": "忽略两视图轨迹", "kind": "choice",
     "choices": _ON_OFF, "default": "1",
     "hint": "忽略只被两张图看到的 3D 点（COLMAP 默认开启）；\n"
             "关掉可增密，但会引入噪声点"},
    {"key": "mapper_filter_max_reproj_error", "label": "三角化最大重投影误差 (px)", "kind": "float",
     "lo": 0.0, "hi": 100.0, "step": 0.1, "decimals": 2, "default": 4,
     "hint": "三角化后按重投影误差滤点（COLMAP 默认 4）"},
    {"key": "mapper_filter_min_tri_angle", "label": "最小三角化角 (°)", "kind": "float",
     "lo": 0.0, "hi": 89.9, "step": 0.1, "decimals": 2, "default": 1.5,
     "hint": "三角化光线夹角下限，太小会产出不稳定点（COLMAP 默认 1.5）"},
]

SFM_MAPPER_BA_PARAMS = [
    {"key": "mapper_ba_refine_focal_length", "label": "精修焦距", "kind": "choice",
     "choices": _ON_OFF, "default": "1",
     "hint": "BA 中优化焦距（COLMAP 默认开启）；\n"
             "相机内参标定准确时可关闭以提速并避免漂移"},
    {"key": "mapper_ba_refine_principal_point", "label": "精修主点", "kind": "choice",
     "choices": _ON_OFF, "default": "0",
     "hint": "BA 中优化主点（COLMAP 默认关闭）；\n"
             "图像少时打开极易发散，一般保持关闭"},
    {"key": "mapper_ba_refine_extra_params", "label": "精修畸变参数", "kind": "choice",
     "choices": _ON_OFF, "default": "1",
     "hint": "BA 中优化畸变系数（COLMAP 默认开启）"},
    {"key": "mapper_ba_local_num_images", "label": "局部 BA 邻近图像数", "kind": "int",
     "lo": 1, "hi": 200, "step": 1, "default": 6,
     "hint": "每轮局部 BA 用多少张邻近图像（COLMAP 默认 6）；\n"
             "调大更准但明显变慢"},
    {"key": "mapper_ba_local_max_num_iterations", "label": "局部 BA 最大迭代", "kind": "int",
     "lo": -1, "hi": 10000, "step": 10, "default": -1,
     "hint": "局部 BA 迭代上限，-1 = 自动（COLMAP 默认 -1）"},
    {"key": "mapper_ba_global_max_num_iterations", "label": "全局 BA 最大迭代", "kind": "int",
     "lo": -1, "hi": 10000, "step": 10, "default": -1,
     "hint": "全局 BA 迭代上限，-1 = 自动（COLMAP 默认 -1）；\n"
             "设成正数可强行多迭代提升精度"},
]

#: (组名, 标题, spec 列表)。组名同时用作「是否覆盖」的标识与 result_params 的分组依据。
SFM_PARAM_GROUPS: list[tuple[str, str, list[dict]]] = [
    ("sift", "特征提取 SiftExtraction", SFM_SIFT_PARAMS),
    ("match", "特征匹配与两视图几何", SFM_MATCH_PARAMS),
    ("seq", "顺序匹配（仅顺序匹配器生效）", SFM_SEQ_PARAMS),
    ("mapper", "Mapper 注册与三角化", SFM_MAPPER_PARAMS),
    ("mapper_ba", "Mapper 束调整 BA", SFM_MAPPER_BA_PARAMS),
]

#: 参数键 → 中文短标签。MVS 页收到 409（工作区陈旧）时用它把变化的字段列成人话。
SFM_PARAM_LABELS: dict[str, str] = {
    spec["key"]: spec["label"]
    for _name, _title, specs in SFM_PARAM_GROUPS
    for spec in specs
}

#: 后端 config 里（而非 ColmapSFMParams 里）参与指纹的字段，同样需要中文标签。
SFM_CONFIG_LABELS: dict[str, str] = {
    "camera_model": "相机模型",
    "single_camera": "单相机内参",
    "matcher": "匹配器",
    "vocab_tree_path": "词汇树路径",
    "reconstruction_name": "稀疏模型名",
    "max_num_features": "特征点数上限",
    "undistort_max_image_size": "去畸变最大边长",
    "num_patch_match_src_images": "PatchMatch 源视图数",
}


def sfm_param_label(key: str) -> str:
    """参数键的人话名字，未知键原样返回（宁可显示英文键，也别漏报）。"""
    return SFM_PARAM_LABELS.get(key) or SFM_CONFIG_LABELS.get(key) or key


#: 各控件初值 = COLMAP 自己的默认值，用来判断某个字段是否真的被改过。
_SPEC_DEFAULTS: dict[str, object] = {
    spec["key"]: spec["default"]
    for _name, _title, specs in SFM_PARAM_GROUPS
    for spec in specs
}


def _differs_from_default(value, default) -> bool:
    """按数值比较，避免 '10.0'（QDoubleSpinBox 的输出）被当成跟默认值 10 不同。"""
    try:
        return float(value) != float(default)
    except (TypeError, ValueError):
        return str(value) != str(default)


def count_overrides(params: dict[str, str]) -> int:
    """参数里真正偏离 COLMAP 默认值的项数——摘要标签就用这个数。

    勾选一个参数组会把该组 **全部** 字段一起发出（组是覆盖的最小粒度，与 MVS 页
    一致），其中大部分就等于 COLMAP 默认值。若按字段个数报数，「快速预览」这种
    只动 4 个值的预设会显示成「已覆盖 11 项」，反而误导用户。
    """
    return sum(1 for key, value in params.items()
               if _differs_from_default(value, _SPEC_DEFAULTS.get(key)))


#: 预设。``values`` 里出现过的键所属的组会被自动勾选，所以新增预设不必手写
#: override 列表：「均衡」「恢复默认」清空覆盖，其余按 values 推导。
SFM_PRESETS: dict[str, dict] = {
    "快速预览": {
        "values": {"sift_max_image_size": 1600, "sift_num_octaves": 3,
                   "mapper_ba_local_num_images": 4,
                   "mapper_ba_global_max_num_iterations": 20},
        "reset_values": False,
    },
    "均衡": {"values": {}, "reset_values": False},
    "高精度": {
        "values": {"sift_max_image_size": -1, "sift_num_octaves": 5,
                   "sift_peak_threshold": 0.004, "match_max_ratio": 0.75,
                   "tv_max_error": 3.0, "tv_min_num_inliers": 20,
                   "mapper_ba_local_num_images": 10,
                   "mapper_ba_global_max_num_iterations": 100},
        "reset_values": False,
    },
    "恢复默认": {"values": {}, "reset_values": True},
}


def _group_of(key: str) -> str | None:
    for name, _title, specs in SFM_PARAM_GROUPS:
        if any(spec["key"] == key for spec in specs):
            return name
    return None


class SfmAdvancedDialog(QDialog):
    """SfM 24 个超参数的模态编辑器。

    用法::

        dlg = SfmAdvancedDialog(self, current=self._sfm_params)
        if dlg.exec() == QDialog.Accepted:
            self._sfm_params = dlg.result_params()

    ``current`` 是上次确认过的覆盖值（键 → 字符串）。传回来时对应组的勾选框会自动
    勾上，所以关掉再打开不会丢设置。
    """

    def __init__(self, parent: QWidget | None = None,
                 current: dict[str, str] | None = None):
        super().__init__(parent)
        self.setWindowTitle("SfM 高级参数")
        self.setModal(True)
        self.setMinimumWidth(720)

        self._widgets: dict[str, QWidget] = {}
        self._groups: dict[str, QGroupBox] = {}
        self._specs_of: dict[str, list[dict]] = {}

        outer = QVBoxLayout(self)
        outer.setSpacing(10)

        intro = QLabel(
            "勾选某个参数组 = 把该组的值发给 COLMAP；不勾 = 该组全部沿用 COLMAP 默认值。\n"
            "控件里显示的初值就是 COLMAP 的默认值。"
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#8b949e;")
        outer.addWidget(intro)

        # ── 预设按钮排 ──
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("预设:"))
        for name in SFM_PRESETS:
            btn = QPushButton(name)
            btn.setObjectName("actionBtn")
            btn.setToolTip("一键填入一组常用参数组合（可再手动调整）")
            btn.clicked.connect(lambda _=False, n=name: self._apply_preset(n))
            preset_row.addWidget(btn)
        preset_row.addStretch()
        self._summary = QLabel("")
        self._summary.setStyleSheet("color:#8b949e;")
        preset_row.addWidget(self._summary)
        outer.addLayout(preset_row)

        # ── 参数区（两列，按流水线顺序排列；外层滚动条兜小窗口）──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        columns = QHBoxLayout(holder)
        columns.setSpacing(12)
        columns.setAlignment(Qt.AlignTop)
        left = QVBoxLayout()
        left.setAlignment(Qt.AlignTop)
        right = QVBoxLayout()
        right.setAlignment(Qt.AlignTop)
        # 左列：提取/匹配；右列：Mapper。与重建流程的先后一致，便于对照。
        for name, title, specs in SFM_PARAM_GROUPS:
            target = left if name in ("sift", "match", "seq") else right
            target.addWidget(self._build_group(name, title, specs))
        columns.addLayout(left, 1)
        columns.addLayout(right, 1)
        scroll.setWidget(holder)
        outer.addWidget(scroll, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("确定")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        # 回填上次的设置，并让摘要标签先有值
        if current:
            for key, value in current.items():
                if key in self._widgets:
                    set_param_value(self._widgets[key], value)
            for key in current:
                gname = _group_of(key)
                if gname:
                    self._groups[gname].setChecked(True)
        self._refresh_summary()

    # ------------------------------------------------------------------

    def _build_group(self, name: str, title: str, specs: list[dict]) -> QGroupBox:
        group = QGroupBox(title)
        group.setCheckable(True)
        group.setChecked(False)  # 默认不覆盖 → 一个字段都不发
        form = QFormLayout(group)
        for spec in specs:
            w = build_param_control(spec)
            form.addRow(spec["label"] + ":", w)
            self._widgets[spec["key"]] = w
        # 勾选框只影响「发不发」，控件始终可编辑：允许先调好数值再决定是否覆盖。
        group.toggled.connect(lambda _on: self._refresh_summary())
        self._groups[name] = group
        self._specs_of[name] = specs
        return group

    def _apply_preset(self, name: str):
        preset = SFM_PRESETS[name]
        if preset["reset_values"]:
            for _n, _t, specs in SFM_PARAM_GROUPS:
                for spec in specs:
                    set_param_value(self._widgets[spec["key"]], spec["default"])
        for key, value in preset["values"].items():
            set_param_value(self._widgets[key], value)
        # 勾选状态最后设：控件值先就位，避免 toggled 期间瞬时启用半套参数。
        # 覆盖哪些组直接从 values 推导——预设改了也不会忘记同步勾选状态。
        wanted = {_group_of(key) for key in preset["values"]}
        for gname, group in self._groups.items():
            group.setChecked(gname in wanted)
        self._refresh_summary()

    def _refresh_summary(self):
        n = count_overrides(self.result_params())
        self._summary.setText("将覆盖 0 项（全部使用 COLMAP 默认）" if n == 0
                              else f"将覆盖 {n} 项参数")

    def result_params(self) -> dict[str, str]:
        """勾选组里的字段（键 → 字符串）。未勾选的组一个字段都不返回。"""
        out: dict[str, str] = {}
        for gname, specs in self._specs_of.items():
            if not self._groups[gname].isChecked():
                continue
            for spec in specs:
                w = self._widgets[spec["key"]]
                out[spec["key"]] = (w.currentData() if hasattr(w, "currentData")
                                    else str(w.value()))
        return out
