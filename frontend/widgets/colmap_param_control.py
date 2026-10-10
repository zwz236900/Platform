"""按 spec 造 COLMAP 参数控件的共享实现。

MVS 页（PatchMatch / 深度融合两组）与 SfM 高级参数对话框共用同一套 spec 约定：

    {"key", "label", "kind", "lo", "hi", "step", "decimals", "default", "hint"}

``kind`` 取 ``"int"`` / ``"float"`` / ``"choice"``；choice 用 ``choices``
（``[(值, 显示文本)]``）代替数值范围。``default`` 一律填 **COLMAP 自己的默认值**，
这样「没勾选覆盖」时控件显示的也是真实生效值，不会误导。
"""

from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox, QWidget


def build_param_control(spec: dict) -> QWidget:
    """按 spec 造控件，初值即 COLMAP 自己的默认值。"""
    if spec["kind"] == "choice":
        combo = QComboBox()
        for value, label in spec["choices"]:
            combo.addItem(label, value)
        idx = combo.findData(spec["default"])
        if idx >= 0:
            combo.setCurrentIndex(idx)
        combo.setToolTip(spec["hint"])
        return combo

    if spec["kind"] == "float":
        spin: QSpinBox | QDoubleSpinBox = QDoubleSpinBox()
        spin.setDecimals(spec.get("decimals", 2))
    else:
        spin = QSpinBox()
    spin.setRange(spec["lo"], spec["hi"])
    spin.setSingleStep(spec["step"])
    spin.setKeyboardTracking(False)  # 输入过程中不发 valueChanged
    spin.setValue(spec["default"])
    spin.setToolTip(spec["hint"])
    return spin


def set_param_value(widget: QWidget, value) -> None:
    """把预设值/默认值写进控件。

    值可以是数字（预设表里写的就是数字）也可以是字符串——``result_params()``
    回传的是表单字符串，重新打开对话框回填时走的正是这条路径，所以要在这里统一
    转成控件需要的类型，否则 ``QSpinBox.setValue("4")`` 会直接抛 TypeError。
    无法解析的值原样忽略，保持控件当前值，不至于让对话框打不开。
    """
    if isinstance(widget, QComboBox):
        idx = widget.findData(str(value))
        if idx >= 0:
            widget.setCurrentIndex(idx)
        return

    if isinstance(value, str):
        try:
            value = float(value) if isinstance(widget, QDoubleSpinBox) else int(float(value))
        except ValueError:
            return
    widget.setValue(value)


def collect_group_values(specs: list[dict], widgets: dict[str, QWidget]) -> dict[str, str]:
    """读一组控件，全部转成字符串——表单字段一律以字符串发送。

    只返回传入 specs 覆盖的键；调用方负责决定这组是否要发（未勾选覆盖的组
    一个字段都不发，后端据此套用 COLMAP 默认值）。
    """
    out: dict[str, str] = {}
    for spec in specs:
        w = widgets[spec["key"]]
        out[spec["key"]] = (w.currentData() if isinstance(w, QComboBox) else str(w.value()))
    return out
