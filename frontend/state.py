"""跨页面共享的应用状态 — 解决「切页后选中项目丢失」问题."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class ProjectState(QObject):
    """全局共享的「当前项目」选择。

    任一页面选中项目（点项目卡片 / 改下拉框）时调用 set_current()，
    其余页面通过 current_changed 信号同步各自的下拉框 / 高亮，
    从而在切换页面后保持选中项目不变。

    - current       当前项目名；空串表示未选择
    - set_current   更新当前项目（值不变时不发信号，避免广播风暴）
    """

    current_changed = Signal(str)  # 项目名，空串 = 未选择

    _instance: "ProjectState | None" = None

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._current = ""

    @classmethod
    def instance(cls) -> "ProjectState":
        if cls._instance is None:
            cls._instance = ProjectState()
        return cls._instance

    @property
    def current(self) -> str:
        return self._current

    def set_current(self, name: str) -> None:
        name = name or ""
        if name != self._current:
            self._current = name
            self.current_changed.emit(name)


def fill_project_combo(
    combo,
    projects,
    text_key: str = "project_name",
) -> str:
    """用项目列表填充下拉框，并选中「全局当前项目」。

    清空 / 重填期间 blockSignals，避免首项自动选中或清空触发用户的
    currentTextChanged 回调；没有可选项时置为空白（-1），不自动选第一个。
    返回最终选中的项目名（可能为空串），调用方据此触发各自的数据加载。
    """
    desired = ProjectState.instance().current
    combo.blockSignals(True)
    combo.clear()
    for p in projects:
        combo.addItem(str(p.get(text_key, "")))
    idx = combo.findText(desired) if desired else -1
    combo.setCurrentIndex(idx)  # -1 → 空白，不自动选第一个
    combo.blockSignals(False)
    return combo.currentText()
