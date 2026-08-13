"""SAM 分割页面 — 框选分割 + 点选分割 + 语义分割."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QComboBox,
    QSpinBox,
    QAbstractSpinBox,
    QTextEdit,
    QMessageBox,
    QWidget,
    QSplitter,
    QButtonGroup,
)
from PySide6.QtCore import Qt, Signal, QUrl, QRect, QPoint, QEvent
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QMouseEvent, QPaintEvent

from .base_page import BasePage
from state import ProjectState, fill_project_combo

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

API_BASE = "http://localhost:8000"


# ---------------------------------------------------------------------------
# HTTP 客户端
# ---------------------------------------------------------------------------

class SamApiClient(QWidget):
    """SAM 页面专用的 HTTP 客户端（单例）。"""

    finished = Signal(str, str, object)  # (method, path, payload)

    _instance: "SamApiClient | None" = None

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._mgr = QNetworkAccessManager(self)
        self.setVisible(False)

    @classmethod
    def instance(cls) -> "SamApiClient":
        if cls._instance is None:
            cls._instance = SamApiClient()
        return cls._instance

    def get(self, path: str, tag: str = ""):
        url = QUrl(API_BASE + path)
        req = QNetworkRequest(url)
        reply = self._mgr.get(req)
        self._tag(reply, "GET", path, tag)

    def post(self, path: str, form_data: dict | None = None, tag: str = ""):
        url = QUrl(API_BASE + path)
        req = QNetworkRequest(url)
        if form_data:
            body = urlencode(form_data).encode("utf-8")
            req.setRawHeader(b"Content-Type", b"application/x-www-form-urlencoded")
            reply = self._mgr.post(req, body)
        else:
            reply = self._mgr.post(req, b"")
        self._tag(reply, "POST", path, tag)

    def fetch_image(self, path: str, tag: str = ""):
        url = QUrl(API_BASE + path)
        req = QNetworkRequest(url)
        reply = self._mgr.get(req)
        self._tag(reply, "GET_IMAGE", path, tag)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _tag(self, reply: QNetworkReply, method: str, path: str, tag: str):
        reply.setProperty("_tag", tag)
        reply.setProperty("_method", method)
        reply.setProperty("_path", path)
        reply.finished.connect(lambda r=reply, s=self: s._on_finished(r))

    def _on_finished(self, reply: QNetworkReply):
        method = reply.property("_method")
        path = reply.property("_path")
        tag = reply.property("_tag")
        try:
            if reply.error() != QNetworkReply.NoError:
                err_text = bytes(reply.readAll()).decode("utf-8", errors="replace")
                try:
                    err_obj = json.loads(err_text)
                    err_msg = err_obj.get("detail", err_text)
                except Exception:
                    err_msg = err_text or reply.errorString()
                self.finished.emit(method, path, {"_error": err_msg, "_tag": tag})
            elif method == "GET_IMAGE":
                data = bytes(reply.readAll())
                self.finished.emit(method, path, {"_image_data": data, "_tag": tag})
            else:
                data = bytes(reply.readAll()).decode("utf-8", errors="replace")
                try:
                    obj = json.loads(data)
                    obj["_tag"] = tag
                    self.finished.emit(method, path, obj)
                except Exception:
                    self.finished.emit(method, path, {"_raw": data, "_tag": tag})
        finally:
            reply.deleteLater()


# ---------------------------------------------------------------------------
# 图片显示画布（支持框选）
# ---------------------------------------------------------------------------

class ImageCanvas(QLabel):
    """可绘制矩形框的图片显示控件。"""

    box_changed = Signal(int, int, int, int)  # x1, y1, x2, y2

    def __init__(self, interactive: bool = True, parent: QWidget | None = None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(200, 200)
        self.setStyleSheet("background-color:#0d1117; border:1px solid #21262d;")
        self.setMouseTracking(interactive)

        self._pixmap: QPixmap | None = None
        self._drawing = False
        self._box = QRect()
        self._interactive = interactive

    def set_image(self, pixmap: QPixmap):
        self._pixmap = pixmap
        self._update_display()

    def set_box(self, x1: int, y1: int, x2: int, y2: int):
        """直接设置框（来自缓存）。"""
        self._box = QRect(QPoint(x1, y1), QPoint(x2, y2))
        self.update()

    def current_box(self) -> tuple[int, int, int, int]:
        r = self._box.normalized()
        if r.isNull():
            return (0, 0, 0, 0)
        return (r.left(), r.top(), r.right(), r.bottom())

    def has_valid_box(self) -> bool:
        x1, y1, x2, y2 = self.current_box()
        return (x2 - x1) >= 5 and (y2 - y1) >= 5

    def clear_box(self):
        self._box = QRect()
        self.update()

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _update_display(self):
        if self._pixmap and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            self.setPixmap(scaled)
        else:
            self.setText("暂无图片")
            self.setStyleSheet(
                "background-color:#0d1117; border:1px solid #21262d;"
                " color:#484f58; font-size:14px;"
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()

    # ------------------------------------------------------------------
    # 坐标映射
    # ------------------------------------------------------------------

    def _image_to_widget(self, pos: QPoint) -> QPoint:
        if not self._pixmap or self._pixmap.isNull():
            return pos
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()
        scale = min(ww / pw, wh / ph) if pw > 0 and ph > 0 else 1.0
        dx = int((ww - pw * scale) / 2)
        dy = int((wh - ph * scale) / 2)
        return QPoint(int(pos.x() * scale + dx), int(pos.y() * scale + dy))

    def _widget_to_image(self, pos: QPoint) -> QPoint:
        if not self._pixmap or self._pixmap.isNull():
            return pos
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()
        scale = min(ww / pw, wh / ph) if pw > 0 and ph > 0 else 1.0
        dx = int((ww - pw * scale) / 2)
        dy = int((wh - ph * scale) / 2)
        return QPoint(
            max(0, int((pos.x() - dx) / scale)),
            max(0, int((pos.y() - dy) / scale)),
        )

    # ------------------------------------------------------------------
    # 鼠标事件
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent):
        if self._interactive and event.button() == Qt.LeftButton and self._pixmap:
            img_pos = self._widget_to_image(event.pos())
            self._box = QRect(img_pos, img_pos)
            self._drawing = True
            self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._interactive and self._drawing:
            img_pos = self._widget_to_image(event.pos())
            self._box.setBottomRight(img_pos)
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._interactive and event.button() == Qt.LeftButton and self._drawing:
            self._drawing = False
            img_pos = self._widget_to_image(event.pos())
            self._box.setBottomRight(img_pos)
            self.update()
            x1, y1, x2, y2 = self.current_box()
            if x2 - x1 >= 5 and y2 - y1 >= 5:
                self.box_changed.emit(x1, y1, x2, y2)
        super().mouseReleaseEvent(event)

    # ------------------------------------------------------------------
    # 绘制
    # ------------------------------------------------------------------

    def paintEvent(self, event: QPaintEvent):
        super().paintEvent(event)
        if self._interactive and self._pixmap and not self._pixmap.isNull() and not self._box.isNull():
            r = self._box.normalized()
            if r.isNull() or r.width() < 1:
                return
            painter = QPainter(self)
            pen = QPen(QColor("#58a6ff"), 2, Qt.SolidLine)
            painter.setPen(pen)
            painter.setBrush(QColor(31, 111, 235, 30))
            top_left = self._image_to_widget(r.topLeft())
            bottom_right = self._image_to_widget(r.bottomRight())
            painter.drawRect(QRect(top_left, bottom_right).normalized())
            painter.end()


# ---------------------------------------------------------------------------
# 点选分割 — 点击式画布
# ---------------------------------------------------------------------------

class PointCanvas(ImageCanvas):
    """多点选择控件 — 左键添加正点（绿），右键添加负点（红），按钮触发分割。"""

    points_changed = Signal()  # 点集合发生变化

    def __init__(self, parent: QWidget | None = None):
        super().__init__(interactive=True, parent=parent)
        self._points: list[tuple[int, int, int]] = []  # (x, y, label) 1=正 0=负

    # -- 访问器 ----------------------------------------------------------

    def points(self) -> list[tuple[int, int, int]]:
        return list(self._points)

    def set_points(self, pts: list[tuple[int, int, int]]):
        self._points = list(pts)
        self.update()

    def clear_points(self):
        self._points.clear()
        self.update()

    def add_point(self, x: int, y: int, label: int):
        self._points.append((x, y, label))
        self.update()

    # -- 鼠标事件 --------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent):
        if not self._interactive or not self._pixmap:
            super().mousePressEvent(event)
            return

        label = None
        if event.button() == Qt.LeftButton:
            label = 1   # 正点（前景）
        elif event.button() == Qt.RightButton:
            label = 0   # 负点（背景）

        if label is not None:
            img_pos = self._widget_to_image(event.pos())
            self.add_point(img_pos.x(), img_pos.y(), label)
            self.points_changed.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    # -- 绘制 ------------------------------------------------------------

    def paintEvent(self, event: QPaintEvent):
        super().paintEvent(event)
        if not self._pixmap or self._pixmap.isNull():
            return

        painter = QPainter(self)
        for (x, y, label) in self._points:
            widget_pos = self._image_to_widget(QPoint(x, y))
            if label == 1:
                color = QColor("#3fb950")  # 绿色正点
                sym = "+"
            else:
                color = QColor("#f85149")  # 红色负点
                sym = "−"

            pen = QPen(color, 2, Qt.SolidLine)
            painter.setPen(pen)
            # 十字线
            painter.drawLine(widget_pos.x() - 12, widget_pos.y(),
                             widget_pos.x() + 12, widget_pos.y())
            painter.drawLine(widget_pos.x(), widget_pos.y() - 12,
                             widget_pos.x(), widget_pos.y() + 12)
            # 外圈
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(widget_pos, 8, 8)
            # 序号标签
            font = QFont("Consolas", 11, QFont.Bold)
            painter.setFont(font)
            painter.setPen(color)
            painter.drawText(widget_pos.x() + 14, widget_pos.y() + 5,
                             f"#{len(self._points)} {sym}")

        painter.end()


# ---------------------------------------------------------------------------
# 框选分割面板
# ---------------------------------------------------------------------------

class BoxSelectionPanel(QWidget):
    """框选分割 — 图片浏览、框选、SAM 推理。"""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._api = SamApiClient.instance()
        self._api.finished.connect(self._on_response)
        self._state = ProjectState.instance()
        self._state.current_changed.connect(self._on_shared_project_changed)

        self._project = ""
        self._images: list[str] = []
        self._current_idx = -1
        self._model_loaded = False
        # 每张图独立缓存其框坐标: {index: (x1, y1, x2, y2)}
        self._box_cache: dict[int, tuple[int, int, int, int]] = {}
        # 每张图独立缓存分割结果图
        self._result_cache: dict[int, QPixmap] = {}
        # 最近一次分割的框，作为无缓存图片的默认框
        self._last_segmented_box: tuple[int, int, int, int] | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # ── 项目选择行 ──
        proj_row = QHBoxLayout()
        proj_row.addWidget(QLabel("项目:"))
        self._proj_combo = QComboBox()
        self._proj_combo.setMinimumWidth(160)
        self._proj_combo.currentTextChanged.connect(self._on_project_changed)
        proj_row.addWidget(self._proj_combo)

        self._refresh_proj_btn = QPushButton("🔄")
        self._refresh_proj_btn.setObjectName("actionBtn")
        self._refresh_proj_btn.setFixedWidth(40)
        self._refresh_proj_btn.clicked.connect(self._load_projects)
        proj_row.addWidget(self._refresh_proj_btn)

        proj_row.addSpacing(16)
        proj_row.addWidget(QLabel("第"))
        self._page_spin = QSpinBox()
        self._page_spin.setMinimum(1)
        self._page_spin.setValue(1)
        self._page_spin.setFixedWidth(60)
        self._page_spin.setEnabled(False)
        self._page_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self._page_spin.valueChanged.connect(self._on_page_spin)
        proj_row.addWidget(self._page_spin)

        proj_row.addWidget(QLabel("张"))
        self._total_label = QLabel("/ 0 张")
        self._total_label.setStyleSheet("color:#8b949e; font-size:12px;")
        proj_row.addWidget(self._total_label)

        proj_row.addStretch()
        layout.addLayout(proj_row)

        # ── 控制按钮行 ──
        ctrl_row = QHBoxLayout()

        self._load_model_btn = QPushButton("🚀 加载 SAM 模型")
        self._load_model_btn.setObjectName("primaryBtn")
        self._load_model_btn.clicked.connect(self._on_toggle_model)
        ctrl_row.addWidget(self._load_model_btn)

        self._segment_btn = QPushButton("✂ 分割")
        self._segment_btn.setObjectName("primaryBtn")
        self._segment_btn.clicked.connect(self._on_segment)
        self._segment_btn.setEnabled(False)
        ctrl_row.addWidget(self._segment_btn)

        ctrl_row.addStretch()

        self._box_info_label = QLabel("框: 未选择")
        self._box_info_label.setStyleSheet("color:#8b949e; font-size:12px;")
        ctrl_row.addWidget(self._box_info_label)

        self._status_label = QLabel("请先选择项目")
        self._status_label.setStyleSheet("color:#8b949e; font-size:12px;")
        ctrl_row.addWidget(self._status_label)

        layout.addLayout(ctrl_row)

        # ── 双栏图片区域 ──
        splitter = QSplitter(Qt.Horizontal)

        # 左：原图 + 框选
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_label = QLabel("原图（拖拽鼠标画框 · ↑↓ 键切换）")
        left_label.setStyleSheet("color:#8b949e; font-size:11px; font-weight:600;")
        left_layout.addWidget(left_label)
        self._left_canvas = ImageCanvas(interactive=True)
        self._left_canvas.box_changed.connect(self._on_box_changed)
        self._left_canvas.setFocusPolicy(Qt.StrongFocus)
        self._left_canvas.installEventFilter(self)
        left_layout.addWidget(self._left_canvas, 1)
        splitter.addWidget(left_panel)

        # 右：分割结果
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_label = QLabel("分割结果")
        right_label.setStyleSheet("color:#8b949e; font-size:11px; font-weight:600;")
        right_layout.addWidget(right_label)
        self._right_canvas = ImageCanvas(interactive=False)
        right_layout.addWidget(self._right_canvas, 1)
        splitter.addWidget(right_panel)

        layout.addWidget(splitter, 1)

        # ── 导航按钮行 ──
        nav_row = QHBoxLayout()
        nav_row.addStretch()
        self._prev_btn = QPushButton("◀ 上一张")
        self._prev_btn.setObjectName("actionBtn")
        self._prev_btn.clicked.connect(self._prev_image)
        nav_row.addWidget(self._prev_btn)

        self._next_btn = QPushButton("下一张 ▶")
        self._next_btn.setObjectName("actionBtn")
        self._next_btn.clicked.connect(self._next_image)
        nav_row.addWidget(self._next_btn)
        layout.addLayout(nav_row)

        # ── 日志 ──
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setObjectName("logArea")
        self._log.setMaximumHeight(100)
        self._log.setFont(QFont("Consolas", 9))
        layout.addWidget(self._log)

    # ------------------------------------------------------------------
    # 显示 / 键盘切换图片
    # ------------------------------------------------------------------

    def showEvent(self, event):
        super().showEvent(event)
        # 进入页面时重新拉取列表，让「别处选中的项目」也能同步到下拉框
        self._load_projects()

    def eventFilter(self, obj, event):
        if obj is self._left_canvas and event.type() == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key_Up, Qt.Key_Left):
                self._prev_image()
                return True
            elif event.key() in (Qt.Key_Down, Qt.Key_Right):
                self._next_image()
                return True
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------
    # 项目列表
    # ------------------------------------------------------------------

    def _load_projects(self):
        self._refresh_proj_btn.setEnabled(False)
        self._api.get("/api/projects", tag="box_list_projects")

    def _on_project_changed(self, project: str):
        self._state.set_current(project)
        if project == self._project:
            return
        self._project = project
        self._images = []
        self._current_idx = -1
        self._box_cache.clear()
        self._result_cache.clear()
        self._last_segmented_box = None
        self._left_canvas.clear_box()
        self._page_spin.setEnabled(False)
        if project:
            self._status_label.setText("加载图片列表...")
            self._api.get(f"/api/projects/{project}/images/list", tag="box_list_images")

    def _on_shared_project_changed(self, project: str):
        """其余页面选中了项目 → 同步本页下拉框。"""
        if self._proj_combo.currentText() == project:
            return
        if project:
            idx = self._proj_combo.findText(project)
            if idx >= 0:
                self._proj_combo.setCurrentIndex(idx)
        else:
            self._proj_combo.setCurrentIndex(-1)

    # ------------------------------------------------------------------
    # 图片导航
    # ------------------------------------------------------------------

    def _on_page_spin(self, num: int):
        """从页码输入跳转图片。"""
        idx = num - 1
        if 0 <= idx < len(self._images) and idx != self._current_idx:
            self._goto_image(idx)

    def _prev_image(self):
        if self._current_idx > 0:
            self._goto_image(self._current_idx - 1)

    def _next_image(self):
        if self._current_idx < len(self._images) - 1:
            self._goto_image(self._current_idx + 1)

    def _goto_image(self, idx: int):
        """切换到指定索引的图片，保存当前框到缓存。"""
        # 保存当前框到缓存
        if self._current_idx >= 0 and self._left_canvas.has_valid_box():
            self._box_cache[self._current_idx] = self._left_canvas.current_box()

        self._current_idx = idx
        self._page_spin.blockSignals(True)
        self._page_spin.setValue(idx + 1)
        self._page_spin.blockSignals(False)

        # 恢复该图片的框
        cached = self._box_cache.get(idx)
        if cached is not None:
            self._left_canvas.set_box(*cached)
            self._update_box_info()
        elif self._last_segmented_box is not None:
            self._left_canvas.set_box(*self._last_segmented_box)
            self._update_box_info()
        else:
            self._left_canvas.clear_box()
            self._box_info_label.setText("框: 未选择")

        # 恢复该图片的分割结果（如果有缓存）
        cached_result = self._result_cache.get(idx)
        if cached_result is not None:
            self._right_canvas.set_image(cached_result)
        else:
            self._right_canvas.set_image(QPixmap())
            # 尝试从磁盘加载已有的分割结果（上次打开项目时保存的）
            if 0 <= idx < len(self._images):
                stem = Path(self._images[idx]).stem
                self._api.fetch_image(
                    f"/api/projects/{self._project}/sam/box-result/{stem}_contour.png",
                    tag="box_right_image",
                )

        # 加载原图
        self._load_current_image()

    def _load_current_image(self):
        if self._current_idx < 0 or not self._project:
            return
        fname = self._images[self._current_idx]
        self._api.fetch_image(
            f"/api/projects/{self._project}/images/view/{fname}",
            tag="box_left_image",
        )
        self._status_label.setText(f"加载中... {fname}")

    # ------------------------------------------------------------------
    # 框选回调
    # ------------------------------------------------------------------

    def _on_box_changed(self, x1: int, y1: int, x2: int, y2: int):
        """用户画完框后立即缓存。"""
        if self._current_idx >= 0:
            self._box_cache[self._current_idx] = (x1, y1, x2, y2)
        self._update_box_info()

    def _update_box_info(self):
        x1, y1, x2, y2 = self._left_canvas.current_box()
        if x2 - x1 >= 5 and y2 - y1 >= 5:
            self._box_info_label.setText(
                f"框: ({x1},{y1}) → ({x2},{y2})  {x2-x1}×{y2-y1}"
            )
        else:
            self._box_info_label.setText("框: 未选择")

    # ------------------------------------------------------------------
    # 操作
    # ------------------------------------------------------------------

    def _on_toggle_model(self):
        """Toggle between load and release."""
        if self._model_loaded:
            self._on_release_model()
        else:
            self._on_load_model()

    def _on_load_model(self):
        if not self._project:
            QMessageBox.warning(self, "提示", "请先选择项目。")
            return
        self._load_model_btn.setEnabled(False)
        self._load_model_btn.setText("⏳ 正在加载...")
        self._log_msg("正在启动 sam3 容器并加载模型到 GPU...")
        self._status_label.setText("正在加载 SAM 模型...")
        self._api.post(
            f"/api/projects/{self._project}/sam/warmup",
            tag="warmup",
        )

    def _on_release_model(self):
        if not self._project:
            return
        self._load_model_btn.setEnabled(False)
        self._load_model_btn.setText("⏳ 正在释放...")
        self._log_msg("正在释放 SAM 模型显存...")
        self._status_label.setText("正在释放 SAM 模型...")
        self._api.post(
            f"/api/projects/{self._project}/sam/server-release",
            tag="box_release",
        )

    def _on_segment(self):
        if not self._project or self._current_idx < 0:
            QMessageBox.warning(self, "提示", "请先选择项目和图片。")
            return
        x1, y1, x2, y2 = self._left_canvas.current_box()
        if x2 - x1 < 5 or y2 - y1 < 5:
            QMessageBox.warning(self, "提示", "请先在原图上拖拽一个有效的矩形框。")
            return
        fname = self._images[self._current_idx]
        self._segment_btn.setEnabled(False)
        self._segment_btn.setText("⏳ 分割中...")
        self._log_msg(f"分割: {fname} box=({x1},{y1},{x2},{y2})")
        self._status_label.setText(f"分割中... {fname}")
        self._api.post(
            f"/api/projects/{self._project}/sam/box-segment",
            {
                "filename": fname,
                "x1": str(x1), "y1": str(y1),
                "x2": str(x2), "y2": str(y2),
            },
            tag="box_segment",
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
        if tag == "box_list_projects":
            self._refresh_proj_btn.setEnabled(True)
            if error:
                self._status_label.setText(f"加载项目失败: {error}")
                return
            projects = payload.get("projects", [])
            previous = self._proj_combo.currentText()
            selected = fill_project_combo(self._proj_combo, projects)
            # fill 期间屏蔽了信号：若全局选中项与下拉框现值不同，手动触发加载
            if selected != previous:
                self._on_project_changed(selected)

        # ── 图片列表 ──
        elif tag == "box_list_images":
            if error:
                self._status_label.setText(f"加载失败: {error}")
            else:
                images = payload.get("images", [])
                self._images = images
                self._box_cache.clear()
                self._last_segmented_box = None
                total = len(images)
                self._page_spin.setMaximum(max(1, total))
                self._page_spin.setEnabled(total > 0)
                self._total_label.setText(f"/ {total} 张")
                if images:
                    self._goto_image(0)

        # ── 左图（原图）──
        elif tag == "box_left_image":
            image_data = payload.get("_image_data")
            if image_data:
                pix = QPixmap()
                pix.loadFromData(image_data)
                self._left_canvas.set_image(pix)
                fname = self._images[self._current_idx] if self._current_idx >= 0 else ""
                self._status_label.setText(
                    f"✓ {self._current_idx + 1}/{len(self._images)}  {fname}"
                )
                # 框已在 _goto_image 中恢复，这里只需更新 segment btn
                self._segment_btn.setEnabled(self._model_loaded)

        # ── 模型预热 ──
        elif tag == "warmup":
            if error:
                self._load_model_btn.setEnabled(True)
                self._load_model_btn.setText("🚀 加载 SAM 模型")
                self._log_msg(f"✗ 模型加载失败: {error}")
                self._status_label.setText("模型加载失败")
                QMessageBox.warning(self, "SAM 模型加载失败", str(error))
            else:
                self._load_model_btn.setEnabled(True)
                self._load_model_btn.setText("🗑 释放 SAM 模型")
                self._model_loaded = True
                self._segment_btn.setEnabled(True)
                self._log_msg("✓ SAM 模型已加载到 GPU")
                self._status_label.setText("SAM 就绪，请在原图上画框")

        elif tag == "box_release":
            self._load_model_btn.setEnabled(True)
            if error:
                self._log_msg(f"✗ 释放模型失败: {error}")
                self._status_label.setText("释放失败")
            else:
                self._load_model_btn.setText("🚀 加载 SAM 模型")
                self._model_loaded = False
                self._segment_btn.setEnabled(False)
                self._log_msg("✓ SAM 模型已释放，显存已清空")
                self._status_label.setText("模型已释放")

        # ── 分割结果 ──
        elif tag == "box_segment":
            self._segment_btn.setEnabled(True)
            self._segment_btn.setText("✂ 分割")
            if error:
                self._log_msg(f"✗ 分割失败: {error}")
                self._status_label.setText("分割失败")
                QMessageBox.warning(self, "分割失败", str(error))
            else:
                resp_data = payload.get("data", {})
                overlay_path = resp_data.get("overlay_path", "")
                contour_path = resp_data.get("contour_path", "")
                box = resp_data.get("box", [])
                score = resp_data.get("score")
                self._log_msg(
                    f"✓ 分割完成 → {Path(overlay_path).name}"
                    if overlay_path else "✓ 分割完成"
                )
                if score is not None:
                    self._status_label.setText(f"✓ 分割完成 (置信度: {score:.2f})")
                else:
                    self._status_label.setText("✓ 分割完成")
                # 缓存当前框并设为后续默认
                cur_box = self._left_canvas.current_box()
                if self._current_idx >= 0:
                    self._box_cache[self._current_idx] = cur_box
                self._last_segmented_box = cur_box
                # 加载结果图 — 优先显示轮廓图（更清晰）
                if contour_path:
                    result_filename = Path(contour_path).name
                    self._api.fetch_image(
                        f"/api/projects/{self._project}/sam/box-result/{result_filename}",
                        tag="box_right_image",
                    )

        # ── 右图（分割结果）──
        elif tag == "box_right_image":
            image_data = payload.get("_image_data")
            if image_data:
                pix = QPixmap()
                pix.loadFromData(image_data)
                self._right_canvas.set_image(pix)
                # 缓存结果图，切换图片后再切回来时保留
                if self._current_idx >= 0:
                    self._result_cache[self._current_idx] = pix

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def _log_msg(self, text: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self._log.append(f"[{ts}] {text}")


# ---------------------------------------------------------------------------
# 点选分割面板
# ---------------------------------------------------------------------------

class PointSelectionPanel(QWidget):
    """点选分割 — 左键正点(绿) + 右键负点(红)，多点选择后手动分割。"""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._api = SamApiClient.instance()
        self._api.finished.connect(self._on_response)
        self._state = ProjectState.instance()
        self._state.current_changed.connect(self._on_shared_project_changed)

        # 状态
        self._project: str = ""
        self._images: list[str] = []
        self._current_idx: int = -1
        self._model_loaded: bool = False
        self._point_cache: dict[int, list[tuple[int, int, int]]] = {}
        self._result_cache: dict[int, QPixmap] = {}
        self._batch_queue: list[int] = []
        self._batch_total: int = 0
        self._batch_done: int = 0
        self._batch_mode: bool = False
        self._pending_points_idx: int = -1  # 等待图片加载完成后恢复标点的索引
        self._disk_loaded: set[int] = set()  # 已从磁盘加载过标点的索引
        # 区分手动/自动（继承）标注
        self._is_manual: dict[int, bool] = {}  # idx -> 该图点是否为手动标注
        self._last_manual_points: list[list[int]] = []  # 最近一次手动标注/修改的点（继承源头）

        # ── 布局 ──
        layout = QVBoxLayout(self)

        # 项目行
        proj_row = QHBoxLayout()
        proj_row.addWidget(QLabel("项目:"))
        self._proj_combo = QComboBox()
        self._proj_combo.setMinimumWidth(160)
        self._proj_combo.setObjectName("projectCombo")
        self._proj_combo.currentTextChanged.connect(self._on_project_changed)
        proj_row.addWidget(self._proj_combo)

        self._refresh_proj_btn = QPushButton("🔄")
        self._refresh_proj_btn.setObjectName("actionBtn")
        self._refresh_proj_btn.setToolTip("刷新项目列表")
        self._refresh_proj_btn.clicked.connect(self._load_projects)
        proj_row.addWidget(self._refresh_proj_btn)

        proj_row.addWidget(QLabel("  第"))
        self._page_spin = QSpinBox()
        self._page_spin.setMinimum(1)
        self._page_spin.setFixedWidth(60)
        self._page_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self._page_spin.setEnabled(False)
        self._page_spin.valueChanged.connect(self._on_page_spin)
        proj_row.addWidget(self._page_spin)
        self._total_label = QLabel("张")
        proj_row.addWidget(self._total_label)
        proj_row.addStretch()
        layout.addLayout(proj_row)

        # 控制行
        ctrl_row = QHBoxLayout()
        self._load_model_btn = QPushButton("🚀 加载 SAM 模型")
        self._load_model_btn.setObjectName("primaryBtn")
        self._load_model_btn.clicked.connect(self._on_toggle_model)
        ctrl_row.addWidget(self._load_model_btn)

        self._segment_btn = QPushButton("✂ 分割")
        self._segment_btn.setObjectName("primaryBtn")
        self._segment_btn.setEnabled(False)
        self._segment_btn.clicked.connect(self._on_segment)
        ctrl_row.addWidget(self._segment_btn)

        self._batch_btn = QPushButton("📦 批量分割")
        self._batch_btn.setObjectName("primaryBtn")
        self._batch_btn.setEnabled(False)
        self._batch_btn.setToolTip("对所有已标点的图片执行分割")
        self._batch_btn.clicked.connect(self._on_batch_segment)
        ctrl_row.addWidget(self._batch_btn)

        self._clear_pts_btn = QPushButton("✕ 清除点")
        self._clear_pts_btn.setObjectName("actionBtn")
        self._clear_pts_btn.clicked.connect(self._on_clear_points)
        ctrl_row.addWidget(self._clear_pts_btn)

        ctrl_row.addStretch()
        self._point_info_label = QLabel("左键正点 · 右键负点")
        self._point_info_label.setStyleSheet("color:#8b949e; font-size:12px;")
        ctrl_row.addWidget(self._point_info_label)
        self._status_label = QLabel("就绪")
        self._status_label.setStyleSheet("color:#8b949e; font-size:12px;")
        ctrl_row.addWidget(self._status_label)
        layout.addLayout(ctrl_row)

        # 双图区域
        splitter = QSplitter(Qt.Horizontal)

        # 左图 — 点选交互
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_header = QLabel("原图（左键正点 · 右键负点 · ↑↓ 键切换）")
        left_header.setStyleSheet("color:#8b949e; font-size:12px; padding:4px;")
        left_layout.addWidget(left_header)
        self._left_canvas = PointCanvas()
        self._left_canvas.points_changed.connect(self._on_points_changed)
        self._left_canvas.setFocusPolicy(Qt.StrongFocus)
        self._left_canvas.installEventFilter(self)
        left_layout.addWidget(self._left_canvas, 1)
        splitter.addWidget(left_container)

        # 右图 — 分割结果
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_header = QLabel("分割结果")
        right_header.setStyleSheet("color:#8b949e; font-size:12px; padding:4px;")
        right_layout.addWidget(right_header)
        self._right_canvas = ImageCanvas(interactive=False)
        right_layout.addWidget(self._right_canvas, 1)
        splitter.addWidget(right_container)

        splitter.setSizes([400, 400])
        layout.addWidget(splitter, 1)

        # 导航行
        nav_row = QHBoxLayout()
        nav_row.addStretch()
        self._prev_btn = QPushButton("◀ 上一张")
        self._prev_btn.setObjectName("actionBtn")
        self._prev_btn.clicked.connect(self._prev_image)
        nav_row.addWidget(self._prev_btn)
        self._next_btn = QPushButton("下一张 ▶")
        self._next_btn.setObjectName("actionBtn")
        self._next_btn.clicked.connect(self._next_image)
        nav_row.addWidget(self._next_btn)
        layout.addLayout(nav_row)

        # 日志
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setObjectName("logArea")
        self._log.setMaximumHeight(100)
        self._log.setFont(QFont("Consolas", 9))
        layout.addWidget(self._log)

    # ------------------------------------------------------------------
    # 显示 / 键盘导航
    # ------------------------------------------------------------------

    def showEvent(self, event):
        super().showEvent(event)
        # 进入页面时重新拉取列表，让「别处选中的项目」也能同步到下拉框
        self._load_projects()

    def eventFilter(self, obj, event):
        if obj is self._left_canvas and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key in (Qt.Key_Up, Qt.Key_Left):
                self._prev_image()
                return True
            elif key in (Qt.Key_Down, Qt.Key_Right):
                self._next_image()
                return True
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------

    def _load_projects(self):
        self._refresh_proj_btn.setEnabled(False)
        self._api.get("/api/projects", tag="point_list_projects")

    def _on_project_changed(self, project: str):
        self._state.set_current(project)
        if project == self._project:
            return
        self._project = project
        self._images.clear()
        self._point_cache.clear()
        self._result_cache.clear()
        self._disk_loaded.clear()
        self._is_manual.clear()
        self._last_manual_points = []
        self._current_idx = -1
        self._left_canvas.set_image(QPixmap())
        self._left_canvas.clear_points()
        self._right_canvas.set_image(QPixmap())
        self._page_spin.setEnabled(False)
        self._total_label.setText("张")
        self._point_info_label.setText("")
        self._status_label.setText("加载图片列表中...")
        if project:
            self._log_msg(f"切换项目: {project}")
            self._api.get(
                f"/api/projects/{project}/images/list",
                tag="point_list_images",
            )

    def _on_shared_project_changed(self, project: str):
        """其余页面选中了项目 → 同步本页下拉框。

        批量分割进行中不跟随：避免切页选择打断正在推进的批量队列。
        """
        if self._batch_mode:
            return
        if self._proj_combo.currentText() == project:
            return
        if project:
            idx = self._proj_combo.findText(project)
            if idx >= 0:
                self._proj_combo.setCurrentIndex(idx)
        else:
            self._proj_combo.setCurrentIndex(-1)

    def _on_page_spin(self, num: int):
        idx = num - 1
        if 0 <= idx < len(self._images):
            self._goto_image(idx)

    def _prev_image(self):
        self._goto_image(self._current_idx - 1)

    def _next_image(self):
        self._goto_image(self._current_idx + 1)

    def _goto_image(self, idx: int):
        if idx < 0 or idx >= len(self._images):
            return

        # 保存当前点集合到缓存 + 磁盘（仅手动标注的图才持久化）
        if self._current_idx >= 0 and self._project:
            if self._is_manual.get(self._current_idx, False):
                pts = self._left_canvas.points()
                self._point_cache[self._current_idx] = [list(p) for p in pts]
                if pts:
                    self._save_points_to_disk(self._current_idx)
                    self._last_manual_points = [list(p) for p in pts]
                else:
                    # 手动清空 → 磁盘同步清空，清除继承源
                    self._save_points_to_disk(self._current_idx)
                    self._last_manual_points = []
            # 自动（继承）的图未手动操作 → 不持久化，保持原状

        self._current_idx = idx
        self._page_spin.blockSignals(True)
        self._page_spin.setValue(idx + 1)
        self._page_spin.blockSignals(False)
        self._total_label.setText(f"/ {len(self._images)} 张")

        # 恢复标点：
        #   1) 该图有缓存点（手动标注或之前继承的）→ 原样恢复
        #   2) 没有 → 继承最近一次手动标注的点（自动点，标记非手动）
        #   3) 无手动源 → 留空；磁盘兜底在图片加载完成后执行
        self._pending_points_idx = idx
        cached = self._point_cache.get(idx)
        if cached is not None:
            self._left_canvas.set_points(cached)
            self._update_points_info()
        elif self._last_manual_points:
            inherited = [list(p) for p in self._last_manual_points]
            self._point_cache[idx] = inherited
            self._is_manual[idx] = False
            self._left_canvas.set_points(inherited)
            self._update_points_info()
        else:
            self._left_canvas.clear_points()
            self._point_info_label.setText("左键正点 · 右键负点")

        # 恢复右图结果
        if idx in self._result_cache:
            self._right_canvas.set_image(self._result_cache[idx])
        else:
            self._right_canvas.set_image(QPixmap())
            # 尝试从磁盘加载已有的分割结果（上次打开项目时保存的）
            if 0 <= idx < len(self._images):
                stem = Path(self._images[idx]).stem
                self._api.fetch_image(
                    f"/api/projects/{self._project}/sam/point-result/{stem}_contour.png",
                    tag="point_right_image",
                )

        self._load_current_image()

    def _load_current_image(self):
        if self._current_idx < 0 or self._current_idx >= len(self._images):
            return
        fname = self._images[self._current_idx]
        self._api.fetch_image(
            f"/api/projects/{self._project}/images/view/{fname}",
            tag="point_left_image",
        )

    # ------------------------------------------------------------------
    # 点选交互
    # ------------------------------------------------------------------

    def _on_points_changed(self):
        """用户在某张图上手动点击添加点 → 标记为手动标注，更新继承源并持久化。"""
        if self._current_idx >= 0:
            pts = self._left_canvas.points()
            self._point_cache[self._current_idx] = [list(p) for p in pts]
            self._is_manual[self._current_idx] = True
            self._last_manual_points = [list(p) for p in pts]
            self._save_points_to_disk(self._current_idx)
        self._update_points_info()

    def _update_points_info(self):
        pts = self._left_canvas.points()
        if not pts:
            self._point_info_label.setText("左键正点 · 右键负点")
            return
        pos_count = sum(1 for _, _, l in pts if l == 1)
        neg_count = sum(1 for _, _, l in pts if l == 0)
        parts = []
        if pos_count:
            parts.append(f"+{pos_count}")
        if neg_count:
            parts.append(f"−{neg_count}")
        self._point_info_label.setText("正点: " + ", ".join(parts) if parts else "左键正点 · 右键负点")
        # 模型已加载且有正点时，启用分割按钮
        self._segment_btn.setEnabled(self._model_loaded and pos_count > 0)
        # 批量按钮：模型已加载且有已标点图片
        self._update_batch_btn()

    def _update_batch_btn(self):
        """根据缓存的标点数量更新批量分割按钮（仅统计手动标注的图）。"""
        if self._model_loaded:
            count = sum(1 for idx, pts in self._point_cache.items()
                        if pts and self._is_manual.get(idx))
            self._batch_btn.setEnabled(count > 0)
        else:
            self._batch_btn.setEnabled(False)

    def _on_clear_points(self):
        """清除所有点（手动清空 → 该图为手动且无点，不再继承也不被继承）。"""
        self._left_canvas.clear_points()
        self._point_info_label.setText("左键正点 · 右键负点")
        self._segment_btn.setEnabled(False)
        # 同步清除磁盘上的点数据
        if self._project and self._current_idx >= 0:
            self._point_cache[self._current_idx] = []
            self._is_manual[self._current_idx] = True
            self._last_manual_points = []
            stem = Path(self._images[self._current_idx]).stem
            self._api.post(
                f"/api/projects/{self._project}/sam/points/save",
                form_data={"stem": stem, "points": "[]"},
                tag="point_save_points",
            )

    def _restore_points_after_image_load(self):
        """图片加载完成后从磁盘加载该图保存的标点（若有则覆盖继承的默认点）。"""
        idx = self._pending_points_idx
        self._pending_points_idx = -1
        if idx < 0 or idx >= len(self._images) or not self._project:
            return

        # 已从磁盘加载过 → 跳过
        if idx in self._disk_loaded:
            return
        self._disk_loaded.add(idx)

        stem = Path(self._images[idx]).stem
        self._api.get(
            f"/api/projects/{self._project}/sam/points/load/{stem}",
            tag="point_load_points",
        )

    def _save_points_to_disk(self, idx: int):
        """将指定索引图片的点保存到后端磁盘。"""
        if not self._project or idx < 0 or idx >= len(self._images):
            return
        stem = Path(self._images[idx]).stem
        pts = self._point_cache.get(idx, [])
        self._api.post(
            f"/api/projects/{self._project}/sam/points/save",
            form_data={"stem": stem, "points": json.dumps(pts)},
            tag="point_save_points",
        )

    def _do_segment(self, batch: bool = False):
        """执行多点分割请求。batch=True 时不修改按钮状态（由批量流程控制）。"""
        if not self._project or self._current_idx < 0:
            return

        pts = self._left_canvas.points()
        if not pts:
            self._log_msg("⚠ 请先添加至少一个正点")
            return

        fname = self._images[self._current_idx]
        if not batch:
            self._segment_btn.setEnabled(False)
            self._segment_btn.setText("⏳ 推理中...")
        self._status_label.setText("SAM 推理中...")
        points_json = json.dumps(pts)
        self._log_msg(f"多点分割: {fname} @ {points_json}")

        self._api.post(
            f"/api/projects/{self._project}/sam/point-segment",
            form_data={"filename": fname, "points": points_json},
            tag="point_segment",
        )

    def _on_segment(self):
        """按钮触发的分割 — 使用所有已选点。"""
        if self._batch_mode:
            self._log_msg("⚠ 批量分割进行中，请稍候")
            return
        if not self._model_loaded:
            self._log_msg("⚠ 请先加载 SAM 模型")
            return
        pts = self._left_canvas.points()
        if not pts:
            self._log_msg("⚠ 请先在图片上添加正点（左键）和负点（右键）")
            return
        if not any(l == 1 for _, _, l in pts):
            self._log_msg("⚠ 至少需要一个正点（左键）")
            return
        self._do_segment()

    def _on_batch_segment(self):
        """批量分割：对所有已标点的图片依次执行分割。"""
        if not self._model_loaded:
            self._log_msg("⚠ 请先加载 SAM 模型")
            return

        # 先保存当前图片的点到缓存（仅手动标注的图）
        if self._current_idx >= 0 and self._is_manual.get(self._current_idx, False):
            pts = self._left_canvas.points()
            self._point_cache[self._current_idx] = [list(p) for p in pts]
            if pts:
                self._save_points_to_disk(self._current_idx)
                self._last_manual_points = [list(p) for p in pts]

        # 收集所有手动标注且带正点的图片索引（自动继承的图不参与，需用户手动确认）
        queue = sorted(idx for idx, pts in self._point_cache.items()
                       if pts and self._is_manual.get(idx)
                       and any(l == 1 for _, _, l in pts))
        if not queue:
            self._log_msg(f"⚠ 没有手动标注的图片（自动继承的点需手动点击确认后才计入）")
            return

        self._batch_queue = queue
        self._batch_total = len(queue)
        self._batch_done = 0
        self._batch_mode = True
        self._load_model_btn.setEnabled(False)
        self._segment_btn.setEnabled(False)
        self._batch_btn.setEnabled(False)
        self._batch_btn.setText("⏳ 批量中...")
        self._page_spin.setEnabled(False)
        self._prev_btn.setEnabled(False)
        self._next_btn.setEnabled(False)
        self._log_msg(f"📦 开始批量分割 {self._batch_total} 张图片 (idx={queue})")
        self._process_next_batch()

    def _process_next_batch(self):
        """处理队列中的下一张图片。"""
        if not self._batch_queue:
            # 全部完成
            self._batch_mode = False
            self._batch_btn.setText("📦 批量分割")
            self._update_batch_btn()
            self._load_model_btn.setEnabled(True)
            self._segment_btn.setEnabled(self._model_loaded)
            self._page_spin.setEnabled(True)
            self._prev_btn.setEnabled(True)
            self._next_btn.setEnabled(True)
            self._status_label.setText(f"✓ 批量完成 ({self._batch_total} 张)")
            self._log_msg(f"✓ 批量分割完成 ({self._batch_total} 张)")
            return

        idx = self._batch_queue.pop(0)
        self._batch_done += 1
        self._status_label.setText(f"批量分割中... {self._batch_done}/{self._batch_total}")
        self._log_msg(f"📦 [{self._batch_done}/{self._batch_total}] 准备分割 idx={idx}")

        # 批量模式：直接切换索引并恢复标点，不等图片加载
        pts = self._point_cache.get(idx, [])
        if not pts:
            self._log_msg(f"⚠ idx={idx} 缓存中无标点，跳过")
            self._process_next_batch()
            return

        # 保存当前图片的点（仅手动标注的图）
        if self._current_idx >= 0 and self._is_manual.get(self._current_idx, False):
            cur_pts = self._left_canvas.points()
            self._point_cache[self._current_idx] = [list(p) for p in cur_pts]
            if cur_pts:
                self._last_manual_points = [list(p) for p in cur_pts]

        # 直接切换（不走 _goto_image，避免延迟恢复标点）
        self._current_idx = idx
        self._page_spin.blockSignals(True)
        self._page_spin.setValue(idx + 1)
        self._page_spin.blockSignals(False)
        self._total_label.setText(f"/ {len(self._images)} 张")
        self._left_canvas.clear_points()
        self._pending_points_idx = -1  # 批量模式不需要延迟恢复

        # 立即恢复标点并触发分割
        self._left_canvas.set_points(pts)
        self._update_points_info()

        # 异步加载图片（仅用于显示，不影响分割）
        self._load_current_image()

        # 触发分割
        self._do_segment(batch=True)

    def _on_toggle_model(self):
        """Toggle between load and release."""
        if self._model_loaded:
            self._on_release_model()
        else:
            self._on_load_model()

    def _on_load_model(self):
        self._load_model_btn.setEnabled(False)
        self._load_model_btn.setText("⏳ 正在加载...")
        self._status_label.setText("正在启动 sam3 容器并加载模型...")
        self._log_msg("正在启动 sam3 容器并加载模型到 GPU...")
        self._api.post(
            f"/api/projects/{self._project}/sam/warmup",
            form_data={},
            tag="point_warmup",
        )

    def _on_release_model(self):
        self._load_model_btn.setEnabled(False)
        self._load_model_btn.setText("⏳ 正在释放...")
        self._log_msg("正在释放 SAM 模型显存...")
        self._status_label.setText("正在释放 SAM 模型...")
        self._api.post(
            f"/api/projects/{self._project}/sam/server-release",
            form_data={},
            tag="point_release",
        )

    # ------------------------------------------------------------------
    # 响应分发
    # ------------------------------------------------------------------

    def _on_response(self, method: str, path: str, payload: dict):
        tag = payload.get("_tag", "")
        error = payload.get("_error")

        # ── 项目列表 ──
        if tag == "point_list_projects":
            self._refresh_proj_btn.setEnabled(True)
            if error:
                self._log_msg(f"✗ 获取项目失败: {error}")
                return
            projects = payload.get("projects", [])
            previous = self._proj_combo.currentText()
            selected = fill_project_combo(self._proj_combo, projects)
            self._log_msg(f"✓ 加载了 {len(projects)} 个项目")
            # fill 期间屏蔽了信号：若全局选中项与下拉框现值不同，手动触发加载
            if selected != previous:
                self._on_project_changed(selected)

        # ── 图片列表 ──
        elif tag == "point_list_images":
            if error:
                self._log_msg(f"✗ 获取图片失败: {error}")
                return
            self._images = payload.get("images", [])
            self._point_cache.clear()
            self._result_cache.clear()
            self._disk_loaded.clear()
            self._is_manual.clear()
            self._last_manual_points = []
            self._page_spin.setMaximum(max(1, len(self._images)))
            self._page_spin.setEnabled(len(self._images) > 0)
            self._log_msg(f"✓ 共 {len(self._images)} 张图片")
            if self._images:
                self._goto_image(0)
            else:
                self._left_canvas.set_image(QPixmap())
                self._right_canvas.set_image(QPixmap())
                self._total_label.setText("/ 0 张")

        # ── 从磁盘加载标点 ──
        elif tag == "point_load_points":
            pts = payload.get("points", [])
            exists = payload.get("exists", False)
            req_stem = Path(path).stem
            if self._current_idx >= 0:
                cur_stem = Path(self._images[self._current_idx]).stem
                if cur_stem == req_stem:
                    if exists and not self._is_manual.get(self._current_idx, False):
                        # 磁盘有保存（含空 = 用户清空过）且本会话尚未手动修改 → 覆盖继承点，标记为手动源
                        self._point_cache[self._current_idx] = [list(p) for p in pts]
                        self._is_manual[self._current_idx] = True
                        if pts:
                            self._last_manual_points = [list(p) for p in pts]
                        else:
                            self._last_manual_points = []
                        self._left_canvas.set_points(pts)
                        self._update_points_info()
                    # 已手动修改过（防磁盘响应覆盖刚点的点）或磁盘无保存 → 保留画布
                else:
                    # 当前已切换到其他图片，缓存到正确的索引（不更新手动源，避免覆盖更新的手动点）
                    for i, fname in enumerate(self._images):
                        if Path(fname).stem == req_stem:
                            if exists:
                                self._point_cache[i] = [list(p) for p in pts]
                                self._is_manual[i] = True
                            break

        # ── 左图 ──
        elif tag == "point_left_image":
            image_data = payload.get("_image_data")
            if image_data:
                pix = QPixmap()
                pix.loadFromData(image_data)
                self._left_canvas.set_image(pix)
                self._status_label.setText(
                    f"图片 {self._current_idx + 1}/{len(self._images)}"
                )
                # 图片加载完成后恢复该图的标点（缓存或磁盘）
                self._restore_points_after_image_load()
                if self._model_loaded and not self._batch_mode:
                    self._segment_btn.setEnabled(True)

        # ── 模型加载 ──
        elif tag == "point_warmup":
            self._load_model_btn.setEnabled(True)
            if error:
                self._load_model_btn.setText("🚀 加载 SAM 模型")
                self._log_msg(f"✗ 模型加载失败: {error}")
                self._status_label.setText("模型加载失败")
                QMessageBox.warning(self, "加载失败", str(error))
            else:
                self._load_model_btn.setText("🗑 释放 SAM 模型")
                self._model_loaded = True
                self._segment_btn.setEnabled(True)
                self._update_batch_btn()
                self._log_msg("✓ SAM 模型已加载到 GPU")
                self._status_label.setText("SAM 就绪，左键正点 · 右键负点，点击分割")

        elif tag == "point_release":
            self._load_model_btn.setEnabled(True)
            if error:
                self._log_msg(f"✗ 释放模型失败: {error}")
                self._status_label.setText("释放失败")
            else:
                self._load_model_btn.setText("🚀 加载 SAM 模型")
                self._model_loaded = False
                self._segment_btn.setEnabled(False)
                self._batch_btn.setEnabled(False)
                self._log_msg("✓ SAM 模型已释放，显存已清空")
                self._status_label.setText("模型已释放")

        # ── 分割结果 ──
        elif tag == "point_segment":
            is_batch = self._batch_mode
            if not is_batch:
                self._segment_btn.setEnabled(True)
                self._segment_btn.setText("✂ 分割")
            if error:
                self._log_msg(f"✗ 分割失败: {error}")
                if not is_batch:
                    self._status_label.setText("分割失败")
                    QMessageBox.warning(self, "分割失败", str(error))
            else:
                resp_data = payload.get("data", {})
                contour_path = resp_data.get("contour_path", "")
                score = resp_data.get("score")
                overlay_path = resp_data.get("overlay_path", "")
                points_used = resp_data.get("points", [])
                self._log_msg(
                    f"✓ 分割完成 → {Path(overlay_path).name}"
                    if overlay_path else "✓ 分割完成"
                )
                if score is not None:
                    self._status_label.setText(f"✓ 分割完成 (置信度: {score:.2f}, {len(points_used)}点)")
                else:
                    self._status_label.setText(f"✓ 分割完成 ({len(points_used)}点)")
                # 缓存当前点集合并保存到磁盘（分割即确认 → 标记为手动）
                if self._current_idx >= 0:
                    pts = self._left_canvas.points()
                    self._point_cache[self._current_idx] = [list(p) for p in pts]
                    self._is_manual[self._current_idx] = True
                    if pts:
                        self._save_points_to_disk(self._current_idx)
                        self._last_manual_points = [list(p) for p in pts]
                    else:
                        self._last_manual_points = []
                # 加载结果图 — 优先轮廓图
                if contour_path:
                    result_filename = Path(contour_path).name
                    self._api.fetch_image(
                        f"/api/projects/{self._project}/sam/point-result/{result_filename}",
                        tag="point_right_image",
                    )

            # 批量模式：继续下一张
            if is_batch:
                self._process_next_batch()

        # ── 右图（分割结果）──
        elif tag == "point_right_image":
            image_data = payload.get("_image_data")
            if image_data:
                pix = QPixmap()
                pix.loadFromData(image_data)
                self._right_canvas.set_image(pix)
                if self._current_idx >= 0:
                    self._result_cache[self._current_idx] = pix

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def _log_msg(self, text: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self._log.append(f"[{ts}] {text}")


# ---------------------------------------------------------------------------
# 语义分割面板（占位）
# ---------------------------------------------------------------------------

class SemanticPanel(QWidget):
    """语义分割 — 待开发。"""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        hint = QLabel("语义分割功能即将上线\n\n通过文本提示词批量分割图像")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet("color:#484f58; font-size:14px; padding:60px;")
        layout.addWidget(hint)


# ---------------------------------------------------------------------------
# SAM 主页面
# ---------------------------------------------------------------------------

class SAMPage(BasePage):
    """SAM 分割 — 框选 / 语义。"""

    def __init__(self):
        super().__init__("SAM 分割")

        nav = QHBoxLayout()
        self._btn_group = QButtonGroup(self)
        self._btn_group.setExclusive(True)

        self._box_btn = QPushButton("✂ 框选分割")
        self._box_btn.setCheckable(True)
        self._box_btn.setChecked(True)
        self._box_btn.setObjectName("actionBtn")
        self._box_btn.setMinimumHeight(32)
        self._btn_group.addButton(self._box_btn, 0)

        self._point_btn = QPushButton("🖱 点选分割")
        self._point_btn.setCheckable(True)
        self._point_btn.setObjectName("actionBtn")
        self._point_btn.setMinimumHeight(32)
        self._btn_group.addButton(self._point_btn, 1)

        self._sem_btn = QPushButton("📝 语义分割")
        self._sem_btn.setCheckable(True)
        self._sem_btn.setObjectName("actionBtn")
        self._sem_btn.setMinimumHeight(32)
        self._btn_group.addButton(self._sem_btn, 2)

        nav.addWidget(self._box_btn)
        nav.addWidget(self._point_btn)
        nav.addWidget(self._sem_btn)
        nav.addStretch()
        self.content_layout.addLayout(nav)

        self._box_panel = BoxSelectionPanel()
        self._point_panel = PointSelectionPanel()
        self._sem_panel = SemanticPanel()
        self._point_panel.setVisible(False)
        self._sem_panel.setVisible(False)

        self.content_layout.addWidget(self._box_panel, 1)
        self.content_layout.addWidget(self._point_panel, 1)
        self.content_layout.addWidget(self._sem_panel, 1)

        self._btn_group.idClicked.connect(self._switch_panel)

    def _switch_panel(self, idx: int):
        self._box_panel.setVisible(idx == 0)
        self._point_panel.setVisible(idx == 1)
        self._sem_panel.setVisible(idx == 2)
