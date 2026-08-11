"""SAM 分割页面 — 框选分割 + 语义分割."""

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
# 框选分割面板
# ---------------------------------------------------------------------------

class BoxSelectionPanel(QWidget):
    """框选分割 — 图片浏览、框选、SAM 推理。"""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._api = SamApiClient.instance()
        self._api.finished.connect(self._on_response)

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
        self._load_model_btn.clicked.connect(self._on_load_model)
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

        # 启动
        self._load_projects()

    # ------------------------------------------------------------------
    # 键盘切换图片
    # ------------------------------------------------------------------

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
        if project == self._project:
            return
        self._project = project
        self._images = []
        self._current_idx = -1
        self._box_cache.clear()
        self._last_segmented_box = None
        self._left_canvas.clear_box()
        self._page_spin.setEnabled(False)
        if project:
            self._status_label.setText("加载图片列表...")
            self._api.get(f"/api/projects/{project}/images/list", tag="box_list_images")

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

    def _on_load_model(self):
        if not self._project:
            QMessageBox.warning(self, "提示", "请先选择项目。")
            return
        self._load_model_btn.setEnabled(False)
        self._load_model_btn.setText("⏳ 加载中（含启动容器）...")
        self._log_msg("正在启动 sam3 容器并加载模型到 GPU...")
        self._status_label.setText("正在加载 SAM 模型...")
        self._api.post(
            f"/api/projects/{self._project}/sam/warmup",
            tag="warmup",
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
            cur = self._proj_combo.currentText()
            self._proj_combo.blockSignals(True)
            self._proj_combo.clear()
            for p in projects:
                self._proj_combo.addItem(p.get("project_name", ""))
            if cur:
                idx = self._proj_combo.findText(cur)
                if idx >= 0:
                    self._proj_combo.setCurrentIndex(idx)
            self._proj_combo.blockSignals(False)

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
                self._load_model_btn.setText("✅ SAM 已就绪")
                self._model_loaded = True
                self._segment_btn.setEnabled(True)
                self._log_msg("✓ SAM 模型已加载到 GPU")
                self._status_label.setText("SAM 就绪，请在原图上画框")

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
                box = resp_data.get("box", [])
                self._log_msg(
                    f"✓ 分割完成 → {Path(overlay_path).name}"
                    if overlay_path else "✓ 分割完成"
                )
                self._status_label.setText("✓ 分割完成")
                # 缓存当前框并设为后续默认
                cur_box = self._left_canvas.current_box()
                if self._current_idx >= 0:
                    self._box_cache[self._current_idx] = cur_box
                self._last_segmented_box = cur_box
                # 加载结果图
                if overlay_path:
                    result_filename = Path(overlay_path).name
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

        self._sem_btn = QPushButton("📝 语义分割")
        self._sem_btn.setCheckable(True)
        self._sem_btn.setObjectName("actionBtn")
        self._sem_btn.setMinimumHeight(32)
        self._btn_group.addButton(self._sem_btn, 1)

        nav.addWidget(self._box_btn)
        nav.addWidget(self._sem_btn)
        nav.addStretch()
        self.content_layout.addLayout(nav)

        self._box_panel = BoxSelectionPanel()
        self._sem_panel = SemanticPanel()
        self._sem_panel.setVisible(False)

        self.content_layout.addWidget(self._box_panel, 1)
        self.content_layout.addWidget(self._sem_panel, 1)

        self._btn_group.idClicked.connect(self._switch_panel)

    def _switch_panel(self, idx: int):
        self._box_panel.setVisible(idx == 0)
        self._sem_panel.setVisible(idx == 1)
