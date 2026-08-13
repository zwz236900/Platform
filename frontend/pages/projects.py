"""项目管理页面 — 完整的项目 CRUD，对接后端 API."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QScrollArea,
    QFrame,
    QDialog,
    QLineEdit,
    QMessageBox,
    QSizePolicy,
    QSplitter,
    QFileDialog,
    QProgressBar,
    QApplication,
)
from PySide6.QtCore import Qt, Signal, QUrl, QTimer, QFile, QIODevice
from PySide6.QtNetwork import (
    QNetworkAccessManager,
    QNetworkRequest,
    QNetworkReply,
    QHttpMultiPart,
    QHttpPart,
)
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QFont

from .base_page import BasePage
from state import ProjectState

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

API_BASE = "http://localhost:8000"

STATUS_LABELS = {
    "pending": "待处理",
    "running": "运行中",
    "completed": "已完成",
    "failed": "失败",
}

STATUS_ORDER = {"running": 0, "pending": 1, "failed": 2, "completed": 3}


# ---------------------------------------------------------------------------
# HTTP 客户端（基于 Qt Network）
# ---------------------------------------------------------------------------

class ApiClient(QWidget):
    """异步 HTTP 客户端，通过信号返回结果。"""

    finished = Signal(str, str, object)  # (method, url_path, response_data | error_str)
    _instance: "ApiClient | None" = None

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._mgr = QNetworkAccessManager(self)
        self.setVisible(False)

    @classmethod
    def instance(cls) -> "ApiClient":
        if cls._instance is None:
            cls._instance = ApiClient()
        return cls._instance

    def get(self, path: str, tag: str = ""):
        """GET 请求。tag 用于调用方识别响应。"""
        self._request("GET", path, tag=tag)

    def post(self, path: str, form_data: dict[str, str] | None = None, tag: str = ""):
        """POST 请求（multipart/form-data）。"""
        self._request("POST", path, form_data=form_data, tag=tag)

    def delete(self, path: str, tag: str = ""):
        """DELETE 请求。"""
        self._request("DELETE", path, tag=tag)

    def upload_files(self, project_name: str, file_paths: list[str], tag: str = ""):
        """上传文件到项目（multipart/form-data）。"""
        multi_part = QHttpMultiPart(QHttpMultiPart.FormDataType)

        for fp in file_paths:
            file_part = QHttpPart()
            filename = Path(fp).name
            file_part.setHeader(
                QNetworkRequest.ContentDispositionHeader,
                f'form-data; name="images"; filename="{filename}"',
            )
            file_part.setHeader(
                QNetworkRequest.ContentTypeHeader,
                "application/octet-stream",
            )
            qfile = QFile(fp)
            if not qfile.open(QIODevice.ReadOnly):
                continue
            qfile.setParent(multi_part)  # lifetime tied to multi_part
            file_part.setBodyDevice(qfile)
            multi_part.append(file_part)

        path = f"/api/projects/{project_name}/images"
        url = QUrl(API_BASE + path)
        req = QNetworkRequest(url)

        reply = self._mgr.post(req, multi_part)
        multi_part.setParent(reply)  # auto-delete with reply

        reply.setProperty("_tag", tag)
        reply.setProperty("_method", "POST")
        reply.setProperty("_path", path)
        reply.finished.connect(lambda: self._on_finished(reply))

    def _request(self, method: str, path: str,
                 form_data: dict[str, str] | None = None,
                 tag: str = ""):
        url = QUrl(API_BASE + path)
        req = QNetworkRequest(url)

        if form_data is not None:
            body = urlencode(form_data).encode("utf-8")
            req.setRawHeader(b"Content-Type", b"application/x-www-form-urlencoded")
            req.setRawHeader(b"Content-Length", str(len(body)).encode())
            if method == "POST":
                reply = self._mgr.post(req, body)
            else:
                reply = self._mgr.sendCustomRequest(req, method.encode(), body)
        else:
            if method == "GET":
                reply = self._mgr.get(req)
            elif method == "DELETE":
                reply = self._mgr.deleteResource(req)
            else:
                reply = self._mgr.post(req, b"")

        reply.setProperty("_tag", tag)
        reply.setProperty("_method", method)
        reply.setProperty("_path", path)
        reply.finished.connect(lambda: self._on_finished(reply))

    def _on_finished(self, reply: QNetworkReply):
        method = reply.property("_method")
        path = reply.property("_path")
        tag = reply.property("_tag")
        try:
            if reply.error() != QNetworkReply.NoError:
                err_text = bytes(reply.readAll()).decode("utf-8", errors="replace")
                # 尝试解析 JSON 错误
                try:
                    err_obj = json.loads(err_text)
                    err_msg = err_obj.get("detail", err_text)
                except Exception:
                    err_msg = err_text or reply.errorString()
                self.finished.emit(method, path, {"_error": err_msg, "_tag": tag})
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
# 新建项目对话框（支持拖拽上传图片 / 视频 / 压缩包）
# ---------------------------------------------------------------------------

# 允许的文件扩展名
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".pgm", ".ppm"}
VIDEO_EXTS  = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv", ".flv"}
ARCHIVE_EXTS = {".zip", ".tar", ".gz", ".7z", ".rar"}
ALLOWED_EXTS = IMAGE_EXTS | VIDEO_EXTS | ARCHIVE_EXTS

# 文件大小上限（单文件 4 GB）
MAX_FILE_SIZE = 4 * 1024 * 1024 * 1024


def _format_size(size_bytes: int) -> str:
    """人类可读的文件大小。"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def _file_icon(ext: str) -> str:
    if ext in IMAGE_EXTS:   return "🖼"
    if ext in VIDEO_EXTS:   return "🎬"
    if ext in ARCHIVE_EXTS: return "📦"
    return "📄"


class DropZone(QFrame):
    """拖拽 / 点击选择文件区域。"""

    files_added = Signal(list)  # list[str] — 文件路径列表

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setObjectName("dropZone")
        self.setMinimumHeight(120)
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        self._icon = QLabel("📁")
        self._icon.setAlignment(Qt.AlignCenter)
        self._icon.setStyleSheet("font-size:32px;")
        layout.addWidget(self._icon)

        self._hint = QLabel("拖拽图片 / 视频 / 压缩包到此处，或点击选择文件")
        self._hint.setAlignment(Qt.AlignCenter)
        self._hint.setStyleSheet("color:#8b949e; font-size:12px;")
        layout.addWidget(self._hint)

        self._sub = QLabel("支持 jpg, png, tif, mp4, avi, mov, zip 等格式")
        self._sub.setAlignment(Qt.AlignCenter)
        self._sub.setStyleSheet("color:#484f58; font-size:11px;")
        layout.addWidget(self._sub)

    # -- stylesheet fallback --
    def _apply_drop_style(self, active: bool):
        if active:
            self.setStyleSheet(
                "#dropZone { border:2px dashed #58a6ff; border-radius:8px;"
                " background-color:#1c2533; }"
            )
        else:
            self.setStyleSheet("")

    def mousePressEvent(self, event):
        self._open_file_dialog()
        super().mousePressEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._apply_drop_style(True)

    def dragLeaveEvent(self, event):
        self._apply_drop_style(False)

    def dropEvent(self, event: QDropEvent):
        self._apply_drop_style(False)
        urls = event.mimeData().urls()
        paths = []
        for url in urls:
            p = url.toLocalFile()
            if p and os.path.isfile(p):
                paths.append(p)
        if paths:
            self.files_added.emit(paths)

    def _open_file_dialog(self):
        filters = (
            "所有支持格式 (*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.pgm *.ppm"
            " *.mp4 *.avi *.mov *.mkv *.webm *.wmv *.flv *.zip *.tar *.gz *.7z *.rar);;"
            "图片 (*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.pgm *.ppm);;"
            "视频 (*.mp4 *.avi *.mov *.mkv *.webm *.wmv *.flv);;"
            "压缩包 (*.zip *.tar *.gz *.7z *.rar);;"
            "所有文件 (*)"
        )
        files, _ = QFileDialog.getOpenFileNames(self, "选择文件", "", filters)
        if files:
            self.files_added.emit(files)


class CreateProjectDialog(QDialog):
    """创建新项目的对话框 — 填写名称 + 可选上传文件。"""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("新建项目")
        self.setMinimumSize(520, 500)
        self.setModal(True)
        self._files: list[str] = []     # 选中的文件路径列表

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 20, 24, 24)

        # 标题
        title = QLabel("创建新项目")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        # 项目名称
        name_layout = QVBoxLayout()
        label = QLabel("项目名称")
        label.setStyleSheet("font-weight:600; font-size:13px; color:#c9d1d9;")
        name_layout.addWidget(label)
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("例如：校园重建、展厅扫描")
        self.name_input.setMinimumHeight(36)
        self.name_input.returnPressed.connect(self._on_create)
        name_layout.addWidget(self.name_input)
        layout.addLayout(name_layout)

        # 拖拽区
        upload_label = QLabel("上传文件（可选，创建后也可添加）")
        upload_label.setStyleSheet("font-weight:600; font-size:13px; color:#c9d1d9;")
        layout.addWidget(upload_label)

        self._drop_zone = DropZone()
        self._drop_zone.files_added.connect(self._add_files)
        layout.addWidget(self._drop_zone)

        # 文件列表
        self._file_list_label = QLabel("")
        self._file_list_label.setStyleSheet("color:#8b949e; font-size:11px;")
        self._file_list_label.setVisible(False)
        layout.addWidget(self._file_list_label)

        self._file_list = QVBoxLayout()
        self._file_list.setSpacing(2)
        layout.addLayout(self._file_list)

        layout.addStretch()

        # 按钮行
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("取消")
        cancel_btn.setObjectName("actionBtn")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        self.create_btn = QPushButton("创建项目")
        self.create_btn.setObjectName("primaryBtn")
        self.create_btn.clicked.connect(self._on_create)
        btn_row.addWidget(self.create_btn)
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------
    # 文件管理
    # ------------------------------------------------------------------

    def _add_files(self, paths: list[str]):
        added = 0
        skipped_size = 0
        skipped_dup = 0
        for p in paths:
            if p in self._files:
                skipped_dup += 1
                continue
            try:
                size = os.path.getsize(p)
            except OSError:
                continue
            if size > MAX_FILE_SIZE:
                skipped_size += 1
                continue
            ext = Path(p).suffix.lower()
            if ext not in ALLOWED_EXTS:
                continue
            self._files.append(p)
            added += 1
            self._add_file_row(p, ext, size)

        if skipped_size:
            QMessageBox.information(
                self, "提示",
                f"跳过了 {skipped_size} 个超过 4GB 的大文件。"
            )

        self._update_summary()

    def _add_file_row(self, path: str, ext: str, size: int):
        row = QHBoxLayout()
        row.setContentsMargins(0, 2, 0, 2)

        icon = QLabel(f"{_file_icon(ext)} {Path(path).name}")
        icon.setStyleSheet("color:#c9d1d9; font-size:12px;")
        row.addWidget(icon, 1)

        size_label = QLabel(_format_size(size))
        size_label.setStyleSheet("color:#8b949e; font-size:11px;")
        row.addWidget(size_label)

        remove_btn = QPushButton("✕")
        remove_btn.setFixedSize(20, 20)
        remove_btn.setCursor(Qt.PointingHandCursor)
        remove_btn.setStyleSheet(
            "QPushButton { border:none; color:#8b949e; font-size:12px; }"
            "QPushButton:hover { color:#da3633; }"
        )
        remove_btn.clicked.connect(lambda: self._remove_file(path))
        row.addWidget(remove_btn)

        self._file_list.addLayout(row)

    def _remove_file(self, path: str):
        if path in self._files:
            self._files.remove(path)
        # 重建文件列表 UI
        self._rebuild_file_list()
        self._update_summary()

    def _rebuild_file_list(self):
        """清除并重建文件列表 UI。"""
        while self._file_list.count():
            item = self._file_list.takeAt(0)
            if item.layout():
                # 递归清理子 widget
                self._clear_layout(item.layout())
            if item.widget():
                item.widget().deleteLater()
        for fp in self._files:
            ext = Path(fp).suffix.lower()
            try:
                size = os.path.getsize(fp)
            except OSError:
                size = 0
            self._add_file_row(fp, ext, size)

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            if item.layout():
                self._clear_layout(item.layout())

    def _update_summary(self):
        n = len(self._files)
        if n == 0:
            self._file_list_label.setVisible(False)
        else:
            total_size = sum(
                os.path.getsize(fp) for fp in self._files
                if os.path.isfile(fp)
            )
            images = sum(1 for fp in self._files
                        if Path(fp).suffix.lower() in IMAGE_EXTS)
            videos = sum(1 for fp in self._files
                        if Path(fp).suffix.lower() in VIDEO_EXTS)
            archives = sum(1 for fp in self._files
                          if Path(fp).suffix.lower() in ARCHIVE_EXTS)

            parts = [f"共 {n} 个文件 ({_format_size(total_size)})"]
            if images:   parts.append(f"{images} 图片")
            if videos:   parts.append(f"{videos} 视频")
            if archives: parts.append(f"{archives} 压缩包")
            self._file_list_label.setText(" · ".join(parts))
            self._file_list_label.setVisible(True)

    # ------------------------------------------------------------------
    # 提交
    # ------------------------------------------------------------------

    def _on_create(self):
        name = self.name_input.text().strip()
        if not name:
            self.name_input.setFocus()
            self.name_input.setStyleSheet(
                "border:1px solid #da3633; border-radius:6px; padding:6px 10px;"
            )
            return
        self.accept()

    def project_name(self) -> str:
        return self.name_input.text().strip()

    def files(self) -> list[str]:
        return list(self._files)


# ---------------------------------------------------------------------------
# 项目卡片
# ---------------------------------------------------------------------------

class ProjectCard(QFrame):
    """项目列表卡片，可点击选中。"""

    clicked = Signal(str)  # project_name

    def __init__(self, project_name: str, name: str, status: str,
                 num_images: int, created_at: str = "",
                 elapsed_seconds: float | None = None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._project_name = project_name
        self._status = status

        self.setObjectName("projectCard")
        self.setMinimumHeight(80)
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 12, 20, 12)

        # 左侧信息
        info = QVBoxLayout()
        title = QLabel(name)
        title.setObjectName("cardTitle")
        info.addWidget(title)

        # 格式化创建时间
        time_str = ""
        if created_at:
            try:
                dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                time_str = dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, TypeError):
                time_str = created_at[:16] if len(created_at) >= 16 else created_at

        elapsed_str = ""
        if elapsed_seconds is not None:
            s = float(elapsed_seconds)
            if s < 60:
                elapsed_str = f"  ·  {s:.0f}s"
            elif s < 3600:
                elapsed_str = f"  ·  {s / 60:.1f}min"

        detail_text = f"{num_images} 张图片"
        if time_str:
            detail_text += f"  ·  {time_str}"
        if elapsed_str:
            detail_text += elapsed_str

        detail = QLabel(detail_text)
        detail.setObjectName("cardDetail")
        info.addWidget(detail)
        layout.addLayout(info, 1)

        # 右侧状态标签
        badge = QLabel(STATUS_LABELS.get(status, status))
        badge.setObjectName(f"badge_{status}")
        badge.setAlignment(Qt.AlignCenter)
        badge.setFixedWidth(72)
        badge.setFixedHeight(28)
        layout.addWidget(badge)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.clicked.emit(self._project_name)

    @property
    def project_name(self) -> str:
        return self._project_name

    @property
    def status(self) -> str:
        return self._status


# ---------------------------------------------------------------------------
# 项目详情面板
# ---------------------------------------------------------------------------

class DetailRow(QWidget):
    """详情面板中的键值行。"""

    def __init__(self, label: str, value: str = "",
                 parent: QWidget | None = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)

        lbl = QLabel(label)
        lbl.setStyleSheet("color:#8b949e; font-size:12px; font-weight:600; min-width:80px;")
        layout.addWidget(lbl)

        self._value_label = QLabel(value)
        self._value_label.setStyleSheet("color:#e1e4e8; font-size:13px;")
        self._value_label.setWordWrap(True)
        layout.addWidget(self._value_label, 1)

    def set_value(self, v: str):
        self._value_label.setText(v)


class ProjectDetailPanel(QFrame):
    """项目详情面板 — 右侧展示区域。"""

    delete_requested = Signal(str)   # project_name
    refresh_requested = Signal(str)  # project_name

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("projectCard")  # 复用卡片样式
        self._project_name = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        # --- 头部 ---
        header = QHBoxLayout()
        self._header_label = QLabel("选择一个项目查看详情")
        self._header_label.setStyleSheet(
            "font-size:18px; font-weight:700; color:#f0f6fc;"
        )
        header.addWidget(self._header_label, 1)

        self._badge = QLabel("")
        self._badge.setAlignment(Qt.AlignCenter)
        self._badge.setFixedWidth(80)
        self._badge.setFixedHeight(28)
        self._badge.setVisible(False)
        header.addWidget(self._badge)
        layout.addLayout(header)

        # --- 分割线 ---
        sep = QWidget()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background-color:#21262d;")
        layout.addWidget(sep)

        # --- 详情行 ---
        self._rows: dict[str, DetailRow] = {}
        for key, label in [
            ("project_name", "项目标识"),
            ("status", "状态"),
            ("num_images", "图片数量"),
            ("created_at", "创建时间"),
            ("started_at", "开始时间"),
            ("finished_at", "完成时间"),
            ("elapsed", "总耗时"),
            ("error_message", "错误信息"),
        ]:
            row = DetailRow(label)
            self._rows[key] = row
            layout.addWidget(row)

        layout.addStretch(1)

        # --- 配置摘要 ---
        self._config_label = QLabel("")
        self._config_label.setStyleSheet(
            "color:#8b949e; font-size:12px; padding-top:8px; border-top:1px solid #21262d;"
        )
        self._config_label.setWordWrap(True)
        self._config_label.setVisible(False)
        layout.addWidget(self._config_label)

        # --- 按钮 ---
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self._delete_btn = QPushButton("🗑 删除项目")
        self._delete_btn.setObjectName("actionBtn")
        self._delete_btn.setStyleSheet(
            "#actionBtn { color:#da3633; border-color:#da363333; }"
            "#actionBtn:hover { background-color:#da363322; }"
        )
        self._delete_btn.clicked.connect(self._on_delete)
        self._delete_btn.setVisible(False)
        btn_row.addWidget(self._delete_btn)

        self._refresh_btn = QPushButton("🔄 刷新详情")
        self._refresh_btn.setObjectName("actionBtn")
        self._refresh_btn.clicked.connect(self._on_refresh)
        self._refresh_btn.setVisible(False)
        btn_row.addWidget(self._refresh_btn)

        layout.addLayout(btn_row)

    def show_empty(self):
        """显示空状态。"""
        self._project_name = ""
        self._header_label.setText("选择一个项目查看详情")
        self._badge.setVisible(False)
        for row in self._rows.values():
            row.set_value("")
        self._config_label.setVisible(False)
        self._delete_btn.setVisible(False)
        self._refresh_btn.setVisible(False)

    def show_project(self, data: dict):
        """展示项目详情。"""
        self._project_name = data.get("project_name", "")

        name = data.get("name", self._project_name)
        status = data.get("status", "pending")
        num_images = data.get("num_images", 0)
        created_at = data.get("created_at", "")
        started_at = data.get("started_at")
        finished_at = data.get("finished_at")
        error_message = data.get("error_message")
        config = data.get("config", {})
        pipeline = data.get("pipeline", {})
        steps = data.get("steps", {})

        self._header_label.setText(name)

        # 状态标签
        status_text = STATUS_LABELS.get(status, status)
        self._badge.setText(status_text)
        self._badge.setObjectName(f"badge_{status}")
        self._badge.style().unpolish(self._badge)
        self._badge.style().polish(self._badge)
        self._badge.setVisible(True)

        # 格式化时间
        def fmt_time(s):
            if not s:
                return "-"
            try:
                dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                return str(s)[:19]

        # 填充详情
        self._rows["project_name"].set_value(self._project_name)
        self._rows["status"].set_value(status_text)
        self._rows["num_images"].set_value(str(num_images))
        self._rows["created_at"].set_value(fmt_time(created_at))
        self._rows["started_at"].set_value(fmt_time(started_at) if started_at else "-")
        self._rows["finished_at"].set_value(fmt_time(finished_at) if finished_at else "-")

        # 总耗时
        elapsed_seconds = pipeline.get("elapsed_seconds") if pipeline else None
        if elapsed_seconds is not None:
            s = float(elapsed_seconds)
            if s < 60:
                elapsed = f"{s:.0f} 秒"
            elif s < 3600:
                elapsed = f"{s / 60:.1f} 分钟"
            else:
                elapsed = f"{s / 3600:.1f} 小时"
            self._rows["elapsed"].set_value(elapsed)
        elif started_at and finished_at:
            self._rows["elapsed"].set_value("已结束（含等待）")
        else:
            self._rows["elapsed"].set_value("-")

        # 错误信息
        if error_message:
            self._rows["error_message"].set_value(error_message)
            self._rows["error_message"]._value_label.setStyleSheet(
                "color:#da3633; font-size:13px;"
            )
        else:
            self._rows["error_message"].set_value("（无）")
            self._rows["error_message"]._value_label.setStyleSheet(
                "color:#8b949e; font-size:13px;"
            )

        # 配置摘要
        if config:
            steps_list = config.get("steps", [])
            matcher = config.get("matcher", "")
            camera = config.get("camera_model", "")
            parts = []
            if steps_list:
                parts.append(f"步骤: {' → '.join(steps_list)}")
            if matcher:
                parts.append(f"匹配: {matcher}")
            if camera:
                parts.append(f"相机: {camera}")
            if parts:
                self._config_label.setText("  |  ".join(parts))
                self._config_label.setVisible(True)
            else:
                self._config_label.setVisible(False)
        else:
            self._config_label.setVisible(False)

        # 按钮
        is_running = (status == "running")
        self._delete_btn.setVisible(not is_running)
        self._refresh_btn.setVisible(True)

    def _on_delete(self):
        if self._project_name:
            self.delete_requested.emit(self._project_name)

    def _on_refresh(self):
        if self._project_name:
            self.refresh_requested.emit(self._project_name)

    @property
    def current_project(self) -> str:
        return self._project_name


# ---------------------------------------------------------------------------
# 项目管理页面
# ---------------------------------------------------------------------------

class ProjectsPage(BasePage):
    """项目管理主页面 — 左侧项目列表 + 右侧详情。"""

    def __init__(self):
        super().__init__("项目管理")

        self._api = ApiClient.instance()
        self._api.finished.connect(self._on_api_response)

        self._state = ProjectState.instance()
        self._state.current_changed.connect(self._on_shared_project_changed)

        self._projects: dict[str, dict] = {}          # 缓存列表数据
        self._selected_project: str = ""               # 当前选中
        self._detail_loaded: set[str] = set()          # 已加载详情的项目
        self._pending_uploads: list[str] = []           # 待上传文件（创建后）

        # --- 操作栏 ---
        toolbar = QHBoxLayout()

        self._status_label = QLabel("加载中...")
        self._status_label.setObjectName("cardDetail")
        toolbar.addWidget(self._status_label)

        toolbar.addStretch()

        self._refresh_btn = QPushButton("🔄 刷新列表")
        self._refresh_btn.setObjectName("actionBtn")
        self._refresh_btn.clicked.connect(self._refresh_list)
        toolbar.addWidget(self._refresh_btn)

        self._create_btn = QPushButton("＋ 新建项目")
        self._create_btn.setObjectName("primaryBtn")
        self._create_btn.clicked.connect(self._on_create_project)
        toolbar.addWidget(self._create_btn)

        self.content_layout.addLayout(toolbar)

        # --- 主体：分割器 ---
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)

        # 左侧 — 项目列表
        left_panel = QWidget()
        left_panel.setObjectName("cardScroll")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        list_label = QLabel("项目列表")
        list_label.setStyleSheet(
            "color:#8b949e; font-size:11px; font-weight:600; padding:4px 4px;"
            "text-transform:uppercase;"
        )
        left_layout.addWidget(list_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setObjectName("cardScroll")
        self._cards_widget = QWidget()
        self._cards_layout = QVBoxLayout(self._cards_widget)
        self._cards_layout.setSpacing(6)
        self._cards_layout.setContentsMargins(0, 0, 4, 0)
        self._cards_layout.addStretch()
        scroll.setWidget(self._cards_widget)
        left_layout.addWidget(scroll, 1)

        # 右侧 — 详情面板
        self._detail_panel = ProjectDetailPanel()
        self._detail_panel.delete_requested.connect(self._on_delete_project)
        self._detail_panel.refresh_requested.connect(self._load_project_detail)

        splitter.addWidget(left_panel)
        splitter.addWidget(self._detail_panel)
        splitter.setStretchFactor(0, 2)  # 列表占 2 份
        splitter.setStretchFactor(1, 3)  # 详情占 3 份

        self.content_layout.addWidget(splitter, 1)

        # --- 自动刷新定时器 ---
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._refresh_list)
        self._poll_timer.setInterval(5000)  # 5 秒轮询

        # --- 启动加载 ---
        self._refresh_list()
        self._poll_timer.start()

    # ------------------------------------------------------------------
    # 网络回调
    # ------------------------------------------------------------------

    def _on_api_response(self, method: str, path: str, payload: object):
        if not isinstance(payload, dict):
            return

        tag = payload.get("_tag", "")
        error = payload.get("_error", "")

        # ── 创建项目响应（包含错误处理）──
        if tag == "create_project":
            if error:
                QMessageBox.warning(self, "创建失败", str(error))
                self._reset_create_btn()
            else:
                project_name = payload.get("data", {}).get("project_name", "")
                # 如果有待上传文件，先上传再刷新
                if self._pending_uploads and project_name:
                    self._status_label.setText(f"项目已创建，正在上传 {len(self._pending_uploads)} 个文件...")
                    self._api.upload_files(
                        project_name, self._pending_uploads,
                        tag=f"upload_to:{project_name}",
                    )
                    self._pending_uploads = []
                else:
                    self._status_label.setText(f"✓ 项目 {project_name} 已创建")
                    self._refresh_list()
                    if project_name:
                        self._selected_project = project_name
                        self._load_project_detail(project_name)
                        self._state.set_current(project_name)
                    self._reset_create_btn()

        # ── 文件上传响应 ──
        elif tag.startswith("upload_to:"):
            project_name = tag.split(":", 1)[1]
            if error:
                QMessageBox.warning(
                    self, "上传失败",
                    f"项目已创建，但文件上传失败：{error}",
                )
            else:
                num_images = payload.get("data", {}).get("num_images", 0)
                self._status_label.setText(
                    f"✓ 项目 {project_name} 已创建，{num_images} 张图片已上传"
                )
            self._refresh_list()
            if project_name:
                self._selected_project = project_name
                self._load_project_detail(project_name)
                self._state.set_current(project_name)
            self._reset_create_btn()

        # ── 删除项目响应 ──
        elif tag == "delete_project":
            if error:
                QMessageBox.warning(self, "删除失败", str(error))
            else:
                self._selected_project = ""
                self._detail_panel.show_empty()
                self._status_label.setText("✓ 项目已删除")
                self._state.set_current("")
                self._refresh_list()

        # ── 列表响应 ──
        elif tag == "list":
            if error:
                self._status_label.setText(f"⚠ 无法连接后端: {error}")
                return
            self._on_list_response(payload)

        # ── 详情响应 ──
        elif tag == "load_detail":
            if error:
                self._detail_panel.show_empty()
                self._detail_panel._header_label.setText(f"加载详情失败: {error}")
                return
            self._detail_panel.show_project(payload)
            # 更新缓存
            pn = payload.get("project_name", "")
            if pn:
                self._projects[pn] = {
                    "project_name": pn,
                    "name": payload.get("name", ""),
                    "status": payload.get("status", "pending"),
                    "num_images": payload.get("num_images", 0),
                    "created_at": payload.get("created_at", ""),
                    "elapsed_seconds": (payload.get("pipeline") or {}).get("elapsed_seconds"),
                }
                self._detail_loaded.add(pn)

    # ------------------------------------------------------------------
    # 项目列表
    # ------------------------------------------------------------------

    def _on_list_response(self, payload: dict):
        projects: list[dict] = payload.get("projects", [])

        # 按状态排序：running > pending > failed > completed
        projects.sort(key=lambda p: STATUS_ORDER.get(p.get("status", ""), 99))

        self._projects = {p["project_name"]: p for p in projects}
        self._rebuild_cards(projects)

        # 更新统计
        total = len(projects)
        running = sum(1 for p in projects if p.get("status") == "running")
        completed = sum(1 for p in projects if p.get("status") == "completed")
        failed = sum(1 for p in projects if p.get("status") == "failed")

        parts = [f"共 {total} 个项目"]
        if running:
            parts.append(f"{running} 运行中")
        if completed:
            parts.append(f"{completed} 已完成")
        if failed:
            parts.append(f"{failed} 失败")
        self._status_label.setText("  ·  ".join(parts))

        # 如果当前选中的项目仍在列表中，刷新详情
        if self._selected_project and self._selected_project in self._projects:
            self._load_project_detail(self._selected_project)

    def _rebuild_cards(self, projects: list[dict]):
        """重建项目卡片列表。"""
        # 清除旧卡片
        while self._cards_layout.count() > 1:  # 保留最后的 stretch
            item = self._cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not projects:
            empty = QLabel("暂无项目\n\n点击「＋ 新建项目」按钮创建第一个项目")
            empty.setObjectName("hint")
            empty.setAlignment(Qt.AlignCenter)
            self._cards_layout.insertWidget(0, empty)
            self._detail_panel.show_empty()
            return

        # 创建卡片
        for p in projects:
            card = ProjectCard(
                project_name=p.get("project_name", ""),
                name=p.get("name", ""),
                status=p.get("status", "pending"),
                num_images=p.get("num_images", 0),
                created_at=p.get("created_at", ""),
                elapsed_seconds=p.get("elapsed_seconds"),
            )
            card.clicked.connect(self._on_select_project)
            # 高亮当前选中
            if p.get("project_name") == self._selected_project:
                card.setStyleSheet(
                    "#projectCard { border-color:#58a6ff; background-color:#1c2533; }"
                )
            self._cards_layout.insertWidget(self._cards_layout.count() - 1, card)

    # ------------------------------------------------------------------
    # 用户操作
    # ------------------------------------------------------------------

    def _refresh_list(self):
        self._api.get("/api/projects", tag="list")

    def _on_select_project(self, project_name: str):
        self._selected_project = project_name
        # 刷新高亮
        for i in range(self._cards_layout.count()):
            w = self._cards_layout.itemAt(i).widget()
            if isinstance(w, ProjectCard):
                if w.project_name == project_name:
                    w.setStyleSheet(
                        "#projectCard { border-color:#58a6ff; background-color:#1c2533; }"
                    )
                else:
                    w.setStyleSheet("")
        # 加载详情
        self._load_project_detail(project_name)
        # 广播到全局，其余页面同步选中
        self._state.set_current(project_name)

    def _on_shared_project_changed(self, project: str):
        """其余页面选中了项目 → 同步本页高亮与详情。"""
        if project == self._selected_project:
            return  # 本页自己触发的选择已在 _on_select_project 处理
        self._selected_project = project
        # 刷新卡片高亮
        for i in range(self._cards_layout.count()):
            w = self._cards_layout.itemAt(i).widget()
            if isinstance(w, ProjectCard):
                if w.project_name == project:
                    w.setStyleSheet(
                        "#projectCard { border-color:#58a6ff; background-color:#1c2533; }"
                    )
                else:
                    w.setStyleSheet("")
        if project:
            if self.isVisible():
                self._load_project_detail(project)
        else:
            self._detail_panel.show_empty()

    def _load_project_detail(self, project_name: str):
        self._detail_panel._header_label.setText("加载中...")
        self._api.get(f"/api/projects/{project_name}", tag="load_detail")

    def _on_create_project(self):
        dialog = CreateProjectDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return

        name = dialog.project_name()
        files = dialog.files()

        self._create_btn.setEnabled(False)
        self._create_btn.setText("创建中...")
        self._pending_uploads = files
        self._api.post("/api/projects", {"name": name}, tag="create_project")
        # 兜底：如果 10 秒后还没恢复就重置按钮
        QTimer.singleShot(10000, self._reset_create_btn)

    def _reset_create_btn(self):
        self._create_btn.setEnabled(True)
        self._create_btn.setText("＋ 新建项目")

    def _on_delete_project(self, project_name: str):
        # 二次确认
        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定要删除项目「{project_name}」及其所有数据吗？\n\n此操作不可恢复。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._status_label.setText(f"正在删除 {project_name}...")
        self._api.delete(f"/api/projects/{project_name}", tag="delete_project")

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def showEvent(self, event):
        """页面显示时刷新列表。"""
        super().showEvent(event)
        self._refresh_list()
        if not self._poll_timer.isActive():
            self._poll_timer.start()

    def hideEvent(self, event):
        """页面隐藏时停止轮询，节约资源。"""
        super().hideEvent(event)
        if self._poll_timer.isActive():
            self._poll_timer.stop()
