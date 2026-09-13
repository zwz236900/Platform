"""视频抽帧对话框（新建项目上传视频后弹出）。

不 import ``pages.projects``（防循环依赖）——HTTP 客户端由构造注入
（ProjectsPage 传入的 ApiClient 单例，靠 tag 前缀 ``vx_`` 区分响应），
API 根地址也由构造传入。流程：

  1. 打开时 ``GET /api/system/ffmpeg`` 与 ``GET .../videos/list``；
     ffmpeg 不可用则提示并禁用「开始抽帧」。
  2. 用户点「开始抽帧」→ 对每个视频 ``POST .../videos/extract``（Form），
     再以 ``QTimer`` 轮询 ``GET .../videos/extract/status``，直到 ok/failed。
  3. 每个视频成功后，用本对话框私有的 ``QNetworkAccessManager``
     GET ``.../input-frames/view/{前几张帧}`` 生成缩略图预览。
  4. 全部完成或用户中途关闭（关闭前发 ``.../videos/extract/stop``）后，
     调用方通过 :meth:`result_frame_count` 读取已抽帧总数。

exec() 是嵌套事件循环，QtNetwork 回包 / QTimer 在其中都能正常触发，
故无需任何线程。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QPixmap
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

MAX_PREVIEW_FRAMES = 6      # 每个视频成功后最多预览的帧数
THUMB_W, THUMB_H = 150, 96  # 缩略图显示尺寸


def _fmt_duration(s) -> str:
    try:
        s = float(s or 0)
    except (TypeError, ValueError):
        return "?"
    if s <= 0:
        return "?"
    if s < 60:
        return f"{s:.0f} 秒"
    return f"{int(s // 60)} 分 {int(s % 60)} 秒"


def _fmt_res(w, h) -> str:
    return f"{w}×{h}" if (w and h) else "?"


class VideoExtractDialog(QDialog):
    """新建项目中的视频抽帧对话框。"""

    def __init__(self, project_name: str, api, api_base: str,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._project_name = project_name
        self._api = api
        self._api_base = api_base

        self._videos: list[dict] = []       # GET .../videos/list 的结果
        self._queue: list[str] = []         # 待抽帧文件名（按序）
        self._idx = 0                       # 当前处理到的下标
        self._running = False
        self._frames_total = 0
        self._last_result = ""              # 汇总信息（给调用方留档）

        self._preview_mgr = QNetworkAccessManager(self)
        self._preview_mgr.finished.connect(self._on_preview_reply)
        self._thumb_labels: dict[str, QLabel] = {}  # filename -> 缩略图标签
        self._thumb_rows: list[tuple[QLabel, str]] = []

        self._api.finished.connect(self._on_api)

        self.setWindowTitle(f"视频抽帧 — {project_name}")
        self.setMinimumSize(560, 560)
        self.setModal(True)

        self._build_ui()

        # 打开即拉取 ffmpeg 状态与视频列表
        self._set_hint("正在读取视频信息...")
        self._ffmpeg_ok = False
        self._api.get("/api/system/ffmpeg", tag="vx_ffmpeg")
        self._api.get(f"/api/projects/{project_name}/videos/list", tag="vx_list")

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(12)

        title = QLabel("视频抽帧")
        title.setObjectName("pageTitle")
        root.addWidget(title)

        sub = QLabel(
            f"项目「{self._project_name}」中的视频会按帧抽取为重建输入图片。"
            "抽出的帧与普通照片一样参与后续重建。"
        )
        sub.setWordWrap(True)
        sub.setStyleSheet("color:#8b949e; font-size:12px;")
        root.addWidget(sub)

        # ── 视频列表 ──
        group = QGroupBox("已上传视频")
        group.setStyleSheet(
            "QGroupBox { color:#c9d1d9; font-size:12px; font-weight:600;"
            " border:1px solid #21262d; border-radius:6px; margin-top:8px;"
            " padding-top:6px; }"
            "QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 4px; }"
        )
        vbox = QVBoxLayout(group)
        self._videos_label = QLabel("读取中...")
        self._videos_label.setStyleSheet("color:#c9d1d9; font-size:12px;")
        self._videos_label.setWordWrap(True)
        vbox.addWidget(self._videos_label)
        root.addWidget(group)

        # ── 抽帧参数 ──
        form_group = QGroupBox("抽帧设置")
        form_group.setStyleSheet(group.styleSheet())
        form = QFormLayout(form_group)
        form.setLabelAlignment(Qt.AlignRight)

        self._mode_combo = QComboBox()
        self._mode_combo.addItem("按固定间隔（每秒 N 帧的倒数）", "interval")
        self._mode_combo.addItem("按目标总帧数（整段均匀取样）", "count")
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        form.addRow("抽样方式:", self._mode_combo)

        self._interval_spin = QDoubleSpinBox()
        self._interval_spin.setRange(0.1, 120.0)
        self._interval_spin.setValue(2.0)
        self._interval_spin.setDecimals(1)
        self._interval_spin.setSingleStep(0.5)
        self._interval_spin.setSuffix(" 秒")
        form.addRow("间隔:", self._interval_spin)

        self._count_spin = QSpinBox()
        self._count_spin.setRange(1, 5000)
        self._count_spin.setValue(300)
        self._count_spin.setSingleStep(50)
        self._count_spin.setSuffix(" 帧")
        form.addRow("目标帧数:", self._count_spin)

        self._width_spin = QSpinBox()
        self._width_spin.setRange(0, 8192)
        self._width_spin.setValue(1920)
        self._width_spin.setSingleStep(320)
        self._width_spin.setSpecialValueText("原始分辨率")
        form.addRow("最长边限制:", self._width_spin)

        self._quality_check = QCheckBox("启用质量过滤（丢弃模糊 / 近似重复帧）")
        self._quality_check.setChecked(True)
        form.addRow("", self._quality_check)

        root.addWidget(form_group)
        self._on_mode_changed(0)

        # ── 进度 ──
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        self._hint_label = QLabel("")
        self._hint_label.setStyleSheet("color:#8b949e; font-size:12px;")
        self._hint_label.setWordWrap(True)
        root.addWidget(self._hint_label)

        # ── 结果预览区 ──
        self._preview_wrap = QWidget()
        pv = QVBoxLayout(self._preview_wrap)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(6)
        self._preview_title = QLabel("")
        self._preview_title.setStyleSheet(
            "color:#8b949e; font-size:11px; font-weight:600;"
        )
        pv.addWidget(self._preview_title)
        self._preview_row = QHBoxLayout()
        self._preview_row.setSpacing(6)
        self._preview_row.addStretch()
        pv.addLayout(self._preview_row)
        self._preview_wrap.setVisible(False)
        root.addWidget(self._preview_wrap)

        root.addStretch(1)

        # ── 按钮 ──
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self._run_btn = QPushButton("▶ 开始抽帧")
        self._run_btn.setObjectName("primaryBtn")
        self._run_btn.setMinimumWidth(120)
        self._run_btn.clicked.connect(self._on_run)
        btn_row.insertWidget(0, self._run_btn)

        self._done_btn = QPushButton("完成")
        self._done_btn.setObjectName("actionBtn")
        self._done_btn.setMinimumWidth(100)
        self._done_btn.setEnabled(False)
        self._done_btn.clicked.connect(self.accept)
        btn_row.addWidget(self._done_btn)
        root.addLayout(btn_row)

        # 轮询
        self._poll = QTimer(self)
        self._poll.setInterval(1000)
        self._poll.timeout.connect(self._poll_status)

    # ------------------------------------------------------------------
    # 状态 / 提示
    # ------------------------------------------------------------------

    def _set_hint(self, text: str, color: str = "#8b949e"):
        self._hint_label.setStyleSheet(f"color:{color}; font-size:12px;")
        self._hint_label.setText(text)

    def _on_mode_changed(self, _idx: int):
        mode = self._mode_combo.currentData()
        self._interval_spin.setEnabled(mode == "interval")
        self._count_spin.setEnabled(mode == "count")

    # ------------------------------------------------------------------
    # 初始数据（ffmpeg 状态 / 视频列表）
    # ------------------------------------------------------------------

    def _on_api(self, method: str, path: str, payload: object):
        if not isinstance(payload, dict):
            return
        tag = payload.get("_tag", "")
        error = payload.get("_error", "")
        if tag == "vx_ffmpeg":
            self._ffmpeg_ok = bool(payload.get("available"))
            if not self._ffmpeg_ok:
                self._run_btn.setEnabled(False)
                self._done_btn.setEnabled(True)
                self._set_hint(
                    "后端缺少 FFmpeg，无法抽帧。请先在 WSL 宿主机执行：\n"
                    "sudo apt update && sudo apt install -y ffmpeg",
                    "#da3633",
                )
        elif tag == "vx_list":
            if error:
                self._set_hint(f"读取视频列表失败：{error}", "#da3633")
                return
            self._videos = list(payload.get("videos") or [])
            self._queue = [v.get("filename") for v in self._videos
                           if v.get("filename")]
            self._render_video_list()
        elif tag == "vx_start":
            self._on_start_response(error)
        elif tag == "vx_status":
            self._on_status(payload, error)
        elif tag == "vx_stop":
            # 停止响应：running 线程会自行把状态置 failed
            pass
        elif tag == "vx_frames":
            if not error:
                self._maybe_load_preview(payload)

    def _render_video_list(self):
        if not self._videos:
            self._videos_label.setText("（无视频）")
            return
        lines = []
        for v in self._videos:
            name = v.get("filename", "?")
            dur = _fmt_duration(v.get("duration"))
            res = _fmt_res(v.get("width"), v.get("height"))
            mark = "✓已抽帧" if v.get("frames_exist") else ""
            lines.append(f"🎬 {name}   ({dur} · {res}) {mark}")
        self._videos_label.setText("\n".join(lines))
        if not self._queue:
            self._run_btn.setEnabled(False)
            self._done_btn.setEnabled(True)
            self._set_hint("该项目下没有可抽帧的视频，可直接点击完成。", "#da3633")

    # ------------------------------------------------------------------
    # 运行主流程
    # ------------------------------------------------------------------

    def _on_run(self):
        if self._running:
            return
        if not self._ffmpeg_ok:
            QMessageBox.warning(self, "无法抽帧", "后端未安装 FFmpeg。")
            return
        if not self._queue:
            QMessageBox.warning(self, "无法抽帧", "没有可抽帧的视频。")
            return
        # 从头或从上次失败处重跑
        if self._idx >= len(self._queue):
            self._idx = 0
            self._frames_total = 0
            self._clear_previews()
        self._run_btn.setEnabled(False)
        self._done_btn.setEnabled(False)
        self._begin_extract()

    def _begin_extract(self):
        self._running = True
        name = self._queue[self._idx]
        mode = self._mode_combo.currentData()
        form = {
            "filename": name,
            "mode": mode,
            "interval": str(self._interval_spin.value()),
            "count": str(self._count_spin.value()),
            "max_width": str(self._width_spin.value()),
            "quality_filter": "1" if self._quality_check.isChecked() else "0",
        }
        self._set_hint(f"正在抽帧 {self._idx + 1}/{len(self._queue)}：{name}")
        self.progress.setRange(0, 0)
        self.progress.setVisible(True)
        self._api.post(
            f"/api/projects/{self._project_name}/videos/extract",
            form, tag="vx_start",
        )

    def _on_start_response(self, error):
        if not self._running:
            return
        if error:
            self._running = False
            self._poll.stop()
            self.progress.setVisible(False)
            self._run_btn.setEnabled(True)
            self._done_btn.setEnabled(True)
            self._set_hint(f"启动抽帧失败：{error}", "#da3633")
            return
        self._poll.start()

    def _poll_status(self):
        self._api.get(
            f"/api/projects/{self._project_name}/videos/extract/status",
            tag="vx_status",
        )

    def _on_status(self, payload: dict, error):
        if not self._running:
            return
        if error:
            self._running = False
            self._poll.stop()
            self.progress.setVisible(False)
            self._run_btn.setEnabled(True)
            self._done_btn.setEnabled(True)
            self._set_hint(f"查询抽帧状态失败：{error}", "#da3633")
            return

        status = payload.get("status")
        if status == "running":
            phase = payload.get("phase")
            total = payload.get("total") or 0
            done = payload.get("done") or 0
            if phase == "filter" and total:
                self.progress.setRange(0, int(total))
                self.progress.setValue(int(done))
                self._set_hint(f"正在过滤质量（{done}/{total}）...")
            else:
                self.progress.setRange(0, 0)
                self._set_hint("正在解码抽取帧...")
            return

        self._poll.stop()

        if status == "ok":
            num = int(payload.get("num_frames") or 0)
            self._frames_total += num
            self._last_result += f"{Path(self._queue[self._idx]).name}: {num} 帧\n"
            self._set_hint(
                f"✓ {Path(self._queue[self._idx]).name} 抽帧完成，保留 {num} 帧"
            )
            # 刷新帧清单 → 生成该视频的缩略图预览
            self._api.get(
                f"/api/projects/{self._project_name}/input-frames/list",
                tag="vx_frames",
            )
            self._advance()
        else:  # failed / idle(异常)
            err = payload.get("error") or "未知原因"
            self._running = False
            self.progress.setVisible(False)
            self._run_btn.setEnabled(True)
            self._done_btn.setEnabled(True)
            self._set_hint(
                f"✗ {Path(self._queue[self._idx]).name} 抽帧失败：{err}\n"
                "可调整设置后重新开始，或直接点击完成跳过。",
                "#da3633",
            )

    def _advance(self):
        """当前视频已成功 → 移到下一个；全部完成则收尾。"""
        self._idx += 1
        if self._idx >= len(self._queue):
            self._running = False
            self.progress.setVisible(False)
            self._done_btn.setEnabled(True)
            if self._frames_total:
                self._set_hint(
                    f"✅ 全部视频抽帧完成，共 {self._frames_total} 帧已加入项目。"
                )
            else:
                self._set_hint(
                    "没有保留任何帧（可能质量过滤过严），可关掉过滤重试。",
                    "#da3633",
                )
        else:
            self._begin_extract()

    # ------------------------------------------------------------------
    # 预览
    # ------------------------------------------------------------------

    def _maybe_load_preview(self, payload: dict):
        """取刚抽完的那个视频的前几张帧做缩略图预览。"""
        stem = Path(self._queue[self._idx - 1] if self._idx else "").stem
        if not stem:
            return
        imgs = [n for n in (payload.get("images") or []) if n.startswith(f"{stem}_f")]
        self._render_preview_header(stem, len(imgs))
        for name in imgs[:MAX_PREVIEW_FRAMES]:
            self._fetch_thumb(name)

    def _fetch_thumb(self, filename: str):
        url = QUrl(
            f"{self._api_base}/api/projects/{self._project_name}"
            f"/input-frames/view/{filename}"
        )
        reply = self._preview_mgr.get(QNetworkRequest(url))
        reply.setProperty("_fname", filename)
        # manager 的 finished 信号已统一接到 _on_preview_reply，勿重复连接

    def _on_preview_reply(self, reply: QNetworkReply):
        fname = reply.property("_fname")
        try:
            if reply.error() != QNetworkReply.NoError:
                return
            data = bytes(reply.readAll())
            pix = QPixmap()
            if not pix.loadFromData(data):
                return
            label = QLabel()
            label.setFixedSize(THUMB_W, THUMB_H)
            label.setAlignment(Qt.AlignCenter)
            label.setScaledContents(False)
            label.setStyleSheet(
                "border:1px solid #30363d; border-radius:4px;"
                " background-color:#0d1117;"
            )
            label.setPixmap(
                pix.scaled(THUMB_W, THUMB_H, Qt.KeepAspectRatio,
                           Qt.SmoothTransformation)
            )
            label.setToolTip(fname)
            self._thumb_labels[fname] = label
            # 插入到 stretch 之前，保持加入顺序
            self._preview_row.insertWidget(self._preview_row.count() - 1, label)
        finally:
            reply.deleteLater()

    def _render_preview_header(self, stem: str, total: int):
        if not self._preview_wrap.isVisible():
            self._preview_wrap.setVisible(True)
        shown = min(total, MAX_PREVIEW_FRAMES)
        self._preview_title.setText(
            f"{stem} — 抽取结果预览（前 {shown} 张 / 共 {total} 张）"
        )

    def _clear_previews(self):
        self._preview_wrap.setVisible(False)
        self._thumb_labels.clear()
        while self._preview_row.count() > 1:  # 保留末尾 stretch
            item = self._preview_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    # ------------------------------------------------------------------
    # 关闭 / 结果
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        # 运行中关闭 → 通知后端停掉后台 ffmpeg
        if self._running:
            self._poll.stop()
            self._api.post(
                f"/api/projects/{self._project_name}/videos/extract/stop",
                tag="vx_stop",
            )
            self._running = False
        super().closeEvent(event)

    def result_frame_count(self) -> int:
        return self._frames_total
