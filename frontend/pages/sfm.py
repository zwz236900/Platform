"""SfM 重建页面 — 参数配置、pipeline 启动与产物下载."""

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
    QDialog,
    QSpinBox,
    QGroupBox,
    QFormLayout,
    QProgressBar,
    QTextEdit,
    QMessageBox,
    QFileDialog,
    QSizePolicy,
    QWidget,
)
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply
from PySide6.QtGui import QFont

from .base_page import BasePage
from .projects import ApiClient
from state import ProjectState, fill_project_combo
from widgets.sfm_advanced_dialog import SfmAdvancedDialog, count_overrides

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

API_BASE = "http://localhost:8000"

MATCHERS = ["exhaustive", "sequential", "vocab_tree", "spatial"]
CAMERA_MODELS = ["SIMPLE_RADIAL", "PINHOLE", "SIMPLE_PINHOLE", "RADIAL", "OPENCV"]

STEP_LABELS = {
    "extract": "特征提取",
    "match": "特征匹配",
    "sfm": "稀疏重建",
    "undistort": "去畸变",
}


def _format_size(n: int) -> str:
    """人类可读的文件大小。"""
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.2f} MB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.1f} KB"
    return f"{n} B"


# ---------------------------------------------------------------------------
# SfM 重建页面
# ---------------------------------------------------------------------------

class SfMPage(BasePage):
    """SfM 重建 — 参数配置与 pipeline 启动。"""

    def __init__(self):
        super().__init__("SfM 重建")

        self._api = ApiClient.instance()
        self._api.finished.connect(self._on_response)
        self._state = ProjectState.instance()
        self._state.current_changed.connect(self._on_shared_project_changed)

        # 独立下载管理器（下载二进制产物，避免干扰 JSON 解析）
        self._dl_mgr = QNetworkAccessManager(self)
        self._dl_mgr.finished.connect(self._on_download_finished)

        self._running = False
        self._current_project = ""
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_log)
        self._poll_timer.setInterval(3000)
        self._poll_count = 0

        # ── 项目选择 ──
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("目标项目:"))
        self.project_combo = QComboBox()
        self.project_combo.setMinimumWidth(220)
        self.project_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.project_combo.currentTextChanged.connect(self._on_project_changed)
        row1.addWidget(self.project_combo)

        self._refresh_btn = QPushButton("🔄 刷新")
        self._refresh_btn.setObjectName("actionBtn")
        self._refresh_btn.clicked.connect(self._load_projects)
        row1.addWidget(self._refresh_btn)
        row1.addStretch()
        self.content_layout.addLayout(row1)

        # ── Pipeline 步骤 ──
        step_group = QGroupBox("重建步骤")
        step_layout = QHBoxLayout(step_group)
        self._step_checks: dict[str, QCheckBox] = {}
        for key, label in STEP_LABELS.items():
            cb = QCheckBox(label)
            cb.setChecked(True)
            self._step_checks[key] = cb
            step_layout.addWidget(cb)
        step_layout.addStretch()
        self.content_layout.addWidget(step_group)

        # ── 参数设置 ──
        params = QHBoxLayout()
        params.setSpacing(12)

        # 左列
        left_group = QGroupBox("匹配与相机")
        lform = QFormLayout(left_group)

        self.matcher_combo = QComboBox()
        self.matcher_combo.addItems(MATCHERS)
        lform.addRow("匹配策略:", self.matcher_combo)

        self.camera_combo = QComboBox()
        self.camera_combo.addItems(CAMERA_MODELS)
        lform.addRow("相机模型:", self.camera_combo)

        self.features_spin = QSpinBox()
        self.features_spin.setRange(1000, 100000)
        self.features_spin.setValue(8192)
        self.features_spin.setSingleStep(1000)
        lform.addRow("最大特征点数:", self.features_spin)
        params.addWidget(left_group)

        # 右列
        right_group = QGroupBox("选项")
        rform = QFormLayout(right_group)

        self.gpu_check = QCheckBox("使用 GPU")
        self.gpu_check.setChecked(True)
        rform.addRow("", self.gpu_check)

        self.single_cam_check = QCheckBox("单相机模式（共享内参）")
        self.single_cam_check.setChecked(True)
        rform.addRow("", self.single_cam_check)

        self.geo_check = QCheckBox("禁用几何一致性检查")
        self.geo_check.setChecked(False)
        rform.addRow("", self.geo_check)

        self.gpu_index_spin = QSpinBox()
        self.gpu_index_spin.setRange(0, 7)
        self.gpu_index_spin.setValue(0)
        rform.addRow("GPU 编号:", self.gpu_index_spin)
        params.addWidget(right_group)

        self.content_layout.addLayout(params)

        # ── 高级参数：24 个 COLMAP 稀疏重建超参数收进独立对话框，主页保持简洁 ──
        self._sfm_params: dict[str, str] = {}   # 只在对话框中勾选覆盖的字段
        adv_row = QHBoxLayout()
        self.adv_btn = QPushButton("⚙ 高级参数…")
        self.adv_btn.setObjectName("actionBtn")
        self.adv_btn.setToolTip(
            "调整 COLMAP 稀疏重建的提取 / 匹配 / Mapper / BA 超参数。\n"
            "不打开此对话框时全部使用 COLMAP 默认值。")
        self.adv_btn.clicked.connect(self._on_advanced)
        adv_row.addWidget(self.adv_btn)

        self._adv_summary = QLabel("")
        self._adv_summary.setStyleSheet("color:#8b949e; font-size:13px;")
        adv_row.addWidget(self._adv_summary)
        adv_row.addStretch()
        self.content_layout.addLayout(adv_row)
        self._refresh_adv_summary()

        # ── 操作按钮 ──
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self._status_label = QLabel("就绪，请选择项目")
        self._status_label.setStyleSheet("color:#8b949e; font-size:13px;")
        btn_row.addWidget(self._status_label)

        btn_row.addStretch()

        self.run_btn = QPushButton("▶ 开始重建")
        self.run_btn.setObjectName("primaryBtn")
        self.run_btn.setMinimumWidth(120)
        self.run_btn.clicked.connect(self._on_run)
        btn_row.addWidget(self.run_btn)

        self.stop_btn = QPushButton("⏹ 停止")
        self.stop_btn.setObjectName("actionBtn")
        self.stop_btn.setMinimumWidth(100)
        self.stop_btn.setStyleSheet(
            "#actionBtn { color:#da3633; border-color:#da363333; }"
            "#actionBtn:hover { background-color:#da363322; }"
        )
        self.stop_btn.clicked.connect(self._on_stop)
        self.stop_btn.setVisible(False)
        btn_row.addWidget(self.stop_btn)
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

        # ── 下载区 ──（与 MVS 页面同构：后端列产物 → 下拉选择 → 存到本地）
        dl_group = QGroupBox("下载产物")
        dl_form = QFormLayout(dl_group)

        dl_row = QHBoxLayout()
        self.dl_file_combo = QComboBox()
        self.dl_file_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        dl_row.addWidget(self.dl_file_combo, 1)

        self.dl_refresh_btn = QPushButton("🔄")
        self.dl_refresh_btn.setObjectName("actionBtn")
        self.dl_refresh_btn.setFixedSize(34, 26)
        self.dl_refresh_btn.setToolTip("重新扫描 SFM_output 里的产物")
        self.dl_refresh_btn.clicked.connect(self._update_download_files)
        dl_row.addWidget(self.dl_refresh_btn)
        dl_form.addRow("文件:", dl_row)

        dl_btn_row = QHBoxLayout()
        self.download_btn = QPushButton("⬇ 下载")
        self.download_btn.setObjectName("actionBtn")
        self.download_btn.setMinimumWidth(100)
        self.download_btn.setToolTip("把选中的产物保存到本地")
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
        self._log.setMinimumHeight(150)
        self._log.setFont(QFont("Consolas", 10))
        self.content_layout.addWidget(self._log, 1)

    # ------------------------------------------------------------------
    # 项目列表
    # ------------------------------------------------------------------

    def _load_projects(self):
        self._refresh_btn.setEnabled(False)
        self._api.get("/api/projects", tag="list_sfm_projects")

    def _on_project_changed(self, project: str):
        """下拉框变化（用户选择 / 共享同步）→ 广播到全局。"""
        self._state.set_current(project)
        self._current_project = project
        # 产物列表随项目走，切项目后立刻重新扫一遍
        self._update_download_files()

    def _on_shared_project_changed(self, project: str):
        """其余页面选中了项目 → 同步本页下拉框。

        运行中不跟随：Pipeline 在后台跑时，_current_project 决定轮询目标，
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

    # ------------------------------------------------------------------
    # 高级参数
    # ------------------------------------------------------------------

    def _on_advanced(self):
        """打开模态对话框编辑 SfM 超参数；确定后只保存勾选覆盖的字段。"""
        dlg = SfmAdvancedDialog(self, current=self._sfm_params)
        if dlg.exec() != QDialog.Accepted:
            return
        self._sfm_params = dlg.result_params()
        self._refresh_adv_summary()
        n = count_overrides(self._sfm_params)
        self._log_msg(f"  高级参数：覆盖 {n} 项" if n
                      else "  高级参数：全部使用 COLMAP 默认值")

    def _refresh_adv_summary(self):
        n = count_overrides(self._sfm_params)
        self._adv_summary.setText(
            "SfM 参数：默认（COLMAP 内置值）" if n == 0 else f"SfM 参数：已覆盖 {n} 项")

    # ------------------------------------------------------------------
    # 启动 Pipeline
    # ------------------------------------------------------------------

    def _on_run(self):
        project = self.project_combo.currentText().strip()
        if not project:
            QMessageBox.warning(self, "提示", "请先选择一个项目。")
            return

        # 收集参数
        steps = [k for k, cb in self._step_checks.items() if cb.isChecked()]
        if not steps:
            QMessageBox.warning(self, "提示", "请至少选择一个重建步骤。")
            return

        config = {
            "steps": ",".join(steps),
            "matcher": self.matcher_combo.currentText(),
            "camera_model": self.camera_combo.currentText(),
            "single_camera": str(self.single_cam_check.isChecked()).lower(),
            "use_gpu": str(self.gpu_check.isChecked()).lower(),
            "gpu_index": str(self.gpu_index_spin.value()),
            "max_num_features": str(self.features_spin.value()),
            "no_geom_consistency": str(self.geo_check.isChecked()).lower(),
        }
        # 高级参数：只叠加在对话框里勾选过覆盖的字段（未勾选的键根本不在这里，
        # 后端据此保持 None → CLI 不追加 flag → COLMAP 用默认值）。
        config.update(self._sfm_params)

        self._current_project = project
        self._set_running(True)
        self._log_msg(f"▶ 启动 Pipeline — 项目: {project}")
        self._log_msg(f"  步骤: {' → '.join(steps)}")
        self._log_msg(f"  匹配: {config['matcher']}, 相机: {config['camera_model']}")
        self._log_msg(f"  特征点: {config['max_num_features']}, GPU: {config['use_gpu']}")
        self._status_label.setText("Pipeline 启动中...")

        self._api.post(
            f"/api/projects/{project}/run",
            config,
            tag="run_pipeline",
        )

    def _on_stop(self):
        project = self._current_project
        if not project:
            return
        reply = QMessageBox.question(
            self,
            "确认停止",
            f"确定要停止项目「{project}」的 Pipeline 吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._log_msg("⏹ 正在停止 Pipeline...")
        self._status_label.setText("正在停止...")
        self._api.post(f"/api/projects/{project}/stop", tag="stop_pipeline")

    # ------------------------------------------------------------------
    # 下载
    # ------------------------------------------------------------------

    def _update_download_files(self):
        """从后端读 SFM_output 里实际存在的产物，填充下载下拉框。"""
        if getattr(self, "dl_file_combo", None) is None:
            return
        project = self._current_project or self.project_combo.currentText().strip()
        self.dl_file_combo.clear()
        if not project:
            self.dl_file_combo.addItem("请先选择项目", "")
            return
        self._api.get(f"/api/projects/{project}/sfm/files", tag="sfm_files")

    def _on_download(self):
        if not self._current_project:
            QMessageBox.warning(self, "提示", "请先选择一个项目。")
            return
        rel_path = self.dl_file_combo.currentData()
        if not rel_path:
            QMessageBox.warning(self, "提示", "暂无可下载的产物，请先运行重建。")
            return

        # 产物在 SFM_output 里是嵌套路径（sparse/0/points3D.ply），落盘时取末段命名
        filename = Path(rel_path).name
        default_name = f"{self._current_project}_{filename}"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存 SfM 产物",
            default_name,
            "点云 / 模型 (*.ply *.bin *.json *.log *.db);;所有文件 (*)",
        )
        if not path:
            return

        self.download_btn.setEnabled(False)
        self.download_btn.setText("⏳ 下载中...")
        # 稀疏点云 PLY 由后端首次导出，会多花一点时间——先说明白，避免像卡死
        self._dl_status.setText(f"下载 {filename} ...")
        self._log_msg(f"⬇ 正在下载 SfM 产物 → {path}")

        url = QUrl(
            f"{API_BASE}/api/projects/{self._current_project}/sfm/download/{rel_path}"
        )
        req = QNetworkRequest(url)
        reply = self._dl_mgr.get(req)
        reply.setProperty("_save_path", path)
        reply.setProperty("_filename", filename)

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
            self._log_msg(
                f"✅ 下载完成: {filename} ({_format_size(len(data))}) → {save_path}")
            # 首次下载会就地生成 points3D.ply，刷新一下让「已生成」状态跟上
            self._update_download_files()
        except OSError as e:
            self._log_msg(f"✗ 写入文件失败: {e}")
            QMessageBox.warning(self, "下载失败", str(e))
        finally:
            reply.deleteLater()

    # ------------------------------------------------------------------
    # 轮询日志
    # ------------------------------------------------------------------

    def _poll_log(self):
        if not self._current_project:
            self._poll_timer.stop()
            return
        self._poll_count += 1
        # 每 5 次（15s）检查一次项目状态
        if self._poll_count % 5 == 0:
            self._api.get(
                f"/api/projects/{self._current_project}",
                tag="check_status",
            )
        else:
            self._api.get(
                f"/api/projects/{self._current_project}/log?lines=40",
                tag="poll_log",
            )
        # 每 tick 都查确定性进度（步骤 + MVS 深度图计数），驱动进度条
        self._api.get(
            f"/api/projects/{self._current_project}/progress",
            tag="sfm_progress",
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
        if tag == "list_sfm_projects":
            self._refresh_btn.setEnabled(True)
            if error:
                return
            projects = payload.get("projects", [])
            previous = self.project_combo.currentText()
            selected = fill_project_combo(self.project_combo, projects)
            # fill 期间屏蔽了信号：若全局选中项与下拉框现值不同，手动同步 _current_project
            if selected != previous:
                self._on_project_changed(selected)
            else:
                # 项目没变 → 上面不会触发刷新（例如切回本页）；产物仍要重扫
                self._update_download_files()

        # ── 产物列表 ──
        elif tag == "sfm_files":
            self.dl_file_combo.clear()
            if not error:
                for prod in payload.get("products", []):
                    self.dl_file_combo.addItem(
                        f"{prod.get('step', '')} · {prod.get('label', '')}",
                        prod.get("path", ""),
                    )
            if self.dl_file_combo.count() == 0:
                self.dl_file_combo.addItem("暂无产物，请先运行重建", "")

        # ── Pipeline 启动响应 ──
        elif tag == "run_pipeline":
            if error:
                self._set_running(False)
                self._log_msg(f"✗ 启动失败: {error}")
                self._status_label.setText("启动失败")
                QMessageBox.warning(self, "Pipeline 启动失败", str(error))
            else:
                self._log_msg("✓ Pipeline 已启动，等待完成...")
                self._status_label.setText("Pipeline 运行中...")
                self._poll_log()
                self._poll_timer.start()

        # ── 停止响应 ──
        elif tag == "stop_pipeline":
            if error:
                self._log_msg(f"✗ 停止失败: {error}")
                QMessageBox.warning(self, "停止失败", str(error))
            else:
                self._set_running(False)
                self._log_msg("⏹ Pipeline 已停止")
                self._status_label.setText("Pipeline 已停止")
                self._poll_timer.stop()

        # ── 日志轮询 ──
        elif tag == "poll_log":
            log_text = payload.get("log", "")
            if log_text and log_text != "(no log yet)":
                # 只追加新增的日志
                self._log.setPlainText(log_text)
                # 滚动到底部
                scrollbar = self._log.verticalScrollBar()
                if scrollbar:
                    scrollbar.setValue(scrollbar.maximum())

            # 检查是否已完成（通过检查项目状态）
            # 如果日志最后一行包含错误或完成信息
            if "FAILED" in log_text or "Pipeline exited" in log_text:
                self._set_running(False)
                self._log_msg("✗ Pipeline 失败，请检查日志。")
                self._status_label.setText("Pipeline 失败")

        # ── 确定性进度（步骤标签 + MVS 深度图计数百分比）──
        elif tag == "sfm_progress":
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
                # 无法计数（特征/匹配/SfM）：保持滚动条 + 步骤文案
                self.progress.setRange(0, 0)

        # ── 项目详情（用于检查运行状态）──
        elif tag == "check_status":
            status = payload.get("status", "")
            if status in ("completed",):
                self._set_running(False)
                elapsed = (payload.get("pipeline") or {}).get("elapsed_seconds", 0)
                self._log_msg(f"✓ Pipeline 完成 — 耗时 {elapsed:.0f}s")
                self._status_label.setText("✓ Pipeline 已完成")
                self._poll_timer.stop()
                # 本轮新产出的稀疏点云 / 模型 / 日志，刷新下载列表
                self._update_download_files()
            elif status in ("failed",):
                self._set_running(False)
                err = payload.get("error_message", "未知错误")
                self._log_msg(f"✗ Pipeline 失败: {err}")
                self._status_label.setText("Pipeline 失败")
                self._poll_timer.stop()

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def _log_msg(self, text: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self._log.append(f"[{ts}] {text}")

    def _set_running(self, running: bool):
        self._running = running
        self.run_btn.setEnabled(not running)
        self.stop_btn.setVisible(running)
        self.project_combo.setEnabled(not running)
        self.progress.setVisible(running)
        # 步骤/详情文案只在运行期间显示，先隐藏避免残留上一轮内容
        self._step_label.setVisible(False)
        self._detail_label.setVisible(False)
        if running:
            self.run_btn.setText("⏳ 运行中...")
            self.progress.setRange(0, 0)
        else:
            self.run_btn.setText("▶ 开始重建")
            self.progress.setVisible(False)

    def showEvent(self, event):
        super().showEvent(event)
        self._load_projects()
        if not self._poll_timer.isActive():
            # 如果正在运行，恢复轮询
            pass

    def hideEvent(self, event):
        super().hideEvent(event)
        # 切走页面时也保持轮询（pipeline 在后台跑）
