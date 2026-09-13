"""Background runner for video frame extraction (抽帧).

Mirrors ``mask_obj_runner.py``: the API layer calls ``run_video_extract_async``,
which *synchronously* validates FFmpeg is present and marks the project as
``running`` before returning (so the GUI never sees a dangling idle state), then
runs the synchronous ``video_processor.extract_frames`` in a daemon thread.
Live progress (phase: decode/filter, frame counts) is kept in an in-memory dict
polled via ``GET .../videos/extract/status``.
"""

from __future__ import annotations

import threading

from modules import video_processor
from modules.logger import get_logger
from modules.task_manager import count_images, update_project

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Run-state tracking (per project)
# ---------------------------------------------------------------------------

# status: idle | running | ok | failed ; filled on exit, idempotent.
_video_extract_state: dict[str, dict] = {}
# Live progress shown while running: {phase: "decode"|"filter", done, total}
_video_extract_progress: dict[str, dict] = {}


def _finalize_state(project_name: str, status: str, **fields) -> None:
    """Record the final state of an extraction run (idempotent)."""
    entry = _video_extract_state.get(project_name) or {}
    entry.update({"status": status, **fields})
    _video_extract_state[project_name] = entry


def get_video_extract_status(project_name: str) -> dict:
    """Build the status payload consumed by the poll endpoint."""
    if is_video_extracting(project_name):
        base = {"status": "running"}
    else:
        base = dict(_video_extract_state.get(project_name) or {"status": "idle"})
    base.setdefault("status", "idle")
    progress = dict(_video_extract_progress.get(project_name) or {})
    return {
        "status": base.get("status", "idle"),
        "source": base.get("source"),
        "num_frames": base.get("num_frames"),
        "error": base.get("error"),
        "phase": progress.get("phase"),
        "done": progress.get("done", 0),
        "total": progress.get("total", 0),
    }


def is_video_extracting(project_name: str) -> bool:
    state = _video_extract_state.get(project_name)
    return bool(state and state.get("status") == "running")


def stop_video_extract(project_name: str) -> bool:
    """Ask the running ffmpeg pass to stop; the thread finalizes state as failed."""
    if not is_video_extracting(project_name):
        return False
    terminated = video_processor.terminate_active(project_name)
    logger.info("Project %r: stop requested (terminate=%s)", project_name, terminated)
    return True


# ---------------------------------------------------------------------------
# Async runner
# ---------------------------------------------------------------------------

def run_video_extract_async(
    project_name: str,
    video_filename: str,
    opts: dict,
    on_complete=None,
) -> None:
    """Start extracting frames in a background thread.

    Raises synchronously (before any running state is set) if FFmpeg is not
    available or a run is already in progress for this project.
    """
    # 调用即校验：FFmpeg 缺失时同步抛错，绝不留下悬空 running 状态。
    info = video_processor.ffmpeg_ready()
    if not info.get("available"):
        raise RuntimeError(
            "后端缺少 FFmpeg。请在 WSL 中执行：sudo apt update && sudo apt install -y ffmpeg"
        )
    if is_video_extracting(project_name):
        raise RuntimeError("该项目正在抽帧，请等待当前任务完成")

    source = video_filename
    _video_extract_state[project_name] = {"status": "running", "source": source}
    _video_extract_progress[project_name] = {"phase": "decode", "done": 0, "total": 0}

    def _on_progress(phase: str, done: int, total: int) -> None:
        _video_extract_progress[project_name] = {
            "phase": phase, "done": int(done), "total": int(total),
        }

    def _target() -> None:
        try:
            result = video_processor.extract_frames(
                project_name,
                video_filename,
                mode=str(opts.get("mode", "interval")),
                interval=float(opts.get("interval", 2.0)),
                target_count=int(opts.get("count", 300)),
                max_width=int(opts.get("max_width", 1920) or 0),
                quality_filter=bool(opts.get("quality_filter", True)),
                on_progress=_on_progress,
            )
            num_frames = int(result.get("num_frames", 0))
            update_project(project_name, num_images=count_images(project_name))
            _video_extract_progress.pop(project_name, None)
            _finalize_state(project_name, "ok",
                            source=source, num_frames=num_frames, error=None)
            logger.info("Project %r: video %r extracted to %d frame(s)",
                        project_name, video_filename, num_frames)
            if on_complete:
                on_complete(project_name, result)
        except Exception as e:  # noqa: BLE001 — report any failure to the GUI
            logger.error("Project %r: extraction failed for %r: %s",
                         project_name, video_filename, e)
            _video_extract_progress.pop(project_name, None)
            _finalize_state(project_name, "failed",
                            source=source, num_frames=None,
                            error=str(e) or "抽帧失败")
            if on_complete:
                on_complete(project_name, {"success": False, "error": str(e)})

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()


def is_project_busy(project_name: str) -> bool:
    """True when a pipeline or an extraction is running for this project.

    Used by upload/video endpoints to reject mutations mid-run (mirrors the
    ``is_process_running`` guard in main.py for pipeline runs).
    """
    return is_video_extracting(project_name)
