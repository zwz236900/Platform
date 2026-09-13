"""API routes for video upload + frame extraction (抽帧).

Registered into the app via a single ``app.include_router(video_router)`` in
main.py so the large inline route body stays untouched.  Frame extraction runs
in the background (``video_runner``); clients start it with a POST and poll
``.../videos/extract/status`` — mirroring the pipeline run/status convention.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from modules.docker_runner import is_process_running
from modules.logger import get_logger
from modules.task_manager import (
    _input_image_dir,
    _input_video_dir,
    get_project,
)
from modules import video_processor
from modules.video_processor import VIDEO_EXTS
from modules.video_runner import (
    get_video_extract_status,
    is_video_extracting,
    run_video_extract_async,
    stop_video_extract,
)

logger = get_logger(__name__)

video_router = APIRouter(prefix="/api", tags=["video"])

IMAGE_MEDIA_MAP = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png", ".tif": "image/tiff", ".tiff": "image/tiff",
    ".bmp": "image/bmp", ".pgm": "image/x-portable-graymap",
    ".ppm": "image/x-portable-pixmap",
}
FRAME_EXTS = set(IMAGE_MEDIA_MAP) | {".gif", ".webp"}


def _check_project(project_name: str) -> None:
    if get_project(project_name) is None:
        raise HTTPException(404, "Project not found")


def _check_idle(project_name: str, *, allow_extract_running: bool = False) -> None:
    """Refuse to mutate a project while its SfM pipeline is running."""
    if is_process_running(project_name):
        raise HTTPException(400, "Project pipeline is running — wait for it to finish")
    if not allow_extract_running and is_video_extracting(project_name):
        raise HTTPException(400, "该项目正在抽帧，请等待当前任务完成")


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

@video_router.get("/system/ffmpeg")
async def ffmpeg_status():
    """FFmpeg availability (frontend preflight before showing the extract dialog)."""
    return video_processor.ffmpeg_ready()


# ---------------------------------------------------------------------------
# Video upload / metadata
# ---------------------------------------------------------------------------

@video_router.post("/projects/{project_name}/videos")
async def upload_video(project_name: str, video: UploadFile = File(...)):
    """Save one video into Input/input_video and probe its metadata.

    The file is streamed to disk (not buffered in RAM — videos can be GBs).
    Returns ``{success, filename, duration, width, height, fps, size}``.
    """
    _check_project(project_name)
    _check_idle(project_name)

    filename = os.path.basename(video.filename or "video")
    ext = Path(filename).suffix.lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(400, f"不支持的视频格式：{ext or '未知'}")

    vdir = _input_video_dir(project_name)
    vdir.mkdir(parents=True, exist_ok=True)
    dest = vdir / filename
    try:
        with dest.open("wb") as fh:
            shutil.copyfileobj(video.file, fh)
    except OSError as e:
        raise HTTPException(500, f"写入视频失败：{e}")

    meta = video_processor.probe_video(dest)
    if not meta.get("success"):
        dest.unlink(missing_ok=True)
        raise HTTPException(
            400,
            f"视频「{filename}」无法解码（{meta.get('error', '未知原因')}）。"
            "请确认是有效的视频文件，或改用图片/压缩包上传。",
        )

    logger.info("Project %r: video %s uploaded (%.1fs, %sx%s)",
                project_name, filename,
                float(meta.get("duration") or 0),
                meta.get("width"), meta.get("height"))
    return {
        "success": True,
        "filename": filename,
        "size": dest.stat().st_size,
        "duration": meta.get("duration"),
        "width": meta.get("width"),
        "height": meta.get("height"),
        "fps": meta.get("fps"),
        "codec": meta.get("codec"),
    }


@video_router.get("/projects/{project_name}/videos/list")
async def list_videos(project_name: str):
    """List stored videos with metadata and whether frames already exist."""
    _check_project(project_name)
    vdir = _input_video_dir(project_name)
    img_dir = _input_image_dir(project_name)
    videos = []
    if vdir.is_dir():
        for p in sorted(vdir.iterdir()):
            if not p.is_file() or p.name.startswith(".extract_tmp_"):
                continue
            ext = p.suffix.lower()
            if ext not in VIDEO_EXTS:
                continue
            meta = video_processor.probe_video(p)
            stem = p.stem
            frames_exist = img_dir.is_dir() and any(img_dir.glob(f"{stem}_f*.jpg"))
            videos.append({
                "filename": p.name,
                "size": p.stat().st_size,
                "frames_exist": frames_exist,
                **({k: meta.get(k) for k in ("duration", "width", "height", "fps", "codec")
                    if k in meta}),
            })
    return {"project_name": project_name, "videos": videos}


# ---------------------------------------------------------------------------
# Frame extraction (background)
# ---------------------------------------------------------------------------

@video_router.post("/projects/{project_name}/videos/extract")
async def extract_video_frames(
    project_name: str,
    filename: str = Form(...),
    mode: str = Form("interval"),
    interval: float = Form(2.0),
    count: int = Form(300),
    max_width: int = Form(1920),
    quality_filter: str = Form("1"),
):
    """Start background frame extraction for one uploaded video.

    Sampling modes: "interval" (every N seconds) or "count" (N total frames,
    evenly spaced).  ``max_width=0`` keeps original resolution.  Returns
    immediately; poll ``.../videos/extract/status`` for progress/result.
    """
    _check_project(project_name)
    _check_idle(project_name, allow_extract_running=False)

    if mode not in ("interval", "count"):
        raise HTTPException(400, f"未知抽帧模式：{mode}")
    if not filename or filename != os.path.basename(filename):
        raise HTTPException(400, "无效的视频文件名")

    vdir = _input_video_dir(project_name)
    src = vdir / filename
    if not src.is_file():
        raise HTTPException(404, f"视频文件不存在：{filename}")

    opts = {
        "mode": mode,
        "interval": max(float(interval), 0.001),
        "count": max(int(count), 1),
        "max_width": max(int(max_width), 0),
        "quality_filter": quality_filter.strip().lower() in ("1", "true", "yes"),
    }
    try:
        run_video_extract_async(project_name, filename, opts)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return {"success": True, "message": f"已开始为 {filename} 抽帧"}


@video_router.post("/projects/{project_name}/videos/extract/stop")
async def stop_extract(project_name: str):
    _check_project(project_name)
    stopped = stop_video_extract(project_name)
    return {"success": True, "stopped": stopped}


@video_router.get("/projects/{project_name}/videos/extract/status")
async def extract_status(project_name: str):
    _check_project(project_name)
    payload = get_video_extract_status(project_name)
    source = payload.get("source")
    payload["log_tail"] = (
        video_processor.extract_log_tail(project_name, source) if source else ""
    )
    return payload


# ---------------------------------------------------------------------------
# Raw input-image serving (used by the extract dialog preview / verification)
# ---------------------------------------------------------------------------

@video_router.get("/projects/{project_name}/input-frames/list")
async def list_input_frames(project_name: str):
    """List image files currently in Input/input_image (extracted frames + uploads)."""
    _check_project(project_name)
    img_dir = _input_image_dir(project_name)
    files = []
    if img_dir.is_dir():
        files = sorted(
            p.name for p in img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in FRAME_EXTS
        )
    return {"project_name": project_name, "images": files}


@video_router.get("/projects/{project_name}/input-frames/view/{filename:path}")
async def view_input_frame(project_name: str, filename: str):
    """Serve one raw image from Input/input_image (path-traversal guarded)."""
    _check_project(project_name)

    safe = os.path.normpath(filename)
    if safe.startswith("..") or safe.startswith("/") or "\\" in safe:
        raise HTTPException(400, "Invalid filename")

    img_dir = _input_image_dir(project_name)
    file_path = img_dir / safe
    if not file_path.is_file():
        raise HTTPException(404, f"Image not found: {filename}")

    ext = file_path.suffix.lower()
    return FileResponse(file_path, media_type=IMAGE_MEDIA_MAP.get(ext, "application/octet-stream"))
