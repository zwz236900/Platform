"""Video frame extraction (抽帧) — FFmpeg sampling + optional quality filtering.

Runs on the WSL host (system ``ffmpeg``/``ffprobe``), NOT inside a docker
container.  Exposes synchronous functions shaped like
``image_preprocessor.strip_metadata``: they return ``{success, ...}`` dicts or
raise ``RuntimeError`` with a user-actionable message; the API layer wraps them.

Extracted frames are written as ``{video_stem}_f%05d.jpg`` into the project's
``Input/input_image`` directory so COLMAP / SfM / ``count_images`` treat them as
ordinary project images.  Raw videos stay in ``Input/input_video`` and can be
re-extracted with different settings (stale frames for that stem are cleared
first).

Reference behaviour (ODM Video2Dataset / 360Cam tools): sample by a fixed time
interval, cap the longest edge, and drop motion-blurred / near-duplicate frames
so SfM gets clean, non-redundant input.  Filtering uses numpy+Pillow only — no
extra Python dependencies.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from modules.logger import get_logger
from modules.task_manager import _input_image_dir, _input_video_dir

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv", ".flv"}
FRAME_EXT = ".jpg"          # extracted frames are JPEG (metadata-free, COLMAP-friendly)
SCRATCH_PREFIX = ".extract_tmp_"   # ffmpeg candidate pass lands here, removed on exit

# Quality-filter thresholds — calibrated on generated test clips, keep
# conservative (only drop *obviously* flat / near-identical frames).
#  - Laplacian variance is computed on a 256px-wide grayscale version; a pure
#    flat frame measures ~180 (codec/JPEG noise floor), in-focus detail ~900+.
#    SHARPNESS_MIN_VAR=300 only drops featureless/blurred-to-flat frames.
#  - aHash is over the whole frame's brightness layout, so adjacent frames of a
#    slowly-moving scene score very low (<3).  DEDUP_MIN_HAMMING=1 therefore
#    only removes frames essentially identical to the last one KEPT (static
#    stretches), while genuine motion survives.
SHARPNESS_ANALYZE_WIDTH = 256     # grayscale width used for the Laplacian metric
SHARPNESS_MIN_VAR = 300.0         # drop when grayscale Laplacian variance < this
AHASH_SIZE = 8                    # aHash grid (8x8 -> 64 bit)
DEDUP_MIN_HAMMING = 1             # drop a frame when aHash distance to last kept < this

# Active ffmpeg process per project (allows the runner to stop a run).
_ACTIVE_FFMPEG: dict[str, subprocess.Popen] = {}


# ---------------------------------------------------------------------------
# FFmpeg detection
# ---------------------------------------------------------------------------

def ffmpeg_ready() -> dict:
    """Return whether ffmpeg/ffprobe are on PATH, plus paths/version."""
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    version = ""
    if ffmpeg:
        try:
            out = subprocess.run([ffmpeg, "-version"], capture_output=True,
                                 text=True, timeout=10)
            version = (out.stdout or "").splitlines()[0] if out.stdout else ""
        except (OSError, subprocess.TimeoutExpired):
            pass
    return {
        "available": bool(ffmpeg and ffprobe),
        "ffmpeg_path": ffmpeg or "",
        "ffprobe_path": ffprobe or "",
        "version": version,
    }


def _parse_rational(r: str) -> float | None:
    """Parse ffprobe ``r_frame_rate`` like '30000/1001' -> fps."""
    m = re.match(r"^(\d+)/(\d+)$", str(r).strip())
    if m:
        den = int(m.group(2))
        return int(m.group(1)) / den if den else None
    try:
        return float(r)
    except (TypeError, ValueError):
        return None


def probe_video(path: Path) -> dict:
    """Probe a video's basic metadata via ffprobe.

    Returns ``{success, duration, width, height, fps, codec}`` — or
    ``{success: False, error}`` when the file cannot be decoded as video.
    """
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {"success": False, "error": "ffprobe 不可用，请先安装 ffmpeg"}
    try:
        cmd = [
            ffprobe, "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,codec_name",
            "-show_entries", "format=duration",
            "-of", "json",
            str(path),
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"success": False, "error": f"ffprobe 失败: {e}"}
    if out.returncode != 0:
        return {"success": False, "error": (out.stderr or out.stdout).strip()[:500]}

    try:
        import json
        data = json.loads(out.stdout or "{}")
        streams = data.get("streams") or []
        if not streams:
            return {"success": False, "error": "文件中没有视频流"}
        s = streams[0]
        fmt = data.get("format", {})
        duration = None
        try:
            duration = float(fmt.get("duration") or s.get("duration") or 0.0)
        except (TypeError, ValueError):
            duration = 0.0
        return {
            "success": True,
            "duration": duration,
            "width": int(s.get("width") or 0),
            "height": int(s.get("height") or 0),
            "fps": _parse_rational(s.get("r_frame_rate") or ""),
            "codec": s.get("codec_name", ""),
        }
    except (ValueError, TypeError, json.JSONDecodeError):
        return {"success": False, "error": "无法解析 ffprobe 输出"}


# ---------------------------------------------------------------------------
# Sampling plan
# ---------------------------------------------------------------------------

def plan_interval(mode: str, duration: float, interval: float, target_count: int) -> float:
    """Return the sampling interval (seconds) for a given mode.

    - "interval": as-is.
    - "count": even sample over the duration -> duration / target_count.
    """
    if mode == "count":
        count = max(int(target_count or 0), 2)
        if duration and duration > 0:
            return max(duration / count, 0.001)
        return max(interval, 0.001)
    return max(interval, 0.001)


def clear_stale_frames(project_name: str, stem: str) -> int:
    """Remove previously extracted frames ``{stem}_f*.jpg`` from input_image."""
    img_dir = _input_image_dir(project_name)
    removed = 0
    if img_dir.is_dir():
        for p in img_dir.glob(f"{stem}_f*{FRAME_EXT}"):
            try:
                p.unlink()
                removed += 1
            except OSError:
                logger.warning("Project %r: could not remove stale frame %s",
                               project_name, p)
    if removed:
        logger.info("Project %r: removed %d stale frame(s) for %r",
                    project_name, removed, stem)
    return removed


# ---------------------------------------------------------------------------
# FFmpeg extraction
# ---------------------------------------------------------------------------

def extract_candidates(
    project_name: str,
    src: Path,
    stem: str,
    interval: float,
    max_width: int,
    on_progress=None,
) -> tuple[int, Path]:
    """Run one ffmpeg pass sampling frames at ``1/interval`` fps.

    Writes ``_%06d.jpg`` candidates into a scratch dir under input_video and
    returns ``(candidate_count, scratch_dir)``.
    """
    scratch = _input_video_dir(project_name) / f"{SCRATCH_PREFIX}{stem}"
    if scratch.exists():
        shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    vf = f"fps=1/{interval}"
    if max_width and max_width > 0:
        vf += f",scale=min(iw\\,{int(max_width)}):-2"
    out_pattern = str(scratch / "_%06d.jpg")

    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(src),
        "-vf", vf,
        "-q:v", "2",          # high-quality JPEG (approx q=95)
        "-vsync", "vfr",
        out_pattern,
    ]
    logger.info("Project %r: ffmpeg extract %s -> %s", project_name, src.name, scratch)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    _ACTIVE_FFMPEG[project_name] = proc

    log_path = _extract_log_path(project_name, stem)
    try:
        with open(log_path, "a", encoding="utf-8") as log_f:
            log_f.write(f"\n=== ffmpeg {cmd[0]} {' '.join(cmd[1:])}\n")
            # Stream stderr line-by-line into the log while reporting progress.
            for line in proc.stderr:
                log_f.write(line)
                log_f.flush()
                if on_progress:
                    done = sum(1 for _ in scratch.glob("_*.jpg"))
                    on_progress("decode", done, 0)
            proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg 抽帧失败 (exit {proc.returncode})，详见项目日志"
            )
        count = sum(1 for _ in scratch.glob("_*.jpg"))
        logger.info("Project %r: extracted %d candidate frame(s)", project_name, count)
        return count, scratch
    finally:
        _ACTIVE_FFMPEG.pop(project_name, None)


def terminate_active(project_name: str) -> bool:
    """Kill the running ffmpeg pass for a project (used by stop endpoint)."""
    proc = _ACTIVE_FFMPEG.get(project_name)
    if proc is None or proc.poll() is not None:
        return False
    try:
        proc.terminate()
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Quality filtering (numpy + Pillow)
# ---------------------------------------------------------------------------

def _gray_downscaled(img: Image.Image, target_width: int = SHARPNESS_ANALYZE_WIDTH) -> Image.Image:
    gray = ImageOps.grayscale(img)
    if gray.width > target_width:
        ratio = target_width / gray.width
        gray = gray.resize((target_width, max(1, int(gray.height * ratio))), Image.LANCZOS)
    return gray


def _sharpness_variance(gray: Image.Image) -> float:
    """Variance of a 4-neighbour Laplacian on grayscale (blur proxy)."""
    a = np.asarray(gray, dtype=np.float32)
    lap = 4.0 * a
    lap[1:, :] -= a[:-1, :]
    lap[:-1, :] -= a[1:, :]
    lap[:, 1:] -= a[:, :-1]
    lap[:, :-1] -= a[:, 1:]
    return float(lap.var())


def _ahash(gray: Image.Image) -> int:
    """64-bit average hash for near-duplicate detection."""
    small = gray.resize((AHASH_SIZE, AHASH_SIZE), Image.LANCZOS)
    a = np.asarray(small, dtype=np.float32)
    bits = (a > a.mean()).ravel()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def filter_and_commit(
    project_name: str,
    scratch: Path,
    stem: str,
    quality_filter: bool,
    on_progress=None,
) -> int:
    """Filter candidates and move accepted frames into input_image.

    With ``quality_filter=False`` every candidate is committed in order.
    Otherwise drop low-sharpness (blur) frames and near-duplicate consecutive
    frames.  Accepted frames are renumbered ``{stem}_f%05d.jpg``.
    Returns the number of committed frames.
    """
    img_dir = _input_image_dir(project_name)
    img_dir.mkdir(parents=True, exist_ok=True)

    candidates = sorted(scratch.glob("_*.jpg"))
    kept: list[tuple[Path, float, int]] = []  # (path, sharpness, ahash)
    prev_hash: int | None = None

    total = len(candidates)
    for i, path in enumerate(candidates):
        try:
            with Image.open(path) as im:
                im.load()
                gray = _gray_downscaled(im)
                sharp = _sharpness_variance(gray)
                h = _ahash(gray)
        except (OSError, ValueError) as e:
            logger.warning("Project %r: skip undecodable candidate %s: %s",
                           project_name, path.name, e)
            continue

        if quality_filter:
            if sharp < SHARPNESS_MIN_VAR:
                logger.info("Project %r: drop %s (blur, laplacian_var=%.1f)",
                            project_name, path.name, sharp)
                continue
            if prev_hash is not None:
                if _hamming(h, prev_hash) < DEDUP_MIN_HAMMING:
                    logger.info("Project %r: drop %s (dup, hamming=%d)",
                                project_name, path.name, _hamming(h, prev_hash))
                    continue
        prev_hash = h
        kept.append(path)
        if on_progress:
            on_progress("filter", i + 1, total)

    # Commit kept frames under a clean sequential name.
    n = 0
    for j, path in enumerate(kept, start=1):
        dest = img_dir / f"{stem}_f{j:05d}{FRAME_EXT}"
        try:
            shutil.copyfile(str(path), str(dest))
            n += 1
        except OSError as e:
            logger.error("Project %r: failed copying %s -> %s: %s",
                         project_name, path.name, dest.name, e)

    logger.info("Project %r: committed %d/%d frames for %r",
                project_name, n, total, stem)
    shutil.rmtree(scratch, ignore_errors=True)
    return n


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _extract_log_path(project_name: str, stem: str) -> Path:
    log_dir = _input_video_dir(project_name).parent / ".extract_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{stem}.log"


def extract_log_tail(project_name: str, video_filename: str, lines: int = 60) -> str:
    """Return the last N lines of a video's extraction log (best effort)."""
    lp = _extract_log_path(project_name, Path(video_filename).stem)
    if not lp.is_file():
        return ""
    try:
        with open(lp, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-lines:])
    except OSError:
        return ""


def extract_frames(
    project_name: str,
    video_filename: str,
    mode: str = "interval",
    interval: float = 2.0,
    target_count: int = 300,
    max_width: int = 1920,
    quality_filter: bool = True,
    on_progress=None,
) -> dict:
    """Extract frames from one uploaded video into ``Input/input_image``.

    Returns ``{success, num_frames, video, duration, width, height, fps}``.
    Raises ``RuntimeError`` with a user-actionable message on failure.
    """
    info = ffmpeg_ready()
    if not info["available"]:
        raise RuntimeError(
            "后端缺少 FFmpeg。请在 WSL 中执行：sudo apt update && sudo apt install -y ffmpeg"
        )

    src = _input_video_dir(project_name) / video_filename
    if not src.is_file():
        raise RuntimeError(f"视频文件不存在：{video_filename}")

    stem = Path(video_filename).stem
    meta = probe_video(src)
    if not meta.get("success"):
        raise RuntimeError(meta.get("error", "无法解码该视频文件"))
    duration = float(meta.get("duration") or 0.0)

    effective_interval = plan_interval(mode, duration, interval, target_count)
    logger.info(
        "Project %r: extract %r mode=%s interval=%.3fs max_width=%s quality=%s",
        project_name, video_filename, mode, effective_interval,
        max_width or "orig", quality_filter,
    )

    clear_stale_frames(project_name, stem)
    candidate_count, scratch = extract_candidates(
        project_name, src, stem, effective_interval, max_width, on_progress
    )
    if candidate_count == 0:
        shutil.rmtree(scratch, ignore_errors=True)
        raise RuntimeError("抽帧失败：未能从视频中抽到任何帧")

    num_frames = filter_and_commit(
        project_name, scratch, stem, quality_filter, on_progress
    )
    if num_frames == 0:
        raise RuntimeError(
            "质量过滤后没有保留任何帧。建议调大抽帧间隔，或关闭质量过滤后再试。"
        )

    logger.info("Project %r: %s -> %d frames", project_name, video_filename, num_frames)
    return {
        "success": True,
        "num_frames": num_frames,
        "video": video_filename,
        "duration": duration,
        "width": meta.get("width"),
        "height": meta.get("height"),
        "fps": meta.get("fps"),
    }
