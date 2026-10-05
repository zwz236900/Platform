"""Pipeline progress via host-side artifact counting.

COLMAP runs (sfm_cli.py inside the container) buffer colmap's stdout, so the
only reliable *live* signals on the host are:
  1. the step banners / command echoes that sfm_cli itself prints to
     SFM_output/pipeline.log (streamed line-by-line by docker_runner), and
  2. the per-image depth maps colmap writes incrementally onto the bind-mounted
     DATA dir (MVS_output/<project>/colmap/stereo/depth_maps/...).

This module turns those two signals into a determinate progress for the GUI:
- it locates the current step from the last PIPELINE START segment in the log,
- and for the MVS (PatchMatch Stereo) phase it counts depth_map files
  (photometric pass, then geometric pass when geom_consistency is on) against
  the number of reference images in stereo/patch-match.cfg.

Pure host-side reads — nothing is injected into the container.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from modules.docker_runner import (
    CONTAINER_NAME,
    DATA_DIR_HOST,
    is_process_running,
)
from modules.mvs_runner import is_mvs_running

# ---------------------------------------------------------------------------
# Step banners printed by sfm_cli.py (order as printed during a run).
# (needle, machine key, display label)
# ---------------------------------------------------------------------------

_BANNERS: list[tuple[str, str, str]] = [
    ("Step 1/5: Feature Extraction", "extract", "步骤 1/5 · 特征提取"),
    ("Step 2/5: Feature Matching", "match", "步骤 2/5 · 特征匹配"),
    ("Step 3/5: Sparse Reconstruction", "sfm", "步骤 3/5 · 稀疏重建 (SfM)"),
    ("Step 4/5: Image Undistortion", "undistort", "步骤 4/5 · 影像校正"),
    ("Step: Reuse Undistorted Workspace", "reuse", "复用去畸变结果 (Reuse)"),
    ("Step 5/5: Dense Reconstruction", "mvs", "步骤 5/5 · 稠密重建 (MVS)"),
    ("--- PatchMatch Stereo ---", "mvs_patch", "稠密重建 · PatchMatch 立体匹配"),
    ("--- Stereo Fusion ---", "mvs_fusion", "稠密重建 · 立体融合"),
]

_MVS_STEPS = {"mvs", "mvs_patch", "mvs_fusion"}


def _project_dir(project_name: str) -> Path:
    return DATA_DIR_HOST / project_name


def _log_path(project_name: str) -> Path:
    return _project_dir(project_name) / "SFM_output" / "pipeline.log"


def _colmap_mvs_dir(project_name: str) -> Path:
    return _project_dir(project_name) / "MVS_output" / "colmap"


def _task_status_running(project_name: str) -> bool:
    """task.json status == running (survives backend restarts; Docker registry does not)."""
    task = _project_dir(project_name) / "task.json"
    try:
        import json
        with open(task, encoding="utf-8") as f:
            return json.load(f).get("status") == "running"
    except (OSError, ValueError):
        return False


def _colmap_process_alive(project_name: str) -> bool:
    """Is a colmap sfm_cli.py process alive in the container for this project?

    Covers the orphan case: a pipeline started before a backend restart keeps
    running inside colmap-dev, but the in-memory runner registry is empty so
    ``is_process_running`` lies. Only called when task.json says running yet
    neither runner claims the process, so the ~docker exec overhead is paid
    only in that ambiguous case.
    """
    marker = f"/data/{project_name}/Input/input_image"
    cmd = [
        "docker", "exec", CONTAINER_NAME, "sh", "-c",
        f"ps -eo args | grep '[s]fm_cli.py' | grep -F '{marker}'",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        return bool(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


def _latest_step(log_path: Path) -> tuple[str, str] | None:
    """Return (key, label) of the newest step banner in the newest run segment.

    A new ``=== PIPELINE START ... ===`` header resets the current step so we
    never report a banner from a previous run of the same project.
    """
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return None

    step: tuple[str, str] | None = None
    for line in lines:
        if line.lstrip().startswith("=== PIPELINE START"):
            step = None
            continue
        for needle, key, label in _BANNERS:
            if needle in line:
                step = (key, label)
                break
    return step


def _geom_consistency(log_path: Path) -> bool | None:
    """Read --PatchMatchStereo.geom_consistency from the echoed colmap command.

    True/False → two-pass / single-pass. None → unknown (fall back to whatever
    the file layout implies).
    """
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return None
    # Last match, not first: pipeline.log is appended to across runs, so an
    # earlier run's flag would otherwise mask the newest one -- e.g. a project
    # re-run with geometric consistency off would keep reporting two-pass and
    # the depth-map counts would never add up.
    found = re.findall(r"--PatchMatchStereo\.geom_consistency\s+(\S+)", text)
    if not found:
        return None
    val = found[-1].strip().lower()
    if val == "true":
        return True
    if val == "false":
        return False
    return None


def _patch_match_refs(workspace: Path) -> int | None:
    """Number of unique reference images in stereo/patch-match.cfg.

    COLMAP writes one problem per line whose first token is the reference image
    (the one whose depth map is computed). Returns None when the cfg is absent
    (e.g. fusion-only) or unreadable.
    """
    cfg = workspace / "stereo" / "patch-match.cfg"
    if not cfg.is_file():
        return None
    refs: set[str] = set()
    try:
        with open(cfg, encoding="utf-8", errors="replace") as f:
            for raw in f:
                s = raw.strip()
                if not s or s.startswith("#"):
                    continue
                first = s.split()[0]
                if first:
                    refs.add(first)
    except OSError:
        return None
    return len(refs) or None


def _depth_stats(workspace: Path) -> dict:
    """Count depth-map files, split by pass.

    geom_consistency on  → <img>.photometric.bin then <img>.geometric.bin
    geom_consistency off → a single <img>.photometric.bin per reference image
        (the "geometric" suffix names the second pass, not the first — see
        patch_match.cc, which picks the suffix straight off the flag).
    """
    depth_dir = workspace / "stereo" / "depth_maps"
    stats = {"photometric": 0, "geometric": 0, "present": False}
    if not depth_dir.is_dir():
        return stats
    try:
        names = [p.name for p in depth_dir.iterdir()]
    except OSError:
        return stats
    stats["present"] = True
    for name in names:
        if name.endswith(".photometric.bin"):
            stats["photometric"] += 1
        elif name.endswith(".geometric.bin"):
            stats["geometric"] += 1
    return stats


def _mvs_progress(workspace: Path, geom: bool | None) -> dict:
    """Determinate progress for the PatchMatch/Fusion phase.

    Returns dict with percent/done/total/detail (percent None when unknown).
    """
    refs = _patch_match_refs(workspace)
    stats = _depth_stats(workspace)
    photo, geo = stats["photometric"], stats["geometric"]

    # Decide effective mode from layout if the echoed flag is unavailable.
    two_pass = True if geom is None else geom
    if geom is None and not stats["present"]:
        two_pass = True  # nothing written yet — assume default two-pass

    if two_pass and (stats["present"] or refs is not None):
        total = 2 * refs if refs is not None else None
        done = photo + geo
        if refs is not None and photo >= refs:
            detail = f"几何一致性 深度图 {min(geo, refs)}/{refs}"
        elif refs is not None:
            detail = f"光度一致性 深度图 {photo}/{refs}"
        else:
            detail = f"已生成 {done} 张深度图"
    else:
        # Single pass still lands in the photometric bucket, so the same sum is
        # right here; counting only "plain .bin" files would always read 0.
        total = refs
        done = photo + geo
        detail = f"PatchMatch 深度图 {done}/{refs}" if refs is not None else "正在生成深度图"

    if total is None:
        percent = None
    else:
        total = max(total, 1)
        percent = round(100.0 * min(done, total) / total, 1)
        if done >= total and stats["present"]:
            detail += "（两趟均已生成）" if two_pass else "（深度图已生成）"

    return {"percent": percent, "done": done, "total": total,
            "detail": detail, "geom_two_pass": two_pass}


def compute_pipeline_progress(project_name: str) -> dict:
    """Host-side progress snapshot for a project's running pipeline.

    The GUI polls this while a pipeline is running to drive a determinate
    progress bar + step label. When nothing is running it reports running=False
    and the caller keeps its existing idle/indeterminate behaviour.
    """
    base = {
        "project_name": project_name,
        "running": False,
        "kind": None,
        "step": None,
        "label": None,
        "percent": None,
        "done": None,
        "total": None,
        "detail": None,
    }

    colmap_active = is_process_running(project_name)
    openmvs_active = is_mvs_running(project_name)

    if not colmap_active and not openmvs_active and _task_status_running(project_name):
        # Orphan colmap: backend restarted mid-run, but sfm_cli.py is still
        # alive inside the container writing depth maps. Detect it so the GUI
        # keeps showing live progress instead of a frozen indeterminate bar.
        if _colmap_process_alive(project_name):
            colmap_active = True

    if colmap_active:
        # COLMAP pipeline (docker exec sfm_cli.py) — full 5 steps or MVS-only.
        base["running"] = True
        base["kind"] = "colmap"
        log_path = _log_path(project_name)
        step = _latest_step(log_path)
        if step is None:
            base["label"] = "COLMAP 管线运行中…"
            return base

        key, label = step
        base["step"] = key
        base["label"] = label

        if key in _MVS_STEPS:
            workspace = _colmap_mvs_dir(project_name)
            geom = _geom_consistency(log_path)
            prog = _mvs_progress(workspace, geom)
            base.update(prog)
            if prog["percent"] is not None and prog["total"]:
                base["detail"] = f"{prog['detail']} · 总进度 {prog['percent']:.0f}%"
            elif prog["percent"] is None:
                base["detail"] = prog["detail"]
        return base

    if openmvs_active:
        # OpenMVS runs in its own container; we only surface running state.
        base["running"] = True
        base["kind"] = "openmvs"
        base["step"] = "openmvs"
        base["label"] = "OpenMVS 运行中…"
        return base

    return base
