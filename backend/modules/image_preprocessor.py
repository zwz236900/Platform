"""Image preprocessing — metadata stripping and COLMAP undistortion.

Runs inside the colmap-dev container via ``docker exec``.  Exposes synchronous
functions that are wrapped in ``asyncio.to_thread`` by the async routes in main.py.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from modules.docker_runner import _ensure_colmap_running
from modules.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTAINER_NAME = "colmap-dev"
DATA_DIR_HOST = Path("/home/zwz/Code/DATA")       # mounted at /data in container
DATA_DIR_CONTAINER = Path("/data")

# Canonical set of supported image extensions (used by upload filter, exiftool,
# and image-count helpers — avoid duplicating this list elsewhere).
SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".pgm", ".ppm")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_container_running() -> bool:
    """Check whether the colmap-dev container is currently running."""
    try:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER_NAME],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() == "true"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ---------------------------------------------------------------------------
# Metadata stripping
# ---------------------------------------------------------------------------

def strip_metadata(project_name: str) -> dict:
    """Strip all IPTC/EXIF metadata from uploaded images.

    Prevents OpenImageIO from crashing on invalid IPTC metadata
    (``encode_iptc_iim_one_tag`` assertion failure) during image undistortion.

    Returns a dict with keys:
        success (bool)
        processed (int) — number of files actually touched
        error (str | None) — error message if the command failed

    Note: the colmap-dev container is not persistent — it is started on demand.
    This function ensures it is running before calling exiftool, instead of
    silently skipping (which would let the problematic metadata reach COLMAP
    and crash undistortion with the OpenImageIO IPTC assertion).
    """
    try:
        _ensure_colmap_running()
    except RuntimeError as e:
        logger.warning("Project %r: cannot start colmap-dev container for metadata stripping: %s",
                       project_name, e)
        return {"success": False, "processed": 0, "error": str(e)}

    image_dir = DATA_DIR_CONTAINER / project_name / "Input" / "input_image"
    if not (DATA_DIR_HOST / project_name / "Input" / "input_image").is_dir():
        logger.info("Project %r: input_image dir does not exist yet — skip preprocessing", project_name)
        return {"success": True, "processed": 0, "error": None}

    # Build extension arguments for exiftool
    ext_args: list[str] = []
    for ext in SUPPORTED_EXTS:
        ext_args += ["-ext", ext.lstrip(".")]

    cmd = [
        "docker", "exec", CONTAINER_NAME,
        "exiftool", "-all=", "-overwrite_original",
        *ext_args,
        str(image_dir) + "/",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError:
        logger.warning("Project %r: docker not available for image preprocessing", project_name)
        return {"success": False, "processed": 0, "error": "docker not found"}
    except subprocess.TimeoutExpired:
        logger.warning("Project %r: image preprocessing timed out", project_name)
        return {"success": False, "processed": 0, "error": "timeout"}

    # exiftool outputs "N image files updated" on success; parse that line
    processed = 0
    if result.returncode == 0 and result.stdout:
        for line in result.stdout.splitlines():
            if "image files updated" in line:
                try:
                    processed = int(line.strip().split()[0])
                except ValueError:
                    pass

    # exit code 1 usually means "nothing to do" — not an error
    if result.returncode not in (0, 1):
        logger.warning(
            "Project %r: exiftool exited with %d — stderr: %s",
            project_name, result.returncode, result.stderr[:200] if result.stderr else "",
        )
        return {"success": False, "processed": processed,
                "error": f"exiftool exit code {result.returncode}"}

    logger.info("Project %r: metadata stripped from %d image(s)", project_name, processed)
    return {"success": True, "processed": processed, "error": None}


# ---------------------------------------------------------------------------
# COLMAP image undistortion
# ---------------------------------------------------------------------------

def undistort_images(project_name: str) -> dict:
    """Run COLMAP image_undistorter to remove lens distortion and convert to PINHOLE.

    Requires that SfM has already completed (``SFM_output/sparse/0/`` must exist).
    Output goes to ``SFM_output/undistorted/`` containing both undistorted images
    and a new sparse model with PINHOLE cameras.

    Returns a dict with keys:
        success (bool)
        output_dir (str) — host path to the undistorted directory
        num_images (int) — number of undistorted images written
        error (str | None) — error message if the command failed

    Note: ensures the colmap-dev container is running (started on demand) rather
    than skipping when it is stopped.
    """
    try:
        _ensure_colmap_running()
    except RuntimeError as e:
        logger.warning("Project %r: cannot start colmap-dev container for undistortion: %s",
                       project_name, e)
        return {"success": False, "output_dir": "", "num_images": 0, "error": str(e)}

    # Prerequisites
    image_dir_host = DATA_DIR_HOST / project_name / "Input" / "input_image"
    sparse_dir_host = DATA_DIR_HOST / project_name / "SFM_output" / "sparse" / "0"
    output_dir_host = DATA_DIR_HOST / project_name / "SFM_output" / "undistorted"

    if not image_dir_host.is_dir():
        return {"success": False, "output_dir": "", "num_images": 0,
                "error": f"Image directory not found: {image_dir_host}"}

    if not sparse_dir_host.is_dir():
        return {"success": False, "output_dir": "", "num_images": 0,
                "error": f"SfM sparse model not found: {sparse_dir_host}. Run SfM pipeline first."}

    # Container-side paths
    image_dir_container = DATA_DIR_CONTAINER / project_name / "Input" / "input_image"
    sparse_dir_container = DATA_DIR_CONTAINER / project_name / "SFM_output" / "sparse" / "0"
    output_dir_container = DATA_DIR_CONTAINER / project_name / "SFM_output" / "undistorted"

    # Ensure output directory exists (COLMAP needs it to not exist, or be empty)
    if output_dir_host.exists():
        import shutil
        logger.info("Project %r: removing existing undistorted dir: %s", project_name, output_dir_host)
        shutil.rmtree(output_dir_host)

    cmd = [
        "docker", "exec", CONTAINER_NAME,
        "colmap", "image_undistorter",
        "--image_path", str(image_dir_container),
        "--input_path", str(sparse_dir_container),
        "--output_path", str(output_dir_container),
        "--output_type", "COLMAP",
    ]

    logger.info("Project %r: running colmap image_undistorter...", project_name)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except FileNotFoundError:
        return {"success": False, "output_dir": "", "num_images": 0,
                "error": "docker not found"}
    except subprocess.TimeoutExpired:
        return {"success": False, "output_dir": "", "num_images": 0,
                "error": "timeout (300s)"}

    if result.returncode != 0:
        stderr_tail = result.stderr[-500:] if result.stderr else "(no stderr)"
        logger.error("Project %r: image_undistorter failed (exit %d): %s",
                     project_name, result.returncode, stderr_tail)
        return {"success": False, "output_dir": "", "num_images": 0,
                "error": f"image_undistorter exit code {result.returncode}: {stderr_tail[:200]}"}

    # Count output images
    undistorted_images_dir = output_dir_host / "images"
    num_images = 0
    if undistorted_images_dir.is_dir():
        num_images = sum(1 for p in undistorted_images_dir.iterdir()
                         if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)

    logger.info("Project %r: undistorted %d images → %s", project_name, num_images, output_dir_host)

    return {
        "success": True,
        "output_dir": str(output_dir_host),
        "num_images": num_images,
        "error": None,
    }
