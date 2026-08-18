"""Docker exec wrapper for running SAM_CLI.py inside the sam3 container."""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path

from modules.docker_logger import LOG_DIR as DOCKER_LOG_DIR, start_follow
from modules.image_preprocessor import is_container_running
from modules.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTAINER_NAME = "sam3"
SAM_CLI_PATH = "/workspace/SAM_CLI.py"         # path inside container
DATA_DIR_HOST = Path("/home/zwz/Code/DATA")     # mounted at /data in container
DATA_DIR_CONTAINER = Path("/data")
CONTAINER_LOG_PATH = DOCKER_LOG_DIR / f"{CONTAINER_NAME}.log"

# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------

def build_sam_command(
    project_name: str,
    *,
    prompts: list[str],
    confidence: float = 0.5,
    device: str = "cuda",
    save_masks: bool = True,
    save_masks_rle: bool = False,
    vis: bool = False,
    batch_size: int = 1,
    checkpoint: str = "/workspace/model/sam3.pt",
) -> list[str]:
    """Build the docker exec command to run SAM_CLI.py inside the sam3 container.

    Project layout inside container:
        /data/<project>/SFM_output/dense/images/   →  --input-dir（去畸变图，SAM 在其上分割）
        /data/<project>/SAM_output/                →  -o
    """
    image_dir = DATA_DIR_CONTAINER / project_name / "SFM_output" / "dense" / "images"
    output_dir = DATA_DIR_CONTAINER / project_name / "SAM_output"

    cmd = [
        "docker", "exec", "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "PYTHONUNBUFFERED=1", CONTAINER_NAME,
        "python3", SAM_CLI_PATH,
        "--input-dir", str(image_dir),
        "--prompts", *prompts,
        "-o", str(output_dir),
        "--device", device,
        "--confidence", str(confidence),
        "--batch-size", str(batch_size),
        "--checkpoint", checkpoint,
    ]

    if save_masks:
        cmd.append("--save-masks")
    if save_masks_rle:
        cmd.append("--save-masks-rle")
    if vis:
        cmd.append("--vis")

    return cmd


# ---------------------------------------------------------------------------
# Async runner
# ---------------------------------------------------------------------------

_running_sam_processes: dict[str, subprocess.Popen] = {}


def run_sam_async(
    project_name: str,
    cmd: list[str],
    on_complete: callable | None = None,
) -> subprocess.Popen | None:
    """Run the SAM inference command asynchronously in a background thread."""

    def _target():
        logger.info("Project %r: launching SAM docker exec...", project_name)

        # Ensure container log follower is running
        start_follow(CONTAINER_NAME)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # merge stderr → stdout for real-time line order
            text=True,
            bufsize=1,  # line-buffered
        )
        _running_sam_processes[project_name] = proc

        # Prepare log paths
        sam_log = DATA_DIR_HOST / project_name / "SAM_output" / "sam.log"
        sam_log.parent.mkdir(parents=True, exist_ok=True)

        # Stream output line-by-line to both log files in real time
        DOCKER_LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(sam_log, "a", encoding="utf-8") as proj_f, \
             open(CONTAINER_LOG_PATH, "a", encoding="utf-8") as dock_f:

            header = f"\n{'='*60}\n=== SAM START {project_name} | {' '.join(cmd)}\n{'='*60}\n"
            proj_f.write(header)
            proj_f.flush()
            dock_f.write(header)
            dock_f.flush()

            for line in proc.stdout:
                proj_f.write(line)
                proj_f.flush()
                dock_f.write(line)
                dock_f.flush()

            proc.wait()
            logger.info("Project %r: SAM docker exited with code %d", project_name, proc.returncode)

            footer = f"\n{'='*60}\n=== SAM END {project_name} | exit_code: {proc.returncode}\n{'='*60}\n"
            proj_f.write(footer)
            proj_f.flush()
            dock_f.write(footer)
            dock_f.flush()

        if on_complete:
            on_complete(project_name, proc.returncode)

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(0.1)

    return _running_sam_processes.get(project_name)


# ---------------------------------------------------------------------------
# Process registry
# ---------------------------------------------------------------------------

def get_sam_process(project_name: str) -> subprocess.Popen | None:
    return _running_sam_processes.get(project_name)


def remove_sam_process(project_name: str) -> None:
    _running_sam_processes.pop(project_name, None)


def is_sam_running(project_name: str) -> bool:
    proc = _running_sam_processes.get(project_name)
    if proc is None:
        return False
    return proc.poll() is None
