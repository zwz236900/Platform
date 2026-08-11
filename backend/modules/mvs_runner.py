"""Docker exec wrapper for running mvs_cli.py inside the openmvs container."""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

from modules.docker_logger import LOG_DIR as DOCKER_LOG_DIR, start_follow
from modules.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTAINER_NAME = "openmvs"
MVS_CLI_PATH = "/workspace/MVS/mvs_cli.py"     # path inside container
DATA_DIR_HOST = Path("/home/zwz/Code/DATA")     # mounted at /data in container
DATA_DIR_CONTAINER = Path("/data")
CONTAINER_LOG_PATH = DOCKER_LOG_DIR / f"{CONTAINER_NAME}.log"

# Valid MVS subcommands
MVS_COMMANDS = ["pipeline", "interface", "densify", "reconstruct", "refine", "texture", "check"]

# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------


def build_mvs_command(
    project_name: str,
    *,
    command: str = "pipeline",
    model: str = "0",
    resolution_level: int = 1,
    max_resolution: int = 3200,
    min_resolution: int = 640,
    number_views: int = 5,
    no_cuda: bool = False,
    refine_resolution: int = 1,
    scales: int = 2,
    gradient_step: float = 45.0,
    texture_resolution: int = 1,
    export_type: str = "ply",
) -> list[str]:
    """Build the docker exec command to run mvs_cli.py inside the openmvs container.

    Project layout inside container:
        /data/<project>/SFM_output/                →  --sfm-dir
        /data/<project>/MVS_output/openmvs/        →  --output-dir
    """
    sfm_dir = DATA_DIR_CONTAINER / project_name / "SFM_output"
    output_dir = DATA_DIR_CONTAINER / project_name / "MVS_output" / "openmvs"

    cmd = [
        "docker", "exec", CONTAINER_NAME,
        "python3", MVS_CLI_PATH,
        command,
        "--project", project_name,
    ]

    if command == "pipeline":
        cmd += [
            "--sfm-dir", str(sfm_dir),
            "--output-dir", str(output_dir),
            "--model", model,
            "--resolution-level", str(resolution_level),
            "--max-resolution", str(max_resolution),
            "--min-resolution", str(min_resolution),
            "--number-views", str(number_views),
            "--refine-resolution", str(refine_resolution),
            "--scales", str(scales),
            "--gradient-step", str(gradient_step),
            "--texture-resolution", str(texture_resolution),
            "--export-type", export_type,
        ]
        if no_cuda:
            cmd.append("--no-cuda")
    elif command == "interface":
        cmd += [
            "--sfm-dir", str(sfm_dir),
            "--output-dir", str(output_dir),
            "--model", model,
        ]
    elif command in ("densify", "reconstruct", "refine", "texture"):
        cmd += [
            "--scene", str(output_dir / "scene.mvs"),  # placeholder, caller should override
            "--output-dir", str(output_dir),
            "--workspace", str(sfm_dir),
        ]

    # For densify: additional params
    if command == "densify":
        cmd += [
            "--resolution-level", str(resolution_level),
            "--max-resolution", str(max_resolution),
            "--number-views", str(number_views),
        ]
        if no_cuda:
            cmd.append("--no-cuda")
    # For refine: additional params
    elif command == "refine":
        cmd += [
            "--resolution-level", str(refine_resolution),
            "--scales", str(scales),
        ]
    # For texture: additional params
    elif command == "texture":
        cmd += [
            "--resolution-level", str(texture_resolution),
            "--min-resolution", str(min_resolution),
            "--export-type", export_type,
        ]

    return cmd


# ---------------------------------------------------------------------------
# Async runner
# ---------------------------------------------------------------------------

_running_mvs_processes: dict[str, subprocess.Popen] = {}


def run_mvs_async(
    project_name: str,
    cmd: list[str],
    on_complete: callable | None = None,
) -> subprocess.Popen | None:
    """Run the MVS command asynchronously in a background thread."""

    def _target():
        logger.info("Project %r: launching MVS docker exec...", project_name)

        # Ensure container log follower is running
        start_follow(CONTAINER_NAME)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        _running_mvs_processes[project_name] = proc

        # Prepare log paths
        mvs_log = DATA_DIR_HOST / project_name / "MVS_output" / "openmvs" / "mvs_pipeline.log"
        mvs_log.parent.mkdir(parents=True, exist_ok=True)

        # Stream output to both log files in real time
        DOCKER_LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(mvs_log, "a", encoding="utf-8") as proj_f, \
             open(CONTAINER_LOG_PATH, "a", encoding="utf-8") as dock_f:

            header = f"\n{'='*60}\n=== MVS START {project_name} | {' '.join(cmd)}\n{'='*60}\n"
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
            logger.info("Project %r: MVS docker exited with code %d", project_name, proc.returncode)

            footer = f"\n{'='*60}\n=== MVS END {project_name} | exit_code: {proc.returncode}\n{'='*60}\n"
            proj_f.write(footer)
            proj_f.flush()
            dock_f.write(footer)
            dock_f.flush()

        if on_complete:
            on_complete(project_name, proc.returncode)

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(0.1)

    return _running_mvs_processes.get(project_name)


# ---------------------------------------------------------------------------
# Process registry
# ---------------------------------------------------------------------------

def get_mvs_process(project_name: str) -> subprocess.Popen | None:
    return _running_mvs_processes.get(project_name)


def remove_mvs_process(project_name: str) -> None:
    _running_mvs_processes.pop(project_name, None)


def is_mvs_running(project_name: str) -> bool:
    proc = _running_mvs_processes.get(project_name)
    if proc is None:
        return False
    return proc.poll() is None
