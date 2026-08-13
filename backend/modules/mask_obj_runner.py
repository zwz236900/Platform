"""Docker exec wrapper for 掩码目标重建（mask_obj）的 MVS 阶段。

裁剪 + 建模在 POST 处理器里同步完成（``algorithm.mask_obj.run``），
本模块只负责把 ``sfm_cli.py --steps mvs`` 的 docker exec 跑起来并流式写日志，
复用 colmap-dev 容器与 docker_runner 的按需启动。
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

from modules.docker_logger import LOG_DIR as DOCKER_LOG_DIR, start_follow
from modules.docker_runner import _ensure_colmap_running
from modules.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTAINER_NAME = "colmap-dev"
DATA_DIR_HOST = Path("/home/zwz/Code/DATA")
CONTAINER_LOG_PATH = DOCKER_LOG_DIR / f"{CONTAINER_NAME}.log"


# ---------------------------------------------------------------------------
# Run-state tracking
# ---------------------------------------------------------------------------

# 最近一次 mask_obj 运行的结果状态，keyed by project_name。
# 前端据此判定完成（替代解析日志关键字），进程退出即落定，幂等。
_mask_obj_run_state: dict[str, dict] = {}


def _finalize_mask_obj_state(project_name: str, exit_code: int) -> None:
    """记录一次 mask_obj 运行的最终状态（进程退出时调用，幂等）。"""
    _mask_obj_run_state[project_name] = {
        "status": "ok" if exit_code == 0 else "failed",
        "exit_code": exit_code,
    }


def get_mask_obj_run_status(project_name: str) -> dict | None:
    """返回当前 mask_obj 运行状态；本会话从未运行过返回 None。

    - {"status": "running"}                       进程存活
    - {"status": "ok"|"failed", "exit_code": int} 已结束
    """
    proc = _running_mask_obj_processes.get(project_name)
    if proc is not None and proc.poll() is None:
        return {"status": "running"}
    if proc is not None:
        # 进程已退出而 _target 尚未落定状态（罕见竞态）→ 就地补齐
        _finalize_mask_obj_state(project_name, proc.returncode)
    return _mask_obj_run_state.get(project_name)


# ---------------------------------------------------------------------------
# Async runner
# ---------------------------------------------------------------------------

_running_mask_obj_processes: dict[str, subprocess.Popen] = {}


def run_mask_obj_async(
    project_name: str,
    cmd: list[str],
    on_complete: callable | None = None,
) -> subprocess.Popen | None:
    """Run the mask_obj MVS command asynchronously in a background thread."""

    # 调用即启动：确保 colmap-dev 容器在跑（不常驻），失败则同步抛错
    _ensure_colmap_running()

    # 同步标记运行中：POST 返回时前端立即查询到 run_status=running
    _mask_obj_run_state[project_name] = {"status": "running", "exit_code": None}

    def _target():
        logger.info("Project %r: launching mask_obj docker exec...", project_name)

        # Ensure container log follower is running
        start_follow(CONTAINER_NAME)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # merge stderr → stdout for real-time line order
            text=True,
            bufsize=1,  # line-buffered
        )
        _running_mask_obj_processes[project_name] = proc

        # Prepare log paths
        out_dir = DATA_DIR_HOST / project_name / "algorithm_output" / "mask_obj"
        log_path = out_dir / "mask_obj.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Stream output line-by-line to both log files in real time
        DOCKER_LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as proj_f, \
             open(CONTAINER_LOG_PATH, "a", encoding="utf-8") as dock_f:

            header = f"\n{'='*60}\n=== MASK_OBJ START {project_name} | {' '.join(cmd)}\n{'='*60}\n"
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
            logger.info("Project %r: mask_obj docker exited with code %d",
                        project_name, proc.returncode)
            # 落定运行状态（幂等；get_mask_obj_run_status 也会在观察到期时补齐）
            _finalize_mask_obj_state(project_name, proc.returncode)

            footer = f"\n{'='*60}\n=== MASK_OBJ END {project_name} | exit_code: {proc.returncode}\n{'='*60}\n"
            proj_f.write(footer)
            proj_f.flush()
            dock_f.write(footer)
            dock_f.flush()

        if on_complete:
            on_complete(project_name, proc.returncode)

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(0.1)

    return _running_mask_obj_processes.get(project_name)


# ---------------------------------------------------------------------------
# Process registry
# ---------------------------------------------------------------------------

def get_mask_obj_process(project_name: str) -> subprocess.Popen | None:
    return _running_mask_obj_processes.get(project_name)


def remove_mask_obj_process(project_name: str) -> None:
    _running_mask_obj_processes.pop(project_name, None)


def is_mask_obj_running(project_name: str) -> bool:
    proc = _running_mask_obj_processes.get(project_name)
    if proc is None:
        return False
    return proc.poll() is None
