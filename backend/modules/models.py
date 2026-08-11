"""Pydantic models for the SfM Platform API."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class MatcherType(str, Enum):
    EXHAUSTIVE = "exhaustive"
    SEQUENTIAL = "sequential"
    VOCAB_TREE = "vocab_tree"
    SPATIAL = "spatial"


class CameraModel(str, Enum):
    SIMPLE_RADIAL = "SIMPLE_RADIAL"
    PINHOLE = "PINHOLE"
    SIMPLE_PINHOLE = "SIMPLE_PINHOLE"
    RADIAL = "RADIAL"
    OPENCV = "OPENCV"


class PipelineStep(str, Enum):
    EXTRACT = "extract"
    MATCH = "match"
    SFM = "sfm"
    UNDISTORT = "undistort"
    MVS = "mvs"
    FUSION = "fusion"
    CONVERT = "convert"


class MVSType(str, Enum):
    """Which MVS pipeline to use."""
    COLMAP = "colmap"
    OPENMVS = "openmvs"


class MVSCommand(str, Enum):
    PIPELINE = "pipeline"
    INTERFACE = "interface"
    DENSIFY = "densify"
    RECONSTRUCT = "reconstruct"
    REFINE = "refine"
    TEXTURE = "texture"
    CHECK = "check"


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class PipelineConfig(BaseModel):
    """Configuration for the SfM pipeline."""
    steps: list[PipelineStep] = Field(
        default=[PipelineStep.EXTRACT, PipelineStep.MATCH, PipelineStep.SFM,
                 PipelineStep.UNDISTORT, PipelineStep.MVS],
        description="Pipeline steps to run",
    )
    mvs_type: MVSType = Field(default=MVSType.COLMAP, description="MVS pipeline for the 'mvs' step")
    matcher: MatcherType = Field(default=MatcherType.EXHAUSTIVE)
    camera_model: CameraModel = Field(default=CameraModel.SIMPLE_RADIAL)
    single_camera: bool = Field(default=True)
    use_gpu: bool = Field(default=True)
    gpu_index: int = Field(default=0, ge=0)
    max_num_features: int = Field(default=8192, ge=0, description="Max SIFT features per image (0=use COLMAP default)")
    no_geom_consistency: bool = Field(default=False)
    reconstruction_name: str = Field(default="0")
    vocab_tree_path: Optional[str] = Field(default=None)
    convert_output_suffix: str = Field(default="txt", description="Output format for model_converter step (txt or bin)")


class MVSConfig(BaseModel):
    """Configuration for the MVS pipeline (COLMAP or OpenMVS)."""
    mvs_type: MVSType = Field(default=MVSType.OPENMVS, description="MVS pipeline type")
    command: MVSCommand = Field(default=MVSCommand.PIPELINE)
    model: str = Field(default="0", description="SFM model name")
    resolution_level: int = Field(default=1, ge=0, description="Downsample level (0=full, 1=half, 2=quarter)")
    max_resolution: int = Field(default=3200, ge=1)
    min_resolution: int = Field(default=640, ge=1)
    number_views: int = Field(default=5, ge=1, description="Views for depth estimation")
    no_cuda: bool = Field(default=False)
    refine_resolution: int = Field(default=1, ge=1)
    scales: int = Field(default=2, ge=1)
    gradient_step: float = Field(default=45.0)
    texture_resolution: int = Field(default=1, ge=1)
    export_type: str = Field(default="ply")


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class TaskSummary(BaseModel):
    """Summary of a task for list views."""
    task_id: str
    name: str
    status: TaskStatus
    num_images: int = 0
    created_at: str = ""
    elapsed_seconds: Optional[float] = None


class StepResult(BaseModel):
    status: str
    num_images: Optional[int] = None
    matcher: Optional[str] = None
    num_reconstructions: Optional[int] = None
    fused_ply: Optional[str] = None
    fused_ply_size_mb: Optional[float] = None


class OutputInfo(BaseModel):
    output_dir: str = ""
    database_path: str = ""
    sparse_dir: str = ""
    dense_dir: str = ""
    sparse_models: list[dict] = Field(default_factory=list)
    dense_outputs: dict = Field(default_factory=dict)


class PipelineInfo(BaseModel):
    version: str = ""
    timestamp: str = ""
    command: str = ""
    steps_requested: list[str] = Field(default_factory=list)
    steps_completed: list[str] = Field(default_factory=list)
    elapsed_seconds: float = 0.0


class TaskDetail(BaseModel):
    """Full detail of a task including results."""
    task_id: str
    name: str
    status: TaskStatus
    config: PipelineConfig = Field(default_factory=PipelineConfig)
    num_images: int = 0
    created_at: str = ""
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error_message: Optional[str] = None
    pipeline: Optional[PipelineInfo] = None
    output: Optional[OutputInfo] = None
    steps: dict[str, Any] = Field(default_factory=dict)
    log_tail: Optional[str] = None


class TaskListResponse(BaseModel):
    tasks: list[TaskSummary]


class ApiResponse(BaseModel):
    success: bool
    message: str = ""
    data: Optional[Any] = None
