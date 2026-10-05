"""Pydantic models for the SfM Platform API."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


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
# COLMAP MVS tunables
# ---------------------------------------------------------------------------

#: Maps a ColmapMVSParams field to the CLI flag sfm_cli.py expects. The CLI then
#: turns it into the COLMAP option -- e.g. "pms_window_radius" is passed to the
#: CLI as --pms_window_radius and forwarded as --PatchMatchStereo.window_radius.
#: Kept here as the single source of truth so the API layer, the stored config
#: and the CLI cannot drift apart.
COLMAP_MVS_CLI_FLAGS: dict[str, str] = {
    # PatchMatch Stereo
    "pms_max_image_size": "--pms_max_image_size",
    "pms_window_radius": "--pms_window_radius",
    "pms_window_step": "--pms_window_step",
    "pms_num_samples": "--pms_num_samples",
    "pms_num_iterations": "--pms_num_iterations",
    "pms_cache_size": "--pms_cache_size",
    "pms_geom_consistency_max_cost": "--pms_geom_consistency_max_cost",
    "pms_filter": "--pms_filter",
    "pms_filter_min_ncc": "--pms_filter_min_ncc",
    "pms_filter_min_triangulation_angle": "--pms_filter_min_triangulation_angle",
    "pms_filter_min_num_consistent": "--pms_filter_min_num_consistent",
    "pms_filter_geom_consistency_max_cost": "--pms_filter_geom_consistency_max_cost",
    # Stereo Fusion
    "fusion_input_type": "--fusion_input_type",
    "fusion_max_image_size": "--fusion_max_image_size",
    "fusion_min_num_pixels": "--fusion_min_num_pixels",
    "fusion_max_num_pixels": "--fusion_max_num_pixels",
    "fusion_max_reproj_error": "--fusion_max_reproj_error",
    "fusion_max_depth_error": "--fusion_max_depth_error",
    "fusion_max_normal_error": "--fusion_max_normal_error",
    "fusion_check_num_images": "--fusion_check_num_images",
}


class ColmapMVSParams(BaseModel):
    """Tunables for the COLMAP dense-reconstruction stage.

    Every field defaults to ``None``, meaning "don't pass the flag -- let COLMAP
    use its own default". A run that sets nothing therefore behaves exactly like
    the hardcoded pipeline did before these knobs existed.

    Bounds mirror the ``Check()`` methods in COLMAP's own
    ``mvs/patch_match_options.cc`` and ``mvs/fusion.cc`` so a bad value is
    rejected here with a clear message instead of aborting inside COLMAP.

    NOTE: this is deliberately NOT used as a FastAPI ``Form()`` model. Binding a
    form model alongside the other scalar ``Form(...)`` parameters of
    ``/mvs/run`` makes FastAPI embed the model as one nested field and silently
    drop the flat fields the client sends. The endpoints take flat scalars and
    build this model explicitly instead.
    """

    # ---- PatchMatch Stereo ----
    pms_max_image_size: Optional[int] = Field(
        default=None, ge=-1, description="Max image edge PatchMatch works on (-1 = no limit)")
    pms_window_radius: Optional[int] = Field(
        default=None, ge=1, le=32, description="Matching window radius (COLMAP max 32)")
    pms_window_step: Optional[int] = Field(
        default=None, ge=1, le=2, description="Window slide step (COLMAP allows 1 or 2)")
    pms_num_samples: Optional[int] = Field(
        default=None, ge=1, description="Number of depth samples per iteration")
    pms_num_iterations: Optional[int] = Field(
        default=None, ge=1, description="PatchMatch iterations")
    pms_cache_size: Optional[int] = Field(
        default=None, ge=1, description="Image/depth-map cache size in GB")
    pms_geom_consistency_max_cost: Optional[float] = Field(
        default=None, ge=0, description="Max cost for geometric consistency")
    pms_filter: Optional[int] = Field(
        default=None, ge=0, le=1, description="Enable depth-map filtering (1/0)")
    pms_filter_min_ncc: Optional[float] = Field(
        default=None, ge=-1, le=1, description="Min NCC to keep a pixel")
    pms_filter_min_triangulation_angle: Optional[float] = Field(
        default=None, ge=0, lt=180, description="Min triangulation angle in degrees")
    pms_filter_min_num_consistent: Optional[int] = Field(
        default=None, ge=0, description="Min number of consistent views")
    pms_filter_geom_consistency_max_cost: Optional[float] = Field(
        default=None, ge=0, description="Max cost for filtered geometric consistency")

    # ---- Stereo Fusion ----
    fusion_input_type: Optional[Literal["auto", "geometric", "photometric"]] = Field(
        default=None,
        description=(
            "'auto' follows geom_consistency (geometric when on, photometric when "
            "off); naming the wrong one fuses nothing."
        ),
    )
    fusion_max_image_size: Optional[int] = Field(
        default=None, ge=-1, description="Max image edge fusion works on (-1 = no limit)")
    fusion_min_num_pixels: Optional[int] = Field(
        default=None, ge=0, description="Min pixel observations for a 3D point")
    fusion_max_num_pixels: Optional[int] = Field(
        default=None, ge=1, description="Max pixel observations for a 3D point")
    fusion_max_reproj_error: Optional[float] = Field(
        default=None, ge=0, description="Max reprojection error in pixels")
    fusion_max_depth_error: Optional[float] = Field(
        default=None, ge=0, description="Max relative depth error")
    fusion_max_normal_error: Optional[float] = Field(
        default=None, ge=0, description="Max normal angle error in degrees")
    fusion_check_num_images: Optional[int] = Field(
        default=None, ge=1, description="Number of images checked during fusion")

    @model_validator(mode="after")
    def _check_pixel_bounds(self) -> "ColmapMVSParams":
        """Mirror COLMAP's CHECK_OPTION_LE(min_num_pixels, max_num_pixels).

        Only meaningful when both ends were given; leaving either unset means
        COLMAP's default is in play, so there is nothing to compare.
        """
        lo, hi = self.fusion_min_num_pixels, self.fusion_max_num_pixels
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(
                f"fusion_min_num_pixels ({lo}) must be <= fusion_max_num_pixels ({hi})")
        return self

    def cli_args(self) -> list[str]:
        """Render the set fields as CLI flags for sfm_cli.py.

        Unset (None) fields are omitted entirely -- that omission is what makes
        COLMAP fall back to its own default.
        """
        args: list[str] = []
        for field, flag in COLMAP_MVS_CLI_FLAGS.items():
            value = getattr(self, field)
            if value is None:
                continue
            args += [flag, str(value)]
        return args

    def is_empty(self) -> bool:
        """True when nothing was overridden (the pre-existing behaviour)."""
        return all(getattr(self, f) is None for f in COLMAP_MVS_CLI_FLAGS)


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
