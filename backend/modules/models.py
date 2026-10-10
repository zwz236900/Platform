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
# COLMAP SfM (sparse) tunables
# ---------------------------------------------------------------------------

#: Maps a ColmapSFMParams field to the CLI flag sfm_cli.py expects. The CLI then
#: forwards it as the COLMAP option named in SfM_PARAM_GROUPS there -- e.g.
#: "mapper_filter_min_tri_angle" becomes --Mapper.filter_min_tri_angle.
#:
#: Names and bounds were taken from the real binary (`colmap <cmd> -h`,
#: COLMAP 4.2.0.dev0), not from documentation: 4.2 has no
#: Mapper.triangulation.* and no Mapper.ba_refine_extrinsics, and the max image
#: size lives under FeatureExtraction while the rest of the SIFT knobs live
#: under SiftExtraction.
#:
#: Kept here as the single source of truth so the API layer, the stored config
#: and the CLI cannot drift apart.
COLMAP_SFM_CLI_FLAGS: dict[str, str] = {
    # Feature extraction
    "sift_max_image_size": "--sift_max_image_size",
    "sift_first_octave": "--sift_first_octave",
    "sift_num_octaves": "--sift_num_octaves",
    "sift_peak_threshold": "--sift_peak_threshold",
    "sift_edge_threshold": "--sift_edge_threshold",
    # Feature matching
    "match_max_ratio": "--match_max_ratio",
    "match_cross_check": "--match_cross_check",
    # Two-view geometry
    "tv_max_error": "--tv_max_error",
    "tv_min_num_inliers": "--tv_min_num_inliers",
    # Sequential matching
    "seq_overlap": "--seq_overlap",
    "seq_loop_detection": "--seq_loop_detection",
    # Mapper registration / triangulation
    "mapper_min_num_matches": "--mapper_min_num_matches",
    "mapper_init_min_num_inliers": "--mapper_init_min_num_inliers",
    "mapper_abs_pose_min_num_inliers": "--mapper_abs_pose_min_num_inliers",
    "mapper_max_reg_trials": "--mapper_max_reg_trials",
    "mapper_ignore_two_view_tracks": "--mapper_ignore_two_view_tracks",
    "mapper_filter_max_reproj_error": "--mapper_filter_max_reproj_error",
    "mapper_filter_min_tri_angle": "--mapper_filter_min_tri_angle",
    # Mapper bundle adjustment
    "mapper_ba_refine_focal_length": "--mapper_ba_refine_focal_length",
    "mapper_ba_refine_principal_point": "--mapper_ba_refine_principal_point",
    "mapper_ba_refine_extra_params": "--mapper_ba_refine_extra_params",
    "mapper_ba_local_num_images": "--mapper_ba_local_num_images",
    "mapper_ba_local_max_num_iterations": "--mapper_ba_local_max_num_iterations",
    "mapper_ba_global_max_num_iterations": "--mapper_ba_global_max_num_iterations",
}


class ColmapSFMParams(BaseModel):
    """Tunables for the COLMAP sparse-reconstruction stage.

    Every field defaults to ``None``, meaning "don't pass the flag -- let COLMAP
    use its own default". A run that sets nothing therefore behaves exactly like
    the hardcoded pipeline did before these knobs existed.

    Bounds mirror the ``-h`` output of the COLMAP 4.2.0.dev0 binary this platform
    ships. Unlike the MVS options there is no cross-option ``CHECK_OPTION`` to
    mirror here, so there is no model validator.

    NOTE: like ``ColmapMVSParams`` this is deliberately NOT used as a FastAPI
    ``Form()`` model. Binding a form model alongside the scalar ``Form(...)``
    parameters of the endpoints makes FastAPI embed it as one nested field and
    silently drop the flat fields the client sends. The endpoints take flat
    scalars and build this model explicitly instead.
    """

    # ---- Feature extraction ----
    sift_max_image_size: Optional[int] = Field(
        default=None, ge=-1, description="特征提取前图像最大边长（-1 = 不限制）")
    sift_first_octave: Optional[int] = Field(
        default=None, ge=-1, description="SIFT 起始金字塔层（-1 = 先上采样）")
    sift_num_octaves: Optional[int] = Field(
        default=None, ge=1, description="SIFT 高斯金字塔组数")
    sift_peak_threshold: Optional[float] = Field(
        default=None, ge=0, description="特征点检测峰值阈值（越小特征越密）")
    sift_edge_threshold: Optional[float] = Field(
        default=None, ge=0, description="SIFT 边缘响应阈值")

    # ---- Feature matching ----
    match_max_ratio: Optional[float] = Field(
        default=None, ge=0, le=1, description="最近邻/次近邻距离比（越小越严格）")
    match_cross_check: Optional[int] = Field(
        default=None, ge=0, le=1, description="双向交叉验证（1/0）")

    # ---- Two-view geometry ----
    tv_max_error: Optional[float] = Field(
        default=None, ge=0, description="RANSAC 最大重投影误差，像素")
    tv_min_num_inliers: Optional[int] = Field(
        default=None, ge=0, description="图像对有效所需最小内点数")

    # ---- Sequential matching (only used by --matcher sequential) ----
    seq_overlap: Optional[int] = Field(
        default=None, ge=0, description="每张图匹配的相邻图像数")
    seq_loop_detection: Optional[int] = Field(
        default=None, ge=0, le=1, description="回环检测（1/0）")

    # ---- Mapper registration / triangulation ----
    mapper_min_num_matches: Optional[int] = Field(
        default=None, ge=0, description="图像对进入重建所需最小匹配数")
    mapper_init_min_num_inliers: Optional[int] = Field(
        default=None, ge=0, description="初始化两视图所需最小内点数")
    mapper_abs_pose_min_num_inliers: Optional[int] = Field(
        default=None, ge=0, description="新图像注册（PnP）最小内点数")
    mapper_max_reg_trials: Optional[int] = Field(
        default=None, ge=0, description="单张图像最大注册尝试次数")
    mapper_ignore_two_view_tracks: Optional[int] = Field(
        default=None, ge=0, le=1,
        description="忽略仅两视图可见的 3D 点（1/0；关掉可增密但引入噪声）")
    mapper_filter_max_reproj_error: Optional[float] = Field(
        default=None, ge=0, description="三角化最大重投影误差，像素")
    mapper_filter_min_tri_angle: Optional[float] = Field(
        default=None, ge=0, lt=90, description="最小三角化角，度")

    # ---- Mapper bundle adjustment ----
    mapper_ba_refine_focal_length: Optional[int] = Field(
        default=None, ge=0, le=1, description="精修焦距（1/0）")
    mapper_ba_refine_principal_point: Optional[int] = Field(
        default=None, ge=0, le=1, description="精修主点（1/0；少图场景易发散）")
    mapper_ba_refine_extra_params: Optional[int] = Field(
        default=None, ge=0, le=1, description="精修畸变参数（1/0）")
    mapper_ba_local_num_images: Optional[int] = Field(
        default=None, ge=1, description="局部 BA 使用的邻近图像数")
    mapper_ba_local_max_num_iterations: Optional[int] = Field(
        default=None, ge=-1, description="局部 BA 最大迭代数（-1 = 自动）")
    mapper_ba_global_max_num_iterations: Optional[int] = Field(
        default=None, ge=-1, description="全局 BA 最大迭代数（-1 = 自动）")

    def cli_args(self) -> list[str]:
        """Render the set fields as CLI flags for sfm_cli.py.

        Unset (None) fields are omitted entirely -- that omission is what makes
        COLMAP fall back to its own default.
        """
        args: list[str] = []
        for field, flag in COLMAP_SFM_CLI_FLAGS.items():
            value = getattr(self, field)
            if value is None:
                continue
            args += [flag, str(value)]
        return args

    def is_empty(self) -> bool:
        """True when nothing was overridden (the pre-existing behaviour)."""
        return all(getattr(self, f) is None for f in COLMAP_SFM_CLI_FLAGS)


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
