# SfM Platform API 文档

> Base URL: `http://localhost:8000`  
> Content-Type: `multipart/form-data`（POST 请求） / `application/json`（响应）

---

## 通用

### `GET /api/health`

健康检查。

**响应** `{status: "ok"}`

---

## 项目管理

### `GET /api/projects`

列出所有项目。

**响应**
```json
{
  "projects": [{
    "project_name": "项目名",
    "name": "显示名称",
    "status": "pending|running|completed|failed",
    "num_images": 35,
    "created_at": "ISO 时间",
    "elapsed_seconds": 123.4
  }]
}
```

### `POST /api/projects`

创建新项目（空目录骨架）。

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `name` | `str` | **必填** | 项目显示名称 |

**响应**
```json
{"success": true, "message": "Project xxx created", "data": {"project_name": "xxx"}}
```

### `GET /api/projects/{project_name}`

获取项目详情（含配置、pipeline 结果、日志尾部）。

**响应**
```json
{
  "project_name": "xxx",
  "name": "显示名",
  "status": "completed",
  "config": { "steps": [...], "matcher": "exhaustive", "camera_model": "SIMPLE_RADIAL", ... },
  "num_images": 35,
  "created_at": "...",
  "started_at": "...",
  "finished_at": "...",
  "error_message": null,
  "pipeline": { "elapsed_seconds": 123.4, ... },
  "steps": { "extract": {...}, "match": {...}, ... },
  "log_tail": "最后 80 行日志..."
}
```

### `DELETE /api/projects/{project_name}`

删除项目及所有数据。要求 pipeline 未在运行。

**响应** `{"success": true, "message": "Project xxx deleted"}`

---

## 图像

### `POST /api/projects/{project_name}/images`

上传图像（支持单个文件和 `.zip` 包）。自动过滤 `__MACOSX`，仅保留支持格式。上传后自动去除 EXIF 元数据。

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `images` | `list[UploadFile]` | **必填** | 图像文件或 `.zip` 包 |

支持的格式: `.jpg` `.jpeg` `.png` `.tif` `.tiff` `.bmp` `.pgm` `.ppm`

**响应**
```json
{"success": true, "message": "35 images in project", "data": {"num_images": 35}}
```

---

## 图像预处理

### `POST /api/projects/{project_name}/preprocess/undistort`

运行 COLMAP `image_undistorter`，将图像去畸变并转换为 PINHOLE 相机模型。

**前置条件**: SfM pipeline 已完成（`SFM_output/sparse/0/` 存在）

**无参数**

**输出产物**: `SFM_output/undistorted/`
```
undistorted/
├── images/          ← 去畸变图像
├── sparse/          ← PINHOLE 相机模型的 sparse 重建
└── stereo/          ← MVS 配置
```

**响应**
```json
{
  "success": true,
  "message": "Undistorted 35 images",
  "data": {"output_dir": "/home/zwz/Code/DATA/xxx/SFM_output/undistorted", "num_images": 35}
}
```

---

## SfM Pipeline

### `POST /api/projects/{project_name}/run`

配置并启动 SfM pipeline（extract → match → sfm → undistort → mvs）。

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `steps` | `str` | `extract,match,sfm,undistort,mvs` | 步骤列表（逗号分隔） |
| `matcher` | `str` | `exhaustive` | 匹配器: `exhaustive` `sequential` `vocab_tree` `spatial` |
| `camera_model` | `str` | `SIMPLE_RADIAL` | 相机模型: `SIMPLE_RADIAL` `PINHOLE` `SIMPLE_PINHOLE` `RADIAL` `OPENCV` |
| `single_camera` | `bool` | `true` | 所有图像共享同一相机内参 |
| `use_gpu` | `bool` | `true` | 使用 GPU |
| `gpu_index` | `int` | `0` | GPU 设备编号 |
| `max_num_features` | `int` | `8192` | 每张图最大 SIFT 特征数（0=默认） |
| `no_geom_consistency` | `bool` | `false` | 禁用几何一致性检查 |
| `reconstruction_name` | `str` | `0` | 重建名称 |
| `convert_output_suffix` | `str` | `txt` | model_converter 输出格式（`txt` 或 `bin`） |

**响应** `{"success": true, "message": "Pipeline started for xxx"}`

### `GET /api/projects/{project_name}/log`

获取 SfM pipeline 日志。

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `lines` | `int` | `100` | 返回尾部行数 |

**响应** `{"project_name": "xxx", "log": "日志内容..."}`

### `GET /api/projects/{project_name}/download/{filename:path}`

下载 SFM_output 目录下的文件。

| 参数 | 类型 | 说明 |
|------|------|------|
| `filename` | `path` | SFM_output 下的相对路径，如 `fused.ply`、`pipeline_result.json` |

### `GET /api/projects/{project_name}/sfm/files`

列出 SFM_output 里可下载的产物（SfM 页面下载区的数据源）。稀疏点云按
`sparse/<重建名>/` 逐个列出，重建名不做限制（COLMAP 可能产出 0/1/… 多个模型）。

| 字段 | 类型 | 说明 |
|------|------|------|
| `step` | `str` | 产物所属步骤，如 `稀疏重建` / `运行结果` / `运行日志` |
| `label` | `str` | 展示用名称（稀疏点云附带点数） |
| `path` | `str` | 相对 SFM_output 的路径，直接用于下载端点 |
| `generated` | `bool` | 仅稀疏点云有：PLY 是否已落盘（未落盘时首次下载会自动导出） |

**响应**
```json
{
  "project_name": "xxx",
  "products": [
    {"step": "稀疏重建", "label": "稀疏点云 0（11,838 点）",
     "path": "sparse/0/points3D.ply", "generated": false},
    {"step": "稀疏重建", "label": "sparse/0/points3D.bin", "path": "sparse/0/points3D.bin"},
    {"step": "运行日志", "label": "pipeline.log", "path": "pipeline.log"}
  ]
}
```

### `GET /api/projects/{project_name}/sfm/download/{filename:path}`

下载 SFM_output 目录下的产物，与上面的通用下载端点同义，额外支持**按需导出稀疏点云**。

| 参数 | 类型 | 说明 |
|------|------|------|
| `filename` | `path` | SFM_output 下的相对路径，如 `sparse/0/points3D.ply` |

`sparse/<重建名>/points3D.ply` 是派生产物：磁盘上还没有时，后端用同目录的 COLMAP
稀疏模型（`points3D.bin` / `points3D.txt`）在进程内就地导出（不经容器，见
`algorithm/common.py:ensure_sparse_ply`），随后作为普通文件回传；源模型更新后再次下载会
自动重新导出。源模型不存在时返回 404。

---

## SAM

### `POST /api/projects/{project_name}/sam/run`

运行 SAM3 推理。

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `prompts` | `str` | `object` | 文本提示词（逗号分隔），如 `"car,person,dog"` |
| `confidence` | `float` | `0.5` | 检测置信度阈值（0.0–1.0） |
| `save_masks` | `bool` | `true` | 导出每张 mask 为独立 PNG |
| `save_masks_rle` | `bool` | `false` | 在 JSON 中包含 RLE 编码 |
| `vis` | `bool` | `false` | 生成 matplotlib 可视化（内存开销大） |
| `batch_size` | `int` | `1` | 骨架批处理批量大小 |
| `device` | `str` | `cuda` | 设备: `cuda` 或 `cpu` |

**响应**
```json
{"success": true, "message": "SAM inference started for xxx", "data": {"prompts": [...], "confidence": 0.5, "save_masks": true}}
```

### `GET /api/projects/{project_name}/sam/result`

获取 SAM 推理结果。

**响应**
```json
{
  "project_name": "xxx",
  "ready": true,
  "total_images": 35,
  "total_prompts": 2,
  "results": [...]
}
```

### `GET /api/projects/{project_name}/sam/log`

获取 SAM 推理日志。

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `lines` | `int` | `100` | 返回尾部行数 |

---

## MVS

### `POST /api/projects/{project_name}/mvs/run`

运行 OpenMVS pipeline。

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `command` | `str` | `pipeline` | MVS 命令: `pipeline` `interface` `densify` `reconstruct` `refine` `texture` `check` |
| `model` | `str` | `0` | SfM 模型名称 |
| `resolution_level` | `int` | `1` | 下采样级别 |
| `max_resolution` | `int` | `3200` | 最大分辨率 |
| `min_resolution` | `int` | `640` | 最小分辨率 |
| `number_views` | `int` | `5` | 深度估计视图数 |
| `no_cuda` | `bool` | `false` | 禁用 CUDA |
| `refine_resolution` | `int` | `1` | 精化分辨率级别 |
| `scales` | `int` | `2` | 多尺度数量 |
| `gradient_step` | `float` | `45.0` | 梯度步长 |
| `texture_resolution` | `int` | `1` | 纹理分辨率级别 |
| `export_type` | `str` | `ply` | 导出格式: `ply` 或 `obj` |

**响应** `{"success": true, "message": "MVS pipeline started for xxx", "data": {"command": "pipeline"}}`

### `GET /api/projects/{project_name}/mvs/log`

获取 MVS pipeline 日志。

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `lines` | `int` | `100` | 返回尾部行数 |

### `GET /api/projects/{project_name}/mvs/result`

获取 MVS 结果。

**响应**
```json
{
  "project_name": "xxx",
  "ready": true,
  "status": "ok",
  "project": {...},
  "steps": {...},
  "total_elapsed": 123.4
}
```

### `GET /api/projects/{project_name}/mvs/download/{filename:path}`

下载 MVS_output 目录下的文件。

| 参数 | 类型 | 说明 |
|------|------|------|
| `filename` | `path` | MVS_output 下的相对路径，如 `scene_dense_mesh.ply` |

---

## 3DGS

### `POST /api/projects/{project_name}/3dgs/run`

运行 3D Gaussian Splatting pipeline。

**前置条件**（train/full）: SfM 已完成（`SFM_output/sparse/0/` 存在）

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `command` | `str` | `full` | 子命令: `prepare` `train` `render` `full` |
| `iterations` | `int` | `30000` | 训练迭代数 |
| `resolution` | `int` | `1` | 图像缩放倍率（1=原始分辨率） |
| `white_background` | `bool` | `false` | 白背景 |
| `quiet` | `bool` | `false` | 安静模式 |
| `checkpoint_interval` | `int` | `0` | 每 N 迭代保存 checkpoint（0=仅默认 7000/30000） |
| `start_checkpoint` | `str` | `""` | 从 .pth checkpoint 恢复训练 |

**响应**
```json
{"success": true, "message": "3DGS full started for xxx", "data": {"command": "full", "iterations": 30000, ...}}
```

### `GET /api/projects/{project_name}/3dgs/log`

获取 3DGS pipeline 日志。

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `lines` | `int` | `100` | 返回尾部行数 |

### `GET /api/projects/{project_name}/3dgs/result`

获取 3DGS 结果（渲染图列表、是否有 point cloud）。

**响应**
```json
{
  "project_name": "xxx",
  "ready": true,
  "renders_count": 35,
  "renders": ["00000.png", "00001.png", ...],
  "has_point_cloud": true
}
```

### `GET /api/projects/{project_name}/3dgs/download/{filename:path}`

下载 3DGS_output 目录下的文件。

| 参数 | 类型 | 说明 |
|------|------|------|
| `filename` | `path` | 3DGS_output 下的相对路径，如 `point_cloud/point_cloud.ply` |

---

## 枚举速查

| 枚举 | 可选值 |
|------|--------|
| `TaskStatus` | `pending` `running` `completed` `failed` |
| `MatcherType` | `exhaustive` `sequential` `vocab_tree` `spatial` |
| `CameraModel` | `SIMPLE_RADIAL` `PINHOLE` `SIMPLE_PINHOLE` `RADIAL` `OPENCV` |
| `PipelineStep` | `extract` `match` `sfm` `undistort` `mvs` `fusion` `convert` |
| `MVSCommand` | `pipeline` `interface` `densify` `reconstruct` `refine` `texture` `check` |
| 3DGS command | `prepare` `train` `render` `full` |
