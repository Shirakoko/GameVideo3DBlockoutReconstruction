# Project: Game Video to 3D Blockout Reconstruction Agent

Stage G / Pipeline 进展（2026-11-24）：已实现米制 Chamfer Distance / F-score / 观测表面覆盖、Camera Pose / ATE / RPE / 帧同步、语义 P/R / 类别 / 静动态 / ID Switch 和 SSIM；LPIPS 提供仅使用显式本地权重的可选接口，缺依赖/权重明确 N/A。报告包括 JSON / 离线 HTML / CSV 和阶段耗时、process-tree CPU/RSS、device-wide NVIDIA GPU、失败/重试信息。A–G 分阶段可配置 Pipeline、哈希缓存、断点恢复、输出保护、进程树 timeout 和有界重试已实现；合成 12 秒 / 60 FPS / 720 帧完成端到端执行，真实 Session 仍因 metric Depth / render-pose 合同未验证而停止 B–F，并生成 partial 报告，不标记真实 A–G 验收。完整渲染 Pose 使用重开 `.blend` 中实际相机逐帧导出，不把输入 Track 比对伪装为独立估计精度。详情见 `docs/stage_g.md`。

用户授权的启发式代理（2026-11-24）：已允许超出验证几何范围按原视频布局补齐。独立 `reconstruction.heuristic_proxy --allow-heuristic` 使用50 cm体素，完成连续地面、环境体积及全段相机自由走廊；1401个可编辑区块、九个预览、720帧相机/场景保存重开检查通过。最终场景与两个MP4已生成于 `outputs/stage_ef/heuristic_voxel_proxy/`，均保持60 FPS/720帧/12秒时间轴；Stage E/F和哈希验证后的重复运行复用通过，精简后全项目145/145测试通过。所有补齐几何明确为heuristic/inferred，原物理深度与真实Stage A/B未验收状态不变。详见 `docs/heuristic_proxy.md`。

基本体粗模改为从原视频观测生成，不读体素占用：`reconstruction.primitive_blockout --allow-heuristic` 反投影 RGB/深度码/语义点，拟合方块/圆柱/球体，结果写到 `outputs/stage_ef/heuristic_primitive_from_video/`。体素目录和旧灰模目录保持不动。

当前进展（2026-11-24）：Stage B 反投影、Open3D 世界对齐/体素融合/SOR、重复观测计权、PLY 与来源记录已实现并通过合成验证；独立 Blender 数学检查通过。真实 Session 的 v3 深度物理规格、SDK 旋转字段绑定及 depth render-pose 时刻仍未确认，Stage A / 真实 Stage B 不标记验收。详见 `docs/stage_b.md` 和 `outputs/stage_b/sample_66ad/validation_report.md`。

Stage C/D 进展（2026-11-24）：五类辅助 RGB/Mask 理解、米制深度/逐像素点云的对象几何、持久 ID、静动态/空间关系、带版本 Scene Schema 与 Validation 已实现并通过五类合成端到端测试；真实 RGB-only 示例保留 inferred/unobserved，未标记真实自动识别精度或真实几何验收。可选本地 Grounding DINO/SAM 接口尚未实机验证。详见 `docs/stage_cd.md`；不扩大下载。

Stage E/F 进展（2026-11-24）：Blender Python API / Background Mode 粗模场景、材质、Collections、原始/稀疏相机 Pose、物体动画、准确 FPS/源帧映射、保存后重开验证、MP4/左右对比、可重复执行与批量任务已实现。已补齐道路/墙壁/建筑类别与材质、源 Scene JSON 快照、损坏缓存/批量参数防护及 FFmpeg 超时日志；Stage E/F 26 项、原全项目 115 项测试通过、无跳过。Blender 5.2.1 LTS 合成 12 秒/60 FPS/720 帧及双分辨率批量渲染重新验收，产物在 `outputs/stage_ef/synthetic_demo/`；真实 partial 输入因未验证几何/Pose 合同明确拒绝，未声称真实 Stage A-F 全流程验收。详见 `docs/stage_ef.md`。

低精度优化路线（2026-11-24）：按用户最新选择采用彩色体素环境粗模，不追求高精度贴图/人物动画。已实现 Scene 1.1.0 显式 inferred 分支、本地模型锁和 61 帧推理、经验相机/深度关联校准、TSDF 可选路线、独立观测体素壳/内部面剔除/颜色调色板、后台网格导入打包/预览及哈希缓存。原深度验证仍失败；独立 RGB 三视图在 627 个留出点上通过（中位数 1.99 px、P90 7.05 px），原始焦距/快照关联的调整仍标为 inferred。已生成 125 区块表面版、25 区块实体版及修复完整表面可见性后的 65 区块实体 `.blend`/九帧预览；前两版图像质量失败，最后一版虽 SSIM 较高但 261/720 帧相机位于候选实体内，新增穿模检查拒绝完整 MP4。不能把保存重开或 SSIM 通过等同于可用真实粗模。原物理合同路径保持严格。详见 `docs/refined_blockout.md`。

## 1. 项目目标

开发一套基于 Python + Blender 的游戏视频 3D 场景粗模重建系统。

输入一段 10–15 秒的游戏录屏，以及可选的 Depth、Camera Pose 等辅助数据，通过自动化 Agent 工作流，重建游戏场景的基础几何结构、物体布局和摄像机运动轨迹。

最终使用 Blender 自动生成一个可编辑的 3D 粗模场景，并渲染出与原视频时间轴逐秒对齐的 Blockout Video（导演台视频）。

注意：

- 不使用 Unreal Engine。
- Blender 仅作为 3D 场景生成、动画和渲染工具。
- 核心算法使用 Python 实现。
- 场景描述必须采用结构化 JSON Schema。
- 整个工作流尽可能自动化，避免人工手动摆放模型。

## 2. 数据来源

使用 Hugging Face 数据集：

https://huggingface.co/datasets/originlab/game-recordings-v4

优先使用 samples/ 中的小规模示例 Session，不要下载整个 5.3TB 数据集。

选择一个满足以下条件的 Session：

- RGB 视频完整。
- Depth 数据完整。
- Camera Pose 数据完整。
- 三种数据具有可验证的帧同步关系。
- Depth 最好是 Metric Depth，而不是 Relative Depth。

从中截取连续 10–15 秒的视频。

数据集中的主要文件：

- video/prehud.mp4
- video/posthud.mp4
- depth/depth.hevc
- depth/depth_meta.jsonl
- depth/decode_contract.json
- telemetry/camera.jsonl
- telemetry/world.json
- tables/frames.parquet
- session.json

必须先阅读数据集 README、session.json、decode_contract.json 和 depth_meta.jsonl，理解具体数据格式。

不要假设不同 Session 的数据结构、深度单位、相机坐标系和分辨率完全一致。

特别注意：

- Depth 使用 10-bit HEVC 和 Log Depth 编码。
- 必须遵循对应 Session 的 depth_transform。
- 读取 near_plane_cm 和相机内参。
- 处理天空、Far Clip 和无效深度值。
- 读取 depth 解码帧数和重复帧统计。
- Camera Pose 必须与 RGB Frame Index 正确关联。
- 检查相机坐标系的 Handedness、Up Axis 和 World Scale。
- 不要简单假设 Camera Pose 的位置和旋转格式可以直接映射到 Blender。

遵守数据集的访问、评估、保留和删除条款。不要将受限数据上传至未经授权的第三方服务。

## 3. 技术栈

核心技术：

- Python 3.11+
- NumPy
- OpenCV
- FFmpeg
- Open3D
- Pandas / PyArrow
- Blender Python API
- JSON Schema
- Matplotlib

后续可选 AI 模块：

- GPT Vision：场景语义理解
- Grounding DINO：物体检测
- SAM 2：视频物体分割
- Depth Anything：RGB-only 深度估计

第一版优先使用数据集提供的 Ground Truth Depth 和 Camera Pose。

不需要训练模型。

## 4. 核心工作流

### Stage A：Dataset Preparation

1. 检查数据集目录。
2. 读取 Session Metadata。
3. 自动检查 RGB、Depth、Camera Pose 的完整性。
4. 截取 10–15 秒连续视频。
5. 解码 Depth HEVC。
6. 读取逐帧 Camera Pose。
7. 统一转换成 Frame Index / Timestamp。
8. 生成数据同步报告。

输出：

- RGB Frames
- Depth Frames
- Camera Pose JSON
- Dataset Validation Report

### Stage B：3D Reconstruction

实现 Depth Back-projection。

根据相机内参 K，将像素坐标和深度转换为相机坐标系中的三维点：

P_camera = D(u,v) * inverse(K) * [u,v,1]^T

再利用 Camera Extrinsics 转换为世界坐标：

P_world = T_world_camera * P_camera

注意：

1. 正确处理矩阵乘法方向。
2. 处理 Blender 与数据集之间的坐标系转换。
3. 处理深度尺度和世界单位。
4. 过滤无效深度。
5. 过滤离群点。
6. 对多帧点云进行世界坐标对齐和融合。
7. 避免重复帧造成点云权重异常。
8. 保留原始帧索引和时间信息。

使用 Open3D 完成：

- Point Cloud Generation
- Point Cloud Transformation
- Voxel Downsampling
- Statistical Outlier Removal
- Multi-frame Fusion

输出：

- scene_pointcloud.ply
- reconstruction_metadata.json

### Stage C：Scene Understanding

使用视觉模型分析关键帧。

提取：

- Scene Type
- Object Category
- Object Bounding Box
- Object Mask
- Static / Dynamic Classification
- Object Position
- Object Dimensions
- Object Motion
- Spatial Relationships

同一个物体必须在不同帧中保持 Persistent Object ID。

不要只依赖 VLM 生成物体位置。物体位置和尺寸应尽可能从深度图、Mask 和几何计算获得。

低置信度的推断必须标记为 Inferred，而不能伪装成 Ground Truth。

这一阶段允许先实现简单规则和人工辅助验证，再逐步接入 VLM、Grounding DINO 和 SAM 2。

### Stage D：Scene Schema

定义统一的 JSON Schema。

每个物体至少包括：

- object_id
- category
- primitive_type
- position
- rotation
- scale
- dimensions
- parent_id
- is_static
- confidence
- geometry_source
- source_frames
- motion_track
- visibility

场景级数据包括：

- scene_id
- coordinate_system
- unit_scale
- objects
- cameras
- timeline
- reconstruction_metadata

需要支持：

- Static Objects
- Dynamic Objects
- Camera Animation
- Object Animation
- Multiple Cameras

Schema 必须具备版本号，并实现 JSON Schema Validation。

### Stage E：Blender Scene Generation

通过 Blender Python API 自动生成场景。

要求：

1. 自动创建 Blender Scene。
2. 创建地面、道路、墙壁、建筑等基础几何体。
3. 根据 JSON Transform 创建物体。
4. 使用简单 Primitive Mesh 表示物体。
5. 使用简单材质区分不同物体类别。
6. 将静态物体和动态物体分别组织到 Collection。
7. 导入或生成融合点云，作为空间参考。
8. 创建 Camera。
9. 根据原始 Camera Pose 设置摄像机位置和朝向。
10. 根据时间戳生成摄像机关键帧。
11. 根据物体 Motion Track 生成动画。
12. 设置正确的 FPS、Frame Range 和 Timeline。

建议的视觉风格：

- Greybox / Blockout
- Untextured Geometry
- Neutral Gray Materials
- Simple Lighting
- Clear Object Silhouettes
- No Complex PBR Materials
- No Decorative Details

优先保证空间布局、几何关系和摄像机运动准确。

输出：

- scene.blend
- scene_schema.json

### Stage F：Video Rendering

使用 Blender Background Mode 自动渲染。

要求：

- 与原视频相同的时间长度。
- 默认 60 FPS。
- 保留原始 Frame Index 映射。
- 摄像机动画与原视频同步。
- 输出 MP4。
- 支持 Original / Blockout Split Screen。

最终输出：

- original_video.mp4
- blockout_video.mp4
- original_vs_blockout.mp4

### Stage G：Quality Evaluation

建立自动质量评估模块。

几何质量：

- Chamfer Distance
- F-score
- Point Cloud Coverage

时序质量：

- Camera Pose Error
- Camera Trajectory Error
- Frame Synchronization Error
- Object Tracking ID Switch

语义质量：

- Object Detection Precision / Recall
- Category Accuracy
- Static / Dynamic Classification Accuracy

视觉质量：

- SSIM
- LPIPS

如果数据集提供 Ground Truth，可以直接进行量化对比。

注意区分：

- Ground Truth
- Predicted
- Inferred
- Unobserved

不能用视觉相似度代替几何精度。

输出：

- evaluation_report.json
- evaluation_report.html
- metrics.csv

## 5. 项目目录

建议使用以下结构：

GameVideoReconstruction/
│
├── config/
│   ├── config.yaml
│   └── scene_schema.json
│
├── data/
│   ├── input/
│   ├── rgb/
│   ├── depth/
│   ├── camera/
│   └── metadata/
│
├── agents/
│   ├── video_understanding.py
│   ├── object_detection.py
│   ├── segmentation.py
│   └── scene_builder.py
│
├── reconstruction/
│   ├── depth_decoder.py
│   ├── backprojection.py
│   ├── pointcloud_fusion.py
│   ├── coordinate_transform.py
│   └── geometry.py
│
├── blender/
│   ├── scene_generator.py
│   ├── object_generator.py
│   ├── camera_generator.py
│   ├── animation_generator.py
│   └── render.py
│
├── evaluation/
│   ├── geometry_metrics.py
│   ├── temporal_metrics.py
│   ├── semantic_metrics.py
│   └── report.py
│
├── outputs/
│
├── scripts/
│   ├── prepare_dataset.py
│   ├── reconstruct_scene.py
│   ├── generate_blender_scene.py
│   └── run_pipeline.py
│
├── tests/
│
├── requirements.txt
└── README.md

可以根据实际实现情况调整目录，不要为了符合目录而创建大量空文件。

## 6. 开发原则

非常重要：

1. 先检查数据，再编写代码。
2. 先实现可验证的核心功能，再接入 AI 模型。
3. 每个阶段都要有独立的输入、输出和日志。
4. 不允许使用未经验证的坐标转换。
5. 不允许假设所有深度都是米制。
6. 不允许忽略视频帧和数据流的同步问题。
7. 不允许把点云直接当成完整可编辑场景。
8. 所有重要算法都要提供可视化调试结果。
9. 优先复用已有开源工具，不要重复造轮子。
10. 每个阶段必须有测试，失败时提供明确错误信息。

## 7. 最终验收标准

第一版 MVP 必须做到：

- 成功读取一个 Hugging Face Session。
- 提取连续 10–15 秒 RGB 视频。
- 正确解码对应 Depth。
- 正确读取 Camera Pose。
- 完成多帧点云融合。
- 自动生成 Blender 场景。
- 自动生成摄像机动画。
- 自动渲染粗模视频。
- 输出原视频和粗模视频的对比结果。
- 输出基础几何与时序评价报告。

## 8. 你的工作方式

你是这个项目的主要开发 Agent。

不要一次性实现全部功能。

请按照以下顺序工作：

第一步：检查当前开发环境、Python 版本、Blender 是否安装、FFmpeg 是否可用，以及数据集访问状态。

第二步：分析 Hugging Face 数据集实际目录和一个 Session 的完整数据结构。

第三步：提出具体的技术实施方案，特别说明坐标系转换、Depth 解码和时间同步方法。

第四步：制定分阶段开发计划。

第五步：每完成一个阶段，都执行测试、输出结果并说明尚存问题。

在没有验证上一阶段结果之前，不要直接进入下一阶段。

优先保证第一个完整 Demo 可以运行，不要过度设计复杂的 Agent 框架。
