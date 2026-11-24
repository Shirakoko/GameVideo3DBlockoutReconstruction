# Stage G 与一键 Pipeline

Stage G 已实现几何、时序、语义和补充视觉评价，以及独立 Stage A–G 的执行、资源采样、报告、缓存、断点续跑和重试。不下载新数据，不上传受限帧，也不自动把预测或启发式结果升级为 Ground Truth。

当前真实 Session 的 `log_sentinel_v3` 米制逆公式、像素采样和 depth render-pose 绑定仍未验证。因此真实报告是 **partial**，不是 Stage A–G 真实重建验收。现有数据能确认的 RGB PTS / Depth Metadata delivery 时间差为 0；JSONL / Parquet 相机字段一致性不是 Camera Pose 估计精度。合成成功也不能替代真实验收。

2026-11-24 本机验证：Stage G **40/40**、全项目 **192/192** 测试通过，无跳过，记录在 `outputs/stage_g/tests/` 和 `outputs/stage_g/regression/`。Blender 5.2.1 LTS 的完整合成流程输出 12 秒 / 60 FPS / 720 帧，二次执行 A–G 全部哈希缓存命中。观测点云 CD 为 `0.0460833 m`，5 cm F-score 为 `0.9773903`、coverage 为 `0.9558844`；720 帧保存相机对 GT 输入的平移 RMSE 为 `4.81084e-8 m`、旋转 RMSE 为 `4.87661e-6 deg`，这是复制 GT 后的保存/转换误差。SSIM 为 `0.7552167`，默认 LPIPS 为 N/A。已有真实本地前缀也已独立重做 Stage A（RGB/Depth 均解码 720 帧，时间差 0），原 Stage A 和旧粗模目录保持不动，没有新下载；重建依然因真实合同缺口停止。

## 一键执行

在项目根目录运行：

```powershell
# 完整合成 A–G：12 秒 / 60 FPS / 720 帧，包含实际 Blender 保存、重开和渲染
.\.venv\Scripts\python.exe scripts/run_pipeline.py --config config/pipeline_synthetic.json

# 原命令再次执行即验证输入、代码、依赖版本和产物哈希，然后复用成功阶段
.\.venv\Scripts\python.exe scripts/run_pipeline.py --config config/pipeline_synthetic.json

# 已有受限样本：重新审查合同，阻塞 B–F，但仍输出真实 partial 报告；预期 exit 2
.\.venv\Scripts\python.exe scripts/run_pipeline.py --config config/pipeline_existing_sample.json

# 可选：从已有本地媒体前缀重做 Stage A 解码到独立 attempt，不下载；仍预期合同阻塞 exit 2
.\.venv\Scripts\python.exe scripts/run_pipeline.py --config config/pipeline_dataset_prepare.json

# 独立评价现有真实样本，预期 exit 2
.\.venv\Scripts\python.exe scripts/evaluate_scene.py

# 外部已验证 GT 和独立预测
.\.venv\Scripts\python.exe scripts/evaluate_scene.py --config path/to/evaluation.json --output outputs/stage_g/reviewed

# 显式重算，保留旧 attempt 和失败日志，不删除历史产物
.\.venv\Scripts\python.exe scripts/run_pipeline.py --config config/pipeline_synthetic.json --no-cache
```

根输出目录包含 `evaluation_report.html`、`metrics.csv`、`evaluation_report.json`、`pipeline_report.json`、`pipeline_state.json` 和 `artifact_manifest.json`。HTML 完全离线、无需 CDN，含指标、相机误差曲线、失败信息、阶段/资源明细及本地产物链接。

各阶段产物放在 `stages/<ID>/attempt_<编号>/artifacts/`，日志放在同一个 attempt 的 `stage.log`。根 HTML / JSON 的 artifact 索引给出当前 `.blend`、Scene JSON、点云、原视频、粗模视频和左右对比的确切位置；不强制复制大文件到根目录。不要发布受限数据或派生报告到外部服务。

## 指标定义

| 指标 | 定义与边界 |
| --- | --- |
| Chamfer Distance | `(mean(P→GT) + mean(GT→P))/2`，欧氏最近邻距离，单位 m；另报平方距离均值 m²，二者不混用。 |
| F-score | 每个显式米制阈值下的点精确率和点召回率的调和均值；默认 5 cm / 10 cm。 |
| Point Cloud Coverage | 每个阈值内被预测覆盖的 **已观测 GT 表面样本** 比例，不是隐藏空间、完整体积的覆盖率。 |
| Camera Pose Error | 按 `(camera_id, source frame_index)` 对齐的未对齐平移距离 m 和 SO(3) 测地旋转角 deg。 |
| Trajectory ATE | 位置残差的 mean / RMSE / median / P95 / max；默认不对齐，可显式选择 SE3 刚体对齐，绝不拟合尺度。 |
| Trajectory RPE | `inverse(GT_relative) @ predicted_relative` 的平移和旋转误差。区间按源 Frame Index 的显式 delta，缺帧不会缩短区间。 |
| Synchronization | 相同源 Frame Index、已审查共同原点秒时钟上的绝对误差、带符号偏移、漂移 ppm、匹配率和缺/多帧数。不是 raw QPC 或最近时间匹配。 |
| Detection P/R/F1 | 已穷尽标注帧上的类别一致、IoU 达标、一对一最大匹配。空标注帧也计入，额外检测会产生 FP。 |
| Category / Static-Dynamic Accuracy | 独立的类别无关定位匹配；错类别不从分母消失，预测的未知静动态标签也计为错误。 |
| ID Switch / Fragmentation | 持久 GT ID 上相邻匹配身份变化，保留遮挡前匹配；可显式限制源帧最大间隔，另报匹配覆盖率。 |
| SSIM | 原分辨率 RGB 的 7×7 均匀窗口、总体协方差、通道平均，仅作视觉补充。 |
| LPIPS | 可选 AlexNet LPIPS，仅使用明确指定的本地 backbone 和 linear 权重；缺依赖/权重时 N/A，绝不自动下载。 |

空预测的 CD/precision 为 N/A，coverage/F-score 为 0；空 GT 无法评价。未知真值、无效合同、没有匹配样本时记录 N/A 和原因，JSON 使用 `null`、CSV 留空，不写 NaN/Infinity，也不伪造零误差。CD 用 KD-tree，最大评估样本数和随机种子可配置并在报告披露。

合成演示的点云指标比较的是 Stage B 融合输出与输入 GT Depth 的观测表面，不是最终五个 Primitive 网格的几何精度。语义识别使用 fixture 的颜色规则，GT 是解析渲染器的实例标签 oracle，仅证明测试流程；不证明真实检测性能。正常完整渲染时，相机预测来自 **保存后重新打开的 `.blend` 中实际逐帧 `camera.matrix_world`**，显式转换为 optical camera-to-world，并披露 float32 数值正交化。与复制给 Blender 的 GT Camera Pose 对比仍是 Pose 保留/转换精度，不是独立 Camera Pose 估计精度。

## GT 与预测合同

`evaluation.json` 的所有相对路径以配置文件所在目录为原点：

```json
{
  "schema": "stage_g.config/1",
  "data_source": "Reviewed dataset session / local internal evaluation",
  "ground_truth": {
    "provenance": "ground_truth",
    "verified": true,
    "evidence": "Applicable Session depth inverse, units, pixel sampling and render-pose review reference",
    "metric_manifest": "gt/metric_frames.json",
    "camera_poses": "gt/camera_poses.json"
  },
  "prediction": {
    "provenance": "predicted",
    "coordinate_system": "blender_rh_z_up",
    "unit": "m",
    "point_cloud": "prediction/scene_pointcloud.ply",
    "camera_poses": "prediction/camera.json",
    "scene_json": "prediction/scene_schema.json"
  },
  "geometry": {"min_depth_m": 0.1, "max_depth_m": 35, "stride": 2, "thresholds_m": [0.05, 0.1], "max_points": 200000, "seed": 0},
  "camera": {"alignment": "none", "rpe_delta_frames": 1}
}
```

上面的深度上下限仅为格式示例，必须换成实际 Session 已审查范围，不是新的真实数据假设。

GT `metric_manifest` 复用 `stage_b.metric_frames/1`：NPZ 中有 floating planar Depth、boolean valid mask、native RGB 和 K；DepthSpec 的尺度、像素中心及 render-pose 都必须有明确验证证据。每个数据文件在加载前检查累计 512 MiB 解压预算，GT 点累积还有 2M 点上限。同一 source observation identity 只贡献一次观测。整数 HEVC codes、相对深度或 inferred 合同不会进入米制评价。

可选密集 GT Camera Pose 复用 `stage_e.camera_poses/1`，支持明确的 optical-to-world 或已审查 UE FRotator 适配器。省略时使用 GT metric manifest 的相机样本，不伪造密集真值。

独立预测相机文件使用 `stage_g.camera_prediction/1`，顶层声明 `coordinate_system="blender_rh_z_up"`、`translation_unit="m"`、`pose_direction="world_from_optical_camera"`；`samples` 包含 `camera_id`、`frame_index`、`timestamp_seconds`、`world_from_camera`。矩阵为 row-major JSON / column-vector math，禁止反射、缩放和隐式外推。多 Camera 分开匹配。

语义评价可增加 `semantic` 字段（`provenance="ground_truth"`、`verified=true`、`evidence`、`annotations`、可选 `iou_threshold` / `max_id_gap_frames`）。标注文件为 `stage_g.semantic_truth/1`，也需 GT provenance、verified、evidence、`exhaustive=true`。每个 `frames` 条目有 `frame_index`、`raster_size=[width,height]`、`objects`，空帧明确为 `objects=[]`；每个实例有 `object_id`、`category`、exclusive-upper `bbox_xyxy`、boolean 或 null `is_static`。这是单视图实例基准，不把辅助标注或预测 Scene 复制后当真实 GT。

同步评价增加 `synchronization`：`verified=true`、`unit="s"`、`time_origin`、`evidence`、可选 `reference="rgb"`、`streams`（直接对象，或含该字段的 JSON 路径）。每个流是 `frame_index` / `timestamp_seconds` 记录；缺帧会独立统计，不把 JSONL 行号或 clock tick 当帧索引。

视觉评价增加 `visual.pairs`，每项明确给出同一源 `frame_index` 对应的 `predicted` / `reference` PNG 路径。必须同分辨率，无隐式 resize/重对齐。LPIPS 另指定 `visual.lpips_backbone_weights`（torchvision AlexNet state dict）和 `visual.lpips_linear_weights`（官方 LPIPS AlexNet linear state dict），并在本地安装可选 `lpips` / PyTorch。未实测本机 LPIPS 权重，默认报告为 N/A。

## Pipeline 配置与恢复

三种显式模式：

- `synthetic`：生成独立解析 fixture 的 RGB、米制 Depth、密集 Camera Pose 和实例真值，再执行 B–G。
- `reviewed`：检查已审查的本地 metric manifest、视频时间轴及索引，不猜深度；`annotations` 与 `local_detector` 必须二选一。复用已安装本地模型，不隐式下载权重。Stage B 融合、C 理解、D 单独校验、E 生成和重开、F 渲染和编码、G 评价。
- `existing_sample`：默认复核 `prepared_sample` 和 `sample_source`。若显式设置 `prepare_raw_sample=true`，Stage A 通过现有 `prepare_sample` 内核从 **已经下载的 66ad 本地前缀** 重新解码到当前 attempt，保持旧 Stage A 产物不动；仍不会使用未经验证的 log codes 重建。该适配器仅支持已有已审查的小样本格式，不是假装支持任意 Session 的通用 decoder。准备失败时 G 不回退到旧样本报告。

`reviewed` 的最小顶层配置为 `schema="pipeline.config/1"`、`mode="reviewed"`、`output`、`input_manifest`、`original_video`、`annotations` 或 `local_detector`、显式 `reconstruction.min_depth_m/max_depth_m`；可加 `camera_poses`、`local_sam`、`blender`、视频窗口、`evaluation`（与上述 evaluator 合同相同）。`evaluation.prediction` 可指定独立候选，否则默认评价本次 Stage B 点云和 Stage C Scene / Stage F 导出的实际 Pose。

`render.enabled=false` 只运行 A–D/G；`render.generate_only=true` 生成并验证 E，但不运行 F。`render.resolution/engine/samples` 与既有 E/F 相同。缩放渲染会导致默认原分辨率 SSIM 的 raster mismatch 并明确报 N/A，应提供显式等分辨率参考帧，不能静默放大获得可比结果。

恢复机制：

- 每个阶段为独立子进程，argv 不经 shell 解释；超时会终止该进程树。
- 输入文件（含 manifest 引用 NPZ / annotation masks）、代码、Schema、数值库版本、FFmpeg / Blender 二进制、配置和上游 fingerprint/产物哈希共同绑定缓存。只有完整必需产物和所有 SHA-256 匹配的成功阶段才复用；空 receipt、损坏或丢失输出会重算。
- 每个 attempt 都持久记录 running → completed/failed/blocked/interrupted，checkpoint 原子替换。成功 checkpoint 可断点恢复；未提交成功的半成品永不当缓存。
- 默认一次失败重试（`retries=1`，最多允许 10）；只有退出 1 / timeout 124 重试，合同阻塞 exit 2 不重试。依赖失败时下游跳过，G 仍运行并给出可用证据和失败报告。
- `cache=false` / `--no-cache` 或 `resume=false` / `--no-resume` 强制新 attempt，旧产物和日志保持不动。没有静默删除历史 run 的策略。
- 独占 PID/create-time 锁防止并发写同一输出目录，死进程锁可恢复。不覆盖没有本 Pipeline checkpoint 的非空目录，也不允许输出与输入目录重叠。

`pipeline/runner.py` 的 `run_steps` 还提供小型通用 DAG 接口，支持 `{python}`、`{root}`、`{output}`、`{context}`、`{dep:<ID>}` argv 占位符。顶层 CLI 只接受有边界的 A–G 适配配置，没有额外 Agent 框架。

## 资源和执行状态

每个 attempt 记录 wall time、process-tree CPU seconds、CPU 峰值（单逻辑核 100%，多核可超过 100%）、RSS 峰值、采样数、GPU device-wide 显存/利用率，以及退出码、错误类型/日志和重试历史。NVIDIA 采样依赖 `nvidia-smi`，不可用则 N/A；它包含其他进程，**不声称是本阶段独占 VRAM**。短进程可能漏采样，RSS 相加可能重复共享页，这些限制随 JSON 报告输出。

缓存命中的当前执行耗时与历史 attempt 耗时分开，不把重用当重新渲染。`pipeline_status=completed` 表示所请求阶段完成；`evaluation.status=evaluated` 表示可用数值已经计算，不是自动质量验收。独立 evaluator partial 返回 2，输入失败返回 1；Pipeline completed 返回 0、blocked 返回 2、failed 返回 1。

运行测试：

```powershell
.\.venv\Scripts\python.exe scripts/run_tests.py --pattern test_stage_g.py --output outputs/stage_g/tests
.\.venv\Scripts\python.exe scripts/run_tests.py --output outputs/stage_g/regression
```

测试包含解析 CD / F-score、SO(3) / ATE / RPE、时间偏移/漂移、缺帧、多 Camera、空预测、错误 GT 身份、语义最大匹配/错类别/ID switch、严格 JSON/CSV/HTML 转义、资源缺失、缓存损坏、输入变化、断点、失败重试、timeout、目录保护及实际 Blender 小型 A–G / reviewed replay / 缓存集成。
