# Stage E/F：后台粗模生成与视频渲染

已实现：Scene JSON `1.0.0` → 可编辑 `scene.blend` → `blockout_video.mp4` → 左原始/右粗模 `original_vs_blockout.mp4`。只运行命令行 Blender Background Mode，不连接或操作人工 Blender UI，不下载新数据、不调用付费生成服务。

2026-11-24 本机验收：Blender **5.2.1 LTS**，独立合成输入 **12 秒、60 FPS、720 帧、240×140**，五类物体、四静态一动态、逐帧原始相机 Pose。真实样本的上游深度/朝向/render-pose 合同仍未验证；Stage E 明确拒绝该 partial JSON，未制造真实重建结果。

验证结果：新增 Stage E/F **19/19** 测试通过，全项目 **108/108** 测试通过、无跳过；保存后重开的 720 帧产生 2,160 个独立投影检查，最大归一化投影误差 `2.72137778090098e-7`。所有最终文件 SHA-256 核验通过；源索引为 `7000..7719`，720 个原始 Tick 全部保留。首/中/末对比画面已检查。双分辨率批量配置的两项完整 12 秒渲染均通过，产物在 `outputs/stage_ef/batch/full_resolution/` 与 `half_resolution/`。

## 一条命令运行

在项目根目录执行；所有入口会自行解析脚本位置，内部 Blender 进程使用绝对路径。

```powershell
# 已有合成输入；生成 + 保存后重开验证 + 后台渲染 + 编码 + 逐帧验证：
.\.venv\Scripts\python.exe scripts/render_blockout.py `
  outputs/stage_ef/demo_input/scene_schema.json `
  --original-video outputs/stage_ef/demo_input/original_video.mp4 `
  --camera-poses outputs/stage_ef/demo_input/camera_poses.json `
  --output outputs/stage_ef/synthetic_demo

# 重新准备可独立复现的 12 秒合成输入（不是受限游戏视频）：
.\.venv\Scripts\python.exe scripts/prepare_stage_ef_demo.py

# 输入、代码、参数和所有产物 SHA-256 都一致时才能复用：
.\.venv\Scripts\python.exe scripts/render_blockout.py `
  outputs/stage_ef/demo_input/scene_schema.json `
  --original-video outputs/stage_ef/demo_input/original_video.mp4 `
  --camera-poses outputs/stage_ef/demo_input/camera_poses.json `
  --output outputs/stage_ef/synthetic_demo --reuse

# Stage E 独立执行，仍重开验证 .blend，但不渲染视频：
.\.venv\Scripts\python.exe scripts/generate_blender_scene.py `
  outputs/stage_ef/demo_input/scene_schema.json `
  --original-video outputs/stage_ef/demo_input/original_video.mp4 `
  --output outputs/stage_ef/build_only
```

新输入替换 `scene_json`、`--original-video`、`--output`；`--camera-poses` 可省略，此时使用 Scene 自带的已验证相机 Track。绝不自动读取 `camera.raw_source` 并猜测坐标合同。

依赖为既有 `requirements.txt` 和本机 Blender（建议 4.2+；5.2.1 已实测）。自动查找 PATH 和 Windows `C:/Program Files/Blender Foundation/Blender */blender.exe`。也可传 `--blender "C:/.../blender.exe"` 或设置 `STAGE_EF_BLENDER`。FFmpeg 复用 Stage A 的 PATH/`STAGE_A_FFMPEG`/imageio-ffmpeg 配置，不要求 ffprobe。

## 输入与时间合同

- Scene 必须通过已有结构/语义 Validation，单位米、世界坐标 `blender_rh_z_up`、四元数 xyzw；每个物体必须有可测量的 Transform/Dimensions，几何合同已验证，相机不能为 null。缺少来源 PLY、内参、相机覆盖、非刚性矩阵或未验证几何均报错，不退化成任意摆放。
- 原始视频必须 CFR。FFmpeg 实际解码，使用每帧**整数 PTS × 有理数 time_base** 检查间隔和帧数；保留 `60000/1001` 等精确 FPS，不用显示的 `59.94` 代替。VFR、旋转视频、FPS 不一致明确拒绝，不通过补帧/减帧掩盖问题。
- `timeline.start_seconds` 是源剪辑时间原点，不是 `frame_index / fps`；`end_seconds` 是**最后一帧 PTS（inclusive）**，视频时长为 `N/FPS`。12 秒/60 FPS 的最后一帧 PTS 为 `719/60`，不是 12。源帧可从 7000、2000 或其他非零索引开始。
- Scene 关键帧的 Frame Index 与时间戳必须落在同一 CFR 格点。视频的全帧窗口和 Scene 范围严格一致。显式剪辑用 `--video-start-frame N --video-frame-count M`；输入 Scene 应描述该窗口。按解码帧序号裁剪，不能用关键帧 seek 或四舍五入的秒数代替。
- Blender 从帧 1 开始：`blender_frame = 1 + (source_seconds-start_seconds)*FPS`。每一输出帧保存 `source_frame_index`、`video_frame_index`、源秒数与原视频 PTS。原始 Tick 仅在 Scene/密集 Pose 提供时保留，未知帧为 null，不编造 Tick 或把 QPC 当秒。
- 目标分辨率默认原视频原尺寸；`--width W --height H` 可缩小渲染和对比右侧，K 同步缩放。Scene 的观测原生 raster 必须与原视频一致，不能盲猜内参来源。H.264 yuv420p 要求偶数输出尺寸。

### 密集原始 Camera Pose

支持可选 `stage_e.camera_poses/1`，结构见 `config/camera_pose_schema.json`。这是一份**调用方已经确认**的坐标/单位/同步合同；`verified: true` 本身不是事实正确性的证明。现有真实 `outputs/stage_a/.../camera/poses.json` 是原始遥测，不满足该合同，故不能直接用于验收。

```json
{
  "schema": "stage_e.camera_poses/1",
  "verified": true,
  "evidence": "具体坐标、单位、SDK 字段和源视频帧同步的审查依据",
  "pose_format": "world_from_optical_camera",
  "coordinate_system": "blender_rh_z_up",
  "translation_unit": "m",
  "samples": [{
    "camera_id": "camera_main",
    "frame_index": 7000,
    "timestamp_seconds": 0.0,
    "original_tick": 17000,
    "world_from_camera": [[1,0,0,0],[0,1,0,0],[0,0,1,2],[0,0,0,1]],
    "intrinsics": [[135,0,120],[0,135,70],[0,0,1]]
  }]
}
```

上例仅示范字段；实际 Track 需要覆盖所有源时间，矩阵必须是 `world_from_optical_camera`，不是 view matrix。另一种输入显式声明 `pose_format: ue_frotator_degrees`、`coordinate_system: ue_lh_z_up` 和 `world_units_per_meter`，每帧用 `pose: {position, pitch, yaw, roll}`。复用 Stage B 已测 UE FRotator 公式和尺度转换，不猜旋转顺序、弧度或 SDK render latch。

多个 Camera 都创建并动画化，`--active-camera ID` 选择渲染镜头；不自动推断剪辑切镜。各 Camera 必须覆盖完整视频窗口，内参 fx/fy 比必须兼容同一个渲染像素宽高比。

## Stage E 实现

- `blender/planner.py`：读 Scene、合同校验、参考点云来源、密集原 Pose 覆盖、逐帧 Build Plan。
- `blender/timeline.py`：保留源帧格点、线性位置/尺度/内参插值、最短弧四元数 SLERP；拒绝相机外推。物体 Motion Track 边界之外保持首/末观测 Pose，并记录这一限制。
- `blender/object_generator.py`：单位 box、capsule、cylinder、plane slab；`dimensions × transform.scale` 只乘一次。plane 使用声明的正厚度，capsule 允许 XYZ 非均匀缩放。无纹理，按类别使用五种灰色/低饱和材质；不为不可见体积随意加厚物体。
- `blender/camera_generator.py`：`T_world_blenderCamera = T_world_optical × diag(1,-1,-1,1)`。`lens=fx×sensor_width/W`，支持偏心主点和非等 fx/fy；设置匹配的 pixel aspect、shift_x、shift_y。并非 look-at 近似，也不丢弃 roll。
- `blender/animation_generator.py`：在**每一渲染帧**烘焙 Camera/Object 动画，规避默认 Bezier 的超调，兼容新版分层 Action。相机透视内参也动画化。
- `blender/scene_generator.py`：factory-empty 创建 Scene；Static、Dynamic、UnknownMotion、Cameras、Lighting、Reference、Hierarchy Collections。`parent_id` 通过嵌套 Collection 组织，所有 Transform 仍为世界坐标，不让父物体位移重复作用到子物体。场景含 World/Sun/Fill、灰模 Mesh、相机、动画。
- 融合参考点云可用 `--point-cloud PATH.ply`；否则读每个物体的几何来源 PLY。经 Open3D 验证有限坐标后，确定性采样最多 50,000 点，嵌入 .blend 顶点，**不参与渲染、不当作完整可编辑重建**。不依赖外部纹理/模型文件。
- 源 Scene JSON、Frame Mapping、坐标/插值约定也嵌入 .blend Text datablocks。`.blend` 可移到其他目录编辑或重新渲染，无需源点云。
- `visibility: not_detected` 是未检测，不是隐藏/遮挡真值；本阶段不会据此关闭物体可见性。

## Stage F、批量与可重复运行

```text
blender --background --factory-startup --threads 4 --python-exit-code 1
  --python <absolute>/blender/scene_generator.py -- <plan> <run-dir>
blender --background --factory-startup --threads 4 <run-dir>/scene.blend
  --python-exit-code 1 --python <absolute>/blender/render.py -- <plan> <run-dir>
```

默认 `BLENDER_WORKBENCH`：快速灰模/材质色，后台实测成功。`--engine BLENDER_EEVEE` 或 `--engine CYCLES --samples 16` 可使用灯光渲染（Cycles 默认 CPU）。Workbench 的 studio lighting 不使用场景中的灯光；灯光保存在 .blend 供其他引擎使用。EEVEE 的本机完整视频未验收，不声称所有 GPU 环境都支持。

渲染编号固定为六位 PNG，检查无缺帧/多帧；FFmpeg 精确有理数 FPS 编码 H.264，关闭音频。原视频按源帧窗口另存 `original_video.mp4`，不覆盖输入。Split-screen 左原视频/右粗模，同步 PTS 后 hstack；不把同一粗模渲染同时当作原视频。编码后再次解码三段视频，检查零 PTS 起点、尺寸、完整帧数与 FPS。

批量 manifest 的路径相对 manifest 所在目录，每个任务必须有独立输出；失败继续处理后续任务，总退出码非零。

```json
{
  "schema": "stage_ef.batch/1",
  "jobs": [
    {"scene_json": "demo_input/scene_schema.json", "original_video": "demo_input/original_video.mp4",
     "camera_poses": "demo_input/camera_poses.json", "output": "batch/scene_01"},
    {"scene_json": "demo_input/scene_schema.json", "original_video": "demo_input/original_video.mp4",
     "output": "batch/scene_02", "engine": "CYCLES", "samples": 8, "resolution": [120,70]}
  ]
}
```

```powershell
.\.venv\Scripts\python.exe scripts/render_blockout.py --batch outputs/stage_ef/batch.json
# 仓库也提供可直接执行的双分辨率合成批量配置：
.\.venv\Scripts\python.exe scripts/render_blockout.py --batch config/stage_ef_demo_batch.json
```

正常重复运行会重新生成 factory-empty 场景，不累积物体/材质/关键帧。仅覆盖 receipt 标记为本流程所有的固定产物，拒绝覆盖无回执的同名用户文件；其他笔记/文件保留。排他 `.stage_ef.lock` 阻止两个运行共用目录。异常退出留下锁时，应先确认旧进程已终止，再移除该明确锁文件；不自动解除活跃锁。

构建/渲染/编码都在输出下的唯一 staging 子目录进行，验证通过才发布固定产物，最后写 receipt。渲染或校验失败保留 staging 和完整日志，既有成功产物不被提前覆盖。发布阶段的磁盘/权限错误不是多文件事务，需修复后重跑；receipt 最后发布，旧 receipt 的哈希会识别不完整产物。`--reuse` 校验源视频/JSON/PLY/Pose、实现和 schema 的哈希及所有已发布文件哈希，任何不一致重建。生成后的临时 PNG 自动移除，完整日志保留；`.blend` 中保存 `//frames/frame_######`，可后续无 UI 重新渲染。

## 产物与验证证据

`outputs/stage_ef/synthetic_demo/`：

- `scene.blend`：保存后**另一个 Blender 后台进程重开**验证。
- `blockout_video.mp4`：240×140，720 帧，60 FPS，12 秒。
- `original_vs_blockout.mp4`：480×140，同样的帧数、FPS、时长。
- `original_video.mp4`：独立 ray/AABB 输入视频的同步副本，不是受限游戏数据。
- `build_plan.json` / `frame_mapping.json`：完整逐帧来源与生成参数。
- `build_report.json` / `saved_scene_verification.json`：逐帧摄像机/对象矩阵误差与独立 `world_to_camera_view` 投影误差。
- `pipeline_report.json`：文件 SHA-256、输入/实现指纹、媒体逐帧校验和限制；`real_session_accuracy_validated` 明确为 false。
- `build.log` / `render.log` / `encode_*.log` / `*.probe.log`：后台完整诊断。

```powershell
.\.venv\Scripts\python.exe scripts/run_tests.py --pattern test_stage_ef.py --output outputs/stage_ef/tests
.\.venv\Scripts\python.exe scripts/run_tests.py --output outputs/stage_ef/all_tests
```

新增测试包含真实 Blender 背景生成/重开/渲染/重复执行、变换与父组织、偏心 K / 非等焦距、240×140→96×56 内参缩放、媒体 60000/1001 FPS、非零源索引、末帧 off-by-one、VFR 拒绝、未知合同与缺失点云拒绝、输出篡改后的缓存失效、批量成功/失败隔离及用户文件保护。没有 Blender 时集成测试显式 skipped；纯合同/媒体测试不依赖 UI。

这是 Stage E/F 的工程可运行验收，不是 Stage G 的真实游戏几何/语义精度验收。真实数据需要先解决 Stage A/B 的合同缺口并重新生成有依据的 Scene JSON。
