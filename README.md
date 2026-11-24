# Game Video Reconstruction — Stage A / B / C / D / E / F

用户已明确允许按原视频布局进行启发式体素补齐，独立入口 `python -m reconstruction.heuristic_proxy --allow-heuristic`。50 cm 体素代理已生成连续地面及环境实体、1401 个可编辑区块，720 帧相机穿模检查为 0，九帧实际预览与保存重开通过。最终 `scene.blend`、`blockout_video.mp4` 和 `original_vs_blockout.mp4` 已生成于 `outputs/stage_ef/heuristic_voxel_proxy/`；两个视频均为 60 FPS / 720 帧 / 12 秒，输入/代码/输出哈希匹配的重复运行复用通过，精简后全项目 145/145 测试通过。它不是精确几何恢复，不改变以下旧重建路线的失败结论。运行与限制见 [docs/heuristic_proxy.md](docs/heuristic_proxy.md)。

基本体现在从原视频观测生成，不再读体素占用：`python -m reconstruction.primitive_blockout --allow-heuristic`。输出在 `outputs/stage_ef/heuristic_primitive_from_video/`；旧的体素目录和从体素拟合的灰模目录保持不动。

新增可选真实视频彩色体素粗模路径，保留环境轮廓/粗颜色，不追求精细贴图。模型识别已在本地 GPU 跑完 61 帧；原深度路径仍失败，独立 RGB 三视图留出重投影通过（627 点，中位数 1.99 px、P90 7.05 px），但不能据此证明全部稠密几何有效。已后台生成表面/实体体素试验 `.blend` 和九帧预览；最新实体候选有 261/720 帧相机落入体素，质量拒绝，未生成新版真实 MP4。原严格路径不变。流程、命令和实际证据见 [docs/refined_blockout.md](docs/refined_blockout.md)。

Stage E/F 已实现并补齐 Roadmap：读取已验证 Scene JSON，通过 Blender Background Mode 自动创建地面/道路/墙壁/建筑等粗模、分类材质、Collections、摄像机和逐帧动画，保存后重开校验并渲染 MP4。支持原始密集 Camera Pose、精确 FPS/源 Frame Index 映射、源 Scene JSON 快照、批量任务、受保护的重复运行和完整产物哈希复用，不依赖人工 Blender UI。Stage E/F 26 项、全项目 115 项测试通过（无跳过）。详情见 [docs/stage_ef.md](docs/stage_ef.md)。

```powershell
.\.venv\Scripts\python.exe scripts/prepare_stage_ef_demo.py
.\.venv\Scripts\python.exe scripts/render_blockout.py outputs/stage_ef/demo_input/scene_schema.json --original-video outputs/stage_ef/demo_input/original_video.mp4 --camera-poses outputs/stage_ef/demo_input/camera_poses.json --output outputs/stage_ef/synthetic_demo
.\.venv\Scripts\python.exe scripts/run_tests.py --pattern test_stage_ef.py --output outputs/stage_ef/tests
```

已验收合成产物：`outputs/stage_ef/synthetic_demo/scene.blend`、`blockout_video.mp4`、`original_vs_blockout.mp4`，均为 12 秒 / 60 FPS / 720 帧（独立合成原始输入，不是受限游戏样本）。真实样本仍因几何和 Camera Pose 合同缺口被 Stage E 明确拒绝，没有假定未知坐标或尺度。

Stage C/D 已增加：五类主要物体（角色、树、岩石、地面、栏杆）的辅助 RGB 理解与遮罩接口、Depth/Point Cloud/Mask 几何估计、Persistent ID、静动态分类、空间关系、Camera/Motion Track，以及 `1.0.0` JSON Scene Schema 的结构和语义校验。真实样本先输出明确的 RGB-only partial；完整几何用五类合成 demo 验证，不绕过 Stage A/B 的真实合同缺口。入口、数据合同、证据及限制见 [docs/stage_cd.md](docs/stage_cd.md)。

```powershell
.\.venv\Scripts\python.exe scripts/understand_scene.py
.\.venv\Scripts\python.exe scripts/understand_scene.py --synthetic
.\.venv\Scripts\python.exe scripts/validate_scene.py outputs/stage_cd/synthetic_demo/scene_schema.json
.\.venv\Scripts\python.exe scripts/run_tests.py --output outputs/stage_cd/tests
```

真实产物：`outputs/stage_cd/sample_66ad/scene_schema.json`（辅助识别五类、10 条观测，未知几何为 null，预期 exit 2）。合成产物：`outputs/stage_cd/synthetic_demo/`（五类持久 ID、4 静态+1 动态、30 条观测、带来源的对象点云与逐帧 Mask/debug）。本地 Grounding DINO/SAM 依赖和固定版本权重已安装，新增 refinement 路径已完成 61 帧 GPU 推理；识别和几何仍是 predicted/inferred，不声称真实自动检测精度已获真值验证。

当前状态：**Stage B 反投影与融合内核、单元测试和合成演示已实现；真实样本米制 Depth / render-pose 仍未确认，真实 Stage A/B 尚未验收**。不扩大数据下载；Stage E/F 只对已验证合同输入生成场景，独立 Stage B 后台数学验证不属于真实样本验收。

Stage B 入口为 `scripts/reconstruct_scene.py`，完整坐标定义、证据边界、输入合同及复现命令见 [docs/stage_b.md](docs/stage_b.md)。默认只复核真实样本并返回阻塞状态；`--synthetic` 显式生成已知尺度的演示，不会自动用合成结果代替真实重建。

- 合成单帧 / 融合 PLY、可视化与元数据：`outputs/stage_b/synthetic_demo/`。
- 真实样本验证报告、仅相机位置的轨迹 PLY / 方向假设图：`outputs/stage_b/sample_66ad/`，**不是场景点云**。
- 坐标单测、重复权重、无效深度、体素来源与 PLY 回读测试：`tests/test_stage_b.py`。
- 独立 Blender 数值回执：`outputs/stage_b/validation/blender_coordinate_probe.json`。

```powershell
.\.venv\Scripts\python.exe scripts/run_tests.py --output outputs/stage_b/tests
.\.venv\Scripts\python.exe scripts/reconstruct_scene.py
.\.venv\Scripts\python.exe scripts/reconstruct_scene.py --synthetic
```

最新产物位于 `outputs/stage_a/sample_66ad/`：12 秒 / 60 FPS RGB、720 帧相机记录、6 张原分辨率 RGB PNG 和 6 个原生 Depth code/mask NPZ。实际 RGB 与 Depth 均解码 720 帧；RGB PTS / Depth Metadata 时间误差为 0，相机逐帧与 Parquet 对照通过。具体限制见该目录的 `validation_report.md`。

本次累计接收约 76.13 MB（含两个候选的 JSONL 前缀、小型合同和一个选定 Session 的媒体前缀）。没有下载任何完整视频、完整 HEVC 或完整 Session。最早受限下载删除期限为 **2026-12-24，Asia/Shanghai**。

## 运行

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/inspect_dataset.py
.\.venv\Scripts\python.exe scripts/run_tests.py
# 本地前缀已存在时无需联网，即可重新生成小样本：
.\.venv\Scripts\python.exe scripts/prepare_sample.py
```

测试维护：同类输入校验使用带名称的 `subTest`，移除重复端点检查和写入固定目录的合成诊断测试，保留全部 Blender 后台、相机/时间轴、穿模、缓存及输出保护测试。全项目共145项，回归记录在 `outputs/stage_ef/cleanup_tests/`；可执行 `.\.venv\Scripts\python.exe scripts/run_tests.py --output outputs/stage_ef/cleanup_tests`。历史验收记录保留。

访问检查是只读的：公开 API/README 展示页、固定 commit 下的 `samples/` 文件树，以及受限文件的 HEAD 请求。默认总响应上限 8 MiB，不下载视频、HEVC、JSONL 或 Parquet；返回码 2 代表需要获批登录，1 代表检查本身失败。认证从 HF_TOKEN / HUGGING_FACE_HUB_TOKEN 或标准本机 HF token 文件读取，不打印、不写入报告。

## 已实现

- `stage_a/access.py`：受限访问检查、字节预算、固定版本、sample 成本排序、JSON / Markdown 报告。
- `stage_a/media.py`：本地 RGB 读取；原生 `yuv420p10le` HEVC 读取，以 `uint16` 保存原始 **0–1023** luma codes；不调用 `gray16le` 范围转换。逐帧生成器带索引、解码字节上限和缺帧错误。当前只支持这一个原生 Depth 像素布局，其他格式明确拒绝。
- `stage_a/depth.py`：公开 README 相对 log-depth 数学函数、near plane 和 pinhole K 数值检查，以及原生 v3 code / reserved-code mask 输出。**实际为 log_sentinel_v3，缺少明确逆公式；米制转换故意拒绝执行**。
- `stage_a/sync.py`：流式 JSONL、可选 PyArrow 帧表读取、显式 Frame Index 关联、多 Camera QPC 记录保留、缺帧/重复索引/时间差检查。保留原始位置和旋转字段，不转换坐标系，不把原始 QPC 当秒。
- `tests/`：纯合成 12 秒 / 60 FPS / 32×32 HEVC + RGB 集成测试，覆盖所有 1024 luma codes、重复帧、非零起始索引、缺帧及拒绝未验证米制公式。

`pyarrow` 已加入依赖，用于真实帧表的有限列/首批数据读取。`stage_a/mp4.py` 检查 ISO-BMFF 的 moov/chunk/sample table，仅下载早期媒体范围；中间 partial.mp4 的完整索引仅用于有限早期解码，不能当完整 Session 使用。

`scripts/inspect_metadata.py` 的总接收上限为 8 MiB；`scripts/fetch_sample.py` 固定到当前 sample、总接收上限 128 MiB，HTTP Range 不被严格兑现时在读取 payload 前停止。前者读取三个 sample 的小型合同和一个候选的 JSONL 前缀，后者读取选定 Session 的 JSONL 前缀、MP4 索引/前15秒媒体范围、24 MiB HEVC 前缀和5.2 MiB帧表。它们需要获批的本机 HF 凭据，不会存储 token。普通重跑只需 `prepare_sample.py`，不要重复下载。

## 重要边界

公开 README 示例为 `D_rel = expm1(q * log1p(K)) / K`，并示范从 FFmpeg gray16le 归一化到 q。这里不把该示例当作每个 Session 的完整物理解码合同：原生 10-bit codes 如何归一化、full/limited range、sky/far-clip/invalid masks、depth_transform、near_plane_cm 和最终单位，均需对照实际 Session 确认。代码输出的 raw codes **不是米制 Depth Frames**。

decoded ordinal / delivery Frame Index / original capture tick / clip ordinal 是不同概念。`join_frames` 要求调用者提供已审查的源帧映射，不使用 JSONL 行号或 QPC 时间戳猜测 Frame Index；索引关联成功也不代表 timestamp、映射来源和相机单采样规则已经验证。相机帧表 binning 应复用实际 `tables/conversion_manifest.json`，并交叉检查 `camera.jsonl`。

重复帧统计定义为解码后相邻像素完全一致，不等于合同中的 `dup_count`，也不能据此判定源相机重复；静止场景可能相同，有损重编码的源重复帧也可能不同。真实剪辑需要读取前一帧以验证首帧的重复状态。

真实样本与合成测试分开：真实样本在 `outputs/stage_a/sample_66ad/`，合成测试在 `outputs/stage_a/tests/`。当前真实样本的 near_plane_cm=10、fx=fy=960、cx=960、cy=540；RGB / Depth 同为 1920×1080。所有相机字段仍处于原始 UE5 / lh_z_up 坐标，不转换到 Blender。

已发现并保留：旧 loader 使用 transform=2 / 旧目录 / 30 FPS 描述，与实际 v4 Session 不兼容；实际 depth_transform=3，sentinel=1023、max_valid=1020、far_plane_encode=100000；Depth Metadata.camera_frame 比 depth_frame 滞后0–13帧，不可当作RGB索引；两项 manifest size/hash 与实际固定版本文件不符；全局 RGB 与 Depth/帧表数量相差1，但0–719窗口无缺帧。解码后精确相邻重复为 RGB 2 / Depth 0，不代表源编码端重复数为0。

## 仍未完成的验证（仅 Stage A）

1. 需要数据提供方明确 v3 逆公式、code 归一化分母、far_plane_encode 单位、near_plane 的作用以及零码意义；不能用旧示例或假设值生成米制 Depth。
2. 需要进一步解释 Depth Metadata 的相机快照引用时刻和源 dup_count。当前保留 RGB 帧相机和 Depth 引用相机两份原始记录，不声称几何对应时刻已完全验证。
3. 不扩大下载、不开启未经验证的真实几何重建。Stage B 内核可在已知尺度/外参的合成输入上运行；真实样本必须确认以上合同及坐标/像素采样定义后才准备几何重建输入。

受限样本仅供内部评估，不再分发，不上传第三方，不公开发布；首次授权下载需记录 30 天删除期限并以完整 LICENSE.md 为准。

本次公开 `LICENSE.md` 顶部标有 DRAFT / pending legal review。程序不会自动申请访问或接受条款；内部评估之外的使用需单独明确授权。
