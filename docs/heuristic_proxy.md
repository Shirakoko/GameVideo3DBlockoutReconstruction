# 原视频布局的体素代理

2026-11-24 用户明确允许超出已验证几何范围，按原视频画面补齐地面、岩石与树木。本入口是**启发式粗模代理**，不是恢复真实几何；原深度失败回执和物理合同门槛不变。

## 运行

在项目根目录执行，全部使用 Blender Background Mode，无需 Blender UI。使用现有本地视频、Camera Pose、61 帧预测语义缓存及已通过的 RGB 相机对应点校准，不重新下载数据或调用远程服务。

```powershell
# 50 cm 体素，先生成场景并检查九个预览视角
.\.venv\Scripts\python.exe -m reconstruction.heuristic_proxy --allow-heuristic --preview-only
# 检查预览后，从同一 Scene JSON 生成完整视频
.\.venv\Scripts\python.exe -m reconstruction.heuristic_proxy --allow-heuristic --render-only
# 校验输入/代码/资产/输出哈希后重复复用
.\.venv\Scripts\python.exe -m reconstruction.heuristic_proxy --allow-heuristic --render-only --reuse
# 导出中断后复用未改变的体素数据，重新执行安全/覆盖检查
.\.venv\Scripts\python.exe -m reconstruction.heuristic_proxy --allow-heuristic --preview-only --resume
# 仅当 Blender 渲染/验证代码改动后，从未改变的 Scene JSON 重建预览
.\.venv\Scripts\python.exe -m reconstruction.heuristic_proxy --allow-heuristic --refresh-previews
```

默认输出 `outputs/stage_ef/heuristic_voxel_proxy/`，可用 `--output` 指定独立目录。可调 `--voxel-size 0.25..1.0`；恢复导出必须保留先前体素尺寸。正常重生成会保留所有先前的 `source_bundle_*`。同目录并发写被锁拒绝；不要删除另一个正在运行的进程的锁。

直接使用 Stage E/F 入口时，代理场景同时要求 `--allow-inferred --allow-heuristic`。后者必须绑定独立的 `heuristic.proxy/1` 安全/覆盖回执；不能仅靠旧的 RGB 相机校准回执让启发式几何自动通过。源 Scene JSON、GLB/PLY、预测遮罩和回执都有明确的推断来源。

## 做了什么

1. 冻结已选中的原始相机快照关联、坐标转换及 inferred 焦距，720 条相机姿态、源 tick/秒时间和 60 FPS 不变。没有重新设计相机路径。
2. 在相机最低高度下约 1.85 m 假设地面，使用训练帧的地面/草地遮罩射线提供代理尺度锚点。原生深度码仅是单调的远近轮廓提示，不套用未知物理逆公式，不升级米制深度校验结果。
3. 固定粗颜色、至少两个训练帧支持的静态体素，再补齐连续地面、岩石/建筑/栏杆体积、草灌木柱和树冠。填充可超过原始点云可观测范围，均明确标为 `heuristic_voxel` / `inferred_proxy_aabb`。补齐部分可无直接源观测，不伪造观测回执；非代理 Scene 的观测约束不变。
4. 对全部 720 条相机位置预留 2.5 m 的非地面自由走廊，并保守计入体素半对角线。检查所有相机都不在占用体素内。
5. 只导出跨类别全局外表面，按 8 m 分块；GLB 内嵌 4096 色调色板。原视频不逐帧投射到场景，不做人物模型/骨骼、复杂 PBR 或精细贴图。
6. 九个视角检查环境覆盖和近距离墙面，再后台保存、独立重开 `.blend`、渲染 PNG 并检查实际渲染覆盖。SSIM 仅记录为参考，不冒充几何验证。旧重建路线的 SSIM/重投影门槛仍保留。
7. 完整渲染、编码、FFmpeg 回读检查：粗模 1920×1080，对比 3840×1080，均 720 帧、60 FPS、12 秒；源帧/tick 映射单独保存。视频无音频。

`proxy_run.json` 记录本次状态、源文件哈希、预览 PNG 哈希与完成产物哈希；`proxy_validation.json` 记录代理假设、单元预算、自由走廊和覆盖检查。原 RGB 校准回执只证明抽样相机对应关系，不证明这个场景的形状/尺寸正确。Blender 保存重开检查只证明生成文件中的相机/网格/时间轴一致。

限制：推断地面可能不符合原地形高差，原始灯光/雾效不能恢复，物体可能合并或偏移，树干/枝叶只保留粗体素轮廓；删除近相机体素是可用性补齐，不是游戏碰撞几何恢复。

所有源数据、缓存、场景和派生视频仅限本地内部评估，不再分发、不上传第三方，在 **2026-12-24 Asia/Shanghai** 前随受限 Session 删除。`game-dev` 本机不可用，采用项目 Python/Open3D/Blender 后台流程，没有 canonical 包认证或游戏项目 vendoring 声明。

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_heuristic_proxy -v
.\.venv\Scripts\python.exe scripts/run_tests.py --output outputs/stage_ef/heuristic_proxy_tests
```

覆盖代理 LUT 的单调性/非物理标签、全段相机自由空间、岩石与树冠不同填充规则、独立授权、回执/相机哈希绑定、补齐部分不能伪装成已观测表面。原 Stage E/F 后台、重复运行及严格合同测试继续执行。

代理交付时全项目回归 **159/159 通过，无失败、无跳过**，历史回执 `outputs/stage_ef/heuristic_proxy_verified_tests/test_results.json`。后台测试确认静态网格若有驱动/变形不会被优化后的验证跳过；相机仍逐帧验证。这些测试结果不代表真实几何精度通过。

当前实际代理：180,289 个占用体素（含44,946个补齐地面单元），1401个外表面区块；全720帧相机中心均不在占用体素内。九个实际预览的源环境区域覆盖约96.2%–100%，SSIM仅供参考。已观察首/中/末画面有连续地面和环境体积，无先前的相机包围/近距离墙面封堵；仍有细柱、局部悬浮块和物体合并/位置偏差，符合启发式代理而非真实几何还原的证据边界。

## 最终交付与重复运行（2026-11-24）

以下文件已实际生成，`proxy_run.json` 的状态为 `passed`，`pipeline_report.json` 的 Stage E 与 Stage F 均为 `passed`：

| 文件 | 实际大小 | 媒体回读 |
| --- | ---: | --- |
| `outputs/stage_ef/heuristic_voxel_proxy/scene.blend` | 159,171,968 bytes | Blender 后台保存与独立重开验证通过 |
| `outputs/stage_ef/heuristic_voxel_proxy/blockout_video.mp4` | 4,827,818 bytes | 1920×1080，60 FPS，720 帧，12 秒 |
| `outputs/stage_ef/heuristic_voxel_proxy/original_vs_blockout.mp4` | 37,241,068 bytes | 3840×1080，60 FPS，720 帧，12 秒 |

源视频回读同为1920×1080、60 FPS、720帧、12秒；三个视频的全部PTS均已检查。相机与时间轴和冻结的 `real_voxel_rgb/scene_schema.json` 一致，保留原先的 `depth_referenced_camera_raw` 关联、tick偏移+1及inferred焦距缩放0.915；这不证明SDK物理相机合同或真实几何精度。

完整渲染结束后实际执行 `--allow-heuristic --render-only --reuse`，退出码为0，返回 `verified matching input/code/output hashes; reusing`。它核验输入、代码及完成产物后复用本目录，没有再次渲染。权威最终回执为 `proxy_run.json`、`pipeline_report.json`、`saved_scene_verification.json` 和 `frame_mapping.json`；历史中断/失败回执保留，不作为当前代理状态。

## 灰模基本体（并行输出）

体素壳保留轮廓，基本体才是可单独移动的导演粗模。`reconstruction.primitive_blockout` 只读取上面已经通过穿模和覆盖检查的 `proxy_voxels.npz`，不重跑深度，也不改写 `heuristic_voxel_proxy/`。

地面用一块薄板，树用圆柱树干加球体树冠，草/灌木用圆柱，岩石、建筑和其余实体用实心方块。材质是统一灰模。相机、源 tick 和 60 FPS 从体素场景原样复制。几何来源是 `heuristic_primitive`，和体素不能写进同一个 Scene。

```powershell
.\.venv\Scripts\python.exe -m reconstruction.primitive_blockout --allow-heuristic --preview-only
.\.venv\Scripts\python.exe -m reconstruction.primitive_blockout --allow-heuristic
```

默认输出 `outputs/stage_ef/heuristic_primitive_proxy/`。通过后才有 `scene.blend`、`blockout_video.mp4`、`original_vs_blockout.mp4`。这仍是启发式代理，不是测量几何。

## 测试与死代码清理（2026-11-24）

测试方法由159个精简至145个：Stage A的端点检查已被1024值往返测试覆盖，重复同步诊断及其固定目录图片/报告生成已删除；Scene Schema和渲染/批量参数的负例合并为带名称的子测试，原负例断言保留。全部Blender后台集成、穿模、静态驱动拒绝、相机/时间轴和哈希复用测试继续执行。

删除未被入口、模块或文档引用的 `read_frame_table`、未使用的导入、`make_scene` 的curve参数、`bake_texture` 的category参数、TSDF构造器的width/height参数及字段，以及纹理选帧中的无效重复运算。实际Parquet适配仍在样本准备入口使用，PyArrow依赖保留；旧重建路线有真实引用，不作为死代码删除。

清理后全项目 **145/145通过，无失败、无错误、无跳过**，记录在 `outputs/stage_ef/cleanup_tests/test_results.json`。另对精简后的贴图烘焙及TSDF接口执行内存合成检查通过。清理不修改Blender生成/渲染代码、Scene JSON或已交付的场景/视频文件，也不改写历史验收结果。
