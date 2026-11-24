# 真实视频的彩色体素粗模（实验性 inferred 路径）

目标是比单个箱体/平面更完整的 **low-poly 粗模**，不是精细模型或写实材质恢复。默认使用 20 cm 体素，保留已观测地形起伏、岩石、树木、建筑、栏杆、草/灌木的轮廓和粗粒度颜色。人物仅作为动态排除遮罩；本轮不做人物模型、骨骼动画或不可见区域补全。

`reconstruction.refine` 是独立可选入口；现有 Scene 1.0.0 的严格物理合同路径不变。Scene 1.1.0 的推断结果必须明确 `--allow-inferred`、所有几何/摄像机置信度标为 inferred，并绑定 SHA-256 校验的通过回执，不能把 `geometry_contract_verified` 改成 true。

## 本地运行

在项目根目录执行；不需要打开 Blender UI。模型 bootstrap 只下载下列公共模型，不下载其他 Session，不上传视频，不调用付费服务。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-refined.txt --extra-index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe scripts/prepare_refined_models.py

# 默认：原始本地样本，20 cm 彩色体素；先执行几何校验和首/中/末+留出帧预览
.\.venv\Scripts\python.exe -m reconstruction.refine --allow-inferred --preview-only
# 更粗的 30 cm 体素
.\.venv\Scripts\python.exe -m reconstruction.refine --allow-inferred --voxel-size 0.3 --preview-only
# 校验和预览均通过后才会渲染完整 1080p / 60 FPS / 720 帧视频
.\.venv\Scripts\python.exe -m reconstruction.refine --allow-inferred
# 仅重新校验，不生成网格/Blender/视频
.\.venv\Scripts\python.exe -m reconstruction.refine --allow-inferred --calibrate-only
# 保留先前纹理表面路线作为可选模式，非默认目标
.\.venv\Scripts\python.exe -m reconstruction.refine --allow-inferred --style textured --preview-only
```

可通过 `--source-session`、`--prepared`、`--models`、`--output`、`--blender` 指定路径。当前真实适配器只支持已准备的该样本：1920×1080、60 FPS、720 帧以及同编号 RGB/Depth/tick 映射；其他合同明确拒绝，不能自动猜测。

模型固定版本：Grounding DINO tiny `a2bb814dd30d776dcf7e30523b00659f4f141c71`、SAM base `70c1a07f894ebb5b307fd9eaaee97b9dfc16068f`；模型卡声明 Apache-2.0，下载 safetensors 并生成 `models.lock.json`，推理时再次核对全部文件哈希。已在本机 RTX 4070 / CUDA 上执行全部 61 个采样帧的识别；识别结果是预测，不是语义真值。

## 管线与复跑

1. 流式原生 RGB / 10-bit Depth 解码，保留每 12 帧及末帧，共 61 帧。识别缓存按源文件、模型锁、识别代码、解码代码及步长独立绑定，修改体素尺寸/校准代码不要求重新识别。复用必须核对准确帧集合和每个 NPZ 的 SHA-256；旧版缓存只在输入与识别代码一致时安全复用。
2. 训练帧 SIFT 双向比例匹配；比较角度单位、yaw/pitch 符号及两种原始摄像机快照。使用多个时间间隔，排除负深度、低视差和重投影异常点。
3. 用训练内部分割比较 same-raster / depth-referenced-snapshot 关联与单调 PCHIP / 平滑 robust log1p 曲线。平滑曲线仅是自由参数的经验拟合，不是借用其他数据集的物理逆公式。最终留出帧不参与拟合和模型选择。不同 depth 摄像机的结果经最近表面 z-buffer 对齐至 RGB，不填洞。
4. 只有最终留出验证通过才融合几何。默认体素每独立训练观测一票、至少两次支持；精确重复输入不重复计权，人物不进入体素。一个网格单元只选择一个语义类别，剔除跨类别内部面，不创建海量独立 cube objects。默认最多 400,000 个占用单元，超过预算报错，要求增大体素尺寸，不静默截断场景。
5. 按 8 m 区块生成真实表面网格；块边界根据体素归属处理，避免丢失边界面。颜色量化为 4096 色的小型内嵌调色板，每面固定色/平面法线，不投射原视频纹理。每区块记录几何融合输入帧和实际可见的代表观测；局部米制网格维度不会应用两次。
6. 原始源帧/tick/秒时间保持不变；每帧相机矩阵来自被选中的原始快照及明确坐标转换，运动没有人为重设计。快照选择仍是 inferred，不等于 SDK render-latch 已获证明。后台保存 `.blend` 后独立重开验证网格包围盒、UV、打包图片、相机/时间轴和 FPS。
7. 在相同真实源数据、相机、尺寸和时间轴上比较体素结果与单箱体基线，首/中/末和六个留出帧均预览。留出区域 masked SSIM 平均须优于基线，才批量渲染全部帧、编码并检查 MP4 帧数/PTS。SSIM 是图像一致性指标，不是几何真值证明。

输出目录默认为 `outputs/stage_ef/real_refined/`。通过全流程后才有 `scene.blend`、`blockout_video.mp4`、`original_vs_blockout.mp4`；报告、日志、源 Scene JSON、GLB/PLY/遮罩与缓存均在该独立目录。源视频不被覆盖。锁防止同目录并发写；阶段异常持久记录失败类型，校验失败返回 2，执行异常返回 1。

## 当前真实验证结果（2026-11-24）

原生深度逆变换路线 **未通过**：603 个对应点，中位数 **6.92 px**、P90 **29.25 px**，未使用该逆变换生成新几何。要求中位数 ≤3 px、P90 ≤10 px、总样本 ≥100、每个留出对 ≥10，门槛没有因为 low-poly 而降低。

独立 `reconstruction.rgb_blockout` 改用至少三个训练视图 SIFT/LK 对应点三角化，不使用原生深度逆公式。训练集选择原始 `depth_referenced_camera_raw`、快照偏移 +1 和焦距系数 0.915（fx=fy=878.4），原始帧/tick/60 FPS 保持不变；该相机绑定是 inferred，不是 SDK 合同证明。六个未参与拟合的 RGB 留出帧共有 627 个点，中位数 **1.99 px**、P90 **7.05 px**，达到原门槛。但后段支持薄弱，660 帧仅 12 个点，不能声称所有帧/稠密点/实体均准确。

实际执行结果：

- `outputs/stage_ef/real_voxel_rgb/`：125 个网格区块，后台生成/保存重开和九帧预览完成；表面太稀疏，图像质量失败。
- `outputs/stage_ef/real_voxel_solid/`：40 cm 的训练语义轮廓约束实体试验；25 个导出区块，图像质量失败，历史产物已于 2026-11-24 清理。
- `outputs/stage_ef/real_voxel_solid_v2/`：修复体素中心投射遗漏，使用完整网格表面射线可见性；65 个网格、`scene.blend`、九帧预览已生成并重开验证。SSIM 平均优于同源同相机同包围盒方块基线，但真实画面仍有大面积缺失和近距离墙面。261/720 帧相机中心在候选实体中，新增 `volume_camera_check` 明确拒绝完整渲染，最终 `quality_rejected`。

`real_voxel_solid_v2/` 的历史产物也已于 2026-11-24 清理；上述记录保留历史验收结论，不表示实验文件仍在本地。`real_refined/` 的共享帧缓存、`real_voxel_rgb/` 的相机和源数据继续保留，供当前 Voxel / Primitive 工作流使用。

三次实验均 **没有新版 `blockout_video.mp4` / `original_vs_blockout.mp4`**。SSIM 单独通过不是可用性证明，不能为输出视频而放宽质量门槛。所有体素内体积、拟合地面和网格可见性深度都是推断；完整表面射线深度仅用于生成网格自身的可见性，绝非源测量深度。

重建实验命令（权重和 61 帧缓存已经本地存在，无需再次下载/分割；实体试验须先重建，不能直接复用已清理的产物）：

```powershell
# RGB 留出验证及表面候选；失败候选不会自动渲染整段视频
.\.venv\Scripts\python.exe -m reconstruction.rgb_blockout --allow-inferred --preview-only
# 独立实体试验；每次复跑保留先前 source_bundle，仍受相机穿模和图像质量约束
.\.venv\Scripts\python.exe -m reconstruction.rgb_hull --allow-inferred --output outputs/stage_ef/real_voxel_solid_v2
# 仅接受两项质量检查均通过的候选，重验源/资产/代码/实体缓存哈希后后台渲染
# 历史 v2 被相机穿模检查拒绝；重建后的候选仍须通过相同质量门槛
.\.venv\Scripts\python.exe scripts/render_rgb_blockout.py --output outputs/stage_ef/real_voxel_solid_v2
# 已通过候选才可用 --reuse 复用经哈希验证的完整输出
```

后续若要生成更完整的视觉代理，需要用户明确接受更广泛的启发式地形/物体补齐；它不再是已验证的几何还原。另一条路线是取得该 Session 的物理深度公式和 SDK render-pose 绑定证据。不要把算法/Blender 合成集成测试通过当作真实视频验收。

报告：`calibration_report.json`、`refinement_report.json`；诊断图：`rgb_depth_diagnostic.jpg`。数据和派生产物仅限本地内部评估，不上传第三方、不再分发，随受限样本在 **2026-12-24 Asia/Shanghai** 前删除。

## 回归测试

2026-11-24 最新全项目回归 **153/153 通过，无失败、无跳过**，包括两项相机穿模及 SSIM 不能绕过穿模的测试、Blender 后台实际渲染/保存重开/重复执行；结果在 `outputs/stage_ef/rgb_solid_final_tests/test_results.json`。这些是算法/合成集成证据，不代表真实粗模质量通过。

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_refined tests.test_voxel_refinement tests.test_rgb_blockout tests.test_rgb_hull -v
.\.venv\Scripts\python.exe scripts/run_tests.py --output outputs/stage_ef/rgb_solid_final_tests
```

覆盖数值投影/三角化、两种深度关联基础、校准训练/留出隔离、未知码值不外推、内部面剔除、单元预算、重复帧和人物排除、分块完整性、调色板 GLB、Scene 合同/来源分离、缓存哈希与失败报告；另有体素调色板后台实际渲染/保存重开/预览像素颜色检查。原有 Stage E/F 后台集成测试继续检查 MP4、保存重开和重复运行。`game-dev` CLI 本机不可用，本实现采用项目 Python/Open3D/Blender 后台流程；没有声称通过该工具的 canonical vendoring/package certification。
