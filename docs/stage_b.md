# Stage B 实现与坐标合同

当前交付是**已验证算法 + 合成端到端产物**，不是已验收的真实游戏场景重建。真实样本的 Stage A 仍存在物理深度与 render-pose 缺口；保留拒绝解码和拒绝融合的安全门。

## 先验证了什么

本地复核 `66ad9535-583f-4a25-b655-6b1697c83a40` 的 Session、decode_contract、world、转换规则、720 帧相机与 Parquet / Depth Metadata、6 个原生码值 NPZ。相机数值与 `last_in_frame` 规则一致。Session、decode_contract 和 world 均为 UE5 / `lh_z_up`，`world_to_meters=100`。这些证据没有定义 v3 码值的物理单位，也没有证明逐帧相机就是渲染该深度的外参。

UE 旋转数学来源：[Epic TRotator 文档](https://dev.epicgames.com/documentation/unreal-engine/API/Runtime/Core/Math/TRotator?application_version=5.5)。角度单位为度，内禀旋转顺序 Yaw→Pitch→Roll，pitch/roll 的正向与 yaw 不统一。已通过独立 closed-form 轴向测试、混合角测试、矩阵逆变换测试以及 Blender 5.2.1 的 `mathutils.Euler` / `world_to_camera_view` 检查。

Blender 独立检查只创建内存中的测试相机，未导入受限数据、未保存 `.blend`、未渲染或实现 Stage E。12 组投影最大误差 `1.430511474609375e-6`，4 组旋转矩阵最大误差 `1.2913095889799386e-7`，容差 `2e-5`。

真实 Session 的 SDK 是否直接输出此 FRotator 定义、像素采样中心、depth snapshot latch 的时刻，仍标记未验证；不将数学单元测试等同于数据源合同证明。

## 坐标定义与矩阵方向

| 空间 | 轴定义 | 尺度 |
|---|---|---|
| Optical camera | +X 向图像右，+Y 向图像下，+Z 向前 | 反投影后统一为 m |
| UE camera-local | +X 向前，+Y 向右，+Z 向上；左手 | 光线基变换，不改变长度 |
| UE world | Session 声明的左手 Z-up | 位置除以 `world_to_meters` |
| Blender world（本项目选择） | UE (X,Y,Z) → (X,-Y,Z)，右手 Z-up | m |
| Blender camera-local | +X 向右，+Y 向上，-Z 为观看方向 | m |

`world_to_meters=100` 的含义是每米 100 个 UE 世界单位。它只约束相机位置尺度；不能代替独立的 `DepthSpec.meters_per_unit`。

所有算法采用**列向量左乘**，JSON 仅按行存储数字：

```text
P_optical = D_m * inverse(K) * [u+pixel_offset, v+pixel_offset, 1]^T
C = UE_FROM_OPTICAL = [[0,0,1], [1,0,0], [0,-1,0]]
B = BLENDER_FROM_UE = diag(1,-1,1)
R_UE = Rz(yaw) * Ry(-pitch) * Rx(-roll)
R_BlenderWorld_Optical = B * R_UE * C
t_BlenderWorld_m = B * position_UE / world_to_meters
P_BlenderWorld = R_BlenderWorld_Optical * P_optical + t_BlenderWorld_m
```

`C`、`B` 的行列式各为 -1，而最终相机旋转的行列式为 +1，不能只转换世界轴而遗漏相机局部轴。UE 的行向量 FMatrix 表达式要转置；本项目不直接复用 UE 内存矩阵。对于 NumPy Nx3 行存储点，等价实现为 `points @ R.T + t`。

反投影使用的是 camera-to-world（`T_world_camera`），不是 view matrix；后者是其逆。两者都可能通过正交性检查，所以读取 manifest 时必须显式声明矩阵方向，不能靠行列式判断方向。

Blender 相机对象矩阵为 `T_BlenderWorld_Optical * diag(1,-1,-1,1)`。一般物体在两个世界基中的变换需要左右两侧都换基，不能把世界反射当作普通 Euler 旋转。

具体单测：UE 相机位置 `(100,200,300)` 世界单位，零旋转，100 世界单位/m；optical 点 `(1,2,5)` m 转为 Blender `(6,-3,1)` m。+90° yaw 把 UE +X 转向 +Y；+90° pitch 把 +X 转向 +Z；+90° roll 把 +Y 转向 -Z。均在 `tests/test_stage_b.py`。

## 算法与输出

- `backprojection.py`：只接受有证据的浮点 `planar_z`，显式单位、单位换算、像素中心偏移、原分辨率 K 和已对齐 RGB。过滤 mask、NaN/Inf、非正深度、near/far 范围；拒绝整数 raw codes / relative depth / ray length。支持 K skew；stride 只采样像素，不错误缩放 K。
- `pointcloud_fusion.py`：Open3D 转换、固定世界网格 `voxel_down_sample_and_trace`、每独立观测每体素一个均值、跨帧融合、Statistical Outlier Removal。每帧的像素数量不直接决定该帧对体素的权重。
- 重复判定：已确认 source observation ID，或完全一致的 depth+mask+RGB+K+capture pose+深度合同。相同深度但相机移动不被当作重复；同 observation ID 但不同像素/外参会报错。不能从 Session 全局 `dup_count` 猜出重复帧位置。
- 保留 delivery/RGB/depth frame index、original tick、秒时间戳、原始观测 ID、K、矩阵和所有重复别名。`point_provenance.npz` 为 CSR：顶点 i 的来源切片是 `offsets[i]:offsets[i+1]`，包含 `frame_indices` 与 `timestamp_seconds`。其顶点顺序与融合 PLY 完全一致，包括 SOR 后的索引更新。
- Binary PLY 包含 `units meter`、Blender 坐标与数据来源注释。写出后 Open3D 回读，检查点数、double 坐标和 RGB 8-bit 量化，并记录 SHA-256。
- 初版是静态点云累积，不处理动态物体运动、TSDF、ICP、网格、语义或场景生成。多帧重影不应被宣称已经解决。

合成 demo 使用独立解析房间/箱体 ray caster，输入深度明确为 cm，半像素中心，帧时间跨度约 12 s；非零源帧索引与 original tick 独立保留。8 个输入 / 7 个独立观测 / 1 个重复跳过，原始有效点 25,172，单帧 3,596 点，0.08 m 体素融合后 SOR 去除 507 点，输出 12,675 点。反投影相对独立真值最大误差约 `3.97e-15` m；融合点到解析表面最大距离约 0.03214 m（边缘处体素均值误差），没有把合成误差当作真实样本精度。

## 运行和复现

在项目根目录运行（无需重新下载数据）：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# 全部测试 / 单独 Stage B 测试
.\.venv\Scripts\python.exe scripts/run_tests.py --output outputs/stage_b/tests
.\.venv\Scripts\python.exe scripts/run_tests.py --pattern test_stage_b.py --output outputs/stage_b/tests/stage_b_only
# 默认仅复核真实样本。Python 返回 2 表示物理合同未确认，不生成真实场景点云。
.\.venv\Scripts\python.exe scripts/reconstruct_scene.py
# 明确选择合成 demo；绝不把它作为默认的真实数据 fallback。
.\.venv\Scripts\python.exe scripts/reconstruct_scene.py --synthetic
# 保存的已知深度 manifest 可走同一通用输入路径，结果仍标注 SYNTHETIC。
.\.venv\Scripts\python.exe scripts/reconstruct_scene.py --input-manifest outputs/stage_b/synthetic_demo/synthetic_input_manifest.json --min-depth-m 0.1 --max-depth-m 40 --output outputs/stage_b/synthetic_manifest_replay
# 独立 Blender 数学验证；不是 Stage E 场景构建。
& 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe' --background --factory-startup --python-exit-code 1 --python scripts/verify_blender_coordinates.py -- outputs/stage_b/validation/blender_coordinate_probe.json
```

`--output` 默认指向本阶段独立目录。重跑会更新该目录内本工具生成的同名产物；不要指定受限原始输入目录或用户其他产物目录。默认真实 audit 的 exit 2 与普通失败 exit 1 区分；PowerShell 如需向父进程透传数值，可在 Python 后 `exit $LASTEXITCODE`。

通用输入合同见生成的 `synthetic_input_manifest.json`：schema=`stage_b.metric_frames/1`，coordinate_system=`blender_rh_z_up`，pose_direction=`world_from_optical_camera`，pose_translation_unit=`m`。`depth_spec` 必须具备 unit、meters_per_unit、depth_type、pixel_center_offset、verified、evidence；`verified` 是调用方审查声明，不是勾选即可把未知数据变成真值的选项。每帧 NPZ 必须含浮点 `depth`、布尔 `valid_mask`、uint8 `rgb`、原分辨率 `K`。`frames` 中提供完整索引、时间、matrix、geometry_pose_verified、sync_evidence 和 source_observation_id。载入上限为 720 帧 / 512 MiB 解压 NPZ，反投影点预算默认 2,000,000。

外部已审查输入必须显式传入 near/far 过滤范围。现有 raw code NPZ 缺少物理合同，不能换字段名或强制转换为 float 后充当 metric 输入。

## 真实样本继续条件

1. 提供适用于当前 Session / transform=3 的逆公式、原生码 bin/归一化规则、输出单位、far_plane_encode 和 near_plane 的尺度关系，以及饱和/零码规则。
2. 明确 SDK pitch/yaw/roll 的单位、旋转顺序/正方向、camera-local 轴，以及 camera.jsonl 与 depth render pose 的对应时刻（不能只看两个记录的 Frame Index 一致）。
3. 确认 K 的像素中心与投影采样规则，完成少量跨帧重投影检验后才执行真实多帧融合。

其他 OriginLab release 的[新 v3 文档线索](https://huggingface.co/datasets/originlab/game-recordings-free-camera-v2/blob/8731e47/README.md)描述了 16-bit code-bin midpoint、far×near_cm 以及 saturation mask，但不与本地固定版本 Session 混用；其近裁面尺度也有部分游戏在复核的说明。

真实输入和派生遥测仍仅限本地内部评估，不上传第三方、不再分发，保留原删除期限 2026-12-24（Asia/Shanghai）。本次遵循 game-asset-production 技能的证据边界，区分数学检查、导出回读与数据源物理验证；本机未安装 game-dev，未执行其生成、规范化或资产包工作流，直接使用项目 Python/Open3D。
