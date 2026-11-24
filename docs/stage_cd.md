# Stage C / D：场景理解与版本化 Scene Schema

实现状态：Stage C/D 核心、五类合成端到端验收、辅助 RGB 样本输出已实现。真实米制几何仍被 Stage A/B 的未验证合同阻塞；**未声称真实五类自动检测精度已达标**。Stage E/F 已在后续实现并通过已验证合成输入验收，见 [stage_ef.md](stage_ef.md)；Stage G 未实现，无新样本下载、无受限数据外传。

## 五类范围与证据

当前支持 `person / tree / rock / ground / barrier`。角色使用 capsule，树用 cylinder，岩石/栏杆用 box，地面用 plane。Stage C 输出位置与尺寸来自 Mask 内的世界点，不来自语言模型编造的位置。

默认入口先读取已提取的 6 张真实 RGB 关键帧：0、120、240、360、600、719。`config/sample_66ad_annotations.json` 保存 Codex 对本地 RGB 的辅助视觉审查、粗略 polygon mask、置信度、角色身份提示及静动态提示。五类主要物体在首帧有示例，角色在六帧中保持一个 ID；首帧之外的树/石/地面/栏杆没有被强行认作首帧实例。240 帧角色只有框代理，不能用于几何。此样例是人工辅助流程接口的示范，**不是精确分割真值、完整场景标注、已基准评测的 VLM 结果或人工认证**；所有这些推断为 `inferred`。静态首帧实例只凭审查提示标记，非测量结果。

默认真实分支输出五物体、10 条观测、6 条相机时间记录，未知世界位置、尺寸及外参为 `null`，几何 Confidence 为 `unobserved / 0`；无虚假 Motion Track 或世界空间关系。原始 720 帧遥测路径保留在 camera `raw_source`，不将未验证旋转转换为可用 Camera Track。原样本及派生产物仍限本地内部评估，沿用 2026-12-24（Asia/Shanghai）删除期限。

可选 `LocalVisionProvider` 使用本地 Grounding DINO 的 RGB 分类/检测及本地 SAM 的实例遮罩。模型路径必须已存在，`local_files_only=True`，不隐式下载、不调用远程模型。当前环境没有 torch/transformers/权重，因此该适配器**未实机验收**；这是后续自动识别入口，不代表真实识别可靠性已验证。SAM 2/VLM 视频推理尚未集成。无 SAM 时保留 bbox_proxy，不从框内背景估计几何。模型原始分数为 `predicted`，不冒充校准的准确率。

## 数据与几何链路

1. `scripts/understand_scene.py` 接受现有 Stage B `stage_b.metric_frames/1`，复用已审查的 DepthSpec、optical camera-to-world、原始帧/tick/seconds。未知 Depth 单位、未审查姿态或输入不一致明确失败，不降级到 RGB-only。
2. `reconstruction/object_geometry.py` 复用 `backproject`，取 Mask 内的有效米制深度点，转为 Blender 世界右手 Z-up。Mask 必须与 RGB/Depth/K 在原生分辨率对齐，不静默缩放。
3. 对 Mask 子点云做 SOR，取 0.5/99.5% 分位数 AABB；中心为边界中心。`dimensions` 是**可见表面范围**，不是隐藏完整物体体积。退化轴最小厚度 0.02m 是显式粗模推断；真实旋转未估计，identity quaternion 仅是世界轴对齐粗模约定。几何源始终为 `inferred`。
4. 可在 Stage B manifest 每帧增加 `point_cloud_npz`，文件必须含 `points_world_m / pixels_uv / frame_index`（兼容 Stage B `single_frame_points.npz`）。回查点云与当前 K/Depth/Pose 的一致性，拒绝错误帧、无效/重复像素和陈旧点云。不把无逐像素对应的融合 PLY 整体投影进动态遮罩，以免累积重影。
5. 静态 Track 才合并跨帧对象点，每独立观测每体素一均值；动态及未知 Track 的代表点云仅用首次有效观测。动态/未知位置样本存 Motion Track，静态 Track 不输出动画。

## Persistent ID 与运动

`agents/tracking.py` 用场景级哈希生成稳定 `object_id`：同输入/scene_id 重跑可复现，跨场景不同。不是跨 Session 全局身份，也不是跨进程继续追踪的数据库。

辅助 `identity_hint` 只在同类别中保持身份；同帧重复 hint、身份变类别明确失败。自动分支按类别、世界位置/常速度预测、RGB 直方图、尺寸比及时间间隔进行 Hungarian 一对一关联。RGB-only 用 IoU/外观，无法仅凭图像位移区分相机与物体运动。自动候选存在歧义时拒绝合并，可产生新 ID；不宣称任意遮挡/相同外观交叉都零切换。默认遮挡保持窗口 6s，超出窗口无提示时新建 ID。

静动态分类至少需 3 个独立观测、跨度 >=1s。世界中心最大位移 <=0.30m 或跨帧 80% 表面近邻残差 <=0.12m 推断静态；残差不满足静态且中心位移 >=0.55m 推断动态，中间区间为未知。表面支持检验用于抵抗相机移动造成地面可见范围变化，但不是精密刚体运动估计；遮挡/对称表面仍可能误判。审查提示与测量冲突时保留未知。来源相同的重复交付不增加几何权重或运动证据；各别名帧索引/时间仍保留。

空间关系只比较同帧/时间/相机的已观测几何，输出 `negative_x_of / above / near / intersects`。左右采用世界 X，不混淆屏幕方向；`intersects` 只指 AABB 相交，非物理碰撞认证。Schema 预留 `supported_by`；当前不自动宣称支撑关系，不由关系推断 parent。

## Stage D 合同

定义文件：`config/scene_schema.json`，Draft 2020-12，固定 `schema_version=1.0.0`。未知版本拒绝，暂无隐式迁移。

- 场景：scene_id、scene_type + confidence、created_at、coordinate_system、unit_scale、objects、cameras、timeline、relationships、reconstruction_metadata。
- 物体：object_id、category、primitive_type、transform、dimensions、parent_id、is_static、classification/semantic/tracking confidence、geometry_source、source_frames、motion_track、visibility、observations。
- Transform：`position` 米制世界坐标，`rotation` 单位四元数 **xyzw**，`scale` 无量纲正值。dimensions 为未缩放 primitive 的尺寸。当前 scale=1。所有物体 Transform 均 local-to-world，parent_id 仅组织关系，不把子节点变成 parent-local。
- Camera Track：多 camera_id，世界从 optical camera 的刚体矩阵，逐样本 K、原帧号与秒时间。Blender camera-local 基转换属于 Stage E；不要把 optical 矩阵直接赋给 Blender 摄像机。
- 时间：原始 frame_index、original_tick、source_observation_id 与 timestamp_seconds 分开保留；FPS 显式。关键帧稀疏采样，end_seconds 是最后观测时刻，不代表补齐连续全片；不从 QPC 猜秒。
- 来源：`ground_truth / predicted / inferred / unobserved` 四态。未知数据用 null 与 0 分；几何由已审查输入计算仍是 inferred，不因合同已验证就成为真实几何 Ground Truth。
- Visibility：`visible / not_detected`。未检测不等于已确认遮挡，不承诺两关键帧间持续可见。

`scene_schema/validation.py` 除 JSON Schema 类型/范围/必填项/禁止额外字段，还检查有限数字、带时区时间、唯一 ID、父节点引用/循环、相机引用、矩阵刚体与单位四元数、时间单调/对应、source_frames、一致的可见性、几何来源与 null、bbox 原生边界、禁止 bbox_proxy 几何、静动态 Motion Track、一致的关系观测和完整状态。Schema 验证不能证明模型识别正确或输入合同真实。

## 运行

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# 真实 RGB 辅助输出，预期 exit 2 = partial，不是程序错误
.\.venv\Scripts\python.exe scripts/understand_scene.py
# 独立五类合成演示，预期 exit 0
.\.venv\Scripts\python.exe scripts/understand_scene.py --synthetic
# 合成数据通过通用 metric + annotation 接口重放（不是另一套算法）
.\.venv\Scripts\python.exe scripts/understand_scene.py --input-manifest outputs/stage_cd/synthetic_demo/synthetic_input_manifest.json --annotations outputs/stage_cd/synthetic_demo/synthetic_annotations.json --scene-id synthetic-replay --fps 60 --min-depth-m 0.1 --max-depth-m 35 --output outputs/stage_cd/synthetic_replay
# 全部测试 / Stage C/D 独立测试
.\.venv\Scripts\python.exe scripts/run_tests.py --output outputs/stage_cd/tests
.\.venv\Scripts\python.exe scripts/run_tests.py --pattern test_stage_cd.py --output outputs/stage_cd/tests/cd_only
# 重新验证导出的 Scene JSON
.\.venv\Scripts\python.exe scripts/validate_scene.py outputs/stage_cd/synthetic_demo/scene_schema.json
.\.venv\Scripts\python.exe scripts/validate_scene.py outputs/stage_cd/sample_66ad/scene_schema.json
# 使用已审查的 metric input + 对应 RGB/Mask 标注
.\.venv\Scripts\python.exe scripts/understand_scene.py --input-manifest YOUR_METRIC_MANIFEST.json --annotations YOUR_ANNOTATIONS.json --scene-id reviewed-session --fps 60 --min-depth-m 0.1 --max-depth-m 35 --output outputs/stage_cd/reviewed_input
```

退出码：0 = 几何结构化链路 complete；2 = 明确 partial（真实合同未验证，或对象缺有效几何）；1 = 输入/执行/Schema 校验失败。`complete` 不代表真实语义识别达到准确率门槛。输出包含 `scene_schema.json / validation_report.json / masks/*.png / objects/*.ply / debug/frame_*.png / debug/world_tracks.png`；真实 RGB-only 不输出对象 PLY 或 world_tracks 图。

合成演示是五类独立 ray/AABB 场景，5 个独立观测+1 个重复交付，移动相机、移动角色、4 个静态物体。颜色规则**只用于合成 fixture**。验证报告中的 `fixture_checks` 给出每类 ID 数、切换、静动态分类、Mask IoU、反投影误差及可见中心相对完整体积中心误差，明确不等价于真实模型精度。

RGB-only 外部输入用 `--rgb-manifest`（`stage_c.rgb_frames/1`，data_source、frames[]：frame_index/timestamp_seconds/original_tick/source_observation_id/rgb_path），同时显式 scene_id/fps/annotations。标注格式参考样例；`file` mask 必须二值 0/255，`polygon` 用原像素顶点，`bbox_proxy` 只供语义。

可选本地视觉（需另行准备合法本地权重和 torch/transformers）：`--local-detector LOCAL_DINO_DIR --local-sam LOCAL_SAM_DIR` 替代 annotations。该入口未在当前环境运行，须先用一组审查关键帧评估。地面等开放词类及游戏画风很可能需要修正提示词/阈值或人工复核，不能仅看模型分数。

输出目录只应使用专用 Stage C/D 产物目录；重跑更新本工具的同名文件，不删除旧孤立生成文件，不覆盖原始数据。持久 ID 与 Mask/几何应连同 Scene JSON 一起保留供追溯。
