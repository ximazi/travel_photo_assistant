# Travel Photo Assistant v0.1 (`travel_photo_assistant`)

基于提交的 `travel_photo_assistant_v0_1` 原型设计材料构建的完整工程化 Python 项目代码，实现**场景观察与拍摄规划分离**、**慢语义规划 + 快端侧对齐双循环**、**JSON Schema Draft 2020-12 与 16 项服务端业务校验**、**预览坐标与镜像变换链**、**元数据过滤知识卡检索（A/B/C 对照）**以及 **E01–E16 离线契约评测框架**。

## 项目结构

```text
travel_photo_assistant/
├── pyproject.toml
├── requirements.txt
├── README.md
├── assets/                                # 完整保留原始 v0.1 设计资产
│   ├── prompts/                           # observer.system.txt / planner.system.txt
│   ├── schemas/                           # scene_observation & shot_plan Draft 2020-12 Schema
│   ├── knowledge/shot_cards.json          # 9 张拍摄知识卡
│   ├── examples/                          # 3 个协议夹具示例
│   │   └── photos/                        # 真实/合成测试照片 + photos.json 观察提示清单
│   ├── evaluation/                        # evaluation_cases.json (E01-E16) & evaluation_protocol.txt
│   └── docs/                              # README.txt / integration_notes.txt / sources.txt
├── src/travel_photo_assistant/
│   ├── config.py                          # 资源路径与运行参数
│   ├── models.py                          # 运行时输入、镜头能力、端侧快照、知识卡数据模型
│   ├── schema_validator.py                # Draft 2020-12 校验 + 供应商 Schema 适配（Gemini / OpenAI / Qwen）
│   ├── validators.py                      # 16 项业务规则校验、生产夹具隔离、物理语义与设备能力守卫
│   ├── coordinates.py                     # 规范化坐标 -> 预览控件 (contain/cover/镜像) 变换链与方向映射
│   ├── knowledge.py                       # 知识卡仓库与元数据/证据约束检索器（A/B/C 三种实验模式）
│   ├── adapters/
│   │   ├── base.py                        # 统一 VisionModelAdapter 抽象接口
│   │   ├── providers.py                   # OpenAICompatibleVisionAdapter & GeminiVisionAdapter
│   │   └── mock.py                        # RuleBasedOfflineVisionAdapter（离线确定性测试与演示）
│   ├── controller.py                      # 双循环运行时控制器（计划锁定、过期丢弃、防抖滞回、预定义完成检查）
│   ├── evaluation.py                      # E01-E16 离线契约与状态机评测器
│   ├── env_loader.py                      # 纯标准库 .env 解析（真实环境变量优先）
│   ├── normalizer.py                      # 真实 VLM 词表归一化（enum 近义词折叠 + 审计日志）
│   ├── photo_pipeline.py                  # 照片 -> 运行时输入 -> 慢/快循环 -> 产物落盘
│   ├── render.py                          # 把方案渲染回原图（复现手机端预览控件的绘制）
│   ├── api.py                             # FastAPI HTTP 服务
│   └── cli.py                             # 命令行入口
├── output/                                # 结果图与 JSON 产物（.gitignore 已忽略）
├── .env                                   # 本地密钥配置（.gitignore 已忽略）
└── tests/                                 # 完整单元测试套件
```

## 快速开始

### 1. 配置 `.env`

复制 `.env.example` 为 `.env` 并填入真实 key（`.env` 已被 `.gitignore` 忽略）：

```bash
TPA_VISION_PROVIDER=gemini     # offline | gemini | openai | qwen
GEMINI_API_KEY=...              # 或 OPENAI_API_KEY=...
```

`.env` 由 `env_loader.py` 用标准库解析（无第三方依赖）。**真实环境变量优先级高于 `.env`**，
所以 `TPA_VISION_PROVIDER=gemini python -m travel_photo_assistant ...` 会临时覆盖 `.env` 里的设置。

```bash
# 查看当前生效的配置（永远不打印 key 内容，只显示 set/unset）
PYTHONPATH=src python -m travel_photo_assistant env
```

### 2. 运行全部单元测试

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

### 3. CLI 常用命令

```bash
# 校验原始 examples 夹具及生产隔离守卫
PYTHONPATH=src python -m travel_photo_assistant validate-examples

# 运行离线慢循环 + 快循环双循环演示
PYTHONPATH=src python -m travel_photo_assistant demo

# 运行 E01-E16 离线契约与状态机评测
PYTHONPATH=src python -m travel_photo_assistant evaluate

# 测试知识卡元数据与视觉证据检索
PYTHONPATH=src python -m travel_photo_assistant check-cards --tags coast leading_line

# 测试预览坐标变换（含前置镜像）
PYTHONPATH=src python -m travel_photo_assistant transform-coord --bbox 0.2,0.3,0.4,0.8 --mirrored --direction left --direction-ref input_image
```

### 4. 真实照片流水线与结果图输出

```bash
# 列出本地测试图
PYTHONPATH=src python -m travel_photo_assistant list-photos

# 对一张真实照片跑完整双循环，并把结果图写进 output/<照片名>/
PYTHONPATH=src python -m travel_photo_assistant run-photo --photo 5.png
```

产物目录结构：

```text
output/5/
├── observation.json   # 结构化场景观察（Draft 2020-12 校验过）
├── plan.json          # 拍摄方案（含动作链、完成检查、失效条件）
└── result.jpg         # 渲染结果图：目标框 / 当前人物框 / 锚点框 / 引导箭头 / 指令横幅
```

结果图上：
- 🔴 **红框 TARGET** — 人物应该到达的位置
- 🟢 **绿框 NOW** — 人物当前的位置
- 🩵 **青色人形轮廓 虚框·站位** — 画面无人时，人物应该站的位置（人形轮廓，按全身/半身画法）
- 🟡 **黄框 裁剪 16:9** — 建议的成片裁剪区，区外压暗
- 🔵 **蓝框** — 需要保留的环境锚点
- 🟠 **橙箭头** — 快循环给出的实时引导方向（前置镜像时箭头同步翻转）

常用参数：

```bash
--goal "..."                 # 覆盖拍摄目标（默认取 photos.json 里的每图目标）
--scene-tags coast forest    # 覆盖观察到的场景标签
--subject-bbox x0,y0,x1,y1   # 覆盖人物框
--mirrored                   # 按前置镜像预览渲染（箭头会翻转）
--no-metadata                # 模拟设备缺少镜头元数据
--out <目录>                 # 覆盖输出目录
```

### 5. 照片清单 `assets/examples/photos/photos.json`

每张照片的离线观察提示写在这个 JSON 清单里，**新增照片不需要改 Python 代码**：

```json
{
  "photos": {
    "5.png": {
      "scene_tags": ["park", "lake", "clean_background"],
      "detected_person_count": 1,
      "subject_bbox": [0.228, 0.800, 0.236, 0.840],
      "face_visibility": "too_small",
      "anchors": [{"anchor_id": "tree_line", "kind": "horizon", "bbox": [0, 0.6, 1, 0.8]}]
    }
  }
}
```

`subject_id` 设为 `null` 可强制触发「多人待选择」；`detected_person_count` 设为 `0` 可验证
「画面无人时不编造人物」。

### 6. 启动 HTTP API 服务

```bash
# 默认使用离线规则适配器；在 .env 里设 TPA_VISION_PROVIDER 切换真实 VLM
PYTHONPATH=src python -m travel_photo_assistant serve --host 127.0.0.1 --port 8000
```

核心接口：
- `GET /health`：健康检查、当前适配器与 `.env` 状态
- `GET /v1/cards`：查看已加载的 9 张拍摄知识卡
- `POST /v1/session/slow-step`：执行慢循环（观察 → 卡片检索 → 规划 → 业务校验 → 计划锁定）
- `POST /v1/session/fast-tick`：执行快循环（端侧跟踪帧 → 失效检测 → 防抖滞回 → 动作推进）
- `POST /v1/transform`：将规范化 `[xmin, ymin, xmax, ymax]` 与方向映射到预览控件像素
- `GET /v1/photos`：列出本地测试图
- `POST /v1/photos/run`：对一张照片跑完整流水线并渲染结果图

### 7. 人物虚框（场景无人时人物往哪站）

旅行场景经常没有人，但成片要带上人物。因此 `shot_plan` 顶层有一个 `subject_placement_guide`：

```json
"subject_placement_guide": {
  "bbox": [0.428, 0.375, 0.572, 0.9],
  "framing": "full_body",
  "target_height_ratio": [0.45, 0.6],
  "instruction": "被拍者站到虚框位置，身体完整入镜、头顶不贴画面上沿，脚下踩稳后再拍摄。",
  "basis": "rule_of_thirds",
  "fact_ids": ["f1"]
}
```

**虚框是人形轮廓，不是矩形**。App 端按 `framing` 画出头、颈、肩、躯干和腿：

| `framing` | 覆盖范围 | 画面高度占比 | 头身比 |
|---|---|---|---|
| `full_body` | 头顶到脚 | 45%–60% | 7.5 头身 |
| `three_quarter` | 膝盖以上 | 52%–66% | 5.4 头身 |
| `half_body` | 腰部以上 | 50%–62% | 3.75 头身 |
| `headshot` | 头肩 | 26%–38% | 2.2 头身 |

肩宽按站立全高 × 0.26 反推，**不随 framing 变化**（半身和全身的肩一样宽），
所以人物在任何取景下都保持真实比例。近景时被拍者离镜头更近，人也画得更大。

`framing` 由用户目标决定：目标里出现「近景 / 特写 / 半身 / 胸像」→ `half_body`；
出现「全身 / 环境 / 人景」→ `full_body`；默认 `full_body`。

规则：

| 观察结果 | `subject_placement_guide` |
|---|---|
| `observed` 且画面无人 | **必须非空**（否则报 `MISSING_PLACEMENT_GUIDE`）|
| 画面已有人物 | **必须为 null**（否则报 `PLACEMENT_GUIDE_CONFLICTS_WITH_SUBJECT`）|
| `insufficient_input` / `needs_subject_selection` | **必须为 null**（否则报 `PLACEMENT_GUIDE_NOT_ALLOWED`）|

位置依据 `basis` 四选一：`rule_of_thirds`（三分线）、`centered_symmetry`（对称中轴）、
`leading_line_anchor`（顺着引导线）、`open_area`（画面留白处）。

虚框是**构图建议**，不是现场测量——不会因此断言地面可站立、光照条件成立。

### 8. 景观裁剪建议（crop_recommendation）

纯景观画面还需要一张能直接用的裁剪建议：

```json
"crop_recommendation": {
  "crop_rect": [0.005, 0.084, 0.935, 1.0],
  "aspect_ratio": "16:9",
  "rationale": "按 16:9 保留 snow_peak，同时保证人物虚框完整落在裁剪区内。",
  "scene_basis": ["mountain", "lake", "city_view"],
  "focus_anchor_ids": ["snow_peak"]
}
```

规则：

| 观察结果 | `crop_recommendation` |
|---|---|
| `observed` 且画面无人 | **必须非空**（否则报 `MISSING_CROP_RECOMMENDATION`）|
| 画面已有人物 | **必须为 null**（构图由人物决定，否则报 `CROP_CONFLICTS_WITH_SUBJECT`）|
| `insufficient_input` / `needs_subject_selection` | **必须为 null**（否则报 `CROP_NOT_ALLOWED`）|

**裁剪区必须完整包含人物虚框**——否则报 `CROP_EXCLUDES_PLACEMENT_GUIDE`。
裁完人没了就失去意义，这是两者之间最关键的联动约束。

比例跟着场景走：对称建筑 → `1:1`，开阔山水 → `16:9`，传统街巷 → `3:2`。
渲染时裁剪区外会被压暗，区内保持原亮度，黄色边框标出成片范围。

### 9. 真实 VLM 的词表归一化与供应商 schema 适配

用真实 Gemini 跑出来的三个问题（离线测试全都测不到，因为 mock 不走供应商 schema）：

**① 供应商不强制 `enum`。** Gemini 会返回 `status:"ok"`、`source:"vision_hint"`、
`facts.kind:"environment"`、`issues.priority:"critical"` 这类表外取值，导致每次观察都被服务端
校验拒绝。三层应对：

1. **提示词给出完整取值表** — `assets/prompts/observer.system.txt` 与 `planner.system.txt`
   末尾的「必须逐字使用的枚举取值」段落。
2. **词表归一化层**（`normalizer.py`）— 把近义词折叠回规范词（`ok→observed`、
   `vision_hint→vlm_visual`、`environment→scene`、`critical→high` …）。
   **只做闭集映射**：表外的词一律不动，仍由 Schema 校验拒绝，绝不放宽契约。
   每次改写都记录在 `SlowLoopResult.normalization_changes` 里，可审计。
3. **带错误反馈重试一次** — 校验仍不过时，把服务端错误回传给模型让它改一次
   （`RuntimeConfig.retry_on_validation_error`，默认开启）。

**② 适配器把 `properties` 映射当成 schema 节点，导致名为 `title` 的属性被丢弃。**
症状是 Gemini 返回 `HTTP 400: properties[primary_plan].required[0]: property is not defined`——
**这条真实规划路径从来没通过过**。现在 `properties` 按映射（属性名 → 子 schema）递归处理。

**③ Gemini 的状态空间限制。** 适配后的 schema 若带数组长度约束（`minItems`/`maxItems`）和数值边界
（`minimum`/`maximum`），Gemini 会拒绝：
`The specified schema produces a constraint that has too many states for serving`。
这些约束现在**只从给供应商的提示 schema 里剥掉**，服务端仍用完整 Draft 2020-12 Schema 校验，
契约一点没放松。

另外：请求超时默认 **120 秒**（`TPA_REQUEST_TIMEOUT` 可调，30 秒在真实模型+大图+代理下必超时）；
图片 mime 按**文件头字节检测**，不再相信调用方标的 `image/jpeg`。

供应商返回结构正确不代表观察正确——完整服务端校验始终保留。

## 重要说明（遵循 v0.1 协议边界）

- **生产隔离**：`production_mode=True` 时严格拒绝 `example_only=true` 与 `source=synthetic_fixture` 数据；空 `image_attachments` 强制返回 `insufficient_input`。
- **评测口径诚实性**：`evaluate` 命令仅验证离线协议契约、业务规则与双循环状态机行为；所有实拍视觉准确率与盲评成片质量指标均明确保持 `field_photo_status="not_executed"`，不回填虚构实拍成绩。
- **照片来源**：服务端**不产出图片**。输入图片通过 `frame.image_attachments` 传入（支持 `data_base64` / `local_path` / `uri`）；输出是 JSON + 预览像素坐标，取景框和箭头由手机 App 端绘制。`render.py` 只是离线复现 App 端的绘制行为，供离线评审用。
- **`photos.json` 里的 bbox 是人工目测估计值**，误差可达 5-10%，只用于跑通坐标链路；真实视觉准确率必须用真实 VLM + 真实照片重新评测。
