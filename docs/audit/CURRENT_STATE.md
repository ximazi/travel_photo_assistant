# P0 当前状态：旅行人物摄影助手

- 审计对象：`main@fc443ca821ca9118fd49a66ed6c411a815823ecf`，2026-11-25。目标/权限采用 [Draft PR #6](https://github.com/ximazi/travel_photo_assistant/pull/6) 的三份文件；跟踪 [#1](https://github.com/ximazi/travel_photo_assistant/issues/1)。
- **主判断：离线链路可运行；尚未证明系统可以从真实画面独立读出问题、给出可执行建议或改善成片。** 不重写 Python 原型，先收紧证据、动作和评测输入。
- 核查范围：README、assets/docs、两个 Prompt 与两个 Schema、9 张卡、E01–E16、Python 模块、七份测试及图片清单。审计启动时只有 #6 是打开的 Draft PR，main 只有一个提交；GitHub Actions API 返回 main 为 0 次运行，非 CI 已通过。
- 执行机：云端 Linux；未发现已连接的 Windows worker。没有接生产、付费配额或上传样片。

## 审计拆解

| 链路 | 代码 / 职责 | 证据等级 |
|---|---|---|
| 启动与配置 | `__main__`、`cli.py`：demo/eval/run-photo/API 启动；`config.py`、`env_loader.py`：路径、超时与本地配置。依赖版本见 [复跑](BASELINE_RESULTS.md)。 | 实际 CLI/loopback HTTP；干净克隆的一个 .env 单测失败，分支修正 |
| 输入与契约 | `models.py`：帧、拍摄目标、镜头及可移动约束；`photo_pipeline.py`：本地图→RuntimeInput 与落盘；`schema_validator.py`：Draft 2020-12、供应商 schema 适配；`normalizer.py`：枚举/格式归一。 | Schema/单测；照片入口会注入答案式预设、设备模拟和默认主体 |
| 观察 | `adapters/base.py` 协议；`adapters/mock.py` 离线规则；`adapters/providers.py` Gemini、OpenAI 兼容（含路由至合适的 Qwen 兼容端点）；`assets/prompts/observer.system.txt`。 | 本地可收到图片文件但离线规则读的是 hints；真实远端请求未执行 |
| 检索/规划 | `knowledge.py`：metadata/cards 排序及 A/B/C；`planner.system.txt` 与 adapter 生成 `ShotPlan`；`validators.py`：几何、设备、动作及安全业务门槛。 | A/B/C 接线和规则回放已验证；无真实命中率。分支增加引导线几何证据门槛；其他证据仍需完善 |
| 控制与显示 | `controller.py`：慢循环按输入→观察→重试/降级→检索→规划→重试/降级；快循环用调用者提供的人框/旋转/版本推进；`coordinates.py` 坐标转换；`render.py` 静态 JPEG 图层。 | 验证了合成的版本丢弃、快循环与非空渲染；没有端侧追踪或真实重拍验证；确认动作可能被误当作可拍 |
| HTTP | `api.py`：health/cards/session/transform/photo。 | loopback 实测；共享控制器与本地文件接口仅适合开发环境，不能当作已隔离线上服务 |
| 手机与成片 | 没有 CameraX/AVFoundation App、实机摄像头、实况取景或成片盲评。 | 缺失；P2–P4，不以静态叠加替代 |
| 评测 | `evaluation.py` 和 `assets/evaluation/`。 | E01–E16 为契约测试，field=0；[现场协议](FIELD_PROTOCOL.md) 不包含已验证结果 |

## 人工对照片像素的审计与应用结果分开

- **5.png**：远处红衣人约占画面高度 4%，脸/姿态不可判读；地面/树带可见，仅凭此图看不出湖面或实际移步路线。离线记录把预设叙述和脸部明暗标成视觉来源，`face_visibility=too_small` 却未输出问题；允许在保持镜头与原机位时，先仅绕机位转动手机追逐中心位置，同时把主体高度目标设为 50–65%。这个动作的停止条件只检查横向中心；没有把远处主体放大至目标量级的构图方法。这是当前计划自洽性失败，不是模型看图失败的测量。
- **这个样例更好的第一步（人工示范，未重拍）**：先确认红衣人是主体；如果双方都明确脚下是安全、开放的草坪且可走，摄影师可沿已确认路线接近、保持当前镜头，到脸可辨、脚还完整、画面仍保留一段树带时停；重新取景再验。若路线未知或不可达，先保留原机位，说明当前画面不能同时满足脸可辨与完整环境，请用户确认设备可用镜头或接受环境优先；不给固定米数，不说已改善。
- **6.png**：存在多位行人、道路与信号灯；先选人，不进入车道。离线 `hold` 能对齐这条建议，但人数 2 来自预设；不作真值。
- **3.png**：流水、岩石、文字覆盖可见，未见明确主体；照片不能证明岩石或水边可站。离线 `no_subject` 只证明流程。不得把静态布局建议表现成真实可达站位。

以上仓库图片只在本地审阅。部分图片有平台 UI 或水印，未找到逐图授权；不收入评分集，不转发/上传，不把图附进 PR。人工示范只是定义验收预期。
