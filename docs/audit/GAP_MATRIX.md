# P0 差距矩阵：前十项

证据类型：S=源代码；O=离线规则/桩；H=人工看图；F=授权真实模型与原图验收（本轮没有 F）。行号均以 main 基线为准，可按符号定位。

| 优先 | 关键差距与后果 | 证据 / 状态 | 下一步验收 |
|---|---|---|---|
| P0 | 1. 真实照入口被人工预设污染，且 planner JSON 带附件路径；答案可来自文件名而非像素 | S/O: `photo_pipeline.build_runtime_input`、`providers.*.plan`。代码存在，不符合目标 | [#7](https://github.com/ximazi/travel_photo_assistant/issues/7)：同图不同文件名；出站桩检查无路径/预设/真值 |
| P0 | 2. 照片默认有 `person_1` 和模拟 26mm（标成 camera_api），尺寸/EXIF 旋转及基本 MIME 不能可靠表达；已知 MIME 的 provider 有字节嗅探 | S: `photo_pipeline.py:90–162` 与 `providers._detect_mime` | #7：未知保持未知；转正后模型和叠加使用同一份像素，同校验两个 provider |
| P0 | 3. 图片先验未获授权、没有金标准和真实性分层，自评或 schema 合格无法验证建议价值 | H/S: photos.json 为 hand_authored_hints、5 个命名 JPG 明确合成、6 个 PNG 有来源问题 | #2、#3、[#10](https://github.com/ximazi/travel_photo_assistant/issues/10)：登记授权、按场景冻结试点与留出；至少展示五例并保留失败 |
| P0 | 4. 画面标签替代引导线证据，否定关键词也可能误命中其他卡 | O/S: `check-cards --tags coast leading_line` 以前没有锚点也入选；`knowledge._extract_visual_evidence_tags` 抽取关键词 | [#8](https://github.com/ximazi/travel_photo_assistant/issues/8)：此分支仅补引导线定位证据；压头等卡的结构化确证仍未做 |
| P0 | 5. 太小的人物/脸不可读漏报；原地平移取景中心无法把约 4% 的人物变为 50–65% | H/O：5.png，`mock.py`、plan/actions、`photo_pipeline` 的模拟 tick；校验器接受 | [#9](https://github.com/ximazi/travel_photo_assistant/issues/9)、#10：增大主体需要经证据允许的动作或目标调整；横向中心检查不能替代大小/脚/脸复核 |
| P0 | 6. 移动权限及空镜放置错误：`forbidden` 和走动偏好组合可漏拦；无安全站位时会要求 placement/crop | O/S: `validators.py:389–419,523–529`；给禁止平移的主体标走动偏好可过；空镜安全 hold 可被拒 | #9：禁止、未知、未请求分开测试，未经确认不下发物理箭头；允许保守 hold |
| P0 | 7. 确认可移动或末动作完成不等于解决原问题；adapter timeout 会绕开降级 | O/S: `controller.py:139–155,192–216,343–385`；障碍确认后可 `ready_to_shoot`，未复核遮挡 | #9：注入超时与手动确认，证实关闭旧箭头，要求新帧检查，不提前报完成 |
| P1 | 8. 可定位事实缺统一的人类核查合同：脸亮、头部穿杆、切脚、地标保留、失败和弃权未分项量化 | O/H: 同一份 5.png 观察可同时有 too_small 与脸部亮度陈述；记录 schema valid | #2/#3/#10：关键事实原图区域、未知及目标失配分别记，对照保留失败 |
| P2 | 9. FastAPI 原型存在会话单例及面向服务器文件路径的调试接口 | S/loopback: `api.create_app`、`run_photo_endpoint`；仅做良性接口检查，未测试恶意输入 | [#11](https://github.com/ximazi/travel_photo_assistant/issues/11)：隔离会话与调试路由，并以并发旧回复回归 |
| P2 | 10. 手机闭环缺失：无原生预览、设备镜头元数据、取景跟随和实机快门 | S/O: repo 树、`coordinates.py`/静态 `render.py` 与固定帧 controller | #4/#5：单端真机；验证横竖屏、镜头切换、连续提示防抖、断网仍能按快门；P1 先证明建议价值 |

## 项目状态

- **Done（待用户评审）**：冻结 P0 主干、实际复跑、现场协议、标注模板、GitHub 子任务 #7–#11。
- **In Progress**：#8 首个引导线回归进入 P0 Draft；其他摄影卡证据仍开放。
- **Blocked**：#3 真模型/实拍配对无已授权样片、无明确批准的免费推理或用户提供本地运行环境。仓库图和规则适配器都不是替代。
- **Next**：先 #7 去污染入口，并行推进 #9 移动、目标大小与真正的完成判据；同时 #2 准备 5 张已获授权的分层试点，再启动 #3/#10。P2 不抢跑。
