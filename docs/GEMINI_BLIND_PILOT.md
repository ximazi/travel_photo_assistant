# 单张真实照片 Gemini 盲测（#7 第一版）

先在 [Google AI Studio](https://aistudio.google.com/projects) 确认 Key 属于未启用计费的 Free Tier 项目、模型可免费调用。只用已授权给 Gemini 的图。不上传 Git，也不要在聊天或终端命令中粘贴密钥。免费项目确认开关是操作者声明，程序不能代查计费状态。

在 Windows PowerShell 中进入这个分支的项目目录。按已有方式在未跟踪的 .env 中配置 `GEMINI_API_KEY`，准备可用的 Pillow 和项目依赖，然后：

```powershell
$env:PYTHONPATH = 'src'
python -m travel_photo_assistant.field_pilot --photo "D:\private\pilot.jpg" --goal "拍一张人景兼顾的旅行照" --out "$env:TEMP\tpa-pilot-a" --model "替换为已确认适用的 Gemini 模型 ID" --mode prompt_only --confirm-free-project --allow-photo-to-gemini
```

输出目录必须全新并位于 Git 工作区外。输出 observation.json、plan.json、retrieval.json、run.json 和有计划时的 overlay.jpg，均当作私有产物。观察与规划请求只带匿名帧标识、明确的拍摄意图与去元数据的转正照片，不读照片清单，不假定主体、相机镜头或可移动空间。若服务报错，不自动回退到离线造数。转正和缩图不改变原始文件。不填写真实焦距，除非之后增加可核验设备元数据入口。

这一步只做单张端到端；固定观察实验使用下方独立模式。需要的话可分别运行不共享观察的端到端模式，另记区别。没有人评时 run.json 只到 needs_human_review，绝不凭 accepted 断言照片质量。

## 固定观察的对照模式

`--compare` 替换 `--mode prompt_only`，不要同时使用。对同一输入只观察一次，随后分别运行无卡、固定卡、检索卡的独立规划器状态，最多 1 次观察 + 3 次规划，不重试。每组输出独立目录，comparison.json 记录共用的观察摘要、调用次数和跳过原因；第一组耗时含观察，不能直接和后两组相除。B/C 目前卡数和字数未强制匹配，属于流程试点，不能把差值完全归因于检索排序。

当观察无明确主体（包括多人待选、空镜），只保留 A 组和暂停结果，B/C 不请求、不记为效果对照。要测知识卡是否改善摄影建议，请先使用明确授权且主体清晰的新样片，人工按画面证据、第一动作可执行性、站位安全复核，再安排实拍。不要对未授权新照片开外部调用。
