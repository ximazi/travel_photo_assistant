# P0 基线复跑

冻结主干：`fc443ca821ca9118fd49a66ed6c411a815823ecf`。Linux / Git 2.56.1 / Python 3.12.14，jsonschema 4.26.0、FastAPI 0.141.1、uvicorn 0.52.1、Pillow 13.0.0。使用干净 clone、项目根下没有 .env；未输入凭据。

```bash
export PYTHONPATH=src TPA_VISION_PROVIDER=offline
python -m unittest discover -s tests -v
python -m travel_photo_assistant validate-examples
python -m travel_photo_assistant evaluate -o /tmp/tpa-contract.json
for mode in prompt_only fixed_cards retrieved_cards; do
  python -m travel_photo_assistant run-photo --photo coast_person.jpg \
    --retrieval-mode "$mode" --out "/tmp/tpa-smoke/$mode"
done
python -m travel_photo_assistant check-cards --tags coast leading_line
python -m travel_photo_assistant serve --host 127.0.0.1 --port 18971
```

| 命令/观察 | main 原样 | Draft 变更后 |
|---|---|---|
| unittest | 83 运行，82 通过，1 失败 | 83/83 通过，仍无根目录 .env |
| evaluate | 离线 16/16；真实 0 | 离线 16/16；真实 0 |
| validate-examples | 两份 fixture schema 有效；生产模式拒绝 example 输入 | 见上述复跑，拒绝预设与实拍隔离仍需 #7 |
| check-cards（带 leading_line 标签，但锚点为空） | `COAST_LINE_ENV_01` | 不选此卡：`missing required_visual_tags: ['leading_line']` |
| A/B/C 合成海边 | A=0 张；B=固定 3 张；C=1 张；均 accepted | 含有效引导线的 C 复跑仍选 COAST_LINE_ENV_01；JPEG 1080×1440，accepted=True |
| localhost `/health`、`/v1/cards`、`/v1/session/fast-tick`、`/v1/photos/run` | 全部 HTTP 200；规则引擎、9 张 draft_unvalidated 卡、无计划 tick、合成样例 plan_ready | 不涉及线上部署 |
| 外部模型前置 | 显式 unset 密钥后 Gemini、OpenAI CLI 均 exit 2（缺键，在传输前停止）；Qwen 共用兼容 adapter，未独立连接 | 未消费配额、无任何真实模型、无手机真机或成片验证 |

### 实际日志摘录

```text
main  clean clone: Ran 83 tests ... FAILED (failures=1)
test_project_env_file_is_discoverable: assertIsNotNone(found)
AssertionError: unexpectedly None
draft clean clone: Ran 83 tests ... OK
evaluate: offline_contract_passed=16, offline_contract_failed=0, field_photo_executed=0
GET /health 200 ok offline-rule-engine-v0.1
GET /v1/cards 200 count=9 status=draft_unvalidated
POST /v1/session/fast-tick 200 no_active_plan
POST /v1/photos/run 200 accepted=True plan_ready ['COAST_LINE_ENV_01']
```

新版 .env 探测用临时文件，只写离线 provider，不依赖开发机的私密配置；没有改动运行时环境加载。E11 使用误导性场景标签、无锚点的反例；单测另覆盖合法坐标、空/退化坐标、错误锚点类型。

### P0 图片流程与证据级别

```bash
# 仓库图片仅本地规则回放，不发给模型，不计实拍分数
for photo in 3.png 5.png 6.png; do
  TPA_VISION_PROVIDER=offline PYTHONPATH=src python -m travel_photo_assistant \
    run-photo --photo "$photo" --out /tmp/tpa-smoke/repo --no-render
done
```

分别返回 `no_subject`、`plan_ready`、`hold`（三次 CLI exit 0）。5.png 的预设驱动计划虽然通过契约，却未解决主体现有大小不足，详细人工对照见 [CURRENT_STATE](CURRENT_STATE.md)。`accepted` 只表示协议流程受理，不是建议做对了。

逐图观察、人评、复拍和潜在模型消耗以 [FIELD_PROTOCOL](FIELD_PROTOCOL.md) 为准。输出 JSON、样片原图、模型真实响应未在本 PR 外发；评测材料不进入 Git，只提交空白表和方法。
