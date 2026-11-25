# 手机快循环：新帧复核交接（#9）

- 同一计划内，客户端为预览帧提供单调递增的整数 `frame_seq`，并保留 `frame_id` 与版本。进入需要视觉复核的动作时，第一次合格的 fast tick 返回 `requires_reobserve=true` 和 `visual_review_request`，没有方向/箭头。这个信号只发一次；后续 tick 继续等，不能逐帧发云端请求。
- 发起方须提交请求之后的新帧，复核结果通过 fast tick 的 `visual_review` 回传 `review_id`、`action_id`、`check_type`、实际复核的 `frame_id` / `frame_seq`、`status`（passed/failed）。旧版单字符串可传但不算完成。超时、失败或变换更新不误标完成；跟踪丢失会废弃未完成的请求。有新计划必须重新请求，不继承旧令牌。
- `can_shoot_without_confirmation=true` 表示引导未复核，但客户端不能锁快门；`ready_to_shoot` 仍只代表动作复核完成。默认等待 5 秒是可调交互上限，不是模型延迟保证。服务器不在快 tick 里自动调用 VLM。本分支还不包含手机采帧、模型结果真实性验证或真实手机接入；调用方必须正确执行采帧及复核，旧客户端要求更新。
- 这是离线握手架构，不是实机结果，也独立于知识卡评测。固定观察 A/B/C 保持三组 planner 同样可见图像；最多一次观察、三次规划，因而可能传四次相同图片像素，此结果不能用于评价只看结构化观察的规划器。
