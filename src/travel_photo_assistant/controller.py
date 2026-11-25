"""Dual-loop runtime controller: Slow semantic planning + Fast on-device alignment.

Implements README.txt Section III & VIII, integration_notes.txt Section V,
and evaluation_protocol.txt Section VIII (plan locking, staleness rejection,
anti-jitter hysteresis, and predefined completion checks).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .adapters.base import VisionModelAdapter
from .config import AssetPaths, RuntimeConfig
from .knowledge import RetrievalMode, RetrievalResult, ShotCardRepository, ShotCardRetriever
from .models import LocalVisionSnapshot, RuntimeInput
from .normalizer import normalize_observation, normalize_plan
from .schema_validator import SchemaValidator
from .validators import BusinessRuleValidator, ValidationResult


@dataclass
class ActiveSessionState:
    """Locked session state maintained across slow and fast loops."""

    active_plan: dict[str, Any] | None = None
    active_observation: dict[str, Any] | None = None
    active_scene_version: int = -1
    active_goal_version: int = 0
    active_lens_id: str | None = None
    active_orientation: str | None = None
    active_crop_signature: str | None = None
    active_subject_id: str | None = None
    active_action_index: int = 0
    completed_action_ids: list[str] = field(default_factory=list)
    stable_in_range_count: int = 0
    last_guidance_direction: str | None = None
    stale_drops: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SlowLoopResult:
    """Result of one slow-loop observe -> retrieve -> plan -> validate cycle."""

    accepted: bool
    dropped_as_stale: bool
    downgraded_to_fallback: bool
    observation: dict[str, Any] | None
    plan: dict[str, Any] | None
    retrieval: RetrievalResult | None
    input_validation: ValidationResult
    observation_validation: ValidationResult | None
    plan_validation: ValidationResult | None
    reason: str = ""
    normalization_changes: list[str] = field(default_factory=list)


@dataclass
class FastLoopResult:
    """Result of one fast-loop on-device alignment tick."""

    status: str  # "in_progress" | "action_completed" | "ready_to_shoot" | "invalidated" | "paused_tracking_lost" | "no_active_plan"
    active_action_id: str | None
    guidance_direction: str | None  # "left" | "right" | "hold" | None (in canonical input_image space)
    show_overlay_arrows: bool
    invalidation_reasons: list[str] = field(default_factory=list)
    requires_reobserve: bool = False
    message: str = ""


class TravelPhotoController:
    """Coordinates slow VLM planning and fast on-device tracking/alignment."""

    def __init__(
        self,
        adapter: VisionModelAdapter,
        *,
        assets: AssetPaths | None = None,
        config: RuntimeConfig | None = None,
    ) -> None:
        self.adapter = adapter
        self.assets = assets or AssetPaths.default()
        self.config = config or RuntimeConfig()
        self.schema_validator = SchemaValidator(self.assets, production_mode=self.config.production_mode)
        self.validator = BusinessRuleValidator(self.schema_validator, production_mode=self.config.production_mode)
        self.card_repo = ShotCardRepository(self.assets.shot_cards)
        self.retriever = ShotCardRetriever(self.card_repo)

        self._observer_prompt = self.assets.observer_prompt.read_text(encoding="utf-8")
        self._planner_prompt = self.assets.planner_prompt.read_text(encoding="utf-8")
        self.state = ActiveSessionState()

    def reset_session(self) -> None:
        self.state = ActiveSessionState()

    def run_slow_loop(
        self,
        runtime_input: RuntimeInput,
        *,
        retrieval_mode: RetrievalMode | str | None = None,
    ) -> SlowLoopResult:
        """Execute Stage 1 (observe) -> Card Retrieval -> Stage 2 (plan) -> Validation -> Lock."""
        # 1. Input guard
        input_val = self.validator.validate_runtime_input(runtime_input)
        if not input_val.is_valid:
            return SlowLoopResult(
                accepted=False,
                dropped_as_stale=False,
                downgraded_to_fallback=False,
                observation=None,
                plan=None,
                retrieval=None,
                input_validation=input_val,
                observation_validation=None,
                plan_validation=None,
                reason="; ".join(e.message for e in input_val.errors),
            )

        # 2. Early staleness check (E01 / E08)
        if self._is_stale_request(runtime_input):
            self._record_stale_drop(runtime_input, "pre_call_stale")
            return SlowLoopResult(
                accepted=False,
                dropped_as_stale=True,
                downgraded_to_fallback=False,
                observation=None,
                plan=self.state.active_plan,
                retrieval=None,
                input_validation=input_val,
                observation_validation=None,
                plan_validation=None,
                reason="Dropped stale request older than active scene_version or goal_version.",
            )

        # 3. Stage 1: Scene Observation
        #    Real vendors do not reliably enforce enum values in their response
        #    schemas, so the payload is folded onto the application vocabulary
        #    first, then validated, then retried once with the errors if needed.
        observation, obs_changes = normalize_observation(
            self.adapter.observe(
                runtime_input=runtime_input,
                system_prompt=self._observer_prompt,
                output_schema=self.schema_validator.observation_schema,
            )
        )
        obs_val = self.validator.validate_observation(observation, runtime_input)

        if not obs_val.is_valid and self.config.retry_on_validation_error:
            feedback = self._format_feedback(obs_val.errors)
            retry_raw = self.adapter.observe(
                runtime_input=runtime_input,
                system_prompt=self._observer_prompt,
                output_schema=self.schema_validator.observation_schema,
                feedback=feedback,
            )
            retry_norm, retry_changes = normalize_observation(retry_raw)
            retry_val = self.validator.validate_observation(retry_norm, runtime_input)
            if retry_val.is_valid or len(retry_val.errors) < len(obs_val.errors):
                observation, obs_val = retry_norm, retry_val
                obs_changes = obs_changes + retry_changes

        if not obs_val.is_valid:
            fallback = self._build_safe_fallback_plan(
                runtime_input,
                status="insufficient_input" if not runtime_input.frame.image_attachments else "hold",
                reason="场景观察结果未通过服务端业务校验，已降级为暂停引导。",
            )
            return SlowLoopResult(
                accepted=False,
                dropped_as_stale=False,
                downgraded_to_fallback=True,
                observation=observation,
                plan=fallback,
                retrieval=None,
                input_validation=input_val,
                observation_validation=obs_val,
                plan_validation=None,
                reason="; ".join(e.message for e in obs_val.errors),
                normalization_changes=obs_changes,
            )

        # 4. Knowledge Card Retrieval (Modes A / B / C)
        mode = retrieval_mode or self.config.retrieval_mode
        retrieval = self.retriever.retrieve(
            observation,
            runtime_input,
            mode=mode,
            max_cards=self.config.max_retrieved_cards,
        )

        # 5. Stage 2: Shot Planning
        plan, plan_changes = normalize_plan(
            self.adapter.plan(
                runtime_input=runtime_input,
                observation=observation,
                shot_cards=retrieval.selected_cards,
                system_prompt=self._planner_prompt,
                output_schema=self.schema_validator.shot_plan_schema,
            )
        )
        plan_val = self.validator.validate_shot_plan(
            plan=plan,
            observation=observation,
            runtime_input=runtime_input,
            provided_cards=retrieval.selected_cards,
        )

        if not plan_val.is_valid and self.config.retry_on_validation_error:
            feedback = self._format_feedback(plan_val.errors)
            retry_raw = self.adapter.plan(
                runtime_input=runtime_input,
                observation=observation,
                shot_cards=retrieval.selected_cards,
                system_prompt=self._planner_prompt,
                output_schema=self.schema_validator.shot_plan_schema,
                feedback=feedback,
            )
            retry_norm, retry_changes = normalize_plan(retry_raw)
            retry_val = self.validator.validate_shot_plan(
                plan=retry_norm,
                observation=observation,
                runtime_input=runtime_input,
                provided_cards=retrieval.selected_cards,
            )
            if retry_val.is_valid or len(retry_val.errors) < len(plan_val.errors):
                plan, plan_val = retry_norm, retry_val
                plan_changes = plan_changes + retry_changes

        # 6. Post-call staleness check (E08: late response after a newer request already committed)
        if self._is_stale_request(runtime_input):
            self._record_stale_drop(runtime_input, "post_call_late_response")
            return SlowLoopResult(
                accepted=False,
                dropped_as_stale=True,
                downgraded_to_fallback=False,
                observation=observation,
                plan=self.state.active_plan,
                retrieval=retrieval,
                input_validation=input_val,
                observation_validation=obs_val,
                plan_validation=plan_val,
                reason="Late VLM response discarded because a newer scene/goal version is already active.",
            )

        # 7. Downgrade invalid plan to safe fallback (never leak invalid plan to user)
        if not plan_val.is_valid:
            fallback = self._build_safe_fallback_plan(
                runtime_input,
                status="hold",
                reason="拍摄方案未通过服务端业务校验，已安全降级。",
            )
            return SlowLoopResult(
                accepted=False,
                dropped_as_stale=False,
                downgraded_to_fallback=True,
                observation=observation,
                plan=fallback,
                retrieval=retrieval,
                input_validation=input_val,
                observation_validation=obs_val,
                plan_validation=plan_val,
                reason="; ".join(e.message for e in plan_val.errors),
                normalization_changes=obs_changes + plan_changes,
            )

        # 8. Commit and lock active plan
        self._commit_plan(runtime_input, observation, plan)
        return SlowLoopResult(
            accepted=True,
            dropped_as_stale=False,
            downgraded_to_fallback=False,
            observation=observation,
            plan=plan,
            retrieval=retrieval,
            input_validation=input_val,
            observation_validation=obs_val,
            plan_validation=plan_val,
            reason="ok",
            normalization_changes=obs_changes + plan_changes,
        )

    def run_fast_tick(self, snapshot: LocalVisionSnapshot) -> FastLoopResult:
        """Process one on-device tracking tick against the active locked plan."""
        plan = self.state.active_plan
        if plan is None or plan.get("status") != "plan_ready":
            return FastLoopResult(
                status="no_active_plan",
                active_action_id=None,
                guidance_direction=None,
                show_overlay_arrows=False,
                requires_reobserve=True,
                message="当前无已锁定的可执行方案，请先执行场景观察与规划。",
            )

        # 1. Check invalidation conditions (E01, E08, E15)
        invalidation_reasons: list[str] = []
        if snapshot.scene_version != self.state.active_scene_version:
            invalidation_reasons.append("scene_version_changed")
        if snapshot.user_goal_version != self.state.active_goal_version:
            invalidation_reasons.append("user_goal_changed")
        if self.state.active_lens_id is not None and snapshot.current_lens_id != self.state.active_lens_id:
            invalidation_reasons.append("lens_changed")
        if self.state.active_orientation is not None and snapshot.orientation != self.state.active_orientation:
            invalidation_reasons.append("orientation_changed")
        if self.state.active_crop_signature is not None and snapshot.crop_signature != self.state.active_crop_signature:
            invalidation_reasons.append("crop_changed")
        if (
            self.state.active_subject_id is not None
            and snapshot.tracked_subject_id is not None
            and snapshot.tracked_subject_id != self.state.active_subject_id
        ):
            invalidation_reasons.append("selected_subject_changed")
        if snapshot.light_materially_changed:
            invalidation_reasons.append("light_materially_changed")

        if invalidation_reasons:
            return FastLoopResult(
                status="invalidated",
                active_action_id=None,
                guidance_direction=None,
                show_overlay_arrows=False,
                invalidation_reasons=invalidation_reasons,
                requires_reobserve=True,
                message=f"当前方案已失效（{', '.join(invalidation_reasons)}），请重新采帧观察与规划。",
            )

        # 2. Tracking lost -> pause arrows immediately (planner prompt & integration_notes)
        if not snapshot.tracking_ok or snapshot.subject_bbox is None:
            self.state.stable_in_range_count = 0
            return FastLoopResult(
                status="paused_tracking_lost",
                active_action_id=plan.get("active_action_id"),
                guidance_direction=None,
                show_overlay_arrows=False,
                invalidation_reasons=["tracking_lost"],
                requires_reobserve=False,
                message="人物跟踪暂时丢失，已暂停定位箭头；请保持画面稳定或重新观察。",
            )

        actions = plan.get("actions", [])
        if self.state.active_action_index >= len(actions):
            return FastLoopResult(
                status="ready_to_shoot",
                active_action_id=None,
                guidance_direction="hold",
                show_overlay_arrows=False,
                requires_reobserve=False,
                message="全部引导步骤已完成，请由用户确认画面并按下快门。",
            )

        action = actions[self.state.active_action_index]
        action_id = action["action_id"]
        check = action["completion_check"]
        check_type = check["type"]

        # 3. Evaluate predefined completion checks (Rule III.9)
        completed, direction, msg = self._evaluate_completion_check(check_type, check, snapshot, action_id)
        if completed:
            self.state.stable_in_range_count += 1
            required = self.config.stable_frames_required if check.get("check_by") == "local_vision" else 1
            if self.state.stable_in_range_count >= required:
                self.state.completed_action_ids.append(action_id)
                self.state.active_action_index += 1
                self.state.stable_in_range_count = 0
                self.state.last_guidance_direction = None

                if self.state.active_action_index >= len(actions):
                    plan["active_action_id"] = None
                    return FastLoopResult(
                        status="ready_to_shoot",
                        active_action_id=None,
                        guidance_direction="hold",
                        show_overlay_arrows=False,
                        requires_reobserve=False,
                        message=f"动作 {action_id} 已完成，所有步骤已就绪，请拍摄。",
                    )
                next_id = actions[self.state.active_action_index]["action_id"]
                plan["active_action_id"] = next_id
                return FastLoopResult(
                    status="action_completed",
                    active_action_id=next_id,
                    guidance_direction="hold",
                    show_overlay_arrows=True,
                    requires_reobserve=(check.get("check_by") == "vlm_review"),
                    message=f"动作 {action_id} 已完成，激活下一步 {next_id}。",
                )
            return FastLoopResult(
                status="in_progress",
                active_action_id=action_id,
                guidance_direction="hold",
                show_overlay_arrows=True,
                message="已进入目标区，保持稳定以确认完成。",
            )

        # Not completed -> apply anti-jitter hysteresis (E09)
        self.state.stable_in_range_count = 0
        self.state.last_guidance_direction = direction
        return FastLoopResult(
            status="in_progress",
            active_action_id=action_id,
            guidance_direction=direction,
            show_overlay_arrows=True,
            requires_reobserve=False,
            message=msg,
        )

    def _evaluate_completion_check(
        self,
        check_type: str,
        check: dict[str, Any],
        snapshot: LocalVisionSnapshot,
        action_id: str,
    ) -> tuple[bool, str | None, str]:
        xmin, ymin, xmax, ymax = snapshot.subject_bbox  # type: ignore[misc]
        center_x = (xmin + xmax) / 2.0
        height_ratio = ymax - ymin
        deadband = self.config.jitter_deadband

        if check_type == "subject_center_x_in_range":
            lo, hi = check["target_range"]
            if lo <= center_x <= hi:
                return True, "hold", "人物水平位置已进入目标区间。"
            # Hysteresis deadband prevents rapid left/right oscillation under mild jitter (E09)
            if (lo - deadband) <= center_x <= (hi + deadband):
                if self.state.last_guidance_direction in {"hold", None}:
                    return False, "hold", "处于稳定容差带内，保持当前构图。"
            if center_x < lo:
                return False, "left", "缓慢向画面左侧微调取景，让人物向右移入目标区。"
            return False, "right", "缓慢向画面右侧微调取景，让人物向左移入目标区。"

        if check_type == "subject_height_ratio_in_range":
            lo, hi = check["target_range"]
            if lo <= height_ratio <= hi:
                return True, "hold", "人物画面高度占比已符合目标。"
            if (lo - deadband) <= height_ratio <= (hi + deadband):
                return False, "hold", "人物占比接近目标区间，保持稳定。"
            return False, "hold", "请按方案微调取景使人物占比接近目标区间。"

        if check_type == "manual_confirmed":
            if action_id in snapshot.manual_confirmations or "manual_confirmed" in snapshot.manual_confirmations:
                return True, "hold", "用户已手动确认前置条件。"
            return False, None, "等待用户确认现场条件。"

        if check_type in {"head_obstacle_separated", "pose_visually_confirmed", "fresh_frame_received"}:
            if check_type in snapshot.vlm_verified_checks or action_id in snapshot.vlm_verified_checks:
                return True, "hold", f"已通过新帧视觉复核（{check_type}）。"
            return False, None, f"等待新帧视觉复核（{check_type}），不可仅凭端侧猜测完成。"

        if check_type == "camera_metadata_match":
            ref = check.get("reference_id")
            if ref and snapshot.current_lens_id == ref:
                return True, "hold", "相机镜头元数据已匹配目标设置。"
            return False, None, "等待相机元数据匹配目标镜头。"

        return False, "hold", "当前步骤无自动完成判定。"

    @staticmethod
    def _format_feedback(errors: list[Any], limit: int = 12) -> str:
        """Render validation errors as a compact, model-readable correction list."""
        lines = []
        for issue in errors[:limit]:
            location = issue.field_path or "<root>"
            lines.append(f"- {location}: {issue.message}")
        if len(errors) > limit:
            lines.append(f"- …另有 {len(errors) - limit} 条同类错误")
        return "\n".join(lines)

    def _is_stale_request(self, runtime_input: RuntimeInput) -> bool:
        if runtime_input.frame.scene_version < self.state.active_scene_version:
            return True
        if (
            runtime_input.frame.scene_version == self.state.active_scene_version
            and runtime_input.user_goal.goal_version < self.state.active_goal_version
        ):
            return True
        return False

    def _record_stale_drop(self, runtime_input: RuntimeInput, stage: str) -> None:
        self.state.stale_drops.append(
            {
                "frame_id": runtime_input.frame.frame_id,
                "scene_version": runtime_input.frame.scene_version,
                "goal_version": runtime_input.user_goal.goal_version,
                "active_scene_version": self.state.active_scene_version,
                "active_goal_version": self.state.active_goal_version,
                "stage": stage,
            }
        )
        if len(self.state.stale_drops) > self.config.max_stale_drop_log:
            self.state.stale_drops = self.state.stale_drops[-self.config.max_stale_drop_log :]

    def _commit_plan(
        self,
        runtime_input: RuntimeInput,
        observation: dict[str, Any],
        plan: dict[str, Any],
    ) -> None:
        self.state.active_observation = observation
        self.state.active_plan = plan
        self.state.active_scene_version = runtime_input.frame.scene_version
        self.state.active_goal_version = runtime_input.user_goal.goal_version
        self.state.active_lens_id = runtime_input.camera_capabilities.current_lens_id
        self.state.active_orientation = runtime_input.user_goal.orientation_preference
        self.state.active_crop_signature = f"{runtime_input.frame.viewport_fit}:{runtime_input.frame.preview_aspect_ratio}"
        subj = observation.get("selected_subject")
        self.state.active_subject_id = subj["subject_id"] if isinstance(subj, dict) else None
        self.state.active_action_index = 0
        self.state.completed_action_ids = []
        self.state.stable_in_range_count = 0
        self.state.last_guidance_direction = None

    @staticmethod
    def _build_safe_fallback_plan(
        runtime_input: RuntimeInput,
        *,
        status: str,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": "0.1.0",
            "frame_id": runtime_input.frame.frame_id,
            "scene_version": runtime_input.frame.scene_version,
            "plan_id": f"fallback_{runtime_input.frame.frame_id}_v{runtime_input.frame.scene_version}",
            "status": status,
            "coordinate_space": "upright_input_normalized_unmirrored",
            "goal": runtime_input.user_goal.priority,
            "scene_summary": reason,
            "subject_placement_guide": None,
            "crop_recommendation": None,
            "primary_plan": None,
            "actions": [],
            "active_action_id": None,
            "alternative": None,
            "uncertainties": [reason],
            "invalidation_conditions": [
                "scene_version_changed",
                "lens_changed",
                "crop_changed",
                "orientation_changed",
                "selected_subject_changed",
                "light_materially_changed",
                "tracking_lost",
                "user_goal_changed",
            ],
            "fallback_instruction": "请保持当前构图稳定，确认拍摄条件后重新获取一帧画面。",
        }
