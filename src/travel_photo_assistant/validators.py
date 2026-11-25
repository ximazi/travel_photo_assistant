"""Mandatory business-rule validators beyond JSON Schema.

Implements all 16 checks from integration_notes.txt Section III and README.txt Section IX.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .models import RuntimeInput, ShotCard
from .schema_validator import SchemaValidator


PSEUDO_PRECISE_DISTANCE_PATTERN = re.compile(
    r"(?:后退|前进|左移|右移|平移|抬高|降低|距离|约|移动)\s*\d+(?:\.\d+)?\s*(?:米|m\b|厘米|cm\b|公分)",
    re.IGNORECASE,
)

# Predefined completion checks allowed by v0.1 contract (never eval arbitrary text)
ALLOWED_COMPLETION_CHECK_TYPES = {
    "subject_center_x_in_range",
    "subject_height_ratio_in_range",
    "head_obstacle_separated",
    "pose_visually_confirmed",
    "camera_metadata_match",
    "fresh_frame_received",
    "manual_confirmed",
    "none",
}

PHYSICAL_MOVEMENT_OPERATIONS = {"physical_translate"}
NON_PARALLAX_OPERATIONS = {"reframe_pan", "reframe_tilt", "change_zoom"}


@dataclass(frozen=True)
class ValidationIssue:
    """Single business or schema validation finding."""

    code: str
    severity: str  # "error" | "warning"
    message: str
    field_path: str = ""


@dataclass
class ValidationResult:
    """Aggregated validation outcome."""

    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def add_error(self, code: str, message: str, field_path: str = "") -> None:
        self.issues.append(ValidationIssue(code=code, severity="error", message=message, field_path=field_path))

    def add_warning(self, code: str, message: str, field_path: str = "") -> None:
        self.issues.append(ValidationIssue(code=code, severity="warning", message=message, field_path=field_path))


def _check_bbox_order(bbox: list[float] | None, path: str, result: ValidationResult) -> None:
    if bbox is None:
        return
    if len(bbox) != 4:
        result.add_error("INVALID_BBOX_LENGTH", f"bbox must have 4 elements, got {len(bbox)}", path)
        return
    xmin, ymin, xmax, ymax = bbox
    if not (0.0 <= xmin < xmax <= 1.0 and 0.0 <= ymin < ymax <= 1.0):
        result.add_error(
            "INVALID_BBOX_BOUNDS",
            f"bbox must satisfy 0 <= xmin < xmax <= 1 and 0 <= ymin < ymax <= 1, got {bbox}",
            path,
        )


class BusinessRuleValidator:
    """Enforces JSON Schema + all v0.1 business, physics, device, and production rules."""

    def __init__(self, schema_validator: SchemaValidator, production_mode: bool = True) -> None:
        self.schema_validator = schema_validator
        self.production_mode = production_mode

    def validate_runtime_input(self, runtime_input: RuntimeInput) -> ValidationResult:
        """Validate incoming runtime request before invoking VLM."""
        result = ValidationResult()

        # Rule (Section I): production entrypoints must reject example_only / synthetic_fixture
        if self.production_mode:
            if runtime_input.example_only:
                result.add_error(
                    "PRODUCTION_REJECTS_EXAMPLE",
                    "Production mode rejects payloads with example_only=true.",
                    "example_only",
                )
            for idx, lens in enumerate(runtime_input.camera_capabilities.available_lenses):
                if lens.equivalent_focal_length_source == "synthetic_fixture":
                    result.add_error(
                        "PRODUCTION_REJECTS_SYNTHETIC_FIXTURE",
                        "Production mode rejects lens metadata with source=synthetic_fixture.",
                        f"camera_capabilities.available_lenses[{idx}]",
                    )

        # Rule (Section I): real image attachments are required for normal observation
        if not runtime_input.frame.image_attachments:
            result.add_warning(
                "NO_IMAGE_ATTACHMENT",
                "image_attachments is empty; pipeline must return insufficient_input.",
                "frame.image_attachments",
            )

        if runtime_input.frame.coordinate_space != "upright_input_normalized_unmirrored":
            result.add_error(
                "INVALID_COORDINATE_SPACE",
                f"Expected upright_input_normalized_unmirrored, got {runtime_input.frame.coordinate_space}",
                "frame.coordinate_space",
            )

        return result

    def validate_observation(
        self,
        observation: dict[str, Any],
        runtime_input: RuntimeInput,
    ) -> ValidationResult:
        """Validate scene_observation payload against schema and business rules."""
        result = ValidationResult()

        # 1. Draft 2020-12 Schema check
        for err in self.schema_validator.validate_observation(observation):
            result.add_error("SCHEMA_ERROR", err, "scene_observation")

        # 2. frame_id & scene_version consistency (Rule III.1)
        if observation.get("frame_id") != runtime_input.frame.frame_id:
            result.add_error(
                "FRAME_ID_MISMATCH",
                f"Observation frame_id={observation.get('frame_id')} != request frame_id={runtime_input.frame.frame_id}",
                "frame_id",
            )
        if observation.get("scene_version") != runtime_input.frame.scene_version:
            result.add_error(
                "SCENE_VERSION_MISMATCH",
                f"Observation scene_version={observation.get('scene_version')} != request scene_version={runtime_input.frame.scene_version}",
                "scene_version",
            )

        # 3. Missing image must yield insufficient_input (Rule I & observer prompt)
        if not runtime_input.frame.image_attachments and observation.get("status") != "insufficient_input":
            result.add_error(
                "MISSING_IMAGE_MUST_BE_INSUFFICIENT",
                "Empty image_attachments requires status=insufficient_input.",
                "status",
            )

        # 4. Production check for synthetic_fixture in facts
        facts = observation.get("facts", [])
        fact_ids: set[str] = set()
        for idx, fact in enumerate(facts):
            fid = fact.get("fact_id")
            if fid in fact_ids:
                result.add_error("DUPLICATE_FACT_ID", f"Duplicate fact_id: {fid}", f"facts[{idx}].fact_id")
            if isinstance(fid, str):
                fact_ids.add(fid)
            if self.production_mode and fact.get("source") == "synthetic_fixture":
                result.add_error(
                    "PRODUCTION_REJECTS_SYNTHETIC_FIXTURE",
                    "Production mode rejects facts with source=synthetic_fixture.",
                    f"facts[{idx}].source",
                )
            _check_bbox_order(fact.get("region"), f"facts[{idx}].region", result)

        # 5. Selected subject bbox & fact reference integrity (Rule III.2, III.3)
        subject = observation.get("selected_subject")
        if observation.get("status") == "insufficient_input":
            if subject is not None or facts or observation.get("anchors") or observation.get("issues"):
                result.add_error(
                    "INSUFFICIENT_INPUT_NOT_EMPTY",
                    "status=insufficient_input requires empty facts/anchors/issues and selected_subject=null.",
                    "status",
                )

        if isinstance(subject, dict):
            _check_bbox_order(subject.get("bbox"), "selected_subject.bbox", result)
            for fid in subject.get("fact_ids", []):
                if fid not in fact_ids:
                    result.add_error("DANGLING_FACT_REF", f"Unknown fact_id '{fid}' in selected_subject", "selected_subject.fact_ids")

        # 6. Anchors bbox & fact references
        for idx, anchor in enumerate(observation.get("anchors", [])):
            _check_bbox_order(anchor.get("bbox"), f"anchors[{idx}].bbox", result)
            for fid in anchor.get("fact_ids", []):
                if fid not in fact_ids:
                    result.add_error("DANGLING_FACT_REF", f"Unknown fact_id '{fid}' in anchor", f"anchors[{idx}].fact_ids")

        # 7. Light & issues fact references
        light = observation.get("light", {})
        if isinstance(light, dict):
            for fid in light.get("fact_ids", []):
                if fid not in fact_ids:
                    result.add_error("DANGLING_FACT_REF", f"Unknown fact_id '{fid}' in light", "light.fact_ids")

        for idx, issue in enumerate(observation.get("issues", [])):
            for fid in issue.get("fact_ids", []):
                if fid not in fact_ids:
                    result.add_error("DANGLING_FACT_REF", f"Unknown fact_id '{fid}' in issue", f"issues[{idx}].fact_ids")

        return result

    def validate_shot_plan(
        self,
        plan: dict[str, Any],
        observation: dict[str, Any],
        runtime_input: RuntimeInput,
        provided_cards: list[ShotCard],
    ) -> ValidationResult:
        """Validate shot_plan against Draft 2020-12 schema and all 16 business rules."""
        result = ValidationResult()

        # 0. Schema check
        for err in self.schema_validator.validate_shot_plan(plan):
            result.add_error("SCHEMA_ERROR", err, "shot_plan")

        # 1. frame_id / scene_version match (Rule III.1)
        if plan.get("frame_id") != runtime_input.frame.frame_id:
            result.add_error(
                "FRAME_ID_MISMATCH",
                f"Plan frame_id={plan.get('frame_id')} != request frame_id={runtime_input.frame.frame_id}",
                "frame_id",
            )
        if plan.get("scene_version") != runtime_input.frame.scene_version:
            result.add_error(
                "SCENE_VERSION_MISMATCH",
                f"Plan scene_version={plan.get('scene_version')} != request scene_version={runtime_input.frame.scene_version}",
                "scene_version",
            )

        status = plan.get("status")
        primary_plan = plan.get("primary_plan")
        actions = plan.get("actions", [])
        active_action_id = plan.get("active_action_id")
        obs_status = observation.get("status")

        # 2. insufficient_input rule (Rule III.6)
        if obs_status == "insufficient_input" or status == "insufficient_input":
            if status != "insufficient_input":
                result.add_error(
                    "STATUS_MUST_BE_INSUFFICIENT",
                    "Observation is insufficient_input so shot_plan status must be insufficient_input.",
                    "status",
                )
            if primary_plan is not None:
                result.add_error(
                    "INSUFFICIENT_MUST_HAVE_NULL_PLAN",
                    "status=insufficient_input requires primary_plan=null.",
                    "primary_plan",
                )
            for idx, act in enumerate(actions):
                if act.get("operation") in PHYSICAL_MOVEMENT_OPERATIONS:
                    result.add_error(
                        "NO_PHYSICAL_ACTION_WHEN_INSUFFICIENT",
                        "Cannot emit physical movement actions when status=insufficient_input.",
                        f"actions[{idx}]",
                    )

        # 3. needs_subject_selection rule (Rule III.15)
        if obs_status == "needs_subject_selection":
            if status != "hold":
                result.add_error(
                    "NEEDS_SUBJECT_MUST_HOLD",
                    "Observation status=needs_subject_selection requires plan status=hold.",
                    "status",
                )
            if primary_plan is not None:
                result.add_error(
                    "NEEDS_SUBJECT_NULL_PLAN",
                    "needs_subject_selection requires primary_plan=null.",
                    "primary_plan",
                )
            if actions:
                result.add_error(
                    "NEEDS_SUBJECT_EMPTY_ACTIONS",
                    "needs_subject_selection requires actions=[].",
                    "actions",
                )
            if active_action_id is not None:
                result.add_error(
                    "NEEDS_SUBJECT_NULL_ACTIVE_ACTION",
                    "needs_subject_selection requires active_action_id=null.",
                    "active_action_id",
                )
            if not str(plan.get("fallback_instruction", "")).strip():
                result.add_error(
                    "NEEDS_SUBJECT_MISSING_FALLBACK",
                    "needs_subject_selection must prompt user via fallback_instruction.",
                    "fallback_instruction",
                )

        # Build valid ID sets for reference integrity (Rule III.3)
        valid_fact_ids = {f["fact_id"] for f in observation.get("facts", []) if "fact_id" in f}
        valid_anchor_ids = {a["anchor_id"] for a in observation.get("anchors", []) if "anchor_id" in a}
        valid_card_ids = {c.id for c in provided_cards}

        # --- Person ghost box (Rule III.17 / III.18) -------------------------
        guide = plan.get("subject_placement_guide")
        subject_present = observation.get("selected_subject") is not None
        obs_status_for_guide = observation.get("status")

        if guide is not None:
            if not isinstance(guide, dict):
                result.add_error(
                    "INVALID_PLACEMENT_GUIDE",
                    "subject_placement_guide must be an object or null.",
                    "subject_placement_guide",
                )
            else:
                _check_bbox_order(guide.get("bbox"), "subject_placement_guide.bbox", result)
                ratio = guide.get("target_height_ratio")
                if isinstance(ratio, list) and len(ratio) == 2:
                    if not (0.0 <= ratio[0] <= ratio[1] <= 1.0):
                        result.add_error(
                            "INVALID_HEIGHT_RATIO_RANGE",
                            f"subject_placement_guide.target_height_ratio must satisfy 0 <= min <= max <= 1, got {ratio}",
                            "subject_placement_guide.target_height_ratio",
                        )
                for fid in guide.get("fact_ids", []):
                    if fid not in valid_fact_ids:
                        result.add_error(
                            "DANGLING_FACT_REF",
                            f"subject_placement_guide.fact_id '{fid}' not in observation.",
                            "subject_placement_guide.fact_ids",
                        )

        # Rule III.18: the ghost box and the real target box must not both appear.
        guide_forbidden = (
            obs_status_for_guide in {"insufficient_input", "needs_subject_selection"}
            or status in {"insufficient_input", "hold"}
        )
        if subject_present and guide is not None:
            result.add_error(
                "PLACEMENT_GUIDE_CONFLICTS_WITH_SUBJECT",
                "A subject is already in frame; subject_placement_guide must be null so only one "
                "position guide (primary_plan.composition.target_subject_bbox) is active.",
                "subject_placement_guide",
            )

        # --- Landscape crop recommendation (Rule III.19 / III.20 / III.21) ----
        crop = plan.get("crop_recommendation")
        if crop is not None:
            if not isinstance(crop, dict):
                result.add_error(
                    "INVALID_CROP_RECOMMENDATION",
                    "crop_recommendation must be an object or null.",
                    "crop_recommendation",
                )
            else:
                crop_rect = crop.get("crop_rect")
                _check_bbox_order(crop_rect, "crop_recommendation.crop_rect", result)
                # Rule III.21: cropping must not cut the planned subject out.
                if (
                    isinstance(crop_rect, list)
                    and len(crop_rect) == 4
                    and isinstance(guide, dict)
                    and isinstance(guide.get("bbox"), list)
                    and len(guide["bbox"]) == 4
                ):
                    gxmin, gymin, gxmax, gymax = guide["bbox"]
                    cxmin, cymin, cxmax, cymax = crop_rect
                    if not (
                        cxmin <= gxmin and cymin <= gymin and cxmax >= gxmax and cymax >= gymax
                    ):
                        result.add_error(
                            "CROP_EXCLUDES_PLACEMENT_GUIDE",
                            f"crop_rect {crop_rect} does not contain the person ghost figure "
                            f"{guide['bbox']}; the crop would cut the subject out of the picture.",
                            "crop_recommendation.crop_rect",
                        )

        if obs_status_for_guide == "observed" and not subject_present and crop is None:
            result.add_error(
                "MISSING_CROP_RECOMMENDATION",
                "No subject in frame: the plan must provide crop_recommendation so the landscape "
                "can be cropped to its best composition.",
                "crop_recommendation",
            )

        if guide_forbidden and crop is not None:
            result.add_error(
                "CROP_NOT_ALLOWED",
                f"Observation status={obs_status_for_guide}, plan status={status}; "
                "crop_recommendation must be null.",
                "crop_recommendation",
            )

        if subject_present and crop is not None:
            result.add_error(
                "CROP_CONFLICTS_WITH_SUBJECT",
                "A subject is already in frame; the framing is decided by the subject, so "
                "crop_recommendation must be null.",
                "crop_recommendation",
            )

        # Rule III.17: a scene with nobody in frame must still say where to stand.
        if obs_status_for_guide == "observed" and not subject_present and guide is None:
            result.add_error(
                "MISSING_PLACEMENT_GUIDE",
                "No subject in frame: the plan must provide subject_placement_guide so the user "
                "knows where the person should stand.",
                "subject_placement_guide",
            )

        # A scene that cannot be judged must not carry a ghost box.
        if guide_forbidden and guide is not None:
            result.add_error(
                "PLACEMENT_GUIDE_NOT_ALLOWED",
                f"Observation status={obs_status_for_guide}, plan status={status}; "
                "subject_placement_guide must be null.",
                "subject_placement_guide",
            )

        # 4. Primary plan checks
        if isinstance(primary_plan, dict):
            for fid in primary_plan.get("used_fact_ids", []):
                if fid not in valid_fact_ids:
                    result.add_error("DANGLING_FACT_REF", f"used_fact_id '{fid}' not in observation.", "primary_plan.used_fact_ids")
            for cid in primary_plan.get("used_card_ids", []):
                if cid not in valid_card_ids:
                    result.add_error("DANGLING_CARD_REF", f"used_card_id '{cid}' was not provided.", "primary_plan.used_card_ids")

            comp = primary_plan.get("composition", {})
            _check_bbox_order(comp.get("target_subject_bbox"), "primary_plan.composition.target_subject_bbox", result)
            hr = comp.get("target_subject_height_ratio")
            if isinstance(hr, list) and len(hr) == 2 and not (0.0 <= hr[0] <= hr[1] <= 1.0):
                result.add_error(
                    "INVALID_HEIGHT_RATIO_RANGE",
                    f"target_subject_height_ratio must satisfy 0 <= min <= max <= 1, got {hr}",
                    "primary_plan.composition.target_subject_height_ratio",
                )
            for aid in comp.get("preserve_anchor_ids", []):
                if aid not in valid_anchor_ids:
                    result.add_error(
                        "DANGLING_ANCHOR_REF",
                        f"preserve_anchor_id '{aid}' not in observation anchors.",
                        "primary_plan.composition.preserve_anchor_ids",
                    )

            # 5. Device & lens capability validation (Rule III.4 & III.5)
            self._validate_lens(primary_plan.get("lens", {}), runtime_input, result)

        # 6. Actions & active_action_id validation (Rules III.3, III.7, III.8, III.9, III.10, III.16)
        action_ids: list[str] = []
        for idx, act in enumerate(actions):
            aid = act.get("action_id")
            if aid in action_ids:
                result.add_error("DUPLICATE_ACTION_ID", f"Duplicate action_id: {aid}", f"actions[{idx}].action_id")
            if isinstance(aid, str):
                action_ids.append(aid)

            for fid in act.get("evidence_fact_ids", []):
                if fid not in valid_fact_ids:
                    result.add_error(
                        "DANGLING_FACT_REF",
                        f"evidence_fact_id '{fid}' not in observation.",
                        f"actions[{idx}].evidence_fact_ids",
                    )

            op = act.get("operation")
            actor = act.get("actor")
            instruction = str(act.get("instruction", ""))
            check = act.get("completion_check", {})
            check_type = check.get("type")

            # Rule III.9: completion_check must map to predefined validator
            if check_type not in ALLOWED_COMPLETION_CHECK_TYPES:
                result.add_error(
                    "UNKNOWN_COMPLETION_CHECK",
                    f"Unsupported completion_check type: {check_type}",
                    f"actions[{idx}].completion_check.type",
                )

            # Rule III.10: head_obstacle_separated cannot be claimed by pan/tilt/zoom alone
            if check_type == "head_obstacle_separated" and op in NON_PARALLAX_OPERATIONS:
                result.add_error(
                    "INVALID_PARALLAX_CLAIM",
                    f"Operation '{op}' cannot resolve same-line-of-sight head_obstacle_separated; physical parallax is required.",
                    f"actions[{idx}]",
                )

            # Anti-hallucination: pseudo-precise distance in meters/cm from single frame
            if PSEUDO_PRECISE_DISTANCE_PATTERN.search(instruction):
                result.add_error(
                    "PSEUDO_PRECISE_DISTANCE",
                    f"Instruction contains unmeasured metric distance from single frame: '{instruction}'",
                    f"actions[{idx}].instruction",
                )

            # Movement constraint checks (Rule III.8)
            mc = runtime_input.movement_constraints
            if op == "physical_translate" and actor == "photographer":
                if mc.photographer_translation in {"forbidden", "tripod_fixed"}:
                    result.add_error(
                        "PHOTOGRAPHER_TRANSLATION_FORBIDDEN",
                        f"Photographer translation is '{mc.photographer_translation}' but action requests physical_translate.",
                        f"actions[{idx}]",
                    )
                elif mc.photographer_translation == "unknown":
                    if aid == active_action_id:
                        result.add_error(
                            "UNCONFIRMED_MOVEMENT_SPACE_ACTIVE",
                            "Cannot activate photographer physical_translate while photographer_translation is unknown; confirm movement space first.",
                            f"actions[{idx}]",
                        )

            if op == "physical_translate" and actor == "subject":
                if mc.subject_translation in {"forbidden", "not_requested"} and not runtime_input.user_goal.allow_subject_walking:
                    result.add_error(
                        "SUBJECT_TRANSLATION_DISALLOWED",
                        f"Subject translation is '{mc.subject_translation}' and allow_subject_walking=false.",
                        f"actions[{idx}]",
                    )

            # Rule III.16: v0.1 human-guidance boundary (camera control ops must not claim camera_api auto-execution)
            if op in {"change_zoom", "exposure_adjust"} and actor == "camera":
                result.add_warning(
                    "V01_HUMAN_GUIDANCE_ONLY",
                    f"v0.1 '{op}' is human guidance only; do not parse natural language as direct camera hardware commands.",
                    f"actions[{idx}]",
                )

        if active_action_id is not None and active_action_id not in action_ids:
            result.add_error(
                "DANGLING_ACTIVE_ACTION_ID",
                f"active_action_id '{active_action_id}' does not exist in actions.",
                "active_action_id",
            )
        if status == "plan_ready" and actions and active_action_id is None:
            result.add_error(
                "MISSING_ACTIVE_ACTION_ID",
                "plan_ready with non-empty actions must specify active_action_id.",
                "active_action_id",
            )

        return result

    @staticmethod
    def _validate_lens(lens_plan: dict[str, Any], runtime_input: RuntimeInput, result: ValidationResult) -> None:
        caps = runtime_input.camera_capabilities
        target_lens_id = lens_plan.get("target_lens_id")
        eq_mm = lens_plan.get("equivalent_focal_length_mm")
        ui_label = lens_plan.get("ui_label")
        action = lens_plan.get("action")

        if not caps.metadata_available or not caps.available_lenses:
            # Rule III.5: when device metadata is missing, must not fabricate focal length or switch to nonexistent lens
            if eq_mm is not None:
                result.add_error(
                    "FABRICATED_FOCAL_LENGTH",
                    f"Device lens metadata is unavailable; equivalent_focal_length_mm must be null, got {eq_mm}.",
                    "primary_plan.lens.equivalent_focal_length_mm",
                )
            if action == "switch" and target_lens_id is not None:
                result.add_error(
                    "UNSUPPORTED_LENS_SWITCH",
                    "Cannot switch lens when available_lenses metadata is missing.",
                    "primary_plan.lens.target_lens_id",
                )
            return

        if target_lens_id is None:
            if action == "switch":
                result.add_error(
                    "MISSING_TARGET_LENS_ID",
                    "lens.action='switch' requires a valid target_lens_id from available_lenses.",
                    "primary_plan.lens.target_lens_id",
                )
            return

        matched = caps.get_lens(target_lens_id)
        if matched is None:
            result.add_error(
                "LENS_NOT_IN_CAPABILITIES",
                f"target_lens_id '{target_lens_id}' is not in device available_lenses.",
                "primary_plan.lens.target_lens_id",
            )
            return

        if ui_label is not None and ui_label != matched.ui_label:
            result.add_error(
                "LENS_UI_LABEL_MISMATCH",
                f"ui_label '{ui_label}' does not match device lens label '{matched.ui_label}'.",
                "primary_plan.lens.ui_label",
            )

        if matched.equivalent_focal_length_mm is None and eq_mm is not None:
            result.add_error(
                "FABRICATED_FOCAL_LENGTH",
                f"Device lens '{target_lens_id}' has unknown focal length; model must not fabricate {eq_mm}mm.",
                "primary_plan.lens.equivalent_focal_length_mm",
            )
        elif matched.equivalent_focal_length_mm is not None and eq_mm is not None:
            if abs(float(eq_mm) - float(matched.equivalent_focal_length_mm)) > 0.5:
                result.add_error(
                    "FOCAL_LENGTH_MISMATCH",
                    f"Planned focal length {eq_mm}mm does not match device metadata {matched.equivalent_focal_length_mm}mm.",
                    "primary_plan.lens.equivalent_focal_length_mm",
                )
