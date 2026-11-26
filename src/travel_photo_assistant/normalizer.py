"""Vocabulary normalization for real VLM output.

Vendors do not reliably enforce `enum` in their structured-output schemas:
Gemini happily returns `status="ok"` or `source="vision_hint"` even when the
response schema lists the exact allowed values. The server must therefore
keep validating (integration_notes.txt Section II: "供应商输出结构正确不代表
观察、物理动作或美学判断正确"), but first it has to bring the payload back
onto the application vocabulary.

This module performs a *closed* mapping: only values in the tables below are
rewritten. Anything unrecognized is left untouched so the normal schema
validation still rejects it — the normalizer never widens the contract, it
only folds near-miss spellings onto the canonical term.

Every rewrite is recorded so the change is auditable. Known directional-light
values are additionally downgraded when not backed by an applicable visual fact.
"""

from __future__ import annotations

import copy
import re
from typing import Any


# --------------------------------------------------------------------------
# Synonym tables. Keys are (payload_kind, logical_path); values map a
# non-canonical token onto the canonical one.
# --------------------------------------------------------------------------

OBSERVATION_ENUMS: dict[tuple[str, ...], dict[str, str]] = {
    ("status",): {
        "ok": "observed",
        "success": "observed",
        "valid": "observed",
        "normal": "observed",
        "observed": "observed",
        "no_image": "insufficient_input",
        "missing_image": "insufficient_input",
        "no_input": "insufficient_input",
        "insufficient": "insufficient_input",
        "multiple_subjects": "needs_subject_selection",
        "subject_selection": "needs_subject_selection",
        "select_subject": "needs_subject_selection",
    },
    ("facts", "*", "kind"): {
        "environment": "scene",
        "landmark": "scene",
        "place": "scene",
        "scenery": "scene",
        "surroundings": "scene",
        "person": "subject",
        "people": "subject",
        "human": "subject",
        "human_subject": "subject",
        "portrait": "subject",
        "composition": "geometry",
        "structure": "geometry",
        "layout": "geometry",
        "spatial": "geometry",
        "illumination": "light",
        "lighting": "light",
        "sunlight": "light",
        "movement": "constraint",
        "safety": "constraint",
        "restriction": "constraint",
    },
    ("facts", "*", "source"): {
        "vision_hint": "vlm_visual",
        "vision_observation": "vlm_visual",
        "vision": "vlm_visual",
        "vlm": "vlm_visual",
        "model": "vlm_visual",
        "image": "vlm_visual",
        "visual": "vlm_visual",
        "picture": "vlm_visual",
        "hint": "local_vision",
        "detector": "local_vision",
        "tracker": "local_vision",
        "on_device": "local_vision",
        "camera": "device_metadata",
        "device": "device_metadata",
        "exif": "device_metadata",
        "user": "user_input",
        "human": "user_input",
        "operator": "user_input",
    },
    ("facts", "*", "strength"): {
        "strong": "high",
        "certain": "high",
        "bright": "high",
        "weak": "low",
        "dim": "low",
        "uncertain": "low",
        "medium": "medium",
        "moderate": "medium",
    },
    ("anchors", "*", "kind"): {
        "geometry": "frame",
        "surround": "frame",
        "window": "frame",
        "doorway": "frame",
        "environment": "landmark",
        "scenery": "landmark",
        "place": "landmark",
        "viewpoint": "landmark",
        "line": "leading_line",
        "path": "leading_line",
        "road": "leading_line",
        "shoreline": "leading_line",
        "river": "leading_line",
        "background": "clean_background",
        "empty": "clean_background",
        "negative_space": "clean_background",
        "ground": "possible_ground",
        "floor": "possible_ground",
        "surface": "possible_ground",
        "pole": "obstacle",
        "post": "obstacle",
        "tree": "obstacle",
        "sign": "obstacle",
        "utility_pole": "obstacle",
        "skyline": "horizon",
        "sea": "horizon",
        "horizon_line": "horizon",
    },
    ("issues", "*", "priority"): {
        "critical": "high",
        "severe": "high",
        "urgent": "high",
        "major": "high",
        "blocker": "high",
        "normal": "medium",
        "moderate": "medium",
        "minor": "low",
        "slight": "low",
        "trivial": "low",
    },
    ("selected_subject", "face_visibility"): {
        "clear": "visible",
        "fully_visible": "visible",
        "partial": "occluded",
        "partially_visible": "occluded",
        "hidden": "occluded",
        "blocked": "occluded",
        "small": "too_small",
        "far": "too_small",
        "distant": "too_small",
        "unclear": "unknown",
        "indeterminate": "unknown",
    },
    ("light", "direction_estimate"): {
        "from_front": "front",
        "frontal": "front",
        "front_light": "front",
        "from_left": "side",
        "from_right": "side",
        "lateral": "side",
        "side_light": "side",
        "from_back": "back",
        "backlit": "back",
        "rim": "back",
        "overcast": "diffuse",
        "flat": "diffuse",
        "soft": "diffuse",
        "ambiguous": "unknown",
        "uncertain": "unknown",
    },
    ("light", "strength"): {
        "strong": "high",
        "bright": "high",
        "harsh": "high",
        "weak": "low",
        "dim": "low",
        "moderate": "medium",
        "medium": "medium",
    },
}

PLAN_ENUMS: dict[tuple[str, ...], dict[str, str]] = {
    ("status",): {
        "ready": "plan_ready",
        "planning": "plan_ready",
        "ok": "plan_ready",
        "plan": "plan_ready",
        "need_more_info": "need_more_view",
        "need_more_evidence": "need_more_view",
        "insufficient": "insufficient_input",
        "no_input": "insufficient_input",
        "no_person": "no_subject",
        "waiting": "hold",
        "waiting_for_subject": "hold",
    },
    ("primary_plan", "composition", "orientation"): {
        "vertical": "portrait",
        "upright": "portrait",
        "horizontal": "landscape",
        "wide": "landscape",
        "unchanged": "keep",
        "as_is": "keep",
        "current": "keep",
    },
    ("primary_plan", "composition", "shot_size"): {
        "environmental": "environmental_full_body",
        "environmental_full": "environmental_full_body",
        "wide_full_body": "environmental_full_body",
        "full": "full_body",
        "full_length": "full_body",
        "three_quarter_body": "three_quarter",
        "3/4": "three_quarter",
        "three_quarters": "three_quarter",
        "half": "half_body",
        "waist": "half_body",
        "medium": "half_body",
        "head": "headshot",
        "closeup": "headshot",
        "close_up": "headshot",
        "face": "headshot",
        "portrait": "headshot",
    },
    ("primary_plan", "lens", "action"): {
        "stay": "keep",
        "maintain": "keep",
        "current": "keep",
        "none": "keep",
        "change": "switch",
        "switch_lens": "switch",
        "zoom": "switch",
    },
    ("actions", "*", "actor"): {
        "person": "subject",
        "people": "subject",
        "human": "subject",
        "self": "subject",
        "model": "subject",
        "user": "photographer",
        "operator": "photographer",
        "taker": "photographer",
        "camera": "camera",
        "device": "camera",
    },
    ("actions", "*", "operation"): {
        "pan": "reframe_pan",
        "pan_left": "reframe_pan",
        "pan_right": "reframe_pan",
        "tilt": "reframe_tilt",
        "tilt_up": "reframe_tilt",
        "level": "level_camera",
        "level_horizon": "level_camera",
        "move": "physical_translate",
        "translate": "physical_translate",
        "walk": "physical_translate",
        "step": "physical_translate",
        "raise_camera": "raise_lower_camera",
        "lower_camera": "raise_lower_camera",
        "height_change": "raise_lower_camera",
        "switch_lens": "change_lens",
        "zoom": "change_zoom",
        "zoom_in": "change_zoom",
        "rotate": "rotate_body",
        "turn_body": "rotate_body",
        "gaze": "head_gaze",
        "look": "head_gaze",
        "stand": "pose_in_place",
        "pose": "pose_in_place",
        "hold_position": "pose_in_place",
        "exposure": "exposure_adjust",
        "ev": "exposure_adjust",
        "brightness": "exposure_adjust",
        "confirm": "confirm_movement_space",
        "confirm_space": "confirm_movement_space",
        "capture": "capture_new_frame",
        "new_frame": "capture_new_frame",
        "shoot": "shoot",
        "take_photo": "shoot",
        "wait": "hold",
        "none": "hold",
    },
    ("actions", "*", "direction_reference"): {
        "image": "input_image",
        "photo": "input_image",
        "screen": "input_image",
        "frame": "input_image",
        "photographer": "photographer_body",
        "user_body": "photographer_body",
        "shooter": "photographer_body",
        "subject": "subject_body",
        "person_body": "subject_body",
        "device": "device_control",
        "hardware": "device_control",
        "none": "none",
        "not_applicable": "none",
    },
    ("actions", "*", "completion_check", "type"): {
        "center_in_range": "subject_center_x_in_range",
        "center_x": "subject_center_x_in_range",
        "horizontal_position": "subject_center_x_in_range",
        "height_ratio": "subject_height_ratio_in_range",
        "subject_height": "subject_height_ratio_in_range",
        "head_clear": "head_obstacle_separated",
        "head_separated": "head_obstacle_separated",
        "manual": "manual_confirmed",
        "user_confirmed": "manual_confirmed",
        "confirmed": "manual_confirmed",
        "new_frame": "fresh_frame_received",
        "fresh_frame": "fresh_frame_received",
        "lens_match": "camera_metadata_match",
        "metadata_match": "camera_metadata_match",
        "pose": "pose_visually_confirmed",
        "vlm_review": "vlm_review",
    },
    ("actions", "*", "completion_check", "check_by"): {
        "on_device": "local_vision",
        "device_vision": "local_vision",
        "camera": "camera_api",
        "hardware": "camera_api",
        "vlm": "vlm_review",
        "model": "vlm_review",
        "human": "user",
        "manual": "user",
    },
    ("subject_placement_guide", "basis"): {
        "thirds": "rule_of_thirds",
        "rule_of_thirds": "rule_of_thirds",
        "symmetry": "centered_symmetry",
        "centered": "centered_symmetry",
        "symmetric": "centered_symmetry",
        "leading_line": "leading_line_anchor",
        "leading_lines": "leading_line_anchor",
        "line_anchor": "leading_line_anchor",
        "empty": "open_area",
        "negative_space": "open_area",
        "open": "open_area",
    },
    ("subject_placement_guide", "framing"): {
        "full": "full_body",
        "fullbody": "full_body",
        "whole_body": "full_body",
        "environmental_full_body": "full_body",
        "environmental": "full_body",
        "three_quarters": "three_quarter",
        "3/4": "three_quarter",
        "knee": "three_quarter",
        "half": "half_body",
        "waist": "half_body",
        "medium": "half_body",
        "head": "headshot",
        "head_shot": "headshot",
        "closeup": "headshot",
        "close_up": "headshot",
        "face": "headshot",
    },
    ("crop_recommendation", "aspect_ratio"): {
        "square": "1:1",
        "1_1": "1:1",
        "landscape": "3:2",
        "3_2": "3:2",
        "portrait": "2:3",
        "2_3": "2:3",
        "4_3": "4:3",
        "3_4": "3:4",
        "4_5": "4:5",
        "5_4": "4:5",
        "wide": "16:9",
        "16_9": "16:9",
        "cinematic": "16:9",
        "vertical": "9:16",
        "9_16": "9:16",
    },
}

# Array-of-enum fields: an unrecognized entry is dropped (and logged) rather
# than failing the whole payload.
PLAN_ENUM_LISTS: dict[tuple[str, ...], dict[str, str]] = {
    ("invalidation_conditions",): {
        "scene_change": "scene_version_changed",
        "new_scene": "scene_version_changed",
        "goal_change": "user_goal_changed",
        "user_goal_change": "user_goal_changed",
        "subject_change": "selected_subject_changed",
        "light_change": "light_materially_changed",
        "crop_change": "crop_changed",
        "orientation_change": "orientation_changed",
        "lens_change": "lens_changed",
        "tracking_lost": "tracking_lost",
    },
}

# Canonical values per list, used to drop unknown entries.
OBSERVATION_ENUMS["invalidation_conditions",] = {}


def _resolve(container: Any, path: tuple[str, ...]) -> list[tuple[Any, str]]:
    """Return (container, key) pairs addressed by `path`; `*` walks list items."""
    if not path:
        return []
    head, rest = path[0], path[1:]
    if head == "*":
        out: list[tuple[Any, str]] = []
        for item in container if isinstance(container, list) else []:
            out.extend(_resolve(item, rest))
        return out
    if not isinstance(container, dict) or head not in container:
        return []
    value = container[head]
    if not rest:
        return [(container, head)]
    if isinstance(value, list):
        if rest[0] == "*":
            out = []
            for item in value:
                out.extend(_resolve(item, rest[1:]))
            return out
        return []
    return _resolve(value, rest)


def _apply(
    payload: dict[str, Any],
    table: dict[tuple[str, ...], dict[str, str]],
    changes: list[str],
    *,
    canonical_lists: dict[tuple[str, ...], set[str]] | None = None,
) -> None:
    for path, synonyms in table.items():
        if canonical_lists is not None and path in canonical_lists:
            canonical = canonical_lists[path]
            for container, key in _resolve(payload, path):
                value = container.get(key)
                if not isinstance(value, list):
                    continue
                kept = []
                for item in value:
                    if isinstance(item, str) and item not in canonical:
                        replacement = synonyms.get(item)
                        if replacement is not None:
                            kept.append(replacement)
                            changes.append(f"{'.'.join(path)}: '{item}' -> '{replacement}'")
                        else:
                            changes.append(f"{'.'.join(path)}: dropped unknown value '{item}'")
                        continue
                    kept.append(item)
                container[key] = kept
            continue

        for container, key in _resolve(payload, path):
            value = container.get(key)
            if not isinstance(value, str):
                continue
            replacement = synonyms.get(value)
            if replacement is not None and replacement != value:
                container[key] = replacement
                changes.append(f"{'.'.join(path)}: '{value}' -> '{replacement}'")


def _canonical_set(table: dict[tuple[str, ...], dict[str, str]], path: tuple[str, ...]) -> set[str]:
    return set(table.get(path, {}).values()) | set(table.get(path, {}).keys())


# A localized light fact must explicitly describe the claimed direction. These
# conservative cues are not optical inference; unknown is always preferable to
# upgrading an appearance-only observation such as dappled ground shadows.
_DIRECTION_EVIDENCE = {
    "front": re.compile(r"\b(?:front(?:al)?[ -]light(?:ing|ed)?|light (?:comes|falls) from (?:the )?front)\b|正面光|顺光", re.I),
    "side": re.compile(r"\b(?:side[ -]light(?:ing|ed)?|light (?:comes|falls) from (?:the )?(?:left|right|side))\b|侧光|侧面来光|从[左右]侧打光", re.I),
    "back": re.compile(r"\b(?:back[ -]?light(?:ing|ed)?|light (?:comes|falls) from behind)\b|逆光|背光", re.I),
}
_NEGATED_DIRECTION = re.compile(
    r"\b(?:no|not|unknown|unclear|uncertain|ambiguous|cannot|insufficient)\b|无法|不能|并非|不是|不确定|看不出|无证据", re.I
)


def _downgrade_unsupported_direction(payload: dict[str, Any], changes: list[str]) -> None:
    light = payload.get("light")
    facts = payload.get("facts")
    if not isinstance(light, dict) or not isinstance(facts, list):
        return  # Let schema validation report malformed structures.
    direction = light.get("direction_estimate")
    refs = light.get("fact_ids")
    if not isinstance(direction, str) or direction not in _DIRECTION_EVIDENCE or not isinstance(refs, list):
        return
    for fact in facts:
        if not isinstance(fact, dict) or fact.get("fact_id") not in refs:
            continue
        if fact.get("kind") != "light" or fact.get("source") not in ("vlm_visual", "local_vision"):
            continue
        if fact.get("strength") not in ("medium", "high"):
            continue
        region = fact.get("region")
        if (
            not isinstance(region, list) or len(region) != 4
            or not all(type(v) in (int, float) and 0 <= v <= 1 for v in region)
            or region[0] >= region[2] or region[1] >= region[3]
        ):
            continue
        statement = fact.get("statement")
        if not isinstance(statement, str) or _NEGATED_DIRECTION.search(statement):
            continue
        # Contradictory cues are not support for a single direction.
        matches = [key for key, pattern in _DIRECTION_EVIDENCE.items() if pattern.search(statement)]
        if matches == [direction]:
            return
    light["direction_estimate"] = "unknown"
    changes.append(f"light.direction_estimate: '{direction}' -> 'unknown' (no linked directional visual evidence)")


def normalize_observation(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Fold near-miss enum values in a scene_observation onto the canonical set."""
    cloned = copy.deepcopy(payload)
    changes: list[str] = []
    _apply(cloned, OBSERVATION_ENUMS, changes)
    _downgrade_unsupported_direction(cloned, changes)
    return cloned, changes


def normalize_plan(
    payload: dict[str, Any], *, photographer_side_step: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Fold near-miss enum values in a shot_plan onto the canonical set."""
    cloned = copy.deepcopy(payload)
    changes: list[str] = []
    _apply(cloned, PLAN_ENUMS, changes)
    _apply(
        cloned,
        PLAN_ENUM_LISTS,
        changes,
        canonical_lists={
            ("invalidation_conditions",): {
                "scene_version_changed",
                "lens_changed",
                "crop_changed",
                "orientation_changed",
                "selected_subject_changed",
                "light_materially_changed",
                "tracking_lost",
                "user_goal_changed",
            }
        },
    )
    # The operator chose a body-relative side. Never let planner prose reverse it
    # or add a second move. Only canonicalize the physical action, not diagnosis.
    if photographer_side_step in ("left", "right") and isinstance(cloned.get("actions"), list):
        word = "左" if photographer_side_step == "left" else "右"
        for i, action in enumerate(cloned["actions"]):
            if not isinstance(action, dict) or action.get("actor") != "photographer" or action.get("operation") != "physical_translate":
                continue
            check = action.get("completion_check")
            if isinstance(check, dict) and check.get("type") == "head_obstacle_separated" and check.get("check_by") == "vlm_review":
                action["instruction"] = (
                    f"保持镜头不变，拍摄者只向自己的{word}侧小幅侧移一步并停下。"
                    "现场不安全就不要移动；移动后重新取景，复核头部与背景是否分离。"
                    "若仍重叠，不再移动，返回等待。"
                )
                action["prerequisites"] = ["拍摄者已在现场确认该侧仅一步的安全空间"]
                check["description"] = "以移动后新帧复核头部轮廓与背景物分离；未分离、画面不清或过期均不算完成。"
                changes.append(f"actions[{i}].instruction: bounded to operator-confirmed {photographer_side_step} side")
    return cloned, changes
