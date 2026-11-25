"""Deterministic offline VisionModelAdapter for local testing, demos, and contract evaluation.

Strictly honors v0.1 boundaries:
- Returns insufficient_input when image_attachments is empty.
- Returns needs_subject_selection -> hold when multiple subjects exist without selection.
- Never fabricates focal lengths, metric distances, or nonexistent lenses.
- Separates pan/tilt/zoom from physical parallax for head-obstacle overlaps.
"""

from __future__ import annotations

from typing import Any

from ..models import RuntimeInput, ShotCard
from .base import VisionModelAdapter


def _pick_framing(runtime_input: RuntimeInput, observation: dict[str, Any]) -> str:
    """Decide whether the ghost figure shows the whole body or only the upper half."""
    hints = runtime_input.local_vision_hints
    if hints.get("virtual_subject_framing"):
        return str(hints["virtual_subject_framing"])

    text = runtime_input.user_goal.priority
    # Only the priority decides the framing: the style is usually a generic
    # "travel portrait" and says nothing about how tight the shot should be.
    if any(k in text for k in ("近景", "特写", "半身", "胸像", "大头照")):
        return "half_body"
    if any(k in text for k in ("全身", "环境", "人景", "并重", "全身照")):
        return "full_body"
    if "环境" in runtime_input.user_goal.goal_tags or "person_environment_balance" in runtime_input.user_goal.goal_tags:
        return "full_body"
    return "full_body"


def _build_placement_guide(
    runtime_input: RuntimeInput,
    observation: dict[str, Any],
) -> dict[str, Any]:
    """Build the person ghost figure for a scene where nobody is in frame.

    The figure is a composition proposal ("stand here"), not an observation.
    Its width follows from the target height ratio and the real frame aspect so
    the silhouette keeps human proportions, and `framing` says how much of the
    body it covers.
    """
    hints = runtime_input.local_vision_hints
    scene_tags = set(observation.get("scene_tags") or [])
    img_w = max(1, runtime_input.frame.upright_input_width_px)
    img_h = max(1, runtime_input.frame.upright_input_height_px)

    framing = _pick_framing(runtime_input, observation)

    # How much of the body the frame shows, and how tall that visible part
    # should be in the picture. A half-body portrait puts the subject CLOSER,
    # so the visible extent is bigger than for a full-body shot.
    visible_fraction = {"full_body": 1.0, "three_quarter": 0.72, "half_body": 0.50, "headshot": 0.22}
    default_extent = {
        "full_body": [0.45, 0.60],
        "three_quarter": [0.52, 0.66],
        "half_body": [0.50, 0.62],
        "headshot": [0.26, 0.38],
    }[framing]
    height_ratio = hints.get("virtual_subject_height_ratio", default_extent)
    lo, hi = float(height_ratio[0]), float(height_ratio[1])
    mid = (lo + hi) / 2.0

    # Where the person's feet should land: the visible ground line.
    ground_y = float(hints.get("ground_line_y", 0.90))

    # Pick the horizontal placement basis.
    anchors = observation.get("anchors") or []
    leading = next((a for a in anchors if a.get("kind") == "leading_line"), None)
    symmetric = bool({"symmetry", "architecture"} & scene_tags)

    if symmetric:
        basis = "centered_symmetry"
        center_x = 0.50
    elif leading is not None and leading.get("bbox"):
        lb = leading["bbox"]
        # Stand on the line: use the far end of the leading line's box.
        basis = "leading_line_anchor"
        center_x = float(lb[2]) * 0.5 + float(lb[0]) * 0.5
        center_x = min(max(center_x, 0.25), 0.75)
    else:
        basis = "rule_of_thirds"
        center_x = 0.62

    # The visible extent in pixels, and the full standing height it implies.
    visible_h_px = mid * img_h
    full_h_px = visible_h_px / visible_fraction[framing]
    # Shoulders span about a quarter of the standing height and do not change
    # with the framing, so the figure keeps the same width in every framing.
    shoulder_w_px = full_h_px * 0.26
    box_w_px = shoulder_w_px * 1.06
    half_w = (box_w_px / img_w) / 2.0

    xmin = max(0.0, center_x - half_w)
    xmax = min(1.0, center_x + half_w)

    if framing == "full_body":
        ymin = max(0.0, ground_y - mid)
        ymax = min(1.0, ground_y)
    else:
        # Upper-body framings place the head in the upper third; the feet are
        # out of frame, so the figure is anchored by the top of the head.
        head_top = {"three_quarter": 0.26, "half_body": 0.20, "headshot": 0.28}[framing]
        ymin = max(0.0, head_top)
        ymax = min(1.0, ymin + mid)

    fact_ids = [f["fact_id"] for f in observation.get("facts", [])[:1]]
    instruction = (
        "被拍者站到虚框位置，身体完整入镜、头顶不贴画面上沿，脚下踩稳后再拍摄。"
    )
    if framing == "half_body":
        instruction = "被拍者靠近到虚框位置，取到腰部以上，脸部清晰不贴画面边缘。"
    elif framing == "three_quarter":
        instruction = "被拍者站到虚框位置，取到膝盖以上，身体重心自然。"
    elif framing == "headshot":
        instruction = "被拍者靠近到虚框位置，取头肩，脸部占画面主体。"
    if basis == "centered_symmetry":
        instruction += "与建筑中轴对齐。"
    elif basis == "leading_line_anchor":
        instruction += "让引导线指向身体。"

    return {
        "bbox": [round(xmin, 3), round(ymin, 3), round(xmax, 3), round(ymax, 3)],
        "framing": framing,
        "target_height_ratio": [round(lo, 3), round(hi, 3)],
        "instruction": instruction,
        "basis": basis,
        "fact_ids": fact_ids,
    }


def _build_crop_recommendation(
    runtime_input: RuntimeInput,
    observation: dict[str, Any],
    guide: dict[str, Any],
) -> dict[str, Any]:
    """Suggest a crop for an environment-only frame.

    The crop must keep the ghost figure: a crop that cuts the planned subject
    out of the picture defeats the point of the placement guide.
    """
    hints = runtime_input.local_vision_hints
    scene_tags = list(observation.get("scene_tags") or [])
    anchors = observation.get("anchors") or []

    # Aspect ratio follows the scene: wide for expansive landscapes, taller for
    # vertical architecture, square for symmetric scenes.
    tag_set = set(scene_tags)
    if "symmetry" in tag_set or "architecture" in tag_set:
        aspect = "1:1"
    elif tag_set & {"mountain", "lake", "coast", "city_view", "park"}:
        aspect = "16:9"
    elif "old_town" in tag_set or "city_street" in tag_set:
        aspect = "3:2"
    else:
        aspect = "3:2"
    aspect = str(hints.get("crop_aspect_ratio", aspect))

    # Start from the anchors that carry the scene, then make sure the planned
    # subject stays inside the crop.
    keep: list[tuple[float, float, float, float]] = []
    focus_ids: list[str] = []
    for anchor in anchors:
        bbox = anchor.get("bbox")
        if not bbox or anchor.get("kind") not in {"landmark", "leading_line", "frame", "horizon"}:
            continue
        keep.append((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])))
        focus_ids.append(str(anchor.get("anchor_id")))

    gb = guide.get("bbox") or [0.4, 0.35, 0.6, 0.9]
    keep.append((float(gb[0]), float(gb[1]), float(gb[2]), float(gb[3])))

    xmin = min(k[0] for k in keep)
    ymin = min(k[1] for k in keep)
    xmax = max(k[2] for k in keep)
    ymax = max(k[3] for k in keep)

    # Respect the target aspect ratio by widening the short side.
    img_w = max(1, runtime_input.frame.upright_input_width_px)
    img_h = max(1, runtime_input.frame.upright_input_height_px)
    num, den = (int(v) for v in aspect.split(":"))
    target_ratio = (num / den) * (img_w / img_h)  # width/height in normalized units

    width = xmax - xmin
    height = ymax - ymin
    if width / max(height, 1e-6) < target_ratio:
        need = height * target_ratio
        grow = (need - width) / 2.0
        xmin -= grow
        xmax += grow
    else:
        need = width / max(target_ratio, 1e-6)
        grow = (need - height) / 2.0
        ymin -= grow
        ymax += grow

    # Small breathing margin, then clamp into the frame.
    margin = 0.015
    xmin = max(0.0, xmin - margin)
    ymin = max(0.0, ymin - margin)
    xmax = min(1.0, xmax + margin)
    ymax = min(1.0, ymax + margin)

    rationale = (
        f"按 {aspect} 保留{'、'.join(focus_ids) if focus_ids else '画面主体结构'}"
        f"，同时保证人物虚框完整落在裁剪区内。"
    )
    return {
        "crop_rect": [round(xmin, 3), round(ymin, 3), round(xmax, 3), round(ymax, 3)],
        "aspect_ratio": aspect,
        "rationale": rationale,
        "scene_basis": scene_tags,
        "focus_anchor_ids": focus_ids,
    }


class RuleBasedOfflineVisionAdapter(VisionModelAdapter):
    """Offline deterministic adapter driven by local_vision_hints and explicit constraints."""

    def __init__(self, model_name: str = "offline-rule-engine-v0.1") -> None:
        self.model_name = model_name

    def observe(
        self,
        runtime_input: RuntimeInput,
        system_prompt: str,
        output_schema: dict[str, Any],
        feedback: str | None = None,
    ) -> dict[str, Any]:
        frame_id = runtime_input.frame.frame_id
        scene_version = runtime_input.frame.scene_version

        # Guard: no real image attachment -> insufficient_input
        if not runtime_input.frame.image_attachments:
            return {
                "schema_version": "0.1.0",
                "frame_id": frame_id,
                "scene_version": scene_version,
                "coordinate_space": "upright_input_normalized_unmirrored",
                "status": "insufficient_input",
                "scene_tags": [],
                "facts": [],
                "selected_subject": None,
                "anchors": [],
                "light": {
                    "appearance": "未提供图像，无法判断光线",
                    "direction_estimate": "unknown",
                    "strength": "low",
                    "fact_ids": [],
                },
                "issues": [],
                "unknowns": ["缺少真实图像附件", "画面内容未知"],
            }

        hints = runtime_input.local_vision_hints
        detected_people = int(hints.get("detected_person_count", runtime_input.user_goal.subject_count or 1))

        # Guard: multiple people without explicit subject_id selection
        if detected_people > 1 and not runtime_input.user_goal.subject_id:
            return {
                "schema_version": "0.1.0",
                "frame_id": frame_id,
                "scene_version": scene_version,
                "coordinate_space": "upright_input_normalized_unmirrored",
                "status": "needs_subject_selection",
                "scene_tags": list(hints.get("scene_tags", ["city_street"])),
                "facts": [
                    {
                        "fact_id": "f1",
                        "kind": "subject",
                        "statement": f"画面中检测到 {detected_people} 位人物，尚未指定目标主体。",
                        "source": "local_vision",
                        "strength": "high",
                        "region": None,
                    }
                ],
                "selected_subject": None,
                "anchors": [],
                "light": {
                    "appearance": str(hints.get("light_appearance", "光线可见，主光方向未确认")),
                    "direction_estimate": str(hints.get("light_direction", "unknown")),
                    "strength": "medium",
                    "fact_ids": [],
                },
                "issues": [],
                "unknowns": ["目标人物尚未选定", "画外通行空间未知"],
            }

        scene_tags = list(hints.get("scene_tags", ["coast", "leading_line"]))
        facts: list[dict[str, Any]] = []
        anchors: list[dict[str, Any]] = []
        issues: list[dict[str, Any]] = []
        unknowns: list[str] = ["人物距相机的绝对距离", "画外地形与移动空间"]

        # Scene fact
        scene_desc = str(hints.get("scene_description", "当前画面可见旅行环境结构。"))
        facts.append(
            {
                "fact_id": "f1",
                "kind": "scene",
                "statement": scene_desc,
                "source": "vlm_visual",
                "strength": "high",
                "region": hints.get("scene_region"),
            }
        )

        # Subject fact
        selected_subject: dict[str, Any] | None = None
        if detected_people > 0:
            subj_bbox = hints.get("subject_bbox", [0.38, 0.32, 0.62, 0.90])
            if "single_person" not in scene_tags and detected_people == 1:
                scene_tags.append("single_person")
            facts.append(
                {
                    "fact_id": "f2",
                    "kind": "subject",
                    "statement": str(hints.get("subject_statement", "选定人物入镜，边界可见。")),
                    "source": "local_vision",
                    "strength": "high",
                    "region": subj_bbox,
                }
            )
            selected_subject = {
                "subject_id": runtime_input.user_goal.subject_id or "person_1",
                "bbox": subj_bbox,
                "full_body_visible": hints.get("full_body_visible", True),
                "face_visibility": str(hints.get("face_visibility", "visible")),
                "pose_description": str(hints.get("pose_description", "自然站立，未推断被遮挡肢体细节")),
                "fact_ids": ["f2"],
            }
        else:
            facts.append(
                {
                    "fact_id": "f2",
                    "kind": "subject",
                    "statement": "当前画面未观察到人物入镜。",
                    "source": "vlm_visual",
                    "strength": "high",
                    "region": None,
                }
            )

        # Light fact
        light_dir = str(hints.get("light_direction", "unknown"))
        if light_dir == "unknown":
            unknowns.append("太阳或主光准确方位")
        facts.append(
            {
                "fact_id": "f3",
                "kind": "light",
                "statement": str(hints.get("light_statement", "脸部亮度可见，无足够证据确认主光方位。")),
                "source": "vlm_visual",
                "strength": "medium",
                "region": None,
            }
        )

        # Constraint fact
        mc = runtime_input.movement_constraints
        facts.append(
            {
                "fact_id": "f4",
                "kind": "constraint",
                "statement": (
                    f"拍摄者平移={mc.photographer_translation}，"
                    f"人物平移={mc.subject_translation}，"
                    f"原地站立确认={mc.standing_in_place_confirmed}"
                ),
                "source": "user_input",
                "strength": "high",
                "region": None,
            }
        )

        # Optional anchors from hints
        for idx, anc in enumerate(hints.get("anchors", [])):
            anchors.append(
                {
                    "anchor_id": anc.get("anchor_id", f"anchor_{idx + 1}"),
                    "kind": anc.get("kind", "leading_line"),
                    "description": anc.get("description", "可见环境结构"),
                    "bbox": anc.get("bbox"),
                    "points": anc.get("points", []),
                    "fact_ids": ["f1"],
                }
            )

        # Optional issues from hints
        for idx, iss in enumerate(hints.get("issues", [])):
            issues.append(
                {
                    "issue_id": iss.get("issue_id", f"issue_{idx + 1}"),
                    "description": iss.get("description", "存在画面干扰"),
                    "priority": iss.get("priority", "high"),
                    "fact_ids": ["f2"] if selected_subject else ["f1"],
                }
            )

        if not runtime_input.camera_capabilities.metadata_available:
            unknowns.extend(["相机可用镜头清单未知", "当前等效焦距未知"])

        return {
            "schema_version": "0.1.0",
            "frame_id": frame_id,
            "scene_version": scene_version,
            "coordinate_space": "upright_input_normalized_unmirrored",
            "status": "observed",
            "scene_tags": scene_tags,
            "facts": facts,
            "selected_subject": selected_subject,
            "anchors": anchors,
            "light": {
                "appearance": str(hints.get("light_appearance", "漫射光或柔光，脸部明暗可见")),
                "direction_estimate": light_dir,
                "strength": "medium",
                "fact_ids": ["f3"],
            },
            "issues": issues,
            "unknowns": list(dict.fromkeys(unknowns)),
        }

    def plan(
        self,
        runtime_input: RuntimeInput,
        observation: dict[str, Any],
        shot_cards: list[ShotCard],
        system_prompt: str,
        output_schema: dict[str, Any],
        feedback: str | None = None,
    ) -> dict[str, Any]:
        frame_id = runtime_input.frame.frame_id
        scene_version = runtime_input.frame.scene_version
        obs_status = observation.get("status")
        plan_id = f"plan_{frame_id}_v{scene_version}"

        base_invalidations = [
            "scene_version_changed",
            "lens_changed",
            "crop_changed",
            "orientation_changed",
            "selected_subject_changed",
            "light_materially_changed",
            "tracking_lost",
            "user_goal_changed",
        ]

        # Case 1: insufficient_input
        if obs_status == "insufficient_input":
            return {
                "schema_version": "0.1.0",
                "frame_id": frame_id,
                "scene_version": scene_version,
                "plan_id": plan_id,
                "status": "insufficient_input",
                "coordinate_space": "upright_input_normalized_unmirrored",
                "goal": runtime_input.user_goal.priority,
                "scene_summary": "缺少真实图像输入，无法生成拍摄方案。",
                "subject_placement_guide": None,
                "crop_recommendation": None,
                "primary_plan": None,
                "actions": [],
                "active_action_id": None,
                "alternative": None,
                "uncertainties": list(observation.get("unknowns", ["缺少真实图像"])),
                "invalidation_conditions": base_invalidations,
                "fallback_instruction": "请先提供一帧真实取景画面后再生成拍摄指导。",
            }

        # Case 2: needs_subject_selection (Rule III.15)
        if obs_status == "needs_subject_selection":
            return {
                "schema_version": "0.1.0",
                "frame_id": frame_id,
                "scene_version": scene_version,
                "plan_id": plan_id,
                "status": "hold",
                "coordinate_space": "upright_input_normalized_unmirrored",
                "goal": runtime_input.user_goal.priority,
                "scene_summary": "画面中存在多位人物，需先由用户指定主体。",
                "subject_placement_guide": None,
                "crop_recommendation": None,
                "primary_plan": None,
                "actions": [],
                "active_action_id": None,
                "alternative": None,
                "uncertainties": list(observation.get("unknowns", [])),
                "invalidation_conditions": base_invalidations,
                "fallback_instruction": "请在预览画面中点击选定要拍摄的主体人物，再继续生成构图指导。",
            }

        # Case 3: no subject in frame (E03)
        subject = observation.get("selected_subject")
        if subject is None:
            guide = _build_placement_guide(runtime_input, observation)
            return {
                "schema_version": "0.1.0",
                "frame_id": frame_id,
                "scene_version": scene_version,
                "plan_id": plan_id,
                "status": "no_subject",
                "coordinate_space": "upright_input_normalized_unmirrored",
                "goal": runtime_input.user_goal.priority,
                "scene_summary": "当前画面未观察到人物，虚框指示人物应站的位置。",
                "subject_placement_guide": guide,
                "crop_recommendation": _build_crop_recommendation(runtime_input, observation, guide),
                "primary_plan": None,
                "actions": [
                    {
                        "action_id": "a1",
                        "actor": "subject",
                        "operation": "capture_new_frame",
                        "direction_reference": "none",
                        "instruction": "请被拍者站到画面中的虚框位置，确认站位安全后入镜，再重新采帧评估构图与光线。",
                        "prerequisites": ["现场站位安全已由用户确认"],
                        "evidence_fact_ids": [f["fact_id"] for f in observation.get("facts", [])[:1]],
                        "completion_check": {
                            "type": "fresh_frame_received",
                            "target_range": None,
                            "reference_id": None,
                            "check_by": "vlm_review",
                            "description": "收到包含人物的新帧后再开始构图规划。",
                        },
                    }
                ],
                "active_action_id": "a1",
                "alternative": None,
                "uncertainties": list(observation.get("unknowns", [])) + ["虚框为构图建议，站立位置未经实地验证"],
                "invalidation_conditions": base_invalidations,
                "fallback_instruction": "人物进入虚框位置后请重新点击分析。",
            }

        # Case 4: Normal observed planning
        fact_ids = [f["fact_id"] for f in observation.get("facts", [])]
        anchor_ids = [a["anchor_id"] for a in observation.get("anchors", [])]
        used_card_ids = [shot_cards[0].id] if shot_cards else []
        top_card = shot_cards[0] if shot_cards else None

        # Choose safe lens strictly from device capabilities
        caps = runtime_input.camera_capabilities
        cur_lens = caps.get_lens(caps.current_lens_id)
        if caps.metadata_available and cur_lens is not None:
            lens_plan = {
                "action": "keep",
                "target_lens_id": cur_lens.lens_id,
                "ui_label": cur_lens.ui_label,
                "equivalent_focal_length_mm": cur_lens.equivalent_focal_length_mm,
                "note": "保持当前设备可用镜头，不虚构未提供的光学焦段。",
            }
        else:
            lens_plan = {
                "action": "keep",
                "target_lens_id": None,
                "ui_label": None,
                "equivalent_focal_length_mm": None,
                "note": "设备镜头元数据缺失，保持当前镜头，仅依据画面占比进行可观察调整。",
            }

        # Detect priority issues (e.g., head-obstacle merge)
        has_head_merge = any(
            any(k in str(i.get("description", "")) for k in ("电线杆", "路牌", "树干", "头部重合", "头部与"))
            for i in observation.get("issues", [])
        )
        is_symmetry = "symmetry" in observation.get("scene_tags", [])
        wants_closeup = any(k in runtime_input.user_goal.priority for k in ("近景", "特写", "表情"))

        actions: list[dict[str, Any]] = []
        if has_head_merge:
            mc = runtime_input.movement_constraints
            if mc.photographer_translation == "allowed":
                actions.append(
                    {
                        "action_id": "a1",
                        "actor": "photographer",
                        "operation": "physical_translate",
                        "direction_reference": "photographer_body",
                        "instruction": "保持当前镜头，拍摄者向身侧小幅试探平移机位，观察头部与背景杆是否出现间隙，改善即停；若无改善则反向试探或停止。",
                        "prerequisites": ["已确认身侧具备可安全平移空间"],
                        "evidence_fact_ids": fact_ids[:2],
                        "completion_check": {
                            "type": "head_obstacle_separated",
                            "target_range": None,
                            "reference_id": subject["subject_id"],
                            "check_by": "vlm_review",
                            "description": "必须在新帧中复核人物头部轮廓与背景杆状物已分离。",
                        },
                    }
                )
            elif mc.subject_translation == "allowed" and runtime_input.user_goal.allow_subject_walking:
                actions.append(
                    {
                        "action_id": "a1",
                        "actor": "subject",
                        "operation": "physical_translate",
                        "direction_reference": "subject_body",
                        "instruction": "请被拍者在已确认安全的范围内向自身一侧小幅横移一步，观察头部与背景杆分离后停止。",
                        "prerequisites": ["已确认人物侧移空间安全"],
                        "evidence_fact_ids": fact_ids[:2],
                        "completion_check": {
                            "type": "head_obstacle_separated",
                            "target_range": None,
                            "reference_id": subject["subject_id"],
                            "check_by": "vlm_review",
                            "description": "新帧复核头部与背景杆分离且未引入新遮挡。",
                        },
                    }
                )
            else:
                actions.append(
                    {
                        "action_id": "a1",
                        "actor": "photographer",
                        "operation": "confirm_movement_space",
                        "direction_reference": "none",
                        "instruction": "纯转动手机无法消除同视线头部与背景杆重叠；请先确认拍摄者或人物能否向一侧小幅平移。",
                        "prerequisites": [],
                        "evidence_fact_ids": fact_ids[:2],
                        "completion_check": {
                            "type": "manual_confirmed",
                            "target_range": None,
                            "reference_id": None,
                            "check_by": "user",
                            "description": "用户确认现场可安全平移后再激活平移步骤。",
                        },
                    }
                )
            target_bbox = subject.get("bbox")
            target_ratio = None
            shot_size = "keep"
            title = "优先消除头部与背景杆重合"
            reason = "先通过真实视差分离头部与背景杆，不以原地转动替代平移。"
        elif is_symmetry:
            target_bbox = [0.38, 0.30, 0.62, 0.88]
            target_ratio = [0.45, 0.68]
            shot_size = "full_body"
            title = "保留建筑对称中轴构图"
            reason = "画面存在明确对称建筑结构，采用居中构图以保留端正秩序感，不机械套用三分法。"
            actions.append(
                {
                    "action_id": "a1",
                    "actor": "photographer",
                    "operation": "level_camera",
                    "direction_reference": "input_image",
                    "instruction": "保持人物在对称中轴附近，微调手机水平使建筑线条端正后拍摄。",
                    "prerequisites": ["人物已位于对称轴附近"],
                    "evidence_fact_ids": fact_ids[:2],
                    "completion_check": {
                        "type": "subject_center_x_in_range",
                        "target_range": [0.44, 0.56],
                        "reference_id": subject["subject_id"],
                        "check_by": "local_vision",
                        "description": "人物中心保持在对称中轴区间内。",
                    },
                }
            )
        elif wants_closeup:
            target_bbox = [0.30, 0.16, 0.70, 0.90]
            target_ratio = [0.60, 0.82]
            shot_size = "half_body"
            title = "突出人物表情的近景构图"
            reason = "响应用户近景人物优先目标，放大人物主体占比并减少无关背景干扰。"
            actions.append(
                {
                    "action_id": "a1",
                    "actor": "photographer",
                    "operation": "reframe_pan",
                    "direction_reference": "input_image",
                    "instruction": "保持当前可用镜头，微调取景让人物稳定在画面中央偏上并确认脸部清晰。",
                    "prerequisites": ["当前镜头可用"],
                    "evidence_fact_ids": fact_ids[:2],
                    "completion_check": {
                        "type": "subject_height_ratio_in_range",
                        "target_range": [0.60, 0.82],
                        "reference_id": subject["subject_id"],
                        "check_by": "local_vision",
                        "description": "人物高度占比达到近景目标区间后复核表情与光线。",
                    },
                }
            )
        else:
            target_bbox = [0.53, 0.32, 0.77, 0.90]
            target_ratio = [0.50, 0.65]
            shot_size = "environmental_full_body"
            title = top_card.title if top_card else "保守原地环境人像构图"
            reason = "依据当前画面可见结构保留环境与人物平衡，优先原地可验证动作。"
            actions.append(
                {
                    "action_id": "a1",
                    "actor": "photographer",
                    "operation": "reframe_pan",
                    "direction_reference": "input_image",
                    "instruction": "保持原地和当前镜头，向画面左侧缓慢转动手机，让人物中心进入右侧目标区。",
                    "prerequisites": ["人物已选定且跟踪有效"],
                    "evidence_fact_ids": fact_ids[:2],
                    "completion_check": {
                        "type": "subject_center_x_in_range",
                        "target_range": [0.60, 0.70],
                        "reference_id": subject["subject_id"],
                        "check_by": "local_vision",
                        "description": "人物中心进入目标区后停止，新画面复核环境与边缘完整性。",
                    },
                }
            )
            actions.append(
                {
                    "action_id": "a2",
                    "actor": "subject",
                    "operation": "rotate_body",
                    "direction_reference": "none",
                    "instruction": "双脚留在原地，身体轻轻侧转到舒服的角度。",
                    "prerequisites": ["a1已完成并经新帧复核"],
                    "evidence_fact_ids": fact_ids[:2],
                    "completion_check": {
                        "type": "pose_visually_confirmed",
                        "target_range": None,
                        "reference_id": subject["subject_id"],
                        "check_by": "vlm_review",
                        "description": "新画面复核自然姿态与脸部可辨。",
                    },
                }
            )

        return {
            "schema_version": "0.1.0",
            "frame_id": frame_id,
            "scene_version": scene_version,
            "plan_id": plan_id,
            "status": "plan_ready",
            "coordinate_space": "upright_input_normalized_unmirrored",
            "goal": runtime_input.user_goal.priority,
            "scene_summary": observation["facts"][0]["statement"] if observation.get("facts") else "已完成当前帧观察。",
            "subject_placement_guide": None,
            "crop_recommendation": None,
            "primary_plan": {
                "title": title,
                "reason": reason,
                "used_fact_ids": fact_ids,
                "used_card_ids": used_card_ids,
                "composition": {
                    "orientation": runtime_input.user_goal.orientation_preference,
                    "shot_size": shot_size,
                    "target_subject_bbox": target_bbox,
                    "target_subject_height_ratio": target_ratio,
                    "preserve_anchor_ids": anchor_ids,
                },
                "lens": lens_plan,
                "pose_intent": "自然舒适站姿，避免复杂或未经确认的移动动作。",
            },
            "actions": actions,
            "active_action_id": actions[0]["action_id"] if actions else None,
            "alternative": {
                "title": "原地半身备选方案",
                "use_when": "若环境杂物增多或无法调整机位，可转为人物优先构图并重新采帧。",
            },
            "uncertainties": list(observation.get("unknowns", [])),
            "invalidation_conditions": base_invalidations,
            "fallback_instruction": "若人物跟踪丢失或镜头切换，请暂停引导并重新获取画面。",
        }
