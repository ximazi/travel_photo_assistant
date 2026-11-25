"""Run the dual-loop pipeline over a real photo file and render the result.

This is the offline counterpart of the phone app: it takes an actual image
file, runs the same slow loop the HTTP API runs, and writes the observation,
the plan and an annotated result image into a local output directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .adapters.base import VisionModelAdapter
from .config import AssetPaths, RuntimeConfig
from .controller import TravelPhotoController
from .knowledge import RetrievalMode
from .models import LocalVisionSnapshot, RuntimeInput
from .render import RenderResult, render_overlay


DEFAULT_PRIORITY = "人物自然、脸部可辨，同时保留旅行环境"

# Sentinel so an explicit caller argument can win over the manifest default.
_UNSET = object()

# Manifest fields that are forwarded to the model as `local_vision_hints`.
_HINT_FIELDS = {
    "scene_tags",
    "scene_description",
    "detected_person_count",
    "subject_bbox",
    "anchors",
    "issues",
    "light_appearance",
    "light_direction",
    "light_statement",
    "face_visibility",
    "pose_description",
    "full_body_visible",
}


def load_photo_manifest(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Load `assets/examples/photos/photos.json`.

    The manifest maps a photo filename to the `local_vision_hints` the offline
    rule adapter needs, plus optional per-photo overrides (goal priority,
    subject id). Real VLM adapters may ignore the hints and read the image.
    """
    manifest_path = Path(path) if path else AssetPaths.default().photos_dir / "photos.json"
    if not manifest_path.is_file():
        return {}
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    photos = raw.get("photos") or {}
    return {str(k): dict(v) for k, v in photos.items() if isinstance(v, dict)}


# filename -> hints. Loaded once at import time; use load_photo_manifest() to reload.
PHOTO_PRESETS: dict[str, dict[str, Any]] = load_photo_manifest()


def build_runtime_input(
    photo_path: str | Path,
    *,
    frame_id: str | None = None,
    scene_version: int = 1,
    priority: Any = _UNSET,
    subject_id: Any = _UNSET,
    goal_tags: list[str] | None = None,
    hints_override: dict[str, Any] | None = None,
    preview_mirrored: bool = False,
    metadata_available: bool = True,
) -> RuntimeInput:
    """Assemble a runtime input for a photo on disk.

    The image is attached via `local_path`, which both the Gemini and the
    OpenAI-compatible adapter can read and base64-encode.

    Precedence: an explicit argument wins, then the per-photo value in
    `photos.json`, then the built-in default.
    """
    path = Path(photo_path)
    stem = path.stem

    manifest_entry = PHOTO_PRESETS.get(path.name) or PHOTO_PRESETS.get(stem) or {}
    hints: dict[str, Any] = {k: v for k, v in manifest_entry.items() if k in _HINT_FIELDS}
    if hints_override:
        hints.update(hints_override)

    # Per-photo overrides from the manifest, unless the caller was explicit.
    if subject_id is _UNSET:
        subject_id = manifest_entry.get("subject_id", "person_1")
    if priority is _UNSET:
        priority = str(manifest_entry.get("priority") or DEFAULT_PRIORITY)
    if goal_tags is None and manifest_entry.get("goal_tags"):
        goal_tags = list(manifest_entry["goal_tags"])

    # PIL is only needed for the size probe; fall back to the schema default.
    try:
        from PIL import Image

        with Image.open(path) as im:
            img_w, img_h = im.size
    except Exception:  # pragma: no cover - size probe is best-effort
        img_w, img_h = 1080, 1440

    lenses: list[dict[str, Any]] = []
    current_lens_id = None
    if metadata_available:
        lenses = [
            {
                "lens_id": "main_1x",
                "ui_label": "1x",
                "equivalent_focal_length_mm": 26,
                "equivalent_focal_length_source": "camera_api",
                "mode": "optical_primary",
            }
        ]
        current_lens_id = "main_1x"

    return RuntimeInput.from_dict(
        {
            "schema_version": "0.1.0",
            "example_only": False,
            "frame": {
                "frame_id": frame_id or f"photo_{stem}",
                "scene_version": scene_version,
                "image_attachments": [{"mime_type": "image/jpeg", "local_path": str(path.resolve())}],
                "upright_input_width_px": img_w,
                "upright_input_height_px": img_h,
                "coordinate_space": "upright_input_normalized_unmirrored",
                "input_rotation_already_applied": True,
                "preview_mirrored": preview_mirrored,
                "viewport_fit": "contain",
                "preview_aspect_ratio": "3:4",
            },
            "user_goal": {
                "priority": priority,
                "style": "自然旅行人像",
                "subject_id": subject_id,
                "subject_count": 1,
                "orientation_preference": "portrait",
                "allow_subject_walking": False,
                "goal_tags": goal_tags or [],
                "goal_version": 1,
            },
            "camera_capabilities": {
                "current_lens_id": current_lens_id,
                "available_lenses": lenses,
                "manual_aperture": False,
                "exposure_compensation_available": True,
                "metadata_available": metadata_available,
            },
            "movement_constraints": {
                "photographer_translation": "unknown",
                "subject_translation": "not_requested",
                "standing_in_place_confirmed": True,
            },
            "local_vision_hints": hints,
        }
    )


@dataclass
class PhotoRunResult:
    """Everything produced by one photo run."""

    photo_path: str
    output_dir: str
    observation: dict[str, Any] | None
    plan: dict[str, Any] | None
    render: RenderResult | None
    selected_card_ids: list[str] = field(default_factory=list)
    validation_errors: list[dict[str, Any]] = field(default_factory=list)
    accepted: bool = False
    reason: str = ""
    fast_status: str | None = None
    guidance_direction: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "photo_path": self.photo_path,
            "output_dir": self.output_dir,
            "accepted": self.accepted,
            "reason": self.reason,
            "observation_status": (self.observation or {}).get("status"),
            "plan_status": (self.plan or {}).get("status"),
            "plan_title": ((self.plan or {}).get("primary_plan") or {}).get("title"),
            "selected_card_ids": self.selected_card_ids,
            "validation_errors": self.validation_errors,
            "fast_status": self.fast_status,
            "guidance_direction": self.guidance_direction,
            "render": {
                "output_path": self.render.output_path,
                "image_size": self.render.image_size,
                "target_bbox_px": self.render.target_bbox_px,
                "subject_bbox_px": self.render.subject_bbox_px,
                "ghost_bbox_px": self.render.ghost_bbox_px,
                "ghost_basis": self.render.ghost_basis,
                "ghost_framing": self.render.ghost_framing,
                "ghost_instruction": self.render.ghost_instruction,
                "crop_rect_px": self.render.crop_rect_px,
                "crop_aspect_ratio": self.render.crop_aspect_ratio,
                "crop_rationale": self.render.crop_rationale,
                "anchor_boxes_px": self.render.anchor_boxes_px,
                "guidance_arrow": self.render.guidance_arrow,
                "title": self.render.title,
                "instruction": self.render.instruction,
            }
            if self.render
            else None,
            "subject_placement_guide": (self.plan or {}).get("subject_placement_guide"),
            "crop_recommendation": (self.plan or {}).get("crop_recommendation"),
        }


def run_photo(
    photo_path: str | Path,
    *,
    assets: AssetPaths | None = None,
    config: RuntimeConfig | None = None,
    adapter: VisionModelAdapter | None = None,
    output_dir: str | Path | None = None,
    retrieval_mode: RetrievalMode | str = RetrievalMode.RETRIEVED_CARDS,
    goal: str | None = None,
    scene_tags: list[str] | None = None,
    subject_bbox: list[float] | None = None,
    preview_mirrored: bool = False,
    metadata_available: bool = True,
    render: bool = True,
) -> PhotoRunResult:
    """Run slow loop + fast tick over one photo and write the artifacts to disk."""
    from .adapters.mock import RuleBasedOfflineVisionAdapter

    assets = assets or AssetPaths.default()
    config = config or RuntimeConfig()
    adapter = adapter or RuleBasedOfflineVisionAdapter()

    photo = Path(photo_path)
    if not photo.is_file():
        raise FileNotFoundError(f"Photo not found: {photo}")

    out_dir = Path(output_dir) if output_dir else (assets.project_root / "output")
    run_dir = out_dir / photo.stem
    run_dir.mkdir(parents=True, exist_ok=True)

    hints_override: dict[str, Any] = {}
    if scene_tags:
        hints_override["scene_tags"] = scene_tags
    if subject_bbox:
        hints_override["subject_bbox"] = subject_bbox

    runtime_input = build_runtime_input(
        photo,
        priority=goal if goal else _UNSET,
        hints_override=hints_override or None,
        preview_mirrored=preview_mirrored,
        metadata_available=metadata_available,
    )

    controller = TravelPhotoController(adapter=adapter, assets=assets, config=config)
    slow = controller.run_slow_loop(runtime_input, retrieval_mode=retrieval_mode)

    validation_errors = [
        {"stage": stage, "code": issue.code, "message": issue.message, "field": issue.field_path}
        for stage, res in (
            ("input", slow.input_validation),
            ("observation", slow.observation_validation),
            ("plan", slow.plan_validation),
        )
        if res is not None
        for issue in res.errors
    ]

    # Write the raw JSON artifacts next to the rendered image.
    (run_dir / "observation.json").write_text(
        json.dumps(slow.observation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "plan.json").write_text(
        json.dumps(slow.plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    render_result: RenderResult | None = None
    fast_status: str | None = None
    guidance_direction: str | None = None
    subject_bbox_now = None

    if slow.plan:
        subject = (slow.observation or {}).get("selected_subject") or {}
        if isinstance(subject, dict):
            subject_bbox_now = subject.get("bbox")

        # Run one fast-loop tick so the rendered image also carries the live
        # guidance direction (which way the user should move right now).
        if subject_bbox_now and slow.plan.get("status") == "plan_ready":
            snapshot = LocalVisionSnapshot(
                frame_id=runtime_input.frame.frame_id,
                scene_version=runtime_input.frame.scene_version,
                transform_version=runtime_input.frame.transform_version,
                current_lens_id=runtime_input.camera_capabilities.current_lens_id,
                orientation=runtime_input.user_goal.orientation_preference,
                crop_signature=f"{runtime_input.frame.viewport_fit}:{runtime_input.frame.preview_aspect_ratio}",
                tracked_subject_id=(subject.get("subject_id") if isinstance(subject, dict) else None),
                subject_bbox=subject_bbox_now,
                tracking_ok=True,
                user_goal_version=runtime_input.user_goal.goal_version,
            )
            tick = controller.run_fast_tick(snapshot)
            fast_status = tick.status
            guidance_direction = tick.guidance_direction

    if render and slow.plan:
        render_result = render_overlay(
            photo,
            slow.plan,
            run_dir / "result.jpg",
            observation=slow.observation,
            subject_bbox=subject_bbox_now,
            guidance_direction=guidance_direction,
            mirrored=preview_mirrored,
        )

    return PhotoRunResult(
        photo_path=str(photo),
        output_dir=str(run_dir),
        observation=slow.observation,
        plan=slow.plan,
        render=render_result,
        selected_card_ids=[c.id for c in slow.retrieval.selected_cards] if slow.retrieval else [],
        validation_errors=validation_errors,
        accepted=slow.accepted,
        reason=slow.reason,
        fast_status=fast_status,
        guidance_direction=guidance_direction,
    )
