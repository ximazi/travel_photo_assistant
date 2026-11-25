"""Typed domain models matching v0.1 JSON contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


BBox = list[float]  # [xmin, ymin, xmax, ymax] normalized to [0, 1]
Point2D = list[float]  # [x, y] normalized to [0, 1]


@dataclass
class ImageAttachment:
    """Reference to a real camera frame image (base64, URI, or local path)."""

    mime_type: str
    data_base64: str | None = None
    uri: str | None = None
    local_path: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImageAttachment":
        return cls(
            mime_type=data.get("mime_type", "image/jpeg"),
            data_base64=data.get("data_base64"),
            uri=data.get("uri"),
            local_path=data.get("local_path"),
        )


@dataclass
class LensCapability:
    """Single lens or zoom entry reported by device camera metadata."""

    lens_id: str
    ui_label: str
    equivalent_focal_length_mm: float | None = None
    equivalent_focal_length_source: str = "unknown"
    mode: str = "optical_primary"
    digital_zoom_allowed: bool = False
    max_zoom_factor: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LensCapability":
        return cls(
            lens_id=data["lens_id"],
            ui_label=data["ui_label"],
            equivalent_focal_length_mm=data.get("equivalent_focal_length_mm"),
            equivalent_focal_length_source=data.get("equivalent_focal_length_source", "unknown"),
            mode=data.get("mode", "optical_primary"),
            digital_zoom_allowed=bool(data.get("digital_zoom_allowed", False)),
            max_zoom_factor=data.get("max_zoom_factor"),
        )


@dataclass
class CameraCapabilities:
    """Device camera capabilities supplied explicitly by runtime."""

    current_lens_id: str | None = None
    available_lenses: list[LensCapability] = field(default_factory=list)
    manual_aperture: bool = False
    exposure_compensation_available: bool = False
    digital_zoom_available: bool = False
    metadata_available: bool = True

    def get_lens(self, lens_id: str | None) -> LensCapability | None:
        if not lens_id:
            return None
        for lens in self.available_lenses:
            if lens.lens_id == lens_id:
                return lens
        return None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CameraCapabilities":
        if not data:
            return cls(metadata_available=False)
        lenses = [LensCapability.from_dict(x) for x in data.get("available_lenses", [])]
        return cls(
            current_lens_id=data.get("current_lens_id"),
            available_lenses=lenses,
            manual_aperture=bool(data.get("manual_aperture", False)),
            exposure_compensation_available=bool(data.get("exposure_compensation_available", False)),
            digital_zoom_available=bool(data.get("digital_zoom_available", False)),
            metadata_available=bool(data.get("metadata_available", len(lenses) > 0)),
        )


@dataclass
class FrameMetadata:
    """Frame and preview geometry metadata."""

    frame_id: str
    scene_version: int
    image_attachments: list[dict[str, Any]] = field(default_factory=list)
    upright_input_width_px: int = 1080
    upright_input_height_px: int = 1440
    coordinate_space: str = "upright_input_normalized_unmirrored"
    input_rotation_already_applied: bool = True
    preview_mirrored: bool = False
    viewport_fit: str = "contain"  # contain | cover
    preview_aspect_ratio: str = "3:4"
    transform_version: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrameMetadata":
        return cls(
            frame_id=data["frame_id"],
            scene_version=int(data["scene_version"]),
            image_attachments=list(data.get("image_attachments", [])),
            upright_input_width_px=int(data.get("upright_input_width_px", 1080)),
            upright_input_height_px=int(data.get("upright_input_height_px", 1440)),
            coordinate_space=data.get("coordinate_space", "upright_input_normalized_unmirrored"),
            input_rotation_already_applied=bool(data.get("input_rotation_already_applied", True)),
            preview_mirrored=bool(data.get("preview_mirrored", False)),
            viewport_fit=data.get("viewport_fit", "contain"),
            preview_aspect_ratio=data.get("preview_aspect_ratio", "3:4"),
            transform_version=int(data.get("transform_version", 1)),
        )


@dataclass
class UserGoal:
    """User shooting goal and subject preferences."""

    priority: str
    style: str = "自然旅行人像"
    subject_id: str | None = None
    subject_count: int | None = None
    orientation_preference: str = "portrait"
    allow_subject_walking: bool = False
    allow_precomposition: bool = False
    goal_tags: list[str] = field(default_factory=list)
    goal_version: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserGoal":
        return cls(
            priority=data.get("priority", "人物自然、脸部可辨，同时保留旅行环境"),
            style=data.get("style", "自然旅行人像"),
            subject_id=data.get("subject_id"),
            subject_count=data.get("subject_count"),
            orientation_preference=data.get("orientation_preference", "portrait"),
            allow_subject_walking=data.get("allow_subject_walking") is True,
            allow_precomposition=bool(data.get("allow_precomposition", False)),
            goal_tags=list(data.get("goal_tags", [])),
            goal_version=int(data.get("goal_version", 1)),
        )


@dataclass
class MovementConstraints:
    """Explicit physical movement constraints for photographer and subject."""

    photographer_translation: str = "unknown"  # allowed | unknown | forbidden | tripod_fixed
    subject_translation: str = "not_requested"  # allowed | unknown | not_requested | forbidden
    standing_in_place_confirmed: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MovementConstraints":
        if not data:
            return cls()
        return cls(
            photographer_translation=data.get("photographer_translation", "unknown"),
            subject_translation=data.get("subject_translation", "not_requested"),
            standing_in_place_confirmed=bool(data.get("standing_in_place_confirmed", True)),
        )


@dataclass
class SessionContext:
    """Active session state passed into slow-loop planning."""

    previous_plan: dict[str, Any] | None = None
    active_action: dict[str, Any] | None = None
    user_requested_new_plan: bool = False
    confirmed_prerequisites: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SessionContext":
        if not data:
            return cls()
        return cls(
            previous_plan=data.get("previous_plan"),
            active_action=data.get("active_action"),
            user_requested_new_plan=bool(data.get("user_requested_new_plan", False)),
            confirmed_prerequisites=list(data.get("confirmed_prerequisites", [])),
        )


@dataclass
class RuntimeInput:
    """Complete runtime input payload for one slow-loop cycle."""

    schema_version: str
    frame: FrameMetadata
    user_goal: UserGoal
    camera_capabilities: CameraCapabilities
    movement_constraints: MovementConstraints
    session: SessionContext = field(default_factory=SessionContext)
    example_only: bool = False
    notice: str | None = None
    local_vision_hints: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeInput":
        return cls(
            schema_version=data.get("schema_version", "0.1.0"),
            frame=FrameMetadata.from_dict(data["frame"]),
            user_goal=UserGoal.from_dict(data.get("user_goal", {})),
            camera_capabilities=CameraCapabilities.from_dict(data.get("camera_capabilities")),
            movement_constraints=MovementConstraints.from_dict(data.get("movement_constraints")),
            session=SessionContext.from_dict(data.get("session")),
            example_only=bool(data.get("example_only", False)),
            notice=data.get("notice"),
            local_vision_hints=dict(data.get("local_vision_hints", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LocalVisionSnapshot:
    """On-device fast-loop tracker telemetry for a single preview frame."""

    frame_id: str
    scene_version: int
    transform_version: int = 1
    current_lens_id: str | None = None
    orientation: str = "portrait"
    crop_signature: str = "default"
    tracked_subject_id: str | None = None
    subject_bbox: BBox | None = None  # [xmin, ymin, xmax, ymax] upright unmirrored
    tracking_ok: bool = True
    camera_roll_deg: float | None = None
    light_materially_changed: bool = False
    user_goal_version: int = 1
    manual_confirmations: list[str] = field(default_factory=list)
    vlm_verified_checks: list[str] = field(default_factory=list)  # Legacy, ignored for completion.
    frame_seq: int | None = None
    visual_review: dict[str, Any] | None = None


@dataclass
class ShotCard:
    """Executable photography knowledge card loaded from knowledge/shot_cards.json."""

    id: str
    title: str
    version: str
    status: str
    source: str
    scene_tags: list[str]
    goal_tags: list[str]
    required_visual_tags: list[str]
    visual_requirements: list[str]
    not_applicable_when: list[str]
    movement_requirements: str
    device_requirements: list[str]
    composition_intent: str
    subject_height_ratio_hint: list[float] | None
    lens_intent: str
    focal_length_35mm_eq_hint: list[float] | None
    pose_options: list[str]
    action_templates: list[str]
    verification_checks: list[str]
    positive_case: str
    negative_case: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ShotCard":
        return cls(
            id=data["id"],
            title=data["title"],
            version=data.get("version", "0.1.0"),
            status=data.get("status", "authored_unvalidated"),
            source=data.get("source", ""),
            scene_tags=list(data.get("scene_tags", [])),
            goal_tags=list(data.get("goal_tags", [])),
            required_visual_tags=list(data.get("required_visual_tags", [])),
            visual_requirements=list(data.get("visual_requirements", [])),
            not_applicable_when=list(data.get("not_applicable_when", [])),
            movement_requirements=data.get("movement_requirements", ""),
            device_requirements=list(data.get("device_requirements", [])),
            composition_intent=data.get("composition_intent", ""),
            subject_height_ratio_hint=data.get("subject_height_ratio_hint"),
            lens_intent=data.get("lens_intent", ""),
            focal_length_35mm_eq_hint=data.get("focal_length_35mm_eq_hint"),
            pose_options=list(data.get("pose_options", [])),
            action_templates=list(data.get("action_templates", [])),
            verification_checks=list(data.get("verification_checks", [])),
            positive_case=data.get("positive_case", ""),
            negative_case=data.get("negative_case", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
