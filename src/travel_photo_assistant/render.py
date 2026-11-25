"""Render a validated shot plan back onto the source photo.

The service never returns images — the phone app is expected to draw the
overlay itself from the normalized coordinates. This module reproduces that
rendering offline so a plan can actually be *looked at* and reviewed.

Everything here draws in the canonical upright-unmirrored input space
(normalized [0, 1] coordinates scaled by the real image size), so the overlay
lines up with the source photo regardless of viewport fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


# Windows CJK-capable fonts, checked in order. Falls back to the bitmap default.
_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/PingFang.ttc",
)

COLOR_TARGET = (235, 48, 48)      # target subject box
COLOR_ANCHOR = (48, 128, 235)     # preserved anchors
COLOR_SUBJECT = (34, 190, 110)    # currently tracked subject
COLOR_ARROW = (255, 176, 32)      # guidance arrow
COLOR_GHOST = (0, 214, 214)       # person ghost figure (no subject in frame)
COLOR_CROP = (255, 214, 0)        # suggested crop region
COLOR_TEXT = (255, 255, 255)
COLOR_PANEL = (18, 18, 22)


def _load_font(size: int) -> ImageFont.ImageFont:
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).is_file():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has no size argument
        return ImageFont.load_default()


@dataclass
class RenderResult:
    """Metadata about what was drawn, for the JSON sidecar report."""

    output_path: str
    image_size: list[int]
    target_bbox_px: list[float] | None = None
    subject_bbox_px: list[float] | None = None
    ghost_bbox_px: list[float] | None = None
    ghost_basis: str | None = None
    ghost_framing: str | None = None
    ghost_instruction: str = ""
    crop_rect_px: list[float] | None = None
    crop_aspect_ratio: str | None = None
    crop_rationale: str = ""
    anchor_boxes_px: list[list[float]] = field(default_factory=list)
    guidance_arrow: str | None = None
    title: str = ""
    instruction: str = ""
    warnings: list[str] = field(default_factory=list)


def _bbox_to_px(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = bbox
    return (xmin * width, ymin * height, xmax * width, ymax * height)


def _draw_ghost_box(
    draw: ImageDraw.ImageDraw,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    color: tuple[int, int, int],
    width: int,
) -> None:
    """Draw corner brackets — a viewfinder-style placeholder, not a solid target."""
    arm_x = (xmax - xmin) * 0.28
    arm_y = (ymax - ymin) * 0.22
    corners = (
        (xmin, ymin, 1, 1),
        (xmax, ymin, -1, 1),
        (xmin, ymax, 1, -1),
        (xmax, ymax, -1, -1),
    )
    for cx, cy, sx, sy in corners:
        draw.line([(cx, cy), (cx + sx * arm_x, cy)], fill=color, width=width)
        draw.line([(cx, cy), (cx, cy + sy * arm_y)], fill=color, width=width)


def _draw_person_silhouette(
    image: Image.Image,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    color: tuple[int, int, int],
    framing: str = "full_body",
) -> None:
    """Draw a human-shaped ghost figure filling the given extent.

    `framing` decides how much of the body is represented: full_body shows head
    to feet, three_quarter stops at the knee, half_body at the waist and
    headshot shows only head and shoulders.
    """
    w = xmax - xmin
    h = ymax - ymin
    if w <= 0 or h <= 0:
        return

    cx = (xmin + xmax) / 2.0
    line = max(2, int(h * 0.010))

    # Proportions of a standing figure. How many head-heights fit into the
    # drawn extent depends on the framing: a full body is ~7.5 heads tall, a
    # half body ~3.75, head and shoulders ~2.2.
    heads_in_extent = {"full_body": 7.5, "three_quarter": 5.4, "half_body": 3.75, "headshot": 2.2}
    head_h = h / heads_in_extent.get(framing, 7.5)
    # A head is a little taller than it is wide.
    head_w = head_h * 0.76
    chin_y = ymin + head_h * 1.04
    shoulder_y = chin_y + head_h * 0.14
    shoulder_w = w * 0.94
    waist_y = ymin + h * 0.44
    waist_w = w * 0.66

    if framing == "half_body":
        waist_y = ymax
        waist_w = w * 0.80
    elif framing == "headshot":
        shoulder_y = chin_y + head_h * 0.18
        shoulder_w = w * 0.92
        waist_y = ymax
        waist_w = w * 0.88

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    fill = (*color, 135)
    edge = (*color, 250)

    # Head
    od.ellipse(
        [cx - head_w / 2.0, ymin, cx + head_w / 2.0, chin_y],
        fill=fill, outline=edge, width=line,
    )
    # Neck, bridging head and shoulders so the figure reads as one body.
    od.rectangle(
        [cx - head_w * 0.32, chin_y - head_h * 0.26, cx + head_w * 0.32, shoulder_y + head_h * 0.10],
        fill=fill,
    )
    # Torso: shoulders taper down to the waist.
    od.polygon(
        [
            (cx - shoulder_w / 2.0, shoulder_y),
            (cx + shoulder_w / 2.0, shoulder_y),
            (cx + waist_w / 2.0, waist_y),
            (cx - waist_w / 2.0, waist_y),
        ],
        fill=fill, outline=edge,
    )
    # Arms hanging just outside the torso line.
    arm_w = shoulder_w * 0.17
    for sign in (-1, 1):
        od.polygon(
            [
                (cx + sign * shoulder_w / 2.0, shoulder_y + head_h * 0.04),
                (cx + sign * (shoulder_w / 2.0 - arm_w), shoulder_y + head_h * 0.04),
                (cx + sign * (waist_w / 2.0 + arm_w * 0.35), waist_y),
                (cx + sign * (waist_w / 2.0 - arm_w * 0.55), waist_y),
            ],
            fill=fill, outline=edge,
        )
    # Legs, only when the framing actually shows them.
    if framing in {"full_body", "three_quarter"}:
        leg_end = ymax if framing == "full_body" else ymin + h * 0.80
        leg_top_w = waist_w * 0.46
        gap = waist_w * 0.10
        for sign in (-1, 1):
            inner = sign * gap / 2.0
            outer = sign * (gap / 2.0 + leg_top_w)
            od.polygon(
                [
                    (cx + inner, waist_y - head_h * 0.05),
                    (cx + outer, waist_y - head_h * 0.05),
                    (cx + outer * 0.78, leg_end),
                    (cx + inner * 1.5, leg_end),
                ],
                fill=fill, outline=edge,
            )
        # A soft shadow hints that the feet must land on a surface.
        if framing == "full_body":
            od.ellipse(
                [cx - waist_w * 0.52, leg_end - head_h * 0.16, cx + waist_w * 0.52, leg_end + head_h * 0.16],
                fill=(*color, 70),
            )

    if image.mode == "RGBA":
        image.alpha_composite(overlay)
    else:
        image.paste(overlay.convert("RGB"), (0, 0), overlay)


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    color: tuple[int, int, int],
    width: int,
) -> None:
    """Draw a horizontal guidance arrow with a filled arrowhead."""
    draw.line([start, end], fill=color, width=width)
    head = max(14, width * 4)
    dx = 1 if end[0] >= start[0] else -1
    draw.polygon(
        [
            (end[0], end[1]),
            (end[0] - dx * head, end[1] - head * 0.6),
            (end[0] - dx * head, end[1] + head * 0.6),
        ],
        fill=color,
    )


def _wrap(text: str, width_chars: int) -> list[str]:
    """Wrap text by character count (CJK-safe, no word boundaries needed)."""
    lines: list[str] = []
    current = ""
    for ch in text:
        current += ch
        if len(current) >= width_chars:
            lines.append(current)
            current = ""
    if current:
        lines.append(current)
    # Avoid leaving trailing punctuation orphaned on its own line.
    if len(lines) > 1 and len(lines[-1]) <= 2 and lines[-1] in {"。", "，", "、", ".", ",", "；", ";"}:
        tail = lines.pop()
        lines[-1] += tail
    return lines


def render_overlay(
    image_path: str | Path,
    plan: dict[str, Any],
    output_path: str | Path,
    *,
    observation: dict[str, Any] | None = None,
    guidance_direction: str | None = None,
    subject_bbox: list[float] | None = None,
    mirrored: bool = False,
    show_anchors: bool = True,
    banner: bool = True,
) -> RenderResult:
    """Draw the plan's target box, anchors and guidance arrow onto the source photo.

    Args:
        image_path: source photo (the same file that was sent to the model).
        plan: a validated `shot_plan` payload.
        output_path: where to write the annotated JPEG.
        observation: optional `scene_observation` used to draw the anchors.
        guidance_direction: "left" | "right" | "hold" | None, in `input_image` space.
        subject_bbox: current tracked subject box in normalized input space.
        mirrored: set True if the preview is mirrored, so the arrow matches the screen.
        show_anchors: draw the preserved anchor boxes.
        banner: draw the title / instruction banner at the top.
    """
    src = Path(image_path)
    if not src.is_file():
        raise FileNotFoundError(f"Source image not found: {src}")

    image = Image.open(src).convert("RGB")
    width, height = image.size
    result = RenderResult(output_path=str(output_path), image_size=[width, height])

    draw = ImageDraw.Draw(image)
    plan = plan or {}
    primary = plan.get("primary_plan") or {}
    actions = plan.get("actions") or []
    active_action_id = plan.get("active_action_id")
    action = next((a for a in actions if a.get("action_id") == active_action_id), actions[0] if actions else {})

    # --- target subject box -------------------------------------------------
    target_bbox = (primary.get("composition") or {}).get("target_subject_bbox")
    if target_bbox:
        xmin, ymin, xmax, ymax = _bbox_to_px(target_bbox, width, height)
        result.target_bbox_px = [round(xmin, 1), round(ymin, 1), round(xmax, 1), round(ymax, 1)]
        draw.rectangle([xmin, ymin, xmax, ymax], outline=COLOR_TARGET, width=6)
        label = "TARGET"
        font_small = _load_font(26)
        draw.rectangle([xmin, max(0, ymin - 34), xmin + 8 * len(label), ymin], fill=COLOR_TARGET)
        draw.text((xmin + 8, max(0, ymin - 32)), label, fill=COLOR_TEXT, font=font_small)

    # --- person ghost figure (scene has nobody in frame) ---------------------
    guide = plan.get("subject_placement_guide")
    if isinstance(guide, dict) and guide.get("bbox"):
        gxmin, gymin, gxmax, gymax = _bbox_to_px(guide["bbox"], width, height)
        result.ghost_bbox_px = [round(gxmin, 1), round(gymin, 1), round(gxmax, 1), round(gymax, 1)]
        result.ghost_basis = guide.get("basis")
        result.ghost_instruction = guide.get("instruction")
        result.ghost_framing = guide.get("framing")
        _draw_person_silhouette(
            image, gxmin, gymin, gxmax, gymax, COLOR_GHOST, str(guide.get("framing") or "full_body")
        )
        draw = ImageDraw.Draw(image)
        font_small = _load_font(26)
        label = "虚框·站位"
        tag_y = max(0, gymin - 38)
        draw.rectangle([gxmin, tag_y, gxmin + 8 * len(label) + 16, tag_y + 34], fill=COLOR_GHOST)
        draw.text((gxmin + 8, tag_y + 2), label, fill=(0, 40, 40), font=font_small)

    # --- crop recommendation -------------------------------------------------
    crop = plan.get("crop_recommendation")
    if isinstance(crop, dict) and crop.get("crop_rect"):
        cxmin, cymin, cxmax, cymax = _bbox_to_px(crop["crop_rect"], width, height)
        result.crop_rect_px = [round(cxmin, 1), round(cymin, 1), round(cxmax, 1), round(cymax, 1)]
        result.crop_aspect_ratio = crop.get("aspect_ratio")
        result.crop_rationale = crop.get("rationale")

        # Dim everything outside the suggested crop, then frame the crop area.
        shade = Image.new("RGBA", image.size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(shade)
        sd.rectangle([0, 0, width, height], fill=(0, 0, 0, 110))
        sd.rectangle([cxmin, cymin, cxmax, cymax], fill=(0, 0, 0, 0))
        image.paste(shade.convert("RGB"), (0, 0), shade)
        draw = ImageDraw.Draw(image)
        draw.rectangle([cxmin, cymin, cxmax, cymax], outline=COLOR_CROP, width=6)

        font_small = _load_font(26)
        label = f"裁剪 {crop.get('aspect_ratio', '')}"
        # Anchor the tag to the bottom-left of the crop so the banner cannot hide it.
        tag_h = 34
        tag_y = min(height - tag_h - 4, cymax - tag_h)
        draw.rectangle([cxmin, tag_y, cxmin + 8 * len(label) + 16, tag_y + tag_h], fill=COLOR_CROP)
        draw.text((cxmin + 8, tag_y + 2), label, fill=(30, 20, 0), font=font_small)

    # --- currently tracked subject -----------------------------------------
    if subject_bbox:
        sxmin, symin, sxmax, symax = _bbox_to_px(subject_bbox, width, height)
        result.subject_bbox_px = [round(sxmin, 1), round(symin, 1), round(sxmax, 1), round(symax, 1)]
        draw.rectangle([sxmin, symin, sxmax, symax], outline=COLOR_SUBJECT, width=5)
        font_small = _load_font(26)
        draw.text((sxmin + 8, symin + 8), "NOW", fill=COLOR_SUBJECT, font=font_small)

    # --- preserved anchors --------------------------------------------------
    if show_anchors and observation:
        anchor_ids = set((primary.get("composition") or {}).get("preserve_anchor_ids") or [])
        for anchor in observation.get("anchors") or []:
            if anchor_ids and anchor.get("anchor_id") not in anchor_ids:
                continue
            bbox = anchor.get("bbox")
            if not bbox:
                continue
            axmin, aymin, axmax, aymax = _bbox_to_px(bbox, width, height)
            result.anchor_boxes_px.append([round(axmin, 1), round(aymin, 1), round(axmax, 1), round(aymax, 1)])
            draw.rectangle([axmin, aymin, axmax, aymax], outline=COLOR_ANCHOR, width=4)

    # --- guidance arrow -----------------------------------------------------
    arrow_dir = guidance_direction
    if arrow_dir and arrow_dir not in {"hold", "none", None} and target_bbox:
        cx = (target_bbox[0] + target_bbox[2]) / 2.0
        y = (target_bbox[1] + target_bbox[3]) / 2.0
        span = width * 0.16
        # Direction is authored in input_image space; the preview may be mirrored.
        screen_dir = arrow_dir
        if mirrored:
            screen_dir = "right" if arrow_dir == "left" else "left"
        start_x = cx * width - span if screen_dir == "right" else cx * width + span
        end_x = cx * width + span if screen_dir == "right" else cx * width - span
        _draw_arrow(draw, (start_x, y * height), (end_x, y * height), COLOR_ARROW, 8)
        result.guidance_arrow = f"{arrow_dir} (screen: {screen_dir})"

    # --- banner -------------------------------------------------------------
    if banner:
        font_title = _load_font(34)
        font_body = _load_font(24)
        title = str(primary.get("title") or plan.get("scene_summary") or plan.get("status") or "")
        instruction = str(action.get("instruction") or "")
        result.title = title
        result.instruction = instruction

        lines = _wrap(instruction, 34)
        panel_h = 62 + 30 * len(lines)
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, 0, width, panel_h], fill=COLOR_PANEL)
        draw.rectangle([0, 0, width, 5], fill=COLOR_TARGET)

        # Cards chip is right-aligned; truncate the title so the two never overlap.
        chip = f"cards: {','.join(primary.get('used_card_ids') or []) or '-'}"
        chip_w = int(draw.textlength(chip, font=font_body))
        chip_x = max(24, width - 26 - chip_w)
        draw.text((chip_x, 20), chip, fill=(190, 195, 205), font=font_body)

        heading = f"[{plan.get('status', '-')}] {title}"
        max_heading_w = chip_x - 24 - 16
        if draw.textlength(heading, font=font_title) > max_heading_w:
            while heading and draw.textlength(heading + "…", font=font_title) > max_heading_w:
                heading = heading[:-1]
            heading += "…"
        draw.text((24, 14), heading, fill=COLOR_TEXT, font=font_title)

        for i, line in enumerate(lines):
            draw.text((24, 62 + i * 30), line, fill=(225, 225, 230), font=font_body)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out, "JPEG", quality=92)
    return result
