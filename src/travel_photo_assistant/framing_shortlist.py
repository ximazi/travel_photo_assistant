"""P0 empty-scene shortlist: distinguish frames, clean pixels, and standing previews."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from math import ceil, floor, isfinite
from typing import Iterable

ASPECTS = {"1:1", "3:2", "2:3", "4:3", "3:4", "4:5", "16:9", "9:16"}


def box(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 4
            or any(type(x) not in (float, int) or not isfinite(x) for x in value)):
        return None
    x1, y1, x2, y2 = value
    if 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1:
        return tuple(value)
    return None


def contains(a, b):
    return a[0] <= b[0] and a[1] <= b[1] and b[2] <= a[2] and b[3] <= a[3]


def inside(a, p):
    return a[0] <= p[0] <= a[2] and a[1] <= p[1] <= a[3]


def overlaps(a, b):
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


@dataclass(frozen=True)
class Proposal:
    source: str
    crop: tuple
    silhouette: tuple
    aspect: str | None = None
    baseline: bool = False


def shortlist(proposals: Iterable[Proposal], anchors: Iterable[dict],
              size: tuple[int, int]) -> dict:
    """Return provisional full-body options and deduped clean judge crops.

    Coordinates are upright normalized source values. Only geometry is tested:
    a possible-ground anchor is not permission to stand or a measurement of depth.
    """
    w, h = size
    if min(w, h) < 1:
        raise ValueError("Expected positive image dimensions")
    groups = {}
    rejected = []
    landmarks, ground, obstacles = [], [], []
    unresolved_landmark = False
    for a in anchors:
        b = box(a.get("bbox"))
        kind = a.get("kind")
        if kind == "landmark" and not b:
            unresolved_landmark = True
        if b and kind in {"landmark", "possible_ground", "obstacle"}:
            {"landmark": landmarks, "possible_ground": ground, "obstacle": obstacles}[kind].append(b)
    for p in proposals:
        crop, fig = box(p.crop), box(p.silhouette)
        reasons = []
        if not crop or not fig:
            rejected.append({"source": p.source, "reasons": ["invalid_geometry"]})
            continue
        pixels = (floor(crop[0] * w), floor(crop[1] * h),
                  ceil(crop[2] * w), ceil(crop[3] * h))
        cw, ch = pixels[2] - pixels[0], pixels[3] - pixels[1]
        foot = ((fig[0] + fig[2]) / 2, fig[3])
        # One-percent head/foot room in the proposed crop; not a safety margin.
        dx, dy = (crop[2] - crop[0]) * .01, (crop[3] - crop[1]) * .01
        inset = crop[0] + dx, crop[1] + dy, crop[2] - dx, crop[3] - dy
        if not contains(inset, fig):
            reasons.append("silhouette_cut_or_too_close_to_edge")
        if not any(inside(b, foot) for b in ground):
            reasons.append("footpoint_not_on_visible_ground")
        if any(inside(b, foot) for b in obstacles):
            reasons.append("footpoint_on_visible_obstacle")
        if unresolved_landmark and not p.baseline:
            reasons.append("landmark_extent_unknown")
        if any(not contains(inset, b) for b in landmarks):
            reasons.append("landmark_cut_or_too_close_to_edge")
        if any(overlaps(fig, b) for b in landmarks):
            reasons.append("silhouette_overlaps_landmark")
        if not p.baseline and (min(cw, ch) < 480 or cw * ch < .55 * w * h):
            reasons.append("too_few_original_pixels")
        if p.aspect is not None:
            if p.aspect not in ASPECTS:
                reasons.append("invalid_aspect")
            else:
                aw, ah = map(int, p.aspect.split(":"))
                expected_width = ch * aw / ah
                if abs(cw - expected_width) > max(3, .025 * expected_width):
                    reasons.append("pixel_aspect_mismatch")
        if reasons:
            rejected.append({"source": p.source, "reasons": reasons})
            continue
        key = tuple(pixels)
        group = groups.setdefault(key, {"crop_pixels": pixels, "placements": [], "baseline": False})
        group["baseline"] |= p.baseline
        normalized_shape = tuple(round(v, 5) for v in fig)
        if not any(x["bbox"] == normalized_shape for x in group["placements"]):
            group["placements"].append({"source": p.source, "bbox": normalized_shape,
                                        "footpoint": foot, "standing_safety_verified": False})
    return {"status": "needs_review" if groups else "no_supported_candidate",
            "clean_crop_count": len(groups), "groups": list(groups.values()),
            "rejected": rejected, "ranking": "not_run", "placement_winner": None}


def arm_shortlist(results, size):
    """Collect full-frame and proposed crops from accepted arms; no model calls."""
    proposals = []
    observation = {}
    for mode, (result, _) in results.items():
        observation = result.observation or {}
        plan = result.plan or {}
        guide = plan.get("subject_placement_guide")
        if not result.accepted or plan.get("status") != "no_subject" or not guide:
            continue
        if guide.get("framing") != "full_body":
            continue  # We cannot derive a footpoint from an intentionally cropped body.
        fig = tuple(guide["bbox"])
        proposals.append(Proposal(mode + ":unchanged", (0, 0, 1, 1), fig, baseline=True))
        crop = plan.get("crop_recommendation")
        if crop:
            proposals.append(Proposal(mode + ":crop", tuple(crop["crop_rect"]), fig,
                                      aspect=crop["aspect_ratio"]))
    return shortlist(proposals, observation.get("anchors", []), size)


def export_clean_crops(image: bytes, report: dict, target: Path) -> None:
    """Export a single unmarked real-pixel image per shortlisted crop."""
    from PIL import Image
    if not report["groups"]:
        return
    target.mkdir(mode=0o700)
    with Image.open(BytesIO(image)) as src:
        for index, group in enumerate(report["groups"], start=1):
            filename = f"frame-{index:02}.jpg"
            src.crop(tuple(group["crop_pixels"])).save(target / filename, quality=94)
            group["clean_preview"] = filename
