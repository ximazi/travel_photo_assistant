"""Experimental Gemini-only wire convention for the blind pilot."""
from copy import deepcopy
from math import isfinite

from .providers import GeminiVisionAdapter

CANONICAL = "upright_input_normalized_unmirrored"
WIRE = "gemini_grid1000_v1"
OBS_BOX = (("facts", "*", "region"), ("selected_subject", "bbox"), ("anchors", "*", "bbox"))
OBS_POINTS = ("anchors", "*", "points")
PLAN_BOX = (("subject_placement_guide", "bbox"), ("crop_recommendation", "crop_rect"),
            ("primary_plan", "composition", "target_subject_bbox"))
NOTE = """Gemini wire override (takes precedence over all other coordinate rules):
Set coordinate_space to gemini_grid1000_v1. Spatial box fields region, bbox,
target_subject_bbox and crop_rect are INTEGER [ymin,xmin,ymax,xmax] in [0,1000].
Point tuples in anchors.points are INTEGER [x,y] in [0,1000].
For example a box from x=12.5% to 76.0% and y=67.1% to 86.4% is
[671,125,864,760]; its top-left point is [125,671]. Use integer 1000 for an
image edge, not integer 1. If unable to locate a nullable field, return null.
Only the listed spatial fields change; height ratios and action thresholds
remain the original units. The planner's scene_observation uses this same
integer wire system. Keep all other rules and semantic requirements.
"""


def _transform(node, path, fn):
    if node is None:
        return
    key, *rest = path
    if key == "*":
        if isinstance(node, list):
            for item in node:
                _transform(item, rest, fn)
    elif isinstance(node, dict) and key in node:
        if rest:
            _transform(node[key], rest, fn)
        elif node[key] is not None:
            node[key] = fn(node[key])


def _decode_tuple(value, box=False):
    size = 4 if box else 2
    if (not isinstance(value, list) or len(value) != size
            or any(type(v) is not int or not 0 <= v <= 1000 for v in value)):
        raise ValueError("Malformed Gemini grid geometry")
    if box:
        y0, x0, y1, x1 = value
        if x0 >= x1 or y0 >= y1:
            raise ValueError("Unordered Gemini grid rectangle")
        value = [x0, y0, x1, y1]
    return [v / 1000 for v in value]


def _encode_tuple(value, box=False):
    if (not isinstance(value, list) or len(value) != (4 if box else 2)
            or any(type(v) not in (int, float) or not isfinite(v) or not 0 <= v <= 1 for v in value)):
        raise ValueError("Malformed canonical geometry")
    grid = [round(v * 1000) for v in value]
    return [grid[1], grid[0], grid[3], grid[2]] if box else grid


def _geometry(payload, *, observation, encode=False):
    source, target = (CANONICAL, WIRE) if encode else (WIRE, CANONICAL)
    if not isinstance(payload, dict) or payload.get("coordinate_space") != source:
        raise ValueError("Unexpected Gemini coordinate-space declaration")
    payload = deepcopy(payload)
    payload["coordinate_space"] = target
    fn = _encode_tuple if encode else _decode_tuple
    for path in OBS_BOX if observation else PLAN_BOX:
        _transform(payload, path, lambda v: fn(v, box=True))
    if observation:
        def convert_points(points):
            if not isinstance(points, list):
                raise ValueError("Malformed Gemini grid points")
            return [fn(p) for p in points]
        _transform(payload, OBS_POINTS, convert_points)
    return payload


def _wire_schema(schema):
    schema = deepcopy(schema)

    def adapt(node, field=None):
        if not isinstance(node, dict):
            return
        if field in {"region", "bbox", "target_subject_bbox", "crop_rect", "points"} and node.get("type") == "array":
            box = field != "points"
            item = {"type": "integer"}
            node["items"] = item if box else {"type": "array", "items": item}
            node["description"] = ("Integer [ymin,xmin,ymax,xmax] in [0,1000]" if box
                                   else "List of integer [x,y] points in [0,1000]")
        for key, value in node.get("properties", {}).items():
            if key == "coordinate_space":
                value.clear()
                value.update(type="string", const=WIRE)
            else:
                adapt(value, key)
        for branch in node.get("anyOf", []):
            adapt(branch, field)
        adapt(node.get("items"))
    adapt(schema)
    return schema


class GeminiGridPilotAdapter(GeminiVisionAdapter):
    """Only opt-in blind pilot calls use integer geometry; receipts stay canonical."""

    def _generate(self, system_prompt, user_payload, attachments, output_schema, feedback=None):
        is_observation = "scene_observation" not in user_payload
        if not is_observation:
            user_payload = {**user_payload, "scene_observation": _geometry(
                user_payload["scene_observation"], observation=True, encode=True,
            )}
        raw = super()._generate(f"{NOTE}\n{system_prompt}\n{NOTE}", user_payload, attachments,
                                _wire_schema(output_schema), feedback)
        return _geometry(raw, observation=is_observation)
