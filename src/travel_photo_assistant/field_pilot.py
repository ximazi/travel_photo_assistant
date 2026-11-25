"""Run a single authorized, filename-blind Gemini evaluation; no fixture fallback."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from uuid import uuid4

from .adapters.providers import GeminiVisionAdapter
from .config import AssetPaths, RuntimeConfig
from .controller import TravelPhotoController
from .env_loader import load_dotenv
from .knowledge import RetrievalMode
from .models import RuntimeInput
from .render import render_overlay


def prepare_photo(path: Path, goal: str) -> tuple[RuntimeInput, bytes]:
    from PIL import Image, ImageOps

    with Image.open(path) as source:
        if source.format not in {"JPEG", "PNG", "WEBP"} or getattr(source, "is_animated", False):
            raise ValueError("Expected a single JPEG, PNG or WebP image.")
        upright = ImageOps.exif_transpose(source).convert("RGB")
        upright.thumbnail((2048, 2048))
        width, height = upright.size
        # Re-encode only pixels, excluding EXIF, GPS and text metadata.
        clean = Image.frombytes("RGB", upright.size, upright.tobytes())
        stream = io.BytesIO()
        clean.save(stream, format="JPEG", quality=90)
    pixels = stream.getvalue()
    return RuntimeInput.from_dict({
        "frame": {
            "frame_id": f"pilot_{uuid4().hex}",
            "scene_version": 1,
            "image_attachments": [{"mime_type": "image/jpeg", "data_base64": base64.b64encode(pixels).decode("ascii")}],
            "upright_input_width_px": width, "upright_input_height_px": height,
            "input_rotation_already_applied": True, "preview_mirrored": False,
            "viewport_fit": "contain", "preview_aspect_ratio": f"{width}:{height}",
        },
        "user_goal": {
            "priority": goal, "subject_id": None, "subject_count": None,
            "orientation_preference": "portrait" if height >= width else "landscape",
            "allow_subject_walking": False,
        },
        "camera_capabilities": {"metadata_available": False, "available_lenses": [], "current_lens_id": None},
        "movement_constraints": {"photographer_translation": "unknown",
                                 "subject_translation": "not_requested", "standing_in_place_confirmed": False},
        "local_vision_hints": {},
    }), pixels


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--photo", required=True, type=Path)
    p.add_argument("--goal", required=True, help="拍摄意图，不包含人工标注/参考答案")
    p.add_argument("--out", required=True, type=Path, help="新增私有结果目录，不要用 Git checkout 下的目录")
    p.add_argument("--model", required=True, help="AI Studio 中确认可用的免费 Gemini 模型 ID")
    p.add_argument("--mode", choices=[m.value for m in RetrievalMode], default="retrieved_cards")
    p.add_argument("--confirm-free-project", action="store_true",
                   help="已核实 Gemini API 项目没有启用计费且此模型在当前 Free Tier 内")
    p.add_argument("--allow-photo-to-gemini", action="store_true",
                   help="明确同意将这张照片的像素发送给 Gemini")
    a = p.parse_args(argv)
    if not (a.confirm_free_project and a.allow_photo_to_gemini):
        p.error("Both consent and a verified non-billing Free Tier project are required.")
    from re import fullmatch
    if not fullmatch(r"gemini-[a-zA-Z0-9.\-]+", a.model):
        p.error("Expected a Gemini model ID, not a URL.")
    assets = AssetPaths.default()
    if a.out.resolve().is_relative_to(assets.project_root.resolve()):
        p.error("--out must be outside the Git checkout.")
    if a.out.exists():
        p.error("--out must be a new, private result directory.")
    load_dotenv()
    adapter = GeminiVisionAdapter(model_name=a.model)
    if not adapter.api_key:
        p.error("GEMINI_API_KEY is not configured.")
    try:
        req, image = prepare_photo(a.photo, a.goal)
    except (OSError, ValueError) as e:
        p.error(f"Cannot prepare the photo ({type(e).__name__}).")
    a.out.mkdir(parents=True, mode=0o700)
    start = perf_counter()
    controller = TravelPhotoController(adapter, assets=assets, config=RuntimeConfig(retry_on_validation_error=False))
    try:
        result = controller.run_slow_loop(req, retrieval_mode=a.mode)
    except (RuntimeError, OSError, ValueError, KeyError, IndexError, TypeError) as e:
        # Do not put provider response bodies, credentials or an image in logs.
        (a.out / "run.json").write_text(json.dumps({"status": "adapter_failed", "class": type(e).__name__}), encoding="utf-8")
        return 2
    elapsed_ms = round((perf_counter() - start) * 1000)
    run = {"status": "needs_human_review", "provider": "gemini", "model": a.model, "mode": a.mode,
           "elapsed_ms": elapsed_ms, "accepted": result.accepted,
           "fallback": result.downgraded_to_fallback, "validation_retries_enabled": False,
           "image_sha256": hashlib.sha256(image).hexdigest(),
           "versions": {x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in
                        (assets.observer_prompt, assets.planner_prompt, assets.shot_cards)},
           "selected_card_ids": [c.id for c in result.retrieval.selected_cards] if result.retrieval else [],
           "validation_issues": [{"stage": name, "code": issue.code, "field": issue.field_path}
                                 for name, val in (("input", result.input_validation),
                                 ("observation", result.observation_validation), ("plan", result.plan_validation))
                                 if val for issue in val.issues],
           "cost_status": "free_project_operator_attested"}
    artifacts = {"observation": result.observation, "plan": result.plan, "run": run,
                 "retrieval": [asdict(x) for x in result.retrieval.evaluations] if result.retrieval else []}
    for name, data in artifacts.items():
        (a.out / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    if result.plan:
        with TemporaryDirectory() as tmp:
            normalized = Path(tmp) / "normalized.jpg"
            normalized.write_bytes(image)
            render_overlay(normalized, result.plan, a.out / "overlay.jpg", observation=result.observation)
    return 0 if result.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
