"""Run a single authorized, filename-blind Gemini evaluation; no fixture fallback."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from uuid import uuid4

from .adapters.base import VisionModelAdapter
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


class FrozenObserver(VisionModelAdapter):
    """Hold one observer response while the production controller runs each planner arm."""

    def __init__(self, delegate, frame_id):
        self.delegate = delegate
        self.frame_id = frame_id
        self.model_name = delegate.model_name
        self.raw_observation = None
        self.observation_calls = 0
        self.planner_calls = 0

    def observe(self, runtime_input, system_prompt, output_schema, feedback=None):
        if runtime_input.frame.frame_id != self.frame_id or feedback is not None:
            raise ValueError("Comparison cannot change the frame or retry the observer.")
        if self.observation_calls == 0:
            self.observation_calls += 1
            self.raw_observation = deepcopy(self.delegate.observe(runtime_input, system_prompt, output_schema))
        return deepcopy(self.raw_observation)

    def plan(self, runtime_input, observation, shot_cards, system_prompt, output_schema, feedback=None):
        if runtime_input.frame.frame_id != self.frame_id or feedback is not None:
            raise ValueError("Comparison cannot change the frame or retry the planner.")
        self.planner_calls += 1
        return self.delegate.plan(runtime_input, observation, shot_cards, system_prompt, output_schema)


def observation_sha(observation):
    if observation is None:
        return None
    canonical = json.dumps(observation, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run_comparison(adapter, assets, req):
    frozen = FrozenObserver(adapter, req.frame.frame_id)
    results = {}
    skip = {}
    for mode in (RetrievalMode.PROMPT_ONLY, RetrievalMode.FIXED_CARDS, RetrievalMode.RETRIEVED_CARDS):
        controller = TravelPhotoController(frozen, assets=assets, config=RuntimeConfig(retry_on_validation_error=False))
        start = perf_counter()
        result = controller.run_slow_loop(req, retrieval_mode=mode.value)
        results[mode.value] = (result, round((perf_counter() - start) * 1000))
        if mode == RetrievalMode.PROMPT_ONLY:
            obs = result.observation or {}
            if not result.observation_validation or not result.observation_validation.is_valid:
                skip = {"fixed_cards": "invalid_observation", "retrieved_cards": "invalid_observation"}
                break
            if obs.get("status") != "observed" or not obs.get("selected_subject"):
                skip = {"fixed_cards": "no_confirmed_subject_for_portrait_comparison",
                        "retrieved_cards": "no_confirmed_subject_for_portrait_comparison"}
                break
    hashes = {observation_sha(result.observation) for result, _ in results.values()}
    if len(hashes) != 1:
        raise ValueError("Comparison arms did not consume the same observation.")
    return results, {
        "status": "needs_human_review" if not skip else "comparison_not_scored",
        "observation_sha256": next(iter(hashes)),
        "observer_requests": frozen.observation_calls,
        "planner_requests": frozen.planner_calls,
        "evaluated_modes": list(results),
        "skipped_modes": skip,
        "card_budget_matched": False,
        "latency_note": "First arm includes observation latency; subsequent arms do not.",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--photo", required=True, type=Path)
    p.add_argument("--goal", required=True, help="拍摄意图，不包含人工标注/参考答案")
    p.add_argument("--out", required=True, type=Path, help="新增私有结果目录，不要用 Git checkout 下的目录")
    p.add_argument("--model", required=True, help="AI Studio 中确认可用的免费 Gemini 模型 ID")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--mode", choices=[m.value for m in RetrievalMode], default="retrieved_cards")
    mode.add_argument("--compare", action="store_true", help="固定一次观察，对照无卡/固定卡/检索卡（至多四次请求）")
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
    try:
        if a.compare:
            results, comparison = run_comparison(adapter, assets, req)
        else:
            start = perf_counter()
            controller = TravelPhotoController(adapter, assets=assets, config=RuntimeConfig(retry_on_validation_error=False))
            result = controller.run_slow_loop(req, retrieval_mode=a.mode)
            results = {a.mode: (result, round((perf_counter() - start) * 1000))}
            comparison = None
    except (RuntimeError, OSError, ValueError, KeyError, IndexError, TypeError) as e:
        # Do not put provider response bodies, credentials or an image in logs.
        (a.out / "run.json").write_text(json.dumps({"status": "adapter_failed", "class": type(e).__name__}), encoding="utf-8")
        return 2
    for mode, (result, elapsed_ms) in results.items():
        target = a.out / mode if a.compare else a.out
        target.mkdir(mode=0o700, exist_ok=True)
        write_arm(target, result, elapsed_ms, mode, a.model, assets, image)
    if comparison is not None:
        (a.out / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if all(result.accepted for result, _ in results.values()) else 1


def write_arm(out, result, elapsed_ms, mode, model, assets, image):
    run = {"status": "needs_human_review", "provider": "gemini", "model": model, "mode": mode,
           "elapsed_ms": elapsed_ms, "accepted": result.accepted,
           "fallback": result.downgraded_to_fallback, "validation_retries_enabled": False,
           "image_sha256": hashlib.sha256(image).hexdigest(),
           "observation_sha256": observation_sha(result.observation),
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
        (out / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    if result.plan:
        with TemporaryDirectory() as tmp:
            normalized = Path(tmp) / "normalized.jpg"
            normalized.write_bytes(image)
            render_overlay(normalized, result.plan, out / "overlay.jpg", observation=result.observation)


if __name__ == "__main__":
    raise SystemExit(main())
