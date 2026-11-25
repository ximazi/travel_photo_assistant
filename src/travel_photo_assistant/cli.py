"""Command-line interface for Travel Photo Assistant v0.1."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .adapters.mock import RuleBasedOfflineVisionAdapter
from .config import AssetPaths, RuntimeConfig
from .controller import TravelPhotoController
from .coordinates import PreviewMapper, PreviewTransformConfig
from .env_loader import env_status, load_dotenv
from .evaluation import EvaluationHarness
from .knowledge import ShotCardRepository, ShotCardRetriever
from .models import LocalVisionSnapshot, RuntimeInput
from .photo_pipeline import PHOTO_PRESETS, run_photo
from .schema_validator import SchemaValidator
from .validators import BusinessRuleValidator


def _cmd_validate_examples(assets: AssetPaths) -> int:
    ex_dir = assets.examples_dir
    runtime_raw = json.loads((ex_dir / "runtime_input.json").read_text(encoding="utf-8"))
    obs_raw = json.loads((ex_dir / "scene_observation.json").read_text(encoding="utf-8"))
    plan_raw = json.loads((ex_dir / "shot_plan.json").read_text(encoding="utf-8"))

    # Fixture mode (production_mode=False) validates protocol contracts
    fixture_schema = SchemaValidator(assets, production_mode=False)
    obs_schema_errs = fixture_schema.validate_observation(obs_raw)
    plan_schema_errs = fixture_schema.validate_shot_plan(plan_raw)

    # Production mode MUST reject example_only=true and source=synthetic_fixture
    prod_schema = SchemaValidator(assets, production_mode=True)
    prod_validator = BusinessRuleValidator(prod_schema, production_mode=True)
    runtime_input = RuntimeInput.from_dict(runtime_raw)
    prod_input_val = prod_validator.validate_runtime_input(runtime_input)
    prod_obs_val = prod_validator.validate_observation(obs_raw, runtime_input)

    report = {
        "fixture_schema_validation": {
            "scene_observation_valid": len(obs_schema_errs) == 0,
            "scene_observation_errors": obs_schema_errs,
            "shot_plan_valid": len(plan_schema_errs) == 0,
            "shot_plan_errors": plan_schema_errs,
        },
        "production_isolation_guard": {
            "rejects_example_runtime_input": not prod_input_val.is_valid,
            "runtime_input_error_codes": [e.code for e in prod_input_val.errors],
            "rejects_synthetic_fixture_observation": not prod_obs_val.is_valid,
            "observation_error_codes": [e.code for e in prod_obs_val.errors],
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if (not obs_schema_errs and not plan_schema_errs and not prod_input_val.is_valid) else 1


def _cmd_evaluate(assets: AssetPaths, output_path: str | None) -> int:
    harness = EvaluationHarness(assets)
    report = harness.run_all_offline_contract_checks()
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(rendered + "\n")
    print(rendered)
    return 0 if report["summary"]["offline_contract_failed"] == 0 else 1


def _cmd_demo(assets: AssetPaths) -> int:
    ctrl = TravelPhotoController(
        adapter=RuleBasedOfflineVisionAdapter(),
        assets=assets,
        config=RuntimeConfig(production_mode=True),
    )
    req = RuntimeInput.from_dict(
        {
            "schema_version": "0.1.0",
            "example_only": False,
            "frame": {
                "frame_id": "live_f001",
                "scene_version": 1,
                "image_attachments": [{"mime_type": "image/jpeg", "uri": "memory://demo_coast_frame.jpg"}],
                "upright_input_width_px": 1080,
                "upright_input_height_px": 1440,
                "coordinate_space": "upright_input_normalized_unmirrored",
                "preview_mirrored": False,
                "viewport_fit": "contain",
                "preview_aspect_ratio": "3:4",
            },
            "user_goal": {
                "priority": "人物自然、脸部可辨，同时保留海岸旅行环境",
                "style": "自然旅行人像",
                "subject_id": "person_1",
                "subject_count": 1,
                "orientation_preference": "portrait",
                "allow_subject_walking": False,
            },
            "camera_capabilities": {
                "current_lens_id": "main_1x",
                "available_lenses": [
                    {
                        "lens_id": "main_1x",
                        "ui_label": "1×",
                        "equivalent_focal_length_mm": 26,
                        "equivalent_focal_length_source": "camera_api",
                        "mode": "optical_primary",
                    }
                ],
                "manual_aperture": False,
                "exposure_compensation_available": True,
            },
            "movement_constraints": {
                "photographer_translation": "unknown",
                "subject_translation": "not_requested",
                "standing_in_place_confirmed": True,
            },
            "local_vision_hints": {
                "scene_tags": ["coast", "leading_line", "clean_background"],
                "scene_description": "海边场景，左侧可见弯曲海岸线。",
                "subject_bbox": [0.38, 0.32, 0.62, 0.90],
                "anchors": [
                    {
                        "anchor_id": "coast_curve",
                        "kind": "leading_line",
                        "description": "左下向中部延伸的海岸曲线",
                        "bbox": [0.02, 0.46, 0.47, 0.96],
                        "points": [[0.03, 0.94], [0.18, 0.72], [0.45, 0.49]],
                    }
                ],
            },
        }
    )

    slow = ctrl.run_slow_loop(req)
    # Simulate 3 fast ticks moving subject center_x into [0.60, 0.70]
    ticks = [
        asdict(
            ctrl.run_fast_tick(
                LocalVisionSnapshot(
                    frame_id=f"live_t{i}",
                    scene_version=1,
                    current_lens_id="main_1x",
                    orientation="portrait",
                    crop_signature="contain:3:4",
                    tracked_subject_id="person_1",
                    subject_bbox=bbox,
                )
            )
        )
        for i, bbox in enumerate(
            [
                [0.42, 0.32, 0.66, 0.90],  # cx=0.54 -> pan left
                [0.53, 0.32, 0.77, 0.90],  # cx=0.65 -> in range (1/2)
                [0.54, 0.32, 0.78, 0.90],  # cx=0.66 -> in range (2/2) -> action a1 completed!
            ],
            start=1,
        )
    ]

    output: dict[str, Any] = {
        "slow_loop": {
            "accepted": slow.accepted,
            "selected_cards": [c.id for c in slow.retrieval.selected_cards] if slow.retrieval else [],
            "plan_title": slow.plan["primary_plan"]["title"] if slow.plan and slow.plan.get("primary_plan") else None,
            "active_action": slow.plan["actions"][0] if slow.plan and slow.plan.get("actions") else None,
        },
        "fast_loop_ticks": ticks,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def _cmd_check_cards(assets: AssetPaths, scene_tags: list[str], priority: str) -> int:
    repo = ShotCardRepository(assets.shot_cards)
    retriever = ShotCardRetriever(repo)
    dummy_obs = {
        "status": "observed",
        "scene_tags": scene_tags,
        "selected_subject": {"subject_id": "person_1"},
        "anchors": [],
        "issues": [],
        "light": {"appearance": "柔光"},
    }
    dummy_input = RuntimeInput.from_dict(
        {
            "frame": {"frame_id": "cli_f1", "scene_version": 1},
            "user_goal": {"priority": priority},
        }
    )
    res = retriever.retrieve(dummy_obs, dummy_input)
    print(
        json.dumps(
            {
                "selected_card_ids": [c.id for c in res.selected_cards],
                "latency_ms": res.latency_ms,
                "evaluations": [asdict(e) for e in res.evaluations],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _cmd_transform_coord(args: argparse.Namespace) -> int:
    bbox = [float(x) for x in args.bbox.split(",")]
    mapper = PreviewMapper(
        PreviewTransformConfig(
            input_width_px=args.input_width,
            input_height_px=args.input_height,
            viewport_width_px=args.viewport_width,
            viewport_height_px=args.viewport_height,
            viewport_fit=args.fit,
            preview_mirrored=args.mirrored,
        )
    )
    rect = mapper.map_bbox(bbox)
    print(
        json.dumps(
            {
                "viewport_rect": asdict(rect),
                "mapped_direction": mapper.map_horizontal_direction(args.direction, args.direction_ref),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _cmd_env(assets: AssetPaths) -> int:
    status = env_status()
    status["output_dir"] = str(assets.project_root / "output")
    status["photos_dir"] = str(assets.photos_dir)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


def _cmd_list_photos(assets: AssetPaths) -> int:
    photos_dir = assets.photos_dir
    entries = []
    if photos_dir.is_dir():
        for p in sorted(photos_dir.iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".heic"}:
                continue
            entries.append({
                "name": p.name,
                "path": str(p),
                "bytes": p.stat().st_size,
                "has_preset": p.name in PHOTO_PRESETS,
                "preset_scene_tags": (PHOTO_PRESETS.get(p.name) or {}).get("scene_tags"),
                "preset_detected_person_count": (PHOTO_PRESETS.get(p.name) or {}).get("detected_person_count"),
            })
    print(json.dumps({
        "photos_dir": str(photos_dir),
        "exists": photos_dir.is_dir(),
        "count": len(entries),
        "photos": entries,
        "note": "这些是合成的测试图，只用于跑通链路，不能用于验证真实视觉准确率。",
    }, ensure_ascii=False, indent=2))
    return 0


def _cmd_run_photo(assets: AssetPaths, args: argparse.Namespace) -> int:
    adapter = RuleBasedOfflineVisionAdapter()
    provider = os.environ.get("TPA_VISION_PROVIDER", "offline").lower().strip()
    if provider == "gemini":
        from .adapters import GeminiVisionAdapter

        adapter = GeminiVisionAdapter(model_name=os.environ.get("TPA_MODEL_NAME", "gemini-2.5-flash"))
    elif provider in {"openai", "openai_compatible", "qwen"}:
        from .adapters import OpenAICompatibleVisionAdapter

        adapter = OpenAICompatibleVisionAdapter(model_name=os.environ.get("TPA_MODEL_NAME", "gpt-4o-mini"))

    subject_bbox = None
    if args.subject_bbox:
        subject_bbox = [float(x) for x in args.subject_bbox.split(",")]

    try:
        result = run_photo(
            args.photo,
            assets=assets,
            config=RuntimeConfig(production_mode=not args.fixture_mode),
            adapter=adapter,
            output_dir=args.out,
            retrieval_mode=args.retrieval_mode,
            goal=args.goal,
            scene_tags=args.scene_tags,
            subject_bbox=subject_bbox,
            preview_mirrored=args.mirrored,
            metadata_available=not args.no_metadata,
            render=not args.no_render,
        )
    except FileNotFoundError as exc:
        print(json.dumps({"error": "photo_not_found", "message": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    except (RuntimeError, ValueError) as exc:
        # Most common cause: a provider was selected but its API key is missing.
        print(json.dumps(
            {
                "error": "adapter_failed",
                "provider": provider,
                "adapter": adapter.model_name,
                "message": str(exc),
                "hint": "检查 .env 中的 TPA_VISION_PROVIDER 与对应 API key；"
                        "运行 `travel-photo-assistant env` 查看当前生效配置。",
            },
            ensure_ascii=False, indent=2,
        ))
        return 2

    payload = result.to_dict()
    payload["adapter"] = adapter.model_name
    payload["provider"] = provider
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if result.accepted else 1


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="travel-photo-assistant", description="Travel Photo Assistant v0.1 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("validate-examples", help="Validate example fixtures against Draft 2020-12 schemas and production guards")
    sub.add_parser("demo", help="Run an offline slow+fast dual-loop demonstration")

    eval_p = sub.add_parser("evaluate", help="Run offline contract verification across evaluation cases E01-E16")
    eval_p.add_argument("-o", "--output", default=None, help="Optional path to save JSON report")

    cards_p = sub.add_parser("check-cards", help="Inspect shot card retrieval and constraint gating")
    cards_p.add_argument("--tags", nargs="+", required=True, help="Observed scene/visual tags (e.g. coast leading_line)")
    cards_p.add_argument("--priority", default="人物自然、保留环境", help="User shooting goal priority")

    coord_p = sub.add_parser("transform-coord", help="Transform normalized bbox to preview viewport coordinates")
    coord_p.add_argument("--bbox", required=True, help="Comma-separated xmin,ymin,xmax,ymax")
    coord_p.add_argument("--input-width", type=int, default=1080)
    coord_p.add_argument("--input-height", type=int, default=1440)
    coord_p.add_argument("--viewport-width", type=float, default=540)
    coord_p.add_argument("--viewport-height", type=float, default=960)
    coord_p.add_argument("--fit", choices=["contain", "cover"], default="contain")
    coord_p.add_argument("--mirrored", action="store_true")
    coord_p.add_argument("--direction", default="left")
    coord_p.add_argument("--direction-ref", default="input_image")

    sub.add_parser("env", help="Show effective environment configuration (keys are never printed)")

    photos_p = sub.add_parser("list-photos", help="List local test photos available to the photo pipeline")
    photos_p.add_argument("--dir", default=None, help="Optional alternative photos directory")

    photo_p = sub.add_parser("run-photo", help="Run the full pipeline over a photo and render the result image")
    photo_p.add_argument("--photo", required=True, help="Photo file path (or a bare name inside assets/examples/photos)")
    photo_p.add_argument("--out", default=None, help="Output directory (default: <project>/output)")
    photo_p.add_argument("--goal", default=None,
                         help="User shooting goal priority (default: per-photo value in photos.json)")
    photo_p.add_argument("--scene-tags", nargs="+", default=None, help="Override observed scene tags")
    photo_p.add_argument("--subject-bbox", default=None, help="Override subject bbox: xmin,ymin,xmax,ymax")
    photo_p.add_argument("--retrieval-mode", default="retrieved_cards",
                          choices=["prompt_only", "fixed_cards", "retrieved_cards"])
    photo_p.add_argument("--mirrored", action="store_true", help="Preview is mirrored (affects the drawn arrow)")
    photo_p.add_argument("--no-metadata", action="store_true", help="Simulate missing device lens metadata")
    photo_p.add_argument("--no-render", action="store_true", help="Skip writing the annotated result image")
    photo_p.add_argument("--fixture-mode", action="store_true",
                         help="Disable production isolation guards (for local fixtures only)")

    serve_p = sub.add_parser("serve", help="Start FastAPI HTTP server")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    assets = AssetPaths.default()

    if args.command == "validate-examples":
        return _cmd_validate_examples(assets)
    if args.command == "demo":
        return _cmd_demo(assets)
    if args.command == "evaluate":
        return _cmd_evaluate(assets, args.output)
    if args.command == "check-cards":
        return _cmd_check_cards(assets, args.tags, args.priority)
    if args.command == "transform-coord":
        return _cmd_transform_coord(args)
    if args.command == "env":
        return _cmd_env(assets)
    if args.command == "list-photos":
        return _cmd_list_photos(assets)
    if args.command == "run-photo":
        # Allow a bare filename to resolve against the bundled photos directory.
        photo = Path(args.photo)
        if not photo.is_file() and not photo.is_absolute():
            candidate = assets.photos_dir / photo
            if candidate.is_file():
                photo = candidate
        args.photo = str(photo)
        return _cmd_run_photo(assets, args)
    if args.command == "serve":
        import uvicorn

        uvicorn.run("travel_photo_assistant.api:app", host=args.host, port=args.port)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
