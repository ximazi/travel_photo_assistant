"""FastAPI HTTP service exposing the Travel Photo Assistant dual-loop engine."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .adapters import (
    GeminiVisionAdapter,
    OpenAICompatibleVisionAdapter,
    RuleBasedOfflineVisionAdapter,
    VisionModelAdapter,
)
from .config import AssetPaths, RuntimeConfig
from .controller import TravelPhotoController
from .coordinates import PreviewMapper, PreviewTransformConfig
from .env_loader import env_status, load_dotenv
from .models import LocalVisionSnapshot, RuntimeInput
from .photo_pipeline import run_photo

load_dotenv()


def _build_default_adapter() -> VisionModelAdapter:
    provider = os.environ.get("TPA_VISION_PROVIDER", "offline").lower().strip()
    if provider == "gemini":
        return GeminiVisionAdapter(model_name=os.environ.get("TPA_MODEL_NAME", "gemini-2.5-flash"))
    if provider in {"openai", "openai_compatible", "qwen"}:
        return OpenAICompatibleVisionAdapter(model_name=os.environ.get("TPA_MODEL_NAME", "gpt-4o-mini"))
    return RuleBasedOfflineVisionAdapter()


class FastTickRequest(BaseModel):
    frame_id: str
    scene_version: int
    transform_version: int = 1
    current_lens_id: str | None = None
    orientation: str = "portrait"
    crop_signature: str = "contain:3:4"
    tracked_subject_id: str | None = None
    subject_bbox: list[float] | None = None
    tracking_ok: bool = True
    camera_roll_deg: float | None = None
    light_materially_changed: bool = False
    user_goal_version: int = 1
    manual_confirmations: list[str] = []
    vlm_verified_checks: list[str] = []


class CoordTransformRequest(BaseModel):
    input_width_px: int = 1080
    input_height_px: int = 1440
    viewport_width_px: float
    viewport_height_px: float
    viewport_fit: str = "contain"
    preview_mirrored: bool = False
    transform_version: int = 1
    bbox: list[float]
    direction: str = "none"
    direction_reference: str = "input_image"


class PhotoRunRequest(BaseModel):
    photo: str
    out: str | None = None
    goal: str | None = None
    retrieval_mode: str = "retrieved_cards"
    scene_tags: list[str] | None = None
    subject_bbox: list[float] | None = None
    preview_mirrored: bool = False
    no_metadata: bool = False
    no_render: bool = False


def create_app(
    *,
    adapter: VisionModelAdapter | None = None,
    production_mode: bool = True,
) -> FastAPI:
    app = FastAPI(title="Travel Photo Assistant API", version="0.1.0")
    assets = AssetPaths.default()
    controller = TravelPhotoController(
        adapter=adapter or _build_default_adapter(),
        assets=assets,
        config=RuntimeConfig(production_mode=production_mode),
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": "0.1.0",
            "provider": controller.adapter.model_name,
            "production_mode": controller.config.production_mode,
            "env": env_status(),
        }

    @app.get("/v1/cards")
    def list_cards() -> dict[str, Any]:
        cards = [c.to_dict() for c in controller.card_repo.all_cards()]
        return {"count": len(cards), "status": controller.card_repo.status, "cards": cards}

    @app.post("/v1/session/reset")
    def reset_session() -> dict[str, Any]:
        controller.reset_session()
        return {"status": "reset"}

    @app.get("/v1/session/state")
    def get_session_state() -> dict[str, Any]:
        return asdict(controller.state)

    @app.post("/v1/session/slow-step")
    def slow_step(payload: dict[str, Any], retrieval_mode: str = "retrieved_cards") -> dict[str, Any]:
        try:
            runtime_input = RuntimeInput.from_dict(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid runtime_input payload: {exc}") from exc

        res = controller.run_slow_loop(runtime_input, retrieval_mode=retrieval_mode)
        return {
            "accepted": res.accepted,
            "dropped_as_stale": res.dropped_as_stale,
            "downgraded_to_fallback": res.downgraded_to_fallback,
            "reason": res.reason,
            "observation": res.observation,
            "plan": res.plan,
            "selected_card_ids": [c.id for c in res.retrieval.selected_cards] if res.retrieval else [],
            "validation_errors": [
                asdict(i)
                for v in (res.input_validation, res.observation_validation, res.plan_validation)
                if v is not None
                for i in v.errors
            ],
        }

    @app.post("/v1/session/fast-tick")
    def fast_tick(req: FastTickRequest) -> dict[str, Any]:
        snap = LocalVisionSnapshot(**req.model_dump())
        res = controller.run_fast_tick(snap)
        return asdict(res)

    @app.post("/v1/transform")
    def transform_coord(req: CoordTransformRequest) -> dict[str, Any]:
        mapper = PreviewMapper(
            PreviewTransformConfig(
                input_width_px=req.input_width_px,
                input_height_px=req.input_height_px,
                viewport_width_px=req.viewport_width_px,
                viewport_height_px=req.viewport_height_px,
                viewport_fit=req.viewport_fit,
                preview_mirrored=req.preview_mirrored,
                transform_version=req.transform_version,
            )
        )
        rect = mapper.map_bbox(req.bbox, transform_version=req.transform_version)
        mapped_dir = mapper.map_horizontal_direction(req.direction, req.direction_reference)
        return {
            "viewport_rect": asdict(rect),
            "mapped_direction": mapped_dir,
            "direction_reference": req.direction_reference,
            "preview_mirrored": req.preview_mirrored,
        }

    @app.get("/v1/photos")
    def list_photos() -> dict[str, Any]:
        """List the local test photos available to the photo pipeline."""
        photos_dir = assets.photos_dir
        entries = []
        if photos_dir.is_dir():
            for p in sorted(photos_dir.iterdir()):
                if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".heic"}:
                    continue
                entries.append({"name": p.name, "path": str(p), "bytes": p.stat().st_size})
        return {
            "photos_dir": str(photos_dir),
            "count": len(entries),
            "photos": entries,
            "note": "合成测试图，仅用于跑通链路，不能验证真实视觉准确率。",
        }

    @app.post("/v1/photos/run")
    def run_photo_endpoint(req: PhotoRunRequest) -> dict[str, Any]:
        """Run the dual loop over a photo on disk and render the annotated result."""
        # A bare filename resolves against assets/examples/photos, same as the CLI.
        photo = Path(req.photo)
        if not photo.is_file() and not photo.is_absolute():
            candidate = assets.photos_dir / photo
            if candidate.is_file():
                photo = candidate
        try:
            result = run_photo(
                photo,
                assets=assets,
                config=RuntimeConfig(production_mode=controller.config.production_mode),
                adapter=controller.adapter,
                output_dir=req.out,
                retrieval_mode=req.retrieval_mode,
                goal=req.goal,
                scene_tags=req.scene_tags,
                subject_bbox=req.subject_bbox,
                preview_mirrored=req.preview_mirrored,
                metadata_available=not req.no_metadata,
                render=not req.no_render,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return result.to_dict()

    return app


app = create_app()
