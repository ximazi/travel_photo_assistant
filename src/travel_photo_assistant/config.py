"""Configuration and asset path management."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _default_project_root() -> Path:
    # src/travel_photo_assistant/config.py -> project root
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AssetPaths:
    """Resolved paths for prompts, schemas, shot cards, examples, and evaluation assets."""

    root: Path

    @classmethod
    def default(cls, root: str | Path | None = None) -> "AssetPaths":
        base = Path(root).resolve() if root is not None else _default_project_root()
        assets_dir = base / "assets"
        if assets_dir.exists():
            return cls(root=assets_dir)
        return cls(root=base)

    @property
    def observer_prompt(self) -> Path:
        return self.root / "prompts" / "observer.system.txt"

    @property
    def planner_prompt(self) -> Path:
        return self.root / "prompts" / "planner.system.txt"

    @property
    def observation_schema(self) -> Path:
        return self.root / "schemas" / "scene_observation.schema.json"

    @property
    def shot_plan_schema(self) -> Path:
        return self.root / "schemas" / "shot_plan.schema.json"

    @property
    def shot_cards(self) -> Path:
        return self.root / "knowledge" / "shot_cards.json"

    @property
    def examples_dir(self) -> Path:
        return self.root / "examples"

    @property
    def photos_dir(self) -> Path:
        """Local test photos used by the offline photo pipeline."""
        return self.root / "examples" / "photos"

    @property
    def project_root(self) -> Path:
        """Repository root (the parent of the assets directory)."""
        return self.root.parent if self.root.name == "assets" else self.root

    @property
    def evaluation_cases(self) -> Path:
        return self.root / "evaluation" / "evaluation_cases.json"

    @property
    def evaluation_protocol(self) -> Path:
        return self.root / "evaluation" / "evaluation_protocol.txt"


@dataclass
class RuntimeConfig:
    """Runtime execution configuration for the Travel Photo Assistant."""

    production_mode: bool = True
    retrieval_mode: str = "retrieved_cards"  # prompt_only | fixed_cards | retrieved_cards
    max_retrieved_cards: int = 3
    jitter_deadband: float = 0.035  # Normalized coordinate tolerance for fast-loop anti-oscillation
    stable_frames_required: int = 2  # Consecutive frames inside target range before completion
    max_stale_drop_log: int = 50
    # Real vendors do not reliably enforce enum values in their response schemas.
    # Retry the model once with the server-side validation errors as feedback.
    retry_on_validation_error: bool = True
    max_validation_retries: int = 1
