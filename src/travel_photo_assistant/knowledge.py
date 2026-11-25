"""Shot card repository and metadata/constraint-based retriever (Modes A / B / C).

Implements README.txt Section IV, integration_notes.txt Section VI, and
evaluation_protocol.txt Section VII:
- Metadata filtering comes before semantic/vector search.
- Retrieval hit does NOT force adoption; visual requirements and not_applicable_when
  constraints must be explicitly checked.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .config import AssetPaths
from .models import RuntimeInput, ShotCard


class RetrievalMode(str, Enum):
    """Experimental modes defined in README.txt Section IV and evaluation_protocol.txt Section II."""

    PROMPT_ONLY = "prompt_only"        # Mode A
    FIXED_CARDS = "fixed_cards"        # Mode B
    RETRIEVED_CARDS = "retrieved_cards"  # Mode C


@dataclass
class CardEvaluation:
    """Audit record explaining why a shot card was selected or rejected."""

    card_id: str
    accepted: bool
    score: float
    matched_scene_tags: list[str] = field(default_factory=list)
    matched_goal_tags: list[str] = field(default_factory=list)
    matched_visual_tags: list[str] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    """Output of card retrieval including latency and audit trace."""

    mode: RetrievalMode
    selected_cards: list[ShotCard]
    evaluations: list[CardEvaluation]
    latency_ms: float


class ShotCardRepository:
    """Loads and indexes authored shot cards from knowledge/shot_cards.json."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or AssetPaths.default().shot_cards
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.schema_version: str = raw.get("schema_version", "0.1.0")
        self.status: str = raw.get("status", "draft_unvalidated")
        self.cards: list[ShotCard] = [ShotCard.from_dict(c) for c in raw.get("cards", [])]
        self._by_id: dict[str, ShotCard] = {c.id: c for c in self.cards}

    def get(self, card_id: str) -> ShotCard | None:
        return self._by_id.get(card_id)

    def all_cards(self) -> list[ShotCard]:
        return list(self.cards)


class ShotCardRetriever:
    """Filters shot cards by scene_tags, goal_tags, visual evidence, and hard constraints."""

    # Default curated subset for Mode B (fixed small set of general cards)
    DEFAULT_FIXED_CARD_IDS = (
        "COAST_LINE_ENV_01",
        "HEAD_BACKGROUND_MERGE_06",
        "ARCH_SYMMETRY_07",
    )

    def __init__(self, repository: ShotCardRepository) -> None:
        self.repository = repository

    def retrieve(
        self,
        observation: dict[str, Any],
        runtime_input: RuntimeInput,
        *,
        mode: RetrievalMode | str = RetrievalMode.RETRIEVED_CARDS,
        max_cards: int = 3,
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        resolved_mode = RetrievalMode(mode) if isinstance(mode, str) else mode

        if resolved_mode == RetrievalMode.PROMPT_ONLY:
            return RetrievalResult(
                mode=resolved_mode,
                selected_cards=[],
                evaluations=[],
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
            )

        if resolved_mode == RetrievalMode.FIXED_CARDS:
            fixed = [c for c in self.repository.all_cards() if c.id in self.DEFAULT_FIXED_CARD_IDS]
            evals = [CardEvaluation(card_id=c.id, accepted=True, score=1.0) for c in fixed]
            return RetrievalResult(
                mode=resolved_mode,
                selected_cards=fixed[:max_cards],
                evaluations=evals,
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
            )

        # Mode C: metadata + visual evidence + constraint gating
        obs_tags = set(observation.get("scene_tags", []))
        inferred_visual_tags = self._extract_visual_evidence_tags(observation)
        # A scene label is not proof that a composition anchor exists.
        # Restrict leading_line to geometry attached to a structured anchor.
        all_visual_tags = (obs_tags - {"leading_line"}) | inferred_visual_tags
        goal_tags = set(self._infer_goal_tags(runtime_input))

        evaluations: list[CardEvaluation] = []
        accepted_pairs: list[tuple[float, ShotCard]] = []

        for card in self.repository.all_cards():
            ev = self._evaluate_card(card, observation, runtime_input, obs_tags, all_visual_tags, goal_tags)
            evaluations.append(ev)
            if ev.accepted:
                accepted_pairs.append((ev.score, card))

        accepted_pairs.sort(key=lambda pair: (-pair[0], pair[1].id))
        selected = [card for _, card in accepted_pairs[:max_cards]]

        return RetrievalResult(
            mode=resolved_mode,
            selected_cards=selected,
            evaluations=evaluations,
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
        )

    @staticmethod
    def _extract_visual_evidence_tags(observation: dict[str, Any]) -> set[str]:
        """Derive verifiable visual tags from structured anchors, subject, issues, and light."""
        tags: set[str] = set()
        if observation.get("selected_subject") is not None:
            tags.add("single_person")

        for anchor in observation.get("anchors", []):
            kind = anchor.get("kind")
            if kind == "leading_line" and ShotCardRetriever._has_anchor_geometry(anchor):
                tags.add("leading_line")
            elif kind == "landmark":
                tags.add("landmark")
            elif kind == "obstacle":
                tags.add("obstacle")
            desc = str(anchor.get("description", ""))
            if "对称" in desc or "拱门" in desc or "中轴" in desc:
                tags.add("symmetry")

        for issue in observation.get("issues", []):
            desc = str(issue.get("description", ""))
            if any(k in desc for k in ("电线杆", "路牌", "树干", "头部重合", "头部与", "杆状")):
                tags.add("head_background_merge")
            if any(k in desc for k in ("游客", "杂物", "杂乱", "拥挤")):
                tags.add("background_clutter")
            if any(k in desc for k in ("斑驳", "光斑", "阴影不均")):
                tags.add("dappled_face_light")
            if any(k in desc for k in ("弱光", "夜景", "模糊", "暗")):
                tags.add("low_light")

        light_app = str(observation.get("light", {}).get("appearance", ""))
        if any(k in light_app for k in ("弱光", "夜景", "偏暗")):
            tags.add("low_light")
        if any(k in light_app for k in ("斑驳", "光斑")):
            tags.add("dappled_face_light")

        return tags

    @staticmethod
    def _has_anchor_geometry(anchor: dict[str, Any]) -> bool:
        # The controller already checks schema and bbox ordering. Recheck the
        # shape here because the retriever may also be called directly.
        def unit_number(value: Any) -> bool:
            return type(value) in (int, float) and 0.0 <= value <= 1.0

        bbox = anchor.get("bbox")
        if (
            isinstance(bbox, list) and len(bbox) == 4
            and all(unit_number(value) for value in bbox)
            and bbox[0] < bbox[2] and bbox[1] < bbox[3]
        ):
            return True
        points = anchor.get("points")
        if not isinstance(points, list):
            return False
        return len({
            tuple(point) for point in points
            if isinstance(point, list) and len(point) == 2
            and all(unit_number(value) for value in point)
        }) >= 2

    @staticmethod
    def _infer_goal_tags(runtime_input: RuntimeInput) -> list[str]:
        tags = list(runtime_input.user_goal.goal_tags)
        text = f"{runtime_input.user_goal.priority} {runtime_input.user_goal.style}"
        if any(k in text for k in ("环境", "海岸", "旅行", "风景", "并重", "平衡")):
            tags.append("person_environment_balance")
        if any(k in text for k in ("表情", "近景", "特写", "人物优先", "突出人物")):
            tags.append("person_first")
        if any(k in text for k in ("地标", "山峰", "建筑")):
            tags.append("landmark_and_person")
        if any(k in text for k in ("自然",)):
            tags.append("natural")
        if any(k in text for k in ("端正", "对称", "建筑感")):
            tags.append("formal")
        if any(k in text for k in ("夜景", "氛围")):
            tags.append("night_atmosphere")
        return list(dict.fromkeys(tags))

    @staticmethod
    def _evaluate_card(
        card: ShotCard,
        observation: dict[str, Any],
        runtime_input: RuntimeInput,
        obs_tags: set[str],
        all_visual_tags: set[str],
        goal_tags: set[str],
    ) -> CardEvaluation:
        reasons: list[str] = []

        if observation.get("status") != "observed":
            reasons.append(f"observation status is '{observation.get('status')}', not 'observed'")

        matched_scene = sorted(set(card.scene_tags) & obs_tags)
        matched_goal = sorted(set(card.goal_tags) & goal_tags)
        matched_visual = sorted(set(card.required_visual_tags) & all_visual_tags)
        missing_visual = sorted(set(card.required_visual_tags) - all_visual_tags)

        # Scene match check (required unless it's a cross-scene issue card like HEAD_BACKGROUND_MERGE_06)
        if not matched_scene and card.id != "HEAD_BACKGROUND_MERGE_06":
            reasons.append(f"no scene_tags overlap (card={card.scene_tags}, obs={sorted(obs_tags)})")

        # Hard visual prerequisite check (integration_notes.txt Section VI)
        if missing_visual:
            reasons.append(f"missing required_visual_tags: {missing_visual}")

        # Goal conflict check (e.g., user wants close-up face only -> reject environmental wide cards)
        priority_text = runtime_input.user_goal.priority
        if any(k in priority_text for k in ("只要脸部特写", "近景", "特写")) and card.id in {
            "COAST_LINE_ENV_01",
            "FOREST_PATH_DEPTH_03",
            "STREET_DEPTH_05",
        }:
            reasons.append("conflicts with user close-up / person-first goal")

        # Movement constraint check for HEAD_BACKGROUND_MERGE_06
        if card.id == "HEAD_BACKGROUND_MERGE_06":
            mc = runtime_input.movement_constraints
            if mc.photographer_translation in {"forbidden", "tripod_fixed"} and mc.subject_translation in {
                "forbidden",
                "not_requested",
            }:
                reasons.append("both photographer and subject cannot translate to create parallax")

        accepted = len(reasons) == 0
        score = 0.0
        if accepted:
            score = (
                2.0 * len(matched_visual)
                + 1.5 * len(matched_scene)
                + 1.0 * len(matched_goal)
            )
            # Prioritize high-priority issue resolution cards (e.g., head overlap / dappled light)
            if card.id in {"HEAD_BACKGROUND_MERGE_06", "FOREST_DAPPLED_LIGHT_04", "LOW_LIGHT_STATIC_09"}:
                score += 1.5

        return CardEvaluation(
            card_id=card.id,
            accepted=accepted,
            score=round(score, 3),
            matched_scene_tags=matched_scene,
            matched_goal_tags=matched_goal,
            matched_visual_tags=matched_visual,
            rejection_reasons=reasons,
        )
