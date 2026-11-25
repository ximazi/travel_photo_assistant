"""Unified VisionModelAdapter interface (integration_notes.txt Section VII)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..models import RuntimeInput, ShotCard


class VisionModelAdapter(ABC):
    """Provider-agnostic interface separating scene observation and shot planning."""

    model_name: str = "unspecified"

    @abstractmethod
    def observe(
        self,
        runtime_input: RuntimeInput,
        system_prompt: str,
        output_schema: dict[str, Any],
        feedback: str | None = None,
    ) -> dict[str, Any]:
        """Stage 1: Observe factual elements from the current frame.

        `feedback` carries the server-side validation errors from a previous
        attempt so the model can correct its output instead of being retried
        blindly.
        """

    @abstractmethod
    def plan(
        self,
        runtime_input: RuntimeInput,
        observation: dict[str, Any],
        shot_cards: list[ShotCard],
        system_prompt: str,
        output_schema: dict[str, Any],
        feedback: str | None = None,
    ) -> dict[str, Any]:
        """Stage 2: Generate an executable shot plan grounded in observation and capabilities."""
