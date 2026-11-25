"""HTTP-based VisionModelAdapter implementations for Gemini and OpenAI-compatible VLMs.

Uses Python standard library (`urllib.request`) with zero third-party SDK lock-in.
Automatically adapts Draft 2020-12 application schemas via `ProviderSchemaAdapter`.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ..models import RuntimeInput, ShotCard
from ..schema_validator import ProviderSchemaAdapter
from .base import VisionModelAdapter


def _default_timeout() -> float:
    """Request timeout in seconds (TPA_REQUEST_TIMEOUT, default 120s).

    A real VLM call over a proxied connection with a full-resolution photo can
    easily exceed 30 seconds.
    """
    try:
        return float(os.environ.get("TPA_REQUEST_TIMEOUT", "120"))
    except ValueError:
        return 120.0


def _detect_mime(raw: bytes) -> str:
    """Detect the real image type from magic bytes (never trust the label)."""
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw.startswith(b"HEIC") or raw[4:8] == b"ftyp":
        return "image/heic"
    return "image/jpeg"


def _load_attachment_base64(att: dict[str, Any]) -> tuple[str, str]:
    mime_type = att.get("mime_type", "image/jpeg")
    if att.get("data_base64"):
        raw = base64.b64decode(str(att["data_base64"]))
        return _detect_mime(raw), str(att["data_base64"])
    if att.get("local_path"):
        raw = Path(att["local_path"]).read_bytes()
        return _detect_mime(raw), base64.b64encode(raw).decode("ascii")
    raise ValueError("Image attachment must provide data_base64 or local_path.")


def _read_json_response(resp: Any, *, provider: str) -> dict[str, Any]:
    """Read a provider response, surfacing the error body on a 4xx/5xx."""
    try:
        raw = resp.read().decode("utf-8")
    except Exception:  # pragma: no cover - defensive
        raise RuntimeError(f"{provider} returned an unreadable response.")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{provider} returned non-JSON: {raw[:500]}") from exc


def _raise_for_status(err: urllib.error.HTTPError, *, provider: str, what: str) -> None:
    """Turn an HTTP error into a RuntimeError that carries the provider's message."""
    try:
        body = err.read().decode("utf-8")
    except Exception:  # pragma: no cover - defensive
        body = ""
    detail = body
    try:
        parsed = json.loads(body)
        detail = parsed.get("error", {}).get("message") or json.dumps(parsed, ensure_ascii=False)
    except (json.JSONDecodeError, AttributeError):
        pass
    raise RuntimeError(f"{provider} {what} failed with HTTP {err.code}: {detail[:800]}")


class OpenAICompatibleVisionAdapter(VisionModelAdapter):
    """Calls OpenAI / Qwen-VL / vLLM chat completions endpoints with structured JSON output."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str = "gpt-4o-mini",
        timeout_sec: float | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.timeout_sec = timeout_sec if timeout_sec is not None else _default_timeout()
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.model_name = model_name

    def _post_chat(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        attachments: list[dict[str, Any]],
        schema_name: str,
        output_schema: dict[str, Any],
        feedback: str | None = None,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("Missing OPENAI_API_KEY for OpenAICompatibleVisionAdapter.")

        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": json.dumps(user_payload, ensure_ascii=False)}
        ]
        if feedback:
            user_content.append(
                {
                    "type": "text",
                    "text": (
                        "服务端校验未通过，请修正后重新输出完整 JSON。\n"
                        "必须逐字使用规定的枚举取值（见 system prompt 中的取值表），不要自创同义词。\n"
                        f"校验错误：\n{feedback}"
                    ),
                }
            )
        for att in attachments:
            if att.get("uri"):
                user_content.append({"type": "image_url", "image_url": {"url": att["uri"]}})
            else:
                mime, b64 = _load_attachment_base64(att)
                user_content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

        adapted_schema = ProviderSchemaAdapter.adapt(output_schema, provider="openai")
        body = {
            "model": self.model_name,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": adapted_schema,
                },
            },
        }
        req = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                parsed = _read_json_response(resp, provider="OpenAI-compatible endpoint")
        except urllib.error.HTTPError as err:
            _raise_for_status(err, provider="OpenAI-compatible endpoint", what="chat completion")
        content = parsed["choices"][0]["message"]["content"]
        return json.loads(content)

    def observe(
        self,
        runtime_input: RuntimeInput,
        system_prompt: str,
        output_schema: dict[str, Any],
        feedback: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "frame": {
                "frame_id": runtime_input.frame.frame_id,
                "scene_version": runtime_input.frame.scene_version,
                "upright_input_width_px": runtime_input.frame.upright_input_width_px,
                "upright_input_height_px": runtime_input.frame.upright_input_height_px,
                "preview_mirrored": runtime_input.frame.preview_mirrored,
            },
            "user_goal": runtime_input.to_dict()["user_goal"],
            "camera_capabilities": runtime_input.to_dict()["camera_capabilities"],
            "movement_constraints": runtime_input.to_dict()["movement_constraints"],
            "local_vision_hints": runtime_input.local_vision_hints,
        }
        return self._post_chat(
            system_prompt,
            payload,
            runtime_input.frame.image_attachments,
            "scene_observation",
            output_schema,
            feedback,
        )

    def plan(
        self,
        runtime_input: RuntimeInput,
        observation: dict[str, Any],
        shot_cards: list[ShotCard],
        system_prompt: str,
        output_schema: dict[str, Any],
        feedback: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "runtime_input": runtime_input.to_dict(),
            "scene_observation": observation,
            "shot_cards": [c.to_dict() for c in shot_cards],
        }
        return self._post_chat(
            system_prompt,
            payload,
            runtime_input.frame.image_attachments,
            "shot_plan",
            output_schema,
            feedback,
        )


class GeminiVisionAdapter(VisionModelAdapter):
    """Calls Google Gemini REST API with adapted responseSchema."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_name: str = "gemini-2.5-flash",
        timeout_sec: float | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.timeout_sec = timeout_sec if timeout_sec is not None else _default_timeout()
        self.model_name = model_name

    def _generate(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        attachments: list[dict[str, Any]],
        output_schema: dict[str, Any],
        feedback: str | None = None,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("Missing GEMINI_API_KEY for GeminiVisionAdapter.")

        parts: list[dict[str, Any]] = [
            {"text": json.dumps(user_payload, ensure_ascii=False)}
        ]
        if feedback:
            parts.append(
                {
                    "text": (
                        "服务端校验未通过，请修正后重新输出完整 JSON。\n"
                        "必须逐字使用规定的枚举取值（见 system prompt 中的取值表），不要自创同义词。\n"
                        f"校验错误：\n{feedback}"
                    )
                }
            )
        for att in attachments:
            mime, b64 = _load_attachment_base64(att)
            parts.append({"inline_data": {"mime_type": mime, "data": b64}})

        adapted_schema = ProviderSchemaAdapter.adapt(output_schema, provider="gemini")
        body = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
                "responseSchema": adapted_schema,
            },
        }
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:generateContent?key={self.api_key}"
        )
        req = urllib.request.Request(
            url=url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                parsed = _read_json_response(resp, provider="Gemini")
        except urllib.error.HTTPError as err:
            _raise_for_status(err, provider="Gemini", what="generateContent")
        text = parsed["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text)

    def observe(
        self,
        runtime_input: RuntimeInput,
        system_prompt: str,
        output_schema: dict[str, Any],
        feedback: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "frame": {
                "frame_id": runtime_input.frame.frame_id,
                "scene_version": runtime_input.frame.scene_version,
                "upright_input_width_px": runtime_input.frame.upright_input_width_px,
                "upright_input_height_px": runtime_input.frame.upright_input_height_px,
                "preview_mirrored": runtime_input.frame.preview_mirrored,
            },
            "user_goal": runtime_input.to_dict()["user_goal"],
            "camera_capabilities": runtime_input.to_dict()["camera_capabilities"],
            "movement_constraints": runtime_input.to_dict()["movement_constraints"],
            "local_vision_hints": runtime_input.local_vision_hints,
        }
        return self._generate(
            system_prompt, payload, runtime_input.frame.image_attachments, output_schema, feedback
        )

    def plan(
        self,
        runtime_input: RuntimeInput,
        observation: dict[str, Any],
        shot_cards: list[ShotCard],
        system_prompt: str,
        output_schema: dict[str, Any],
        feedback: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "runtime_input": runtime_input.to_dict(),
            "scene_observation": observation,
            "shot_cards": [c.to_dict() for c in shot_cards],
        }
        return self._generate(
            system_prompt, payload, runtime_input.frame.image_attachments, output_schema, feedback
        )
