"""JSON Schema Draft 2020-12 validator and vendor structured-output schema adapter."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .config import AssetPaths


class SchemaValidator:
    """Validates scene_observation and shot_plan payloads against Draft 2020-12 schemas."""

    def __init__(self, assets: AssetPaths | None = None, production_mode: bool = True) -> None:
        self.assets = assets or AssetPaths.default()
        self.production_mode = production_mode
        self._obs_schema = self._load_schema(self.assets.observation_schema)
        self._plan_schema = self._load_schema(self.assets.shot_plan_schema)

        if production_mode:
            self._obs_schema = self._strip_synthetic_fixture(self._obs_schema)

        self._obs_validator = Draft202012Validator(self._obs_schema)
        self._plan_validator = Draft202012Validator(self._plan_schema)

    @staticmethod
    def _load_schema(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _strip_synthetic_fixture(schema: dict[str, Any]) -> dict[str, Any]:
        """Remove synthetic_fixture enum in production mode per integration_notes.txt Section I."""
        cloned = copy.deepcopy(schema)
        try:
            enum_list = cloned["properties"]["facts"]["items"]["properties"]["source"]["enum"]
            if "synthetic_fixture" in enum_list:
                enum_list.remove("synthetic_fixture")
        except (KeyError, TypeError):
            pass
        return cloned

    @property
    def observation_schema(self) -> dict[str, Any]:
        return copy.deepcopy(self._obs_schema)

    @property
    def shot_plan_schema(self) -> dict[str, Any]:
        return copy.deepcopy(self._plan_schema)

    def validate_observation(self, payload: dict[str, Any]) -> list[str]:
        """Return list of human-readable JSON Schema error strings (empty if valid)."""
        errors = sorted(self._obs_validator.iter_errors(payload), key=lambda e: list(e.path))
        return [self._format_error(e) for e in errors]

    def validate_shot_plan(self, payload: dict[str, Any]) -> list[str]:
        """Return list of human-readable JSON Schema error strings (empty if valid)."""
        errors = sorted(self._plan_validator.iter_errors(payload), key=lambda e: list(e.path))
        return [self._format_error(e) for e in errors]

    @staticmethod
    def _format_error(err: Any) -> str:
        path = ".".join(str(p) for p in err.path) if err.path else "<root>"
        return f"{path}: {err.message}"


class ProviderSchemaAdapter:
    """Adapts Draft 2020-12 application schemas to model vendor structured-output subsets.

    Addresses integration_notes.txt Section II: different VLM vendors have different
    support for `$schema`, `$id`, `const`, `anyOf` nullability, and array bounds.
    """

    @classmethod
    def adapt(cls, schema: dict[str, Any], provider: str) -> dict[str, Any]:
        provider_key = provider.lower().strip()
        cloned = copy.deepcopy(schema)
        cloned.pop("$schema", None)
        cloned.pop("$id", None)

        if provider_key == "gemini":
            return cls._adapt_gemini(cloned)
        if provider_key in {"openai", "openai_compatible", "qwen"}:
            return cls._adapt_openai(cloned)
        raise ValueError(f"Unsupported schema provider: {provider}")

    @classmethod
    def _adapt_gemini(cls, node: Any) -> Any:
        """Convert Draft 2020-12 features into Gemini responseSchema compatible subset.

        Gemini additionally rejects schemas whose "state space" is too large:
        "The specified schema produces a constraint that has too many states for
        serving". The usual triggers are array length limits (especially nested
        ones) and numeric minimum/maximum bounds.

        Those constraints are dropped from the *vendor hint only* — the server
        keeps validating against the full Draft 2020-12 schema, so nothing is
        weakened: the vendor schema only has to hint at the shape.
        """
        if isinstance(node, list):
            return [cls._adapt_gemini(item) for item in node]
        if not isinstance(node, dict):
            return node

        result: dict[str, Any] = {}

        # Convert `const` -> single-value `enum` with explicit string type
        if "const" in node:
            val = node["const"]
            result["type"] = "string" if isinstance(val, str) else "number"
            result["enum"] = [val]
            return result

        # Convert `anyOf: [T, {"type": "null"}]` -> T with `nullable: true`
        if "anyOf" in node and isinstance(node["anyOf"], list):
            non_null = [s for s in node["anyOf"] if not (isinstance(s, dict) and s.get("type") == "null")]
            has_null = len(non_null) < len(node["anyOf"])
            if len(non_null) == 1:
                adapted = cls._adapt_gemini(non_null[0])
                if isinstance(adapted, dict) and has_null:
                    adapted["nullable"] = True
                return adapted

        for key, value in node.items():
            # `properties` maps property NAMES to schemas. It must be walked as a
            # map, not as a schema node — otherwise a property literally named
            # "title" (or "$id", "additionalProperties", ...) is dropped and the
            # vendor rejects the schema with "required[0]: property is not defined".
            if key == "properties" and isinstance(value, dict):
                result["properties"] = {
                    prop_name: cls._adapt_gemini(prop_schema)
                    for prop_name, prop_schema in value.items()
                }
                continue
            # Array length limits and numeric bounds blow up Gemini's state space.
            if key in {"minItems", "maxItems", "minimum", "maximum", "exclusiveMinimum"}:
                continue
            if key in {"additionalProperties", "$schema", "$id", "title"}:
                continue
            result[key] = cls._adapt_gemini(value)
        return result

    @classmethod
    def _adapt_openai(cls, node: Any) -> Any:
        """Convert Draft 2020-12 features for OpenAI strict structured outputs.

        OpenAI strict mode rejects `anyOf`, so `anyOf: [T, null]` is flattened to
        `T` (the server still validates nullability itself), and every property
        is listed in `required` as strict mode demands.
        """
        if isinstance(node, list):
            return [cls._adapt_openai(item) for item in node]
        if not isinstance(node, dict):
            return node

        result: dict[str, Any] = {}
        if "const" in node:
            val = node["const"]
            result["type"] = "string" if isinstance(val, str) else "number"
            result["enum"] = [val]
            return result

        # OpenAI strict mode has no `anyOf`: flatten the nullable union.
        if "anyOf" in node and isinstance(node["anyOf"], list):
            non_null = [s for s in node["anyOf"] if not (isinstance(s, dict) and s.get("type") == "null")]
            if len(non_null) == 1:
                return cls._adapt_openai(non_null[0])

        for key, value in node.items():
            # Same map-vs-schema distinction as the Gemini adapter.
            if key == "properties" and isinstance(value, dict):
                result["properties"] = {
                    prop_name: cls._adapt_openai(prop_schema)
                    for prop_name, prop_schema in value.items()
                }
                continue
            if key in {"$schema", "$id", "anyOf"}:
                continue
            result[key] = cls._adapt_openai(value)
        if result.get("type") == "object":
            if "additionalProperties" not in result:
                result["additionalProperties"] = False
            props = result.get("properties")
            if isinstance(props, dict):
                # Strict mode requires every declared property to be required.
                result["required"] = list(props.keys())
        return result
