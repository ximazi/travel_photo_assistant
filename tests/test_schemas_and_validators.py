"""Tests for Draft 2020-12 schema validation, vendor adaptation, and business rules."""

from __future__ import annotations

import copy
import json
import unittest

from travel_photo_assistant.config import AssetPaths
from travel_photo_assistant.models import RuntimeInput
from travel_photo_assistant.schema_validator import ProviderSchemaAdapter, SchemaValidator
from travel_photo_assistant.validators import BusinessRuleValidator


class TestSchemasAndValidators(unittest.TestCase):
    def setUp(self) -> None:
        self.assets = AssetPaths.default()
        self.fixture_schema = SchemaValidator(self.assets, production_mode=False)
        self.prod_schema = SchemaValidator(self.assets, production_mode=True)
        self.prod_validator = BusinessRuleValidator(self.prod_schema, production_mode=True)

        ex = self.assets.examples_dir
        self.ex_runtime = json.loads((ex / "runtime_input.json").read_text(encoding="utf-8"))
        self.ex_obs = json.loads((ex / "scene_observation.json").read_text(encoding="utf-8"))
        self.ex_plan = json.loads((ex / "shot_plan.json").read_text(encoding="utf-8"))

    def test_example_fixtures_pass_draft_2020_12_schema(self) -> None:
        self.assertEqual(self.fixture_schema.validate_observation(self.ex_obs), [])
        self.assertEqual(self.fixture_schema.validate_shot_plan(self.ex_plan), [])

    def test_production_mode_rejects_example_and_synthetic_fixture(self) -> None:
        rt = RuntimeInput.from_dict(self.ex_runtime)
        in_val = self.prod_validator.validate_runtime_input(rt)
        self.assertFalse(in_val.is_valid)
        codes = {e.code for e in in_val.errors}
        self.assertIn("PRODUCTION_REJECTS_EXAMPLE", codes)
        self.assertIn("PRODUCTION_REJECTS_SYNTHETIC_FIXTURE", codes)

        obs_val = self.prod_validator.validate_observation(self.ex_obs, rt)
        self.assertFalse(obs_val.is_valid)
        obs_codes = {e.code for e in obs_val.errors}
        self.assertIn("PRODUCTION_REJECTS_SYNTHETIC_FIXTURE", obs_codes)
        self.assertIn("MISSING_IMAGE_MUST_BE_INSUFFICIENT", obs_codes)

    def test_provider_schema_adapter_gemini_and_openai(self) -> None:
        gemini_schema = ProviderSchemaAdapter.adapt(self.prod_schema.observation_schema, "gemini")
        self.assertNotIn("$schema", gemini_schema)
        self.assertNotIn("$id", gemini_schema)
        self.assertEqual(gemini_schema["properties"]["schema_version"]["enum"], ["0.1.0"])
        self.assertTrue(gemini_schema["properties"]["selected_subject"]["nullable"])

        openai_schema = ProviderSchemaAdapter.adapt(self.prod_schema.shot_plan_schema, "openai")
        self.assertNotIn("$schema", openai_schema)
        self.assertEqual(openai_schema["properties"]["schema_version"]["enum"], ["0.1.0"])
        self.assertFalse(openai_schema["additionalProperties"])

    def test_bbox_ordering_and_dangling_references_rejected(self) -> None:
        rt_dict = copy.deepcopy(self.ex_runtime)
        rt_dict["example_only"] = False
        rt_dict["frame"]["image_attachments"] = [{"mime_type": "image/jpeg", "uri": "memory://a.jpg"}]
        for lens in rt_dict["camera_capabilities"]["available_lenses"]:
            lens["equivalent_focal_length_source"] = "camera_api"
        rt = RuntimeInput.from_dict(rt_dict)

        bad_obs = copy.deepcopy(self.ex_obs)
        for f in bad_obs["facts"]:
            f["source"] = "vlm_visual"
        bad_obs["selected_subject"]["bbox"] = [0.7, 0.3, 0.4, 0.9]  # xmin > xmax!
        bad_obs["selected_subject"]["fact_ids"] = ["nonexistent_fact"]

        obs_val = self.prod_validator.validate_observation(bad_obs, rt)
        self.assertFalse(obs_val.is_valid)
        codes = {e.code for e in obs_val.errors}
        self.assertIn("INVALID_BBOX_BOUNDS", codes)
        self.assertIn("DANGLING_FACT_REF", codes)

    def test_pseudo_precise_distance_and_parallax_violation_rejected(self) -> None:
        rt_dict = copy.deepcopy(self.ex_runtime)
        rt_dict["example_only"] = False
        rt_dict["frame"]["image_attachments"] = [{"mime_type": "image/jpeg", "uri": "memory://a.jpg"}]
        for lens in rt_dict["camera_capabilities"]["available_lenses"]:
            lens["equivalent_focal_length_source"] = "camera_api"
        rt = RuntimeInput.from_dict(rt_dict)

        obs = copy.deepcopy(self.ex_obs)
        for f in obs["facts"]:
            f["source"] = "vlm_visual"

        bad_plan = copy.deepcopy(self.ex_plan)
        bad_plan["actions"][0]["instruction"] = "请拍摄者后退 1.5 米并向右转手机。"
        bad_plan["actions"][0]["completion_check"]["type"] = "head_obstacle_separated"

        plan_val = self.prod_validator.validate_shot_plan(bad_plan, obs, rt, [])
        self.assertFalse(plan_val.is_valid)
        codes = {e.code for e in plan_val.errors}
        self.assertIn("PSEUDO_PRECISE_DISTANCE", codes)
        self.assertIn("INVALID_PARALLAX_CLAIM", codes)
        self.assertIn("DANGLING_CARD_REF", codes)


class TestProviderSchemaAdapterProperties(unittest.TestCase):
    """Regression: a property literally named 'title' must survive adaptation.

    Gemini rejects the schema with "required[0]: property is not defined" when
    the adapter treats the `properties` map as a schema node and drops the
    schema-level 'title' keyword.
    """

    def _plan_schema(self):
        return json.loads(
            (AssetPaths.default().shot_plan_schema).read_text(encoding="utf-8")
        )

    def test_gemini_keeps_properties_named_title(self):
        adapted = ProviderSchemaAdapter.adapt(self._plan_schema(), provider="gemini")
        primary = adapted["properties"]["primary_plan"]
        self.assertIn("title", primary["properties"])
        self.assertIn("title", primary["required"])
        alternative = adapted["properties"]["alternative"]
        self.assertIn("title", alternative["properties"])
        self.assertIn("title", alternative["required"])

    def test_openai_keeps_properties_named_title(self):
        adapted = ProviderSchemaAdapter.adapt(self._plan_schema(), provider="openai")
        primary = adapted["properties"]["primary_plan"]
        self.assertIn("title", primary["properties"])
        self.assertIn("title", primary["required"])

    def test_every_required_property_exists_in_every_vendor_schema(self):
        for provider in ("gemini", "openai"):
            adapted = ProviderSchemaAdapter.adapt(self._plan_schema(), provider=provider)
            self._assert_required_resolvable(adapted, provider)

    def _assert_required_resolvable(self, node, provider):
        if isinstance(node, list):
            for item in node:
                self._assert_required_resolvable(item, provider)
            return
        if not isinstance(node, dict):
            return
        props = node.get("properties")
        if isinstance(props, dict):
            for name in node.get("required", []):
                self.assertIn(
                    name, props,
                    f"[{provider}] required property '{name}' is missing from properties",
                )
        for value in node.values():
            self._assert_required_resolvable(value, provider)


class TestProviderSchemaStateSpace(unittest.TestCase):
    """Gemini rejects schemas whose state space is too large.

    "The specified schema produces a constraint that has too many states for
    serving" is triggered by array length limits and numeric bounds. They are
    stripped from the vendor hint only — the server keeps the full
    Draft 2020-12 schema.
    """

    def _plan_schema(self):
        return json.loads(AssetPaths.default().shot_plan_schema.read_text(encoding="utf-8"))

    def _assert_no_bounds(self, node, path=""):
        if isinstance(node, list):
            for i, item in enumerate(node):
                self._assert_no_bounds(item, f"{path}[{i}]")
            return
        if not isinstance(node, dict):
            return
        for banned in ("minItems", "maxItems", "minimum", "maximum"):
            self.assertNotIn(banned, node, f"{path} still carries '{banned}'")
        for key, value in node.items():
            self._assert_no_bounds(value, f"{path}.{key}")

    def test_gemini_plan_schema_has_no_bounds(self):
        adapted = ProviderSchemaAdapter.adapt(self._plan_schema(), provider="gemini")
        self._assert_no_bounds(adapted)

    def test_gemini_observation_schema_has_no_bounds(self):
        obs = json.loads(AssetPaths.default().observation_schema.read_text(encoding="utf-8"))
        adapted = ProviderSchemaAdapter.adapt(obs, provider="gemini")
        self._assert_no_bounds(adapted)

    def test_server_side_schema_still_enforces_bounds(self):
        """Stripping the vendor hint must NOT weaken the server contract."""
        validator = SchemaValidator(AssetPaths.default(), production_mode=True)
        plan = {
            "schema_version": "0.1.0", "frame_id": "f", "scene_version": 0, "plan_id": "p",
            "status": "hold", "coordinate_space": "upright_input_normalized_unmirrored",
            "goal": "g", "scene_summary": "s", "subject_placement_guide": None,
            "crop_recommendation": None,
            "primary_plan": None, "actions": [], "active_action_id": None, "alternative": None,
            "uncertainties": [], "invalidation_conditions": [],
            "fallback_instruction": "f",
        }
        self.assertEqual(validator.validate_shot_plan(plan), [])

        # A guide bbox outside [0,1] must still be rejected server-side.
        bad = dict(plan)
        bad["subject_placement_guide"] = {
            "bbox": [0.2, 0.2, 1.8, 0.9],
            "target_height_ratio": [0.4, 0.6],
            "instruction": "i", "basis": "rule_of_thirds", "fact_ids": [],
        }
        errors = validator.validate_shot_plan(bad)
        self.assertTrue(errors)
        self.assertTrue(any("maximum" in e or "1" in e for e in errors))

    def test_mime_type_is_detected_from_content(self):
        from travel_photo_assistant.adapters.providers import _detect_mime, _load_attachment_base64

        png = b"\x89PNG\r\n\x1a\n" + b"0" * 32
        jpg = b"\xff\xd8\xff\xe0" + b"0" * 32
        self.assertEqual(_detect_mime(png), "image/png")
        self.assertEqual(_detect_mime(jpg), "image/jpeg")

        import base64
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.png"
            path.write_bytes(png)
            mime, b64 = _load_attachment_base64(
                {"mime_type": "image/jpeg", "local_path": str(path)}
            )
            self.assertEqual(mime, "image/png")
            self.assertEqual(base64.b64decode(b64), png)


if __name__ == "__main__":
    unittest.main()
