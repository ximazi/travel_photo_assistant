"""Tests for the person ghost box (subject_placement_guide).

Scenario under test: travel scenes usually contain nobody, but the shot still
has to include a person. The plan therefore carries a ghost box saying where
the person should stand — mutually exclusive with the real target box.
"""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from travel_photo_assistant.adapters.mock import RuleBasedOfflineVisionAdapter, _build_placement_guide
from travel_photo_assistant.config import AssetPaths, RuntimeConfig
from travel_photo_assistant.controller import TravelPhotoController
from travel_photo_assistant.models import RuntimeInput
from travel_photo_assistant.photo_pipeline import build_runtime_input, run_photo
from travel_photo_assistant.render import render_overlay
from travel_photo_assistant.schema_validator import ProviderSchemaAdapter, SchemaValidator
from travel_photo_assistant.validators import BusinessRuleValidator

ASSETS = AssetPaths.default()


def _controller():
    return TravelPhotoController(
        adapter=RuleBasedOfflineVisionAdapter(),
        assets=ASSETS,
        config=RuntimeConfig(production_mode=True),
    )


class TestPlacementGuideSchema(unittest.TestCase):
    def test_guide_is_a_required_top_level_field(self):
        schema = json.loads(ASSETS.shot_plan_schema.read_text(encoding="utf-8"))
        self.assertIn("subject_placement_guide", schema["required"])
        self.assertIn("subject_placement_guide", schema["properties"])

    def test_guide_is_nullable_and_strict(self):
        schema = json.loads(ASSETS.shot_plan_schema.read_text(encoding="utf-8"))
        variants = schema["properties"]["subject_placement_guide"]["anyOf"]
        self.assertTrue(any(v.get("type") == "null" for v in variants))
        obj = next(v for v in variants if v.get("type") == "object")
        self.assertFalse(obj["additionalProperties"])
        self.assertEqual(
            sorted(obj["required"]),
            ["basis", "bbox", "fact_ids", "framing", "instruction", "target_height_ratio"],
        )

    def test_provider_adapters_keep_the_guide_nullable(self):
        schema = json.loads(ASSETS.shot_plan_schema.read_text(encoding="utf-8"))
        gemini = ProviderSchemaAdapter.adapt(schema, provider="gemini")
        guide = gemini["properties"]["subject_placement_guide"]
        self.assertTrue(guide.get("nullable"))
        self.assertNotIn("anyOf", guide)


class TestPlacementGuideGeneration(unittest.TestCase):
    def test_no_subject_scene_produces_a_ghost_box(self):
        photo = ASSETS.photos_dir / "2.png"
        if not photo.is_file():
            self.skipTest("2.png is not present")
        ctrl = _controller()
        res = ctrl.run_slow_loop(build_runtime_input(photo))
        self.assertEqual(res.plan["status"], "no_subject")
        guide = res.plan["subject_placement_guide"]
        self.assertIsNotNone(guide)
        xmin, ymin, xmax, ymax = guide["bbox"]
        self.assertTrue(0.0 <= xmin < xmax <= 1.0)
        self.assertTrue(0.0 <= ymin < ymax <= 1.0)
        self.assertIn(guide["basis"], {"rule_of_thirds", "centered_symmetry", "leading_line_anchor", "open_area"})
        self.assertTrue(guide["instruction"])
        self.assertTrue(guide["fact_ids"])

    def test_scene_with_a_subject_has_no_ghost_box(self):
        photo = ASSETS.photos_dir / "coast_person.jpg"
        ctrl = _controller()
        res = ctrl.run_slow_loop(build_runtime_input(photo))
        self.assertEqual(res.plan["status"], "plan_ready")
        self.assertIsNone(res.plan["subject_placement_guide"])

    def test_multiple_people_scene_has_no_ghost_box(self):
        photo = ASSETS.photos_dir / "6.png"
        if not photo.is_file():
            self.skipTest("6.png is not present")
        ctrl = _controller()
        res = ctrl.run_slow_loop(build_runtime_input(photo))
        self.assertEqual(res.plan["status"], "hold")
        self.assertIsNone(res.plan["subject_placement_guide"])

    def test_ghost_box_keeps_human_proportions(self):
        """Box width must follow from the height ratio and the real frame aspect."""
        runtime_input = build_runtime_input(ASSETS.photos_dir / "coast_person.jpg")
        runtime_input.local_vision_hints["ground_line_y"] = 0.90
        runtime_input.local_vision_hints["virtual_subject_height_ratio"] = [0.50, 0.50]
        runtime_input.local_vision_hints["virtual_subject_framing"] = "full_body"
        observation = {"scene_tags": ["coast"], "anchors": [], "facts": [{"fact_id": "f1"}]}
        guide = _build_placement_guide(runtime_input, observation)
        xmin, ymin, xmax, ymax = guide["bbox"]
        img_w = runtime_input.frame.upright_input_width_px
        img_h = runtime_input.frame.upright_input_height_px
        expected_width = (0.50 * img_h * 0.26 * 1.06) / img_w
        self.assertAlmostEqual(xmax - xmin, expected_width, delta=0.002)
        self.assertAlmostEqual(ymax - ymin, 0.50, places=3)
        self.assertAlmostEqual(ymax, 0.90, places=3)

    def test_symmetric_scene_centers_the_ghost_box(self):
        runtime_input = build_runtime_input(ASSETS.photos_dir / "coast_person.jpg")
        observation = {"scene_tags": ["architecture", "symmetry"], "anchors": [], "facts": []}
        guide = _build_placement_guide(runtime_input, observation)
        xmin, _, xmax, _ = guide["bbox"]
        self.assertEqual(guide["basis"], "centered_symmetry")
        self.assertAlmostEqual((xmin + xmax) / 2.0, 0.50, places=3)


class TestPlacementGuideValidation(unittest.TestCase):
    def setUp(self):
        self.ctrl = _controller()
        photo = ASSETS.photos_dir / "coast_person.jpg"
        self.req = build_runtime_input(photo)
        res = self.ctrl.run_slow_loop(self.req)
        self.assertTrue(res.accepted, res.reason)
        self.plan = res.plan
        self.observation = res.observation
        self.cards = res.retrieval.selected_cards

    def _validate(self, plan, observation=None, runtime=None):
        return self.ctrl.validator.validate_shot_plan(
            plan, observation or self.observation, runtime or self.req, self.cards
        )

    def test_missing_guide_in_an_empty_scene_is_rejected(self):
        photo = ASSETS.photos_dir / "2.png"
        if not photo.is_file():
            self.skipTest("2.png is not present")
        ctrl = _controller()
        res = ctrl.run_slow_loop(build_runtime_input(photo))
        broken = copy.deepcopy(res.plan)
        broken["subject_placement_guide"] = None
        val = ctrl.validator.validate_shot_plan(
            broken, res.observation, build_runtime_input(photo), res.retrieval.selected_cards
        )
        self.assertFalse(val.is_valid)
        self.assertIn("MISSING_PLACEMENT_GUIDE", [e.code for e in val.errors])

    def test_guide_conflicting_with_an_existing_subject_is_rejected(self):
        broken = copy.deepcopy(self.plan)
        broken["subject_placement_guide"] = {
            "bbox": [0.3, 0.3, 0.6, 0.9],
            "target_height_ratio": [0.5, 0.6],
            "instruction": "站到虚框位置",
            "basis": "rule_of_thirds",
            "fact_ids": [],
        }
        val = self._validate(broken)
        self.assertFalse(val.is_valid)
        self.assertIn("PLACEMENT_GUIDE_CONFLICTS_WITH_SUBJECT", [e.code for e in val.errors])

    def test_invalid_guide_bbox_is_rejected(self):
        photo = ASSETS.photos_dir / "2.png"
        if not photo.is_file():
            self.skipTest("2.png is not present")
        ctrl = _controller()
        req = build_runtime_input(photo)
        res = ctrl.run_slow_loop(req)
        broken = copy.deepcopy(res.plan)
        broken["subject_placement_guide"]["bbox"] = [0.7, 0.3, 0.4, 0.9]
        val = ctrl.validator.validate_shot_plan(broken, res.observation, req, res.retrieval.selected_cards)
        self.assertFalse(val.is_valid)
        self.assertIn("INVALID_BBOX_BOUNDS", [e.code for e in val.errors])

    def test_guide_with_dangling_fact_reference_is_rejected(self):
        photo = ASSETS.photos_dir / "2.png"
        if not photo.is_file():
            self.skipTest("2.png is not present")
        ctrl = _controller()
        req = build_runtime_input(photo)
        res = ctrl.run_slow_loop(req)
        broken = copy.deepcopy(res.plan)
        broken["subject_placement_guide"]["fact_ids"] = ["f_does_not_exist"]
        val = ctrl.validator.validate_shot_plan(broken, res.observation, req, res.retrieval.selected_cards)
        self.assertFalse(val.is_valid)
        self.assertIn("DANGLING_FACT_REF", [e.code for e in val.errors])

    def test_guide_is_forbidden_when_input_is_insufficient(self):
        broken = copy.deepcopy(self.plan)
        broken["status"] = "insufficient_input"
        broken["subject_placement_guide"] = {
            "bbox": [0.3, 0.3, 0.6, 0.9],
            "target_height_ratio": [0.5, 0.6],
            "instruction": "x",
            "basis": "rule_of_thirds",
            "fact_ids": [],
        }
        val = self._validate(broken)
        self.assertFalse(val.is_valid)
        self.assertIn("PLACEMENT_GUIDE_NOT_ALLOWED", [e.code for e in val.errors])

    def test_guide_is_forbidden_when_the_observation_is_insufficient(self):
        broken = copy.deepcopy(self.plan)
        broken["subject_placement_guide"] = {
            "bbox": [0.3, 0.3, 0.6, 0.9],
            "target_height_ratio": [0.5, 0.6],
            "instruction": "x",
            "basis": "rule_of_thirds",
            "fact_ids": [],
        }
        insufficient_obs = copy.deepcopy(self.observation)
        insufficient_obs["status"] = "insufficient_input"
        val = self._validate(broken, observation=insufficient_obs)
        self.assertFalse(val.is_valid)
        self.assertIn("PLACEMENT_GUIDE_NOT_ALLOWED", [e.code for e in val.errors])


class TestPlacementGuideRendering(unittest.TestCase):
    def test_ghost_box_is_rendered(self):
        photo = ASSETS.photos_dir / "2.png"
        if not photo.is_file():
            self.skipTest("2.png is not present")
        with tempfile.TemporaryDirectory() as tmp:
            result = run_photo(photo, output_dir=tmp)
            self.assertEqual(result.plan["status"], "no_subject")
            self.assertIsNotNone(result.render.ghost_bbox_px)
            self.assertIsNotNone(result.render.ghost_basis)
            self.assertTrue((Path(result.output_dir) / "result.jpg").is_file())

    def test_ghost_box_is_absent_when_a_subject_exists(self):
        photo = ASSETS.photos_dir / "coast_person.jpg"
        with tempfile.TemporaryDirectory() as tmp:
            result = run_photo(photo, output_dir=tmp)
            self.assertIsNone(result.render.ghost_bbox_px)

    def test_ghost_box_helper_draws_corner_brackets(self):
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (200, 400), (10, 10, 10))
        draw = ImageDraw.Draw(img)
        from travel_photo_assistant.render import COLOR_GHOST, _draw_ghost_box

        _draw_ghost_box(draw, 40, 80, 120, 340, COLOR_GHOST, 5)
        # The four corners must be lit, the middle of the top edge must stay dark.
        self.assertGreater(sum(img.getpixel((40, 80))), 200)
        self.assertLess(sum(img.getpixel((80, 80))), 120)


if __name__ == "__main__":
    unittest.main()
