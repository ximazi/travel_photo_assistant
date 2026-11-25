"""Tests for the human ghost figure and the landscape crop recommendation.

The ghost figure must be a person-shaped guide (head, neck, torso, legs) and
must say how much of the body it covers. The crop recommendation must never
cut the planned subject out of the picture.
"""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from travel_photo_assistant.config import AssetPaths, RuntimeConfig
from travel_photo_assistant.controller import TravelPhotoController
from travel_photo_assistant.photo_pipeline import build_runtime_input, run_photo
from travel_photo_assistant.render import _draw_person_silhouette, render_overlay
from travel_photo_assistant.schema_validator import SchemaValidator

ASSETS = AssetPaths.default()


def _controller():
    return TravelPhotoController(
        adapter=__import__(
            "travel_photo_assistant.adapters.mock", fromlist=["RuleBasedOfflineVisionAdapter"]
        ).RuleBasedOfflineVisionAdapter(),
        assets=ASSETS,
        config=RuntimeConfig(production_mode=True),
    )


class TestGhostFigureFraming(unittest.TestCase):
    def test_full_body_is_the_default_framing(self):
        photo = ASSETS.photos_dir / "2.png"
        if not photo.is_file():
            self.skipTest("2.png is not present")
        res = _controller().run_slow_loop(build_runtime_input(photo))
        guide = res.plan["subject_placement_guide"]
        self.assertEqual(guide["framing"], "full_body")

    def test_a_close_up_goal_switches_to_half_body(self):
        photo = ASSETS.photos_dir / "2.png"
        if not photo.is_file():
            self.skipTest("2.png is not present")
        runtime_input = build_runtime_input(photo, priority="突出人物表情的近景照片")
        res = _controller().run_slow_loop(runtime_input)
        guide = res.plan["subject_placement_guide"]
        self.assertEqual(guide["framing"], "half_body")

    def test_full_body_stands_on_the_ground_line_but_half_body_does_not(self):
        """A full-body figure is anchored at the feet; a half-body figure is not."""
        from travel_photo_assistant.adapters.mock import _build_placement_guide

        photo = ASSETS.photos_dir / "2.png"
        runtime_input = build_runtime_input(photo)
        runtime_input.local_vision_hints["ground_line_y"] = 0.90
        observation = {"scene_tags": ["coast"], "anchors": [], "facts": []}

        full = _build_placement_guide(runtime_input, observation)
        self.assertAlmostEqual(full["bbox"][3], 0.90, places=3)

        runtime_input.user_goal.priority = "近景特写"
        half = _build_placement_guide(runtime_input, observation)
        # The head sits in the upper third and the figure does not reach the
        # ground line, because the feet are out of frame.
        self.assertLess(half["bbox"][1], 0.35)
        self.assertLess(half["bbox"][3], 0.95)

    def test_shoulder_width_follows_the_implied_standing_height(self):
        """Shoulders span ~26% of the standing height the framing implies.

        A half-body shot puts the subject closer to the camera, so the implied
        standing height (and therefore the shoulder width in the picture) grows.
        """
        from travel_photo_assistant.adapters.mock import _build_placement_guide

        photo = ASSETS.photos_dir / "2.png"
        runtime_input = build_runtime_input(photo)
        img_w = runtime_input.frame.upright_input_width_px
        img_h = runtime_input.frame.upright_input_height_px
        visible_fraction = {"full_body": 1.0, "half_body": 0.50}
        observation = {"scene_tags": ["coast"], "anchors": [], "facts": []}

        for priority, framing in (
            ("人物自然、脸部可辨，同时保留旅行环境", "full_body"),
            ("近景特写", "half_body"),
        ):
            runtime_input.user_goal.priority = priority
            guide = _build_placement_guide(runtime_input, observation)
            self.assertEqual(guide["framing"], framing)
            xmin, ymin, xmax, ymax = guide["bbox"]
            visible_h = (ymax - ymin) * img_h
            implied_full_h = visible_h / visible_fraction[framing]
            expected_w = (implied_full_h * 0.26 * 1.06) / img_w
            self.assertAlmostEqual(xmax - xmin, expected_w, delta=0.003)

    def test_framing_is_required_by_the_schema(self):
        schema = json.loads(ASSETS.shot_plan_schema.read_text(encoding="utf-8"))
        variants = schema["properties"]["subject_placement_guide"]["anyOf"]
        obj = next(v for v in variants if v.get("type") == "object")
        self.assertIn("framing", obj["required"])
        self.assertEqual(
            obj["properties"]["framing"]["enum"],
            ["full_body", "three_quarter", "half_body", "headshot"],
        )


class TestGhostFigureRendering(unittest.TestCase):
    def test_silhouette_draws_head_torso_and_legs(self):
        from PIL import Image

        img = Image.new("RGB", (300, 900), (12, 12, 12))
        _draw_person_silhouette(img, 110, 60, 190, 840, (0, 214, 214), "full_body")

        def cyan_ratio(x, y):
            r, g, b = img.getpixel((x, y))
            return (g > 55 and b > 55 and r < 140)

        head = [cyan_ratio(150, y) for y in range(70, 180)]
        torso = [cyan_ratio(150, y) for y in range(250, 450)]
        legs = [cyan_ratio(150, y) for y in range(600, 830)]

        self.assertTrue(any(head), "no head was drawn")
        self.assertTrue(any(torso), "no torso was drawn")
        self.assertTrue(any(legs), "no legs were drawn")

    def test_headshot_has_no_legs(self):
        from PIL import Image

        img = Image.new("RGB", (300, 900), (12, 12, 12))
        _draw_person_silhouette(img, 110, 60, 190, 300, (0, 214, 214), "headshot")

        def cyan_ratio(x, y):
            r, g, b = img.getpixel((x, y))
            return (g > 55 and b > 55 and r < 140)

        self.assertTrue(any(cyan_ratio(150, y) for y in range(70, 150)))
        self.assertFalse(any(cyan_ratio(150, y) for y in range(400, 800)))

    def test_render_reports_the_framing(self):
        photo = ASSETS.photos_dir / "2.png"
        if not photo.is_file():
            self.skipTest("2.png is not present")
        with tempfile.TemporaryDirectory() as tmp:
            result = run_photo(photo, output_dir=tmp)
            self.assertEqual(result.render.ghost_framing, "full_body")


class TestCropRecommendation(unittest.TestCase):
    def test_landscape_without_a_subject_gets_a_crop(self):
        photo = ASSETS.photos_dir / "3.png"
        if not photo.is_file():
            self.skipTest("3.png is not present")
        res = _controller().run_slow_loop(build_runtime_input(photo))
        crop = res.plan["crop_recommendation"]
        self.assertIsNotNone(crop)
        self.assertIn(crop["aspect_ratio"], {"1:1", "3:2", "2:3", "4:3", "3:4", "4:5", "16:9", "9:16"})
        xmin, ymin, xmax, ymax = crop["crop_rect"]
        self.assertTrue(0.0 <= xmin < xmax <= 1.0)
        self.assertTrue(0.0 <= ymin < ymax <= 1.0)
        self.assertTrue(crop["rationale"])

    def test_crop_must_contain_the_ghost_figure(self):
        photo = ASSETS.photos_dir / "3.png"
        if not photo.is_file():
            self.skipTest("3.png is not present")
        ctrl = _controller()
        req = build_runtime_input(photo)
        res = ctrl.run_slow_loop(req)
        broken = copy.deepcopy(res.plan)
        guide = broken["subject_placement_guide"]["bbox"]
        # Deliberately crop the figure out.
        broken["crop_recommendation"]["crop_rect"] = [
            0.0, 0.0, max(0.05, guide[0] - 0.05), 0.2
        ]
        val = ctrl.validator.validate_shot_plan(
            broken, res.observation, req, res.retrieval.selected_cards
        )
        self.assertFalse(val.is_valid)
        self.assertIn("CROP_EXCLUDES_PLACEMENT_GUIDE", [e.code for e in val.errors])

    def test_scene_with_a_subject_has_no_crop(self):
        photo = ASSETS.photos_dir / "coast_person.jpg"
        res = _controller().run_slow_loop(build_runtime_input(photo))
        self.assertIsNone(res.plan["crop_recommendation"])

    def test_missing_crop_in_an_empty_scene_is_rejected(self):
        photo = ASSETS.photos_dir / "3.png"
        if not photo.is_file():
            self.skipTest("3.png is not present")
        ctrl = _controller()
        req = build_runtime_input(photo)
        res = ctrl.run_slow_loop(req)
        broken = copy.deepcopy(res.plan)
        broken["crop_recommendation"] = None
        val = ctrl.validator.validate_shot_plan(
            broken, res.observation, req, res.retrieval.selected_cards
        )
        self.assertFalse(val.is_valid)
        self.assertIn("MISSING_CROP_RECOMMENDATION", [e.code for e in val.errors])

    def test_crop_is_required_by_the_schema(self):
        schema = json.loads(ASSETS.shot_plan_schema.read_text(encoding="utf-8"))
        self.assertIn("crop_recommendation", schema["required"])
        self.assertIn("crop_recommendation", schema["properties"])

    def test_render_draws_the_crop_region(self):
        photo = ASSETS.photos_dir / "3.png"
        if not photo.is_file():
            self.skipTest("3.png is not present")
        with tempfile.TemporaryDirectory() as tmp:
            result = run_photo(photo, output_dir=tmp)
            self.assertIsNotNone(result.render.crop_rect_px)
            self.assertIsNotNone(result.render.crop_aspect_ratio)
            self.assertTrue((Path(result.output_dir) / "result.jpg").is_file())

    def test_generated_plan_passes_schema_validation(self):
        photo = ASSETS.photos_dir / "3.png"
        if not photo.is_file():
            self.skipTest("3.png is not present")
        res = _controller().run_slow_loop(build_runtime_input(photo))
        validator = SchemaValidator(ASSETS, production_mode=True)
        self.assertEqual(validator.validate_shot_plan(res.plan), [])


if __name__ == "__main__":
    unittest.main()
