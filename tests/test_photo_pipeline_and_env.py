"""Tests for the .env loader, the photo pipeline, and the overlay renderer."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from travel_photo_assistant.config import AssetPaths
from travel_photo_assistant.env_loader import _parse_env_text, env_status, find_env_file, load_dotenv
from travel_photo_assistant.photo_pipeline import PHOTO_PRESETS, build_runtime_input, run_photo
from travel_photo_assistant.render import _wrap, render_overlay


ASSETS = AssetPaths.default()


class TestEnvLoader(unittest.TestCase):
    def test_parse_env_text_handles_comments_export_and_quotes(self):
        parsed = _parse_env_text(
            "\n".join(
                [
                    "# a comment",
                    "",
                    "PLAIN=value",
                    "export EXPORTED=exported_value",
                    'QUOTED="quoted value"',
                    "SINGLE='single value'",
                    "EMPTY=",
                    "NO_EQUALS_SIGN",
                    "  SPACED  =  trimmed  ",
                ]
            )
        )
        self.assertEqual(parsed["PLAIN"], "value")
        self.assertEqual(parsed["EXPORTED"], "exported_value")
        self.assertEqual(parsed["QUOTED"], "quoted value")
        self.assertEqual(parsed["SINGLE"], "single value")
        self.assertEqual(parsed["EMPTY"], "")
        self.assertNotIn("NO_EQUALS_SIGN", parsed)
        self.assertEqual(parsed["SPACED"], "trimmed")

    def test_existing_environment_variables_win_over_dotenv(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("TPA_TEST_KEY=from_file\n", encoding="utf-8")
            os.environ["TPA_TEST_KEY"] = "from_environment"
            try:
                applied = load_dotenv(env_file)
                self.assertNotIn("TPA_TEST_KEY", applied)
                self.assertEqual(os.environ["TPA_TEST_KEY"], "from_environment")
            finally:
                os.environ.pop("TPA_TEST_KEY", None)

    def test_override_true_replaces_existing_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("TPA_TEST_KEY=from_file\n", encoding="utf-8")
            os.environ["TPA_TEST_KEY"] = "from_environment"
            try:
                load_dotenv(env_file, override=True)
                self.assertEqual(os.environ["TPA_TEST_KEY"], "from_file")
            finally:
                os.environ.pop("TPA_TEST_KEY", None)

    def test_missing_env_file_is_not_an_error(self):
        self.assertEqual(load_dotenv(Path("/nonexistent/path/.env")), {})

    def test_env_status_never_leaks_secret_values(self):
        os.environ["GEMINI_API_KEY"] = "super-secret-value"
        try:
            status = env_status()
            self.assertEqual(status["GEMINI_API_KEY"], "set")
            self.assertNotIn("super-secret-value", json.dumps(status))
        finally:
            os.environ.pop("GEMINI_API_KEY", None)

    def test_project_env_file_is_discoverable(self):
        found = find_env_file()
        self.assertIsNotNone(found)
        self.assertEqual(found.name, ".env")


class TestWrap(unittest.TestCase):
    def test_trailing_punctuation_is_not_orphaned(self):
        lines = _wrap("保持原地和当前镜头，向画面左侧缓慢转动手机，让人物中心进入右侧目标区。", 34)
        self.assertTrue(all(len(line) > 2 for line in lines))
        self.assertTrue("".join(lines).endswith("。"))

    def test_short_text_stays_on_one_line(self):
        self.assertEqual(_wrap("短句", 34), ["短句"])


class TestPhotoPipeline(unittest.TestCase):
    def test_every_manifest_entry_has_a_photo(self):
        photos_dir = ASSETS.photos_dir
        self.assertTrue(photos_dir.is_dir(), "assets/examples/photos must exist")
        for filename in PHOTO_PRESETS:
            self.assertTrue(
                (photos_dir / filename).is_file(),
                f"manifest references '{filename}' but the file does not exist",
            )

    def test_manifest_entries_only_use_supported_hint_fields(self):
        from travel_photo_assistant.photo_pipeline import _HINT_FIELDS

        allowed = _HINT_FIELDS | {"subject_id", "priority", "goal_tags"}
        for filename, entry in PHOTO_PRESETS.items():
            unknown = set(entry) - allowed
            self.assertFalse(unknown, f"'{filename}' has unsupported fields: {sorted(unknown)}")

    def test_build_runtime_input_attaches_the_real_file(self):
        photo = ASSETS.photos_dir / "coast_person.jpg"
        runtime_input = build_runtime_input(photo)
        self.assertEqual(len(runtime_input.frame.image_attachments), 1)
        self.assertTrue(runtime_input.frame.image_attachments[0]["local_path"].endswith("coast_person.jpg"))
        self.assertEqual(runtime_input.frame.coordinate_space, "upright_input_normalized_unmirrored")

    def test_build_runtime_input_without_metadata_drops_lenses(self):
        photo = ASSETS.photos_dir / "coast_person.jpg"
        runtime_input = build_runtime_input(photo, metadata_available=False)
        self.assertEqual(runtime_input.camera_capabilities.available_lenses, [])
        self.assertIsNone(runtime_input.camera_capabilities.current_lens_id)

    def test_manifest_subject_id_null_forces_subject_selection(self):
        """A photo with several people must not let the assistant pick one."""
        entry = PHOTO_PRESETS.get("6.png")
        if entry is None:
            self.skipTest("6.png is not in the manifest")
        runtime_input = build_runtime_input(ASSETS.photos_dir / "6.png")
        self.assertIsNone(runtime_input.user_goal.subject_id)
        self.assertEqual(runtime_input.local_vision_hints["detected_person_count"], 2)

    def test_run_photo_writes_artifacts_and_renders(self):
        photo = ASSETS.photos_dir / "coast_person.jpg"
        with tempfile.TemporaryDirectory() as tmp:
            result = run_photo(photo, output_dir=tmp)
            self.assertTrue(result.accepted, result.reason)
            self.assertEqual(result.plan["status"], "plan_ready")

            run_dir = Path(result.output_dir)
            self.assertTrue((run_dir / "observation.json").is_file())
            self.assertTrue((run_dir / "plan.json").is_file())
            self.assertTrue((run_dir / "result.jpg").is_file())

            # The rendered image must be a real, non-trivial JPEG.
            from PIL import Image

            with Image.open(run_dir / "result.jpg") as img:
                self.assertEqual(img.size, (1080, 1440))

            # observation.json must be valid JSON with the expected shape.
            observation = json.loads((run_dir / "observation.json").read_text(encoding="utf-8"))
            self.assertEqual(observation["status"], "observed")
            self.assertEqual(observation["frame_id"], "photo_coast_person")

    def test_run_photo_without_a_subject_does_not_fabricate_one(self):
        """A landscape with no person must yield no_subject, not an invented bbox."""
        photo = ASSETS.photos_dir / "2.png"
        if not photo.is_file():
            self.skipTest("2.png is not present")
        with tempfile.TemporaryDirectory() as tmp:
            result = run_photo(photo, output_dir=tmp)
            self.assertEqual(result.plan["status"], "no_subject")
            self.assertIsNone(result.plan["primary_plan"])
            self.assertIsNone(result.observation["selected_subject"])

    def test_run_photo_with_multiple_people_holds(self):
        photo = ASSETS.photos_dir / "6.png"
        if not photo.is_file():
            self.skipTest("6.png is not present")
        with tempfile.TemporaryDirectory() as tmp:
            result = run_photo(photo, output_dir=tmp)
            self.assertEqual(result.observation["status"], "needs_subject_selection")
            self.assertEqual(result.plan["status"], "hold")
            self.assertIsNone(result.plan["primary_plan"])
            self.assertEqual(result.plan["actions"], [])
            self.assertTrue(result.plan["fallback_instruction"])

    def test_run_photo_rejects_a_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            run_photo(ASSETS.photos_dir / "does_not_exist.jpg")


class TestRenderOverlay(unittest.TestCase):
    def test_render_draws_target_and_subject_boxes(self):
        photo = ASSETS.photos_dir / "coast_person.jpg"
        plan = {
            "status": "plan_ready",
            "primary_plan": {
                "title": "岸线引导的环境人像",
                "used_card_ids": ["COAST_LINE_ENV_01"],
                "composition": {
                    "target_subject_bbox": [0.53, 0.32, 0.77, 0.90],
                    "preserve_anchor_ids": [],
                },
            },
            "actions": [
                {
                    "action_id": "a1",
                    "operation": "reframe_pan",
                    "instruction": "向画面左侧缓慢转动手机。",
                    "completion_check": {"type": "subject_center_x_in_range"},
                }
            ],
            "active_action_id": "a1",
        }
        observation = {"anchors": []}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.jpg"
            result = render_overlay(
                photo, plan, out, observation=observation,
                subject_bbox=[0.38, 0.32, 0.62, 0.90], guidance_direction="left",
            )
            self.assertTrue(out.is_file())
            self.assertEqual(result.image_size, [1080, 1440])
            self.assertEqual(result.target_bbox_px, [572.4, 460.8, 831.6, 1296.0])
            self.assertEqual(result.subject_bbox_px, [410.4, 460.8, 669.6, 1296.0])
            self.assertIn("left", result.guidance_arrow)

    def test_mirrored_preview_flips_the_drawn_arrow(self):
        photo = ASSETS.photos_dir / "coast_person.jpg"
        plan = {
            "status": "plan_ready",
            "primary_plan": {
                "title": "t",
                "composition": {"target_subject_bbox": [0.53, 0.32, 0.77, 0.90]},
            },
            "actions": [],
            "active_action_id": None,
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = render_overlay(
                photo, plan, Path(tmp) / "o.jpg", guidance_direction="left", mirrored=True
            )
            self.assertIn("screen: right", result.guidance_arrow)

    def test_render_raises_for_a_missing_source(self):
        with self.assertRaises(FileNotFoundError):
            render_overlay("/nonexistent/photo.jpg", {}, "/tmp/out.jpg")


if __name__ == "__main__":
    unittest.main()
