"""Tests for the dual-loop controller and all 16 evaluation cases (E01-E16)."""

from __future__ import annotations

import unittest

from travel_photo_assistant.adapters.mock import RuleBasedOfflineVisionAdapter
from travel_photo_assistant.config import AssetPaths, RuntimeConfig
from travel_photo_assistant.controller import TravelPhotoController
from travel_photo_assistant.evaluation import EvaluationHarness
from travel_photo_assistant.models import LocalVisionSnapshot, RuntimeInput


class TestControllerAndEvaluationCases(unittest.TestCase):
    def test_all_16_evaluation_cases_offline_contract(self) -> None:
        harness = EvaluationHarness(AssetPaths.default())
        report = harness.run_all_offline_contract_checks()
        self.assertEqual(report["summary"]["total_cases"], 16)
        self.assertEqual(report["summary"]["offline_contract_failed"], 0, msg=report)
        self.assertEqual(report["summary"]["offline_contract_passed"], 16)
        # Honest reporting: field photo tests are not fabricated
        self.assertEqual(report["summary"]["field_photo_executed"], 0)

    def test_full_slow_and_fast_loop_action_progression(self) -> None:
        ctrl = TravelPhotoController(
            adapter=RuleBasedOfflineVisionAdapter(),
            assets=AssetPaths.default(),
            config=RuntimeConfig(production_mode=True, stable_frames_required=2),
        )
        req = RuntimeInput.from_dict(
            {
                "schema_version": "0.1.0",
                "example_only": False,
                "frame": {
                    "frame_id": "f10",
                    "scene_version": 1,
                    "image_attachments": [{"mime_type": "image/jpeg", "uri": "memory://coast.jpg"}],
                    "viewport_fit": "contain",
                    "preview_aspect_ratio": "3:4",
                },
                "user_goal": {
                    "priority": "人物自然、保留海岸环境",
                    "subject_id": "person_1",
                    "subject_count": 1,
                    "orientation_preference": "portrait",
                },
                "camera_capabilities": {
                    "current_lens_id": "main_1x",
                    "available_lenses": [
                        {
                            "lens_id": "main_1x",
                            "ui_label": "1×",
                            "equivalent_focal_length_mm": 26,
                            "equivalent_focal_length_source": "camera_api",
                            "mode": "optical_primary",
                        }
                    ],
                },
                "local_vision_hints": {
                    "scene_tags": ["coast", "leading_line"],
                    "subject_bbox": [0.38, 0.32, 0.62, 0.90],
                },
            }
        )
        slow = ctrl.run_slow_loop(req)
        self.assertTrue(slow.accepted)
        self.assertEqual(slow.plan["active_action_id"], "a1")

        # Tick 1: tracking lost pauses overlay arrows
        t_lost = ctrl.run_fast_tick(
            LocalVisionSnapshot(
                frame_id="f11",
                scene_version=1,
                current_lens_id="main_1x",
                orientation="portrait",
                crop_signature="contain:3:4",
                tracked_subject_id="person_1",
                subject_bbox=None,
                tracking_ok=False,
            )
        )
        self.assertEqual(t_lost.status, "paused_tracking_lost")
        self.assertFalse(t_lost.show_overlay_arrows)

        # Tick 2 & 3: subject center_x enters [0.60, 0.70] for 2 consecutive frames -> a1 completes, a2 activates
        for fid in ("f12", "f13"):
            tick = ctrl.run_fast_tick(
                LocalVisionSnapshot(
                    frame_id=fid,
                    scene_version=1,
                    current_lens_id="main_1x",
                    orientation="portrait",
                    crop_signature="contain:3:4",
                    tracked_subject_id="person_1",
                    subject_bbox=[0.53, 0.32, 0.77, 0.90],
                )
            )
        self.assertEqual(tick.status, "action_completed")
        self.assertEqual(tick.active_action_id, "a2")

        # Tick 4: a2 requires vlm_review (pose_visually_confirmed), cannot complete without vlm_verified_checks
        t_wait = ctrl.run_fast_tick(
            LocalVisionSnapshot(
                frame_id="f14",
                scene_version=1,
                current_lens_id="main_1x",
                orientation="portrait",
                crop_signature="contain:3:4",
                tracked_subject_id="person_1",
                subject_bbox=[0.53, 0.32, 0.77, 0.90],
            )
        )
        self.assertEqual(t_wait.status, "in_progress")

        # Tick 5: vlm_review confirms pose -> ready_to_shoot
        t_done = ctrl.run_fast_tick(
            LocalVisionSnapshot(
                frame_id="f15",
                scene_version=1,
                current_lens_id="main_1x",
                orientation="portrait",
                crop_signature="contain:3:4",
                tracked_subject_id="person_1",
                subject_bbox=[0.53, 0.32, 0.77, 0.90],
                vlm_verified_checks=["pose_visually_confirmed"],
            )
        )
        self.assertEqual(t_done.status, "ready_to_shoot")


if __name__ == "__main__":
    unittest.main()
