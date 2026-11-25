"""Tests for the VLM vocabulary normalizer.

Real vendors do not reliably enforce `enum` in their response schemas. The
normalizer folds near-miss tokens onto the canonical vocabulary; unknown
tokens are left alone so schema validation still rejects them.
"""

from __future__ import annotations

import unittest

from travel_photo_assistant.config import AssetPaths, RuntimeConfig
from travel_photo_assistant.controller import TravelPhotoController
from travel_photo_assistant.normalizer import normalize_observation, normalize_plan
from travel_photo_assistant.schema_validator import SchemaValidator

ASSETS = AssetPaths.default()


def _obs(**overrides):
    payload = {
        "schema_version": "0.1.0",
        "frame_id": "f1",
        "scene_version": 1,
        "coordinate_space": "upright_input_normalized_unmirrored",
        "status": "observed",
        "scene_tags": ["coast"],
        "facts": [
            {
                "fact_id": "f1",
                "kind": "environment",
                "statement": "海边",
                "source": "vision_hint",
                "strength": "strong",
                "region": None,
            }
        ],
        "selected_subject": None,
        "anchors": [
            {
                "anchor_id": "a1",
                "kind": "geometry",
                "description": "拱门",
                "bbox": [0.1, 0.1, 0.4, 0.5],
                "points": [],
                "fact_ids": ["f1"],
            }
        ],
        "light": {
            "appearance": "柔光",
            "direction_estimate": "overcast",
            "strength": "strong",
            "fact_ids": [],
        },
        "issues": [
            {
                "issue_id": "i1",
                "description": "电线杆与头部重合",
                "priority": "critical",
                "fact_ids": ["f1"],
            }
        ],
        "unknowns": [],
    }
    payload.update(overrides)
    return payload


class TestNormalizeObservation(unittest.TestCase):
    def test_status_ok_becomes_observed(self):
        payload, changes = normalize_observation(_obs(status="ok"))
        self.assertEqual(payload["status"], "observed")
        self.assertTrue(any("status" in c for c in changes))

    def test_fact_kind_and_source_are_folded(self):
        payload, _ = normalize_observation(_obs())
        fact = payload["facts"][0]
        self.assertEqual(fact["kind"], "scene")
        self.assertEqual(fact["source"], "vlm_visual")
        self.assertEqual(fact["strength"], "high")

    def test_anchor_kind_geometry_becomes_frame(self):
        payload, _ = normalize_observation(_obs())
        self.assertEqual(payload["anchors"][0]["kind"], "frame")

    def test_issue_priority_critical_becomes_high(self):
        payload, _ = normalize_observation(_obs())
        self.assertEqual(payload["issues"][0]["priority"], "high")

    def test_light_direction_overcast_becomes_diffuse(self):
        payload, _ = normalize_observation(_obs())
        self.assertEqual(payload["light"]["direction_estimate"], "diffuse")

    def test_unknown_token_is_left_untouched(self):
        payload, changes = normalize_observation(_obs(status="banana"))
        self.assertEqual(payload["status"], "banana")
        self.assertFalse([c for c in changes if c.startswith("status")])

    def test_normalized_payload_passes_schema_validation(self):
        validator = SchemaValidator(ASSETS, production_mode=True)
        normalized, _ = normalize_observation(_obs())
        self.assertEqual(validator.validate_observation(normalized), [])

    def test_unnormalized_payload_fails_schema_validation(self):
        validator = SchemaValidator(ASSETS, production_mode=True)
        self.assertNotEqual(validator.validate_observation(_obs()), [])


class TestNormalizePlan(unittest.TestCase):
    def _plan(self, **overrides):
        plan = {
            "schema_version": "0.1.0",
            "frame_id": "f1",
            "scene_version": 1,
            "plan_id": "p1",
            "status": "plan_ready",
            "coordinate_space": "upright_input_normalized_unmirrored",
            "goal": "g",
            "scene_summary": "s",
            "subject_placement_guide": None,
            "crop_recommendation": None,
            "primary_plan": {
                "title": "t",
                "reason": "r",
                "used_fact_ids": [],
                "used_card_ids": [],
                "composition": {
                    "orientation": "vertical",
                    "shot_size": "environmental",
                    "target_subject_bbox": [0.3, 0.3, 0.6, 0.9],
                    "target_subject_height_ratio": [0.5, 0.6],
                    "preserve_anchor_ids": [],
                },
                "lens": {
                    "action": "stay",
                    "target_lens_id": None,
                    "ui_label": None,
                    "equivalent_focal_length_mm": None,
                    "note": "n",
                },
                "pose_intent": "p",
            },
            "actions": [
                {
                    "action_id": "a1",
                    "actor": "user",
                    "operation": "pan",
                    "direction_reference": "image",
                    "instruction": "i",
                    "prerequisites": [],
                    "evidence_fact_ids": [],
                    "completion_check": {
                        "type": "center_in_range",
                        "target_range": [0.4, 0.6],
                        "reference_id": None,
                        "check_by": "on_device",
                        "description": "d",
                    },
                }
            ],
            "active_action_id": "a1",
            "alternative": None,
            "uncertainties": [],
            "invalidation_conditions": ["scene_change", "lens_changed"],
            "fallback_instruction": "f",
        }
        plan.update(overrides)
        return plan

    def test_enums_are_folded(self):
        plan, _ = normalize_plan(self._plan())
        self.assertEqual(plan["primary_plan"]["composition"]["orientation"], "portrait")
        self.assertEqual(plan["primary_plan"]["composition"]["shot_size"], "environmental_full_body")
        self.assertEqual(plan["primary_plan"]["lens"]["action"], "keep")
        action = plan["actions"][0]
        self.assertEqual(action["actor"], "photographer")
        self.assertEqual(action["operation"], "reframe_pan")
        self.assertEqual(action["direction_reference"], "input_image")
        self.assertEqual(action["completion_check"]["type"], "subject_center_x_in_range")
        self.assertEqual(action["completion_check"]["check_by"], "local_vision")

    def test_unknown_invalidation_condition_is_dropped_and_logged(self):
        plan, changes = normalize_plan(
            self._plan(invalidation_conditions=["scene_change", "something_weird", "lens_changed"])
        )
        self.assertEqual(plan["invalidation_conditions"], ["scene_version_changed", "lens_changed"])
        self.assertTrue(any("dropped unknown value" in c for c in changes))

    def test_normalized_plan_passes_schema_validation(self):
        validator = SchemaValidator(ASSETS, production_mode=True)
        normalized, _ = normalize_plan(self._plan())
        self.assertEqual(validator.validate_shot_plan(normalized), [])

    def test_unnormalized_plan_fails_schema_validation(self):
        validator = SchemaValidator(ASSETS, production_mode=True)
        self.assertNotEqual(validator.validate_shot_plan(self._plan()), [])

    def test_placement_guide_basis_is_folded(self):
        plan = self._plan(
            subject_placement_guide={
                "bbox": [0.3, 0.3, 0.6, 0.9],
                "framing": "full_body",
                "target_height_ratio": [0.5, 0.6],
                "instruction": "i",
                "basis": "thirds",
                "fact_ids": [],
            }
        )
        normalized, _ = normalize_plan(plan)
        self.assertEqual(normalized["subject_placement_guide"]["basis"], "rule_of_thirds")


class TestRetryWithFeedback(unittest.TestCase):
    """A model that gets it wrong must get a second chance with the errors.

    Enum mistakes are repaired by the normalizer, so the retry path is
    exercised with a violation the normalizer deliberately cannot fix: a
    dangling fact reference.
    """

    class _BadThenGoodAdapter:
        model_name = "bad-then-good"

        def __init__(self):
            self.observe_calls = 0
            self.plan_calls = 0
            self.last_observe_feedback = None
            self.last_plan_feedback = None

        def observe(self, runtime_input, system_prompt, output_schema, feedback=None):
            self.observe_calls += 1
            self.last_observe_feedback = feedback
            payload = _obs(status="ok")
            if self.observe_calls == 1:
                # Dangling reference: normalization cannot repair this.
                payload["anchors"][0]["fact_ids"] = ["f_does_not_exist"]
            payload["frame_id"] = runtime_input.frame.frame_id
            payload["scene_version"] = runtime_input.frame.scene_version
            return payload

        def plan(self, runtime_input, observation, shot_cards, system_prompt, output_schema, feedback=None):
            self.plan_calls += 1
            self.last_plan_feedback = feedback
            guide = None
            crop = None
            if observation.get("selected_subject") is None:
                guide = {
                    "bbox": [0.3, 0.3, 0.6, 0.9],
                    "framing": "full_body",
                    "target_height_ratio": [0.5, 0.6],
                    "instruction": "站到虚框位置",
                    "basis": "rule_of_thirds",
                    "fact_ids": [],
                }
                crop = {
                    "crop_rect": [0.1, 0.1, 0.8, 0.95],
                    "aspect_ratio": "3:2",
                    "rationale": "保留引导线与人物位置",
                    "scene_basis": ["coast"],
                    "focus_anchor_ids": [],
                }
            return {
                "schema_version": "0.1.0",
                "frame_id": runtime_input.frame.frame_id,
                "scene_version": runtime_input.frame.scene_version,
                "plan_id": "p1",
                "status": "no_subject" if guide else "plan_ready",
                "coordinate_space": "upright_input_normalized_unmirrored",
                "goal": "g",
                "scene_summary": "s",
                "subject_placement_guide": guide,
                "crop_recommendation": crop,
                "primary_plan": None,
                "actions": [],
                "active_action_id": None,
                "alternative": None,
                "uncertainties": [],
                "invalidation_conditions": [],
                "fallback_instruction": "f",
            }

    def test_observation_is_retried_once_with_feedback(self):
        from travel_photo_assistant.photo_pipeline import build_runtime_input

        adapter = self._BadThenGoodAdapter()
        ctrl = TravelPhotoController(
            adapter=adapter,
            assets=ASSETS,
            config=RuntimeConfig(production_mode=True),
        )
        photo = ASSETS.photos_dir / "coast_person.jpg"
        res = ctrl.run_slow_loop(build_runtime_input(photo))
        self.assertEqual(adapter.observe_calls, 2)
        self.assertIsNotNone(adapter.last_observe_feedback)
        # The feedback must name the actual violation so the model can fix it.
        self.assertIn("fact", adapter.last_observe_feedback.lower())
        self.assertTrue(res.accepted, res.reason)

    def test_retry_can_be_disabled(self):
        from travel_photo_assistant.photo_pipeline import build_runtime_input

        adapter = self._BadThenGoodAdapter()
        ctrl = TravelPhotoController(
            adapter=adapter,
            assets=ASSETS,
            config=RuntimeConfig(production_mode=True, retry_on_validation_error=False),
        )
        photo = ASSETS.photos_dir / "coast_person.jpg"
        ctrl.run_slow_loop(build_runtime_input(photo))
        self.assertEqual(adapter.observe_calls, 1)


if __name__ == "__main__":
    unittest.main()
