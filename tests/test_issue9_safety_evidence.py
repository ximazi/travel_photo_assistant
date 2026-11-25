"""Offline acceptance regressions for issue #9; no provider requests."""
import copy
import unittest
from unittest.mock import patch

from travel_photo_assistant.evaluation import EvaluationHarness, _base_runtime_dict
from travel_photo_assistant.models import RuntimeInput


class TestIssue9(unittest.TestCase):
    def _baseline(self, *, walking=True, subject="allowed", photographer="allowed"):
        read = _base_runtime_dict(subject_translation=subject, photographer_translation=photographer)
        read["user_goal"]["allow_subject_walking"] = walking
        req = RuntimeInput.from_dict(read)
        ctrl = EvaluationHarness()._new_controller()
        result = ctrl.run_slow_loop(req)
        self.assertTrue(result.accepted, result.reason)
        return req, ctrl, result

    def _translation(self, actor, *, queued=False, **kwargs):
        req, ctrl, result = self._baseline(**kwargs)
        plan = copy.deepcopy(result.plan)
        target = plan["actions"][-1 if queued else 0]
        target.update(actor=actor, operation="physical_translate", instruction="Translate in the permitted space.")
        return ctrl.validator.validate_shot_plan(plan, result.observation, req, result.retrieval.selected_cards)

    def test_subject_movement_needs_allowed_space_and_literal_opt_in(self):
        for state, permission in (
            ("forbidden", True), ("not_requested", True), ("unknown", True),
            ("invalid", True), ("allowed", False), ("allowed", "false"), ("allowed", 1),
        ):
            with self.subTest(state=state, permission=permission):
                val = self._translation("subject", subject=state, walking=permission)
                self.assertIn("SUBJECT_TRANSLATION_DISALLOWED", {e.code for e in val.errors})
        self.assertTrue(self._translation("subject").is_valid)

    def test_photographer_move_cannot_be_queued_while_space_unknown(self):
        val = self._translation("photographer", photographer="unknown", queued=True)
        self.assertIn("UNCONFIRMED_MOVEMENT_SPACE_ACTIVE", {e.code for e in val.errors})
        self.assertTrue(self._translation("photographer", photographer="allowed").is_valid)

    def test_translation_by_camera_not_allowed(self):
        val = self._translation("camera")
        self.assertIn("INVALID_TRANSLATION_ACTOR", {e.code for e in val.errors})

    def test_light_direction_only_reaches_planner_with_linked_direction_evidence(self):
        req, ctrl, result = self._baseline()
        for statement, strength, source, region, linked, expected in (
            ("Dappled sunlight creates high contrast on the path.", "high", "vlm_visual", [0.3, 0.6, 0.7, 1], True, "unknown"),
            ("Side lighting cannot be confirmed.", "high", "vlm_visual", [0.3, 0.6, 0.7, 1], True, "unknown"),
            ("Side lighting lights the left side of the face.", "high", "user_input", [0.3, 0.6, 0.7, 1], True, "unknown"),
            ("Side lighting lights the left side of the face.", "high", "vlm_visual", None, True, "unknown"),
            ("Side lighting lights the left side of the face.", "low", "vlm_visual", [0.3, 0.6, 0.7, 1], True, "unknown"),
            ("Side lighting lights the left side of the face.", "high", "vlm_visual", [0.3, 0.6, 0.7, 1], False, "unknown"),
            ("Side lighting lights the left side of the face.", "high", "vlm_visual", [0.3, 0.6, 0.7, 1], True, "side"),
        ):
            with self.subTest(statement=statement, strength=strength, source=source, region=region, linked=linked):
                obs = copy.deepcopy(result.observation)
                obs["facts"].append({"fact_id":"direction1", "kind":"light", "statement":statement,
                                     "strength":strength, "source":source, "region":region})
                obs["light"].update(direction_estimate="side", fact_ids=["direction1"] if linked else [])
                with patch.object(ctrl.adapter, "observe", return_value=obs), patch.object(
                    ctrl.adapter, "plan", wraps=ctrl.adapter.plan
                ) as plan:
                    report = ctrl.run_slow_loop(req)
                self.assertTrue(report.accepted, report.reason)
                self.assertEqual(report.observation["light"]["direction_estimate"], expected)
                self.assertEqual(plan.call_args.kwargs["observation"]["light"]["direction_estimate"], expected)
                self.assertEqual(obs["light"]["direction_estimate"], "side")
                self.assertEqual(bool(report.normalization_changes), expected == "unknown")
