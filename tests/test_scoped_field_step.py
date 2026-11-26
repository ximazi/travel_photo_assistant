import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from travel_photo_assistant.adapters.mock import RuleBasedOfflineVisionAdapter
from travel_photo_assistant.config import AssetPaths
from travel_photo_assistant.evaluation import EvaluationHarness, _base_runtime_dict
from travel_photo_assistant.field_pilot import main, overlay_plan, run_comparison
from travel_photo_assistant.models import RuntimeInput


def request(side=None):
    d = _base_runtime_dict(hints={
        "scene_tags": ["city_street"],
        "issues": [{"issue_id": "i1", "description": "\u5934\u90e8\u4e0e\u7535\u7ebf\u6746\u91cd\u5408", "priority": "high"}],
    })
    d["movement_constraints"]["photographer_side_step"] = side
    if side:
        d["movement_constraints"]["photographer_translation"] = "allowed"
    return RuntimeInput.from_dict(d)


class ObservedOverlapAdapter(RuleBasedOfflineVisionAdapter):
    """Synthetic visual claim with a distinct region and links; no actual model call."""

    def observe(self, *args, **kwargs):
        obs = super().observe(*args, **kwargs)
        if obs["selected_subject"] and obs["issues"]:
            fid = "head-overlap"
            obs["facts"].insert(0, {
                "fact_id": fid, "kind": "geometry", "source": "vlm_visual", "strength": "high",
                "statement": "\u9009\u5b9a\u4eba\u7269\u5934\u90e8\u4e0e\u80cc\u666f\u7535\u7ebf\u6746\u91cd\u5408",
                "region": [.46, .33, .53, .41],
            })
            obs["issues"][0]["fact_ids"].append(fid)
        return obs


class ScopedFieldStepTest(unittest.TestCase):
    def test_same_observation_abc_per_condition_card_selection_is_not_feasibility(self):
        assets = AssetPaths.default()
        adapter = ObservedOverlapAdapter()
        reports = {}
        for side in (None, "left", "right"):
            r = request(side)
            with patch.object(adapter, "observe", wraps=adapter.observe) as observe, \
                 patch.object(adapter, "plan", wraps=adapter.plan) as planner:
                arms, report = run_comparison(adapter, assets, r)
            obs_req = observe.call_args.args[0]
            self.assertIsNone(obs_req.movement_constraints.photographer_side_step)
            self.assertEqual(obs_req.movement_constraints.photographer_translation, "unknown")
            self.assertEqual(report["movement_condition"]["confirmed_photographer_side_step"], side)
            for call in planner.call_args_list:
                self.assertEqual(call.args[0].movement_constraints.photographer_side_step, side)
                self.assertFalse(call.args[0].user_goal.allow_subject_walking)
            self.assertEqual((report["observer_requests"], report["planner_requests"]), (1, 3))
            retrieve, _ = arms["retrieved_cards"]
            self.assertIn("HEAD_BACKGROUND_MERGE_06", [c.id for c in retrieve.retrieval.selected_cards])
            self.assertTrue(all(result.accepted for result, _ in arms.values()))
            for mode, (result, _) in arms.items():
                self.assertFalse(report["arm_movement"][mode]["fresh_frame_verified"])
                self.assertEqual(report["arm_movement"][mode]["active_photographer_step_accepted"], side is not None)
                a = result.plan["actions"][0]
                if side:
                    self.assertEqual(len(result.plan["actions"]), 1)
                    self.assertEqual((a["operation"], a["actor"]), ("physical_translate", "photographer"))
                    self.assertEqual((a["completion_check"]["type"], a["completion_check"]["check_by"]),
                                     ("head_obstacle_separated", "vlm_review"))
                    self.assertIn("\u81ea\u5df1\u7684" + ("\u5de6" if side == "left" else "\u53f3") + "\u4fa7", a["instruction"])
                    self.assertNotIn("\u53cd\u5411", a["instruction"])  # Original fixture recommends opposite if first fails.
                else:
                    self.assertEqual(a["operation"], "confirm_movement_space")
            reports[side] = report
        self.assertEqual(len({r["observation_sha256"] for r in reports.values()}), 1)

    def test_invalid_surrogates_and_extra_moves_fail_closed(self):
        ctrl = EvaluationHarness()._new_controller()
        ctrl.adapter = ObservedOverlapAdapter()
        r = request("left")
        result = ctrl.run_slow_loop(r)
        for bad_case in ("pan", "tilt", "other_actor", "wrong_reference", "wrong_check", "manual", "extra_move"):
            p = copy.deepcopy(result.plan)
            a = p["actions"][0]
            if bad_case in {"pan", "tilt"}:
                a["operation"] = "reframe_" + bad_case
                code = "INVALID_PARALLAX_CLAIM"
            elif bad_case == "other_actor":
                a["actor"] = "subject"
                code = "SUBJECT_TRANSLATION_DISALLOWED"
            else:
                if bad_case == "wrong_reference":
                    a["direction_reference"] = "input_image"
                elif bad_case == "wrong_check":
                    a["completion_check"]["type"] = "camera_metadata_match"
                elif bad_case == "manual":
                    a["completion_check"]["check_by"] = "user"
                else:
                    p["actions"].append({**a, "action_id": "a2"})
                code = "INVALID_SCOPED_SIDE_STEP"
            with self.subTest(bad_case=bad_case):
                val = ctrl.validator.validate_shot_plan(p, result.observation, r, result.retrieval.selected_cards)
                self.assertIn(code, {e.code for e in val.errors})

    def test_no_overlap_claim_uncited_or_unlocalized_claim_blocks_side_step(self):
        req = request("left")
        ctrl = EvaluationHarness()._new_controller()
        # The baseline mock has a selected subject and a head-overlap issue text,
        # but the issue/action point only to generic selected-subject/scene facts.
        baseline = ctrl.run_slow_loop(req)
        self.assertIsNotNone(baseline.observation["selected_subject"])
        self.assertIn("SCOPED_SIDE_STEP_OVERLAP_UNSUPPORTED", {e.code for e in baseline.plan_validation.errors})
        ctrl.adapter = ObservedOverlapAdapter()
        valid = ctrl.run_slow_loop(req)
        self.assertTrue(valid.accepted)
        for bad in ("unlinked_issue", "unlinked_action", "lower_body", "negated", "inferred"):
            p = copy.deepcopy(valid.plan)
            obs = copy.deepcopy(valid.observation)
            overlap = obs["facts"][0]
            if bad == "unlinked_issue":
                obs["issues"][0]["fact_ids"] = ["f2"]
            elif bad == "unlinked_action":
                p["actions"][0]["evidence_fact_ids"] = ["f2"]
            elif bad == "lower_body":
                overlap["region"] = [.46, .65, .53, .73]
            elif bad == "negated":
                overlap["statement"] += "\uff0c\u4f46\u5b9e\u9645\u6ca1\u6709\u91cd\u5408"
            else:
                overlap["source"] = "user_input"
            with self.subTest(bad=bad):
                val = ctrl.validator.validate_shot_plan(p, obs, req, valid.retrieval.selected_cards)
                self.assertIn("SCOPED_SIDE_STEP_OVERLAP_UNSUPPORTED", {e.code for e in val.errors})

    def test_overlay_does_not_echo_model_summary_or_target_geometry(self):
        req = request("left")
        ctrl = EvaluationHarness()._new_controller()
        ctrl.adapter = ObservedOverlapAdapter()
        result = ctrl.run_slow_loop(req)
        result.plan["primary_plan"]["title"] = "Move forward as instructed"
        result.plan["scene_summary"] = "Step to the right."
        result.plan["subject_placement_guide"] = {"bbox": [.1,.2,.3,.4], "instruction": "Move right"}
        result.plan["crop_recommendation"] = {"crop_rect": [.1,.2,.3,.4], "rationale": "Move right"}
        display = overlay_plan(result, req)
        self.assertIsNone(display["subject_placement_guide"])
        self.assertIsNone(display["crop_recommendation"])
        self.assertIsNone(display["primary_plan"]["composition"]["target_subject_bbox"])
        self.assertNotIn("Move forward", display["primary_plan"]["title"])
        self.assertEqual(display["actions"][0]["instruction"], result.plan["actions"][0]["instruction"])
        self.assertEqual(result.plan["primary_plan"]["title"], "Move forward as instructed")

    def test_cli_preserves_typed_condition_outside_goal_and_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "generic.jpg"
            Image.new("RGB", (64, 48), "gray").save(path)
            out = Path(td) / "results"
            adapter = ObservedOverlapAdapter()
            adapter.api_key = "stub-local-only"
            observation = adapter.observe(request(), "", {})
            goal = "A natural travel portrait"
            def offline_fixture(runtime_input, *_):
                self.assertEqual(runtime_input.user_goal.priority, goal)
                self.assertEqual(runtime_input.local_vision_hints, {})
                self.assertIsNone(runtime_input.movement_constraints.photographer_side_step)
                return {**observation, "frame_id": runtime_input.frame.frame_id}
            with patch("travel_photo_assistant.field_pilot.load_dotenv"), \
                 patch("travel_photo_assistant.field_pilot.GeminiVisionAdapter", return_value=adapter), \
                 patch.object(adapter, "observe", side_effect=offline_fixture), \
                 patch("urllib.request.urlopen") as transport:
                rc = main(["--photo", str(path), "--out", str(out), "--model", "gemini-testing",
                           "--goal", goal, "--confirm-free-project", "--allow-photo-to-gemini",
                           "--confirm-photographer-side-step", "left", "--compare"])
                self.assertEqual(rc, 0)
                transport.assert_not_called()
            comparison = json.loads((out/"comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(comparison["movement_condition"]["confirmed_photographer_side_step"], "left")
            for mode in comparison["evaluated_modes"]:
                arm = json.loads((out/mode/"run.json").read_text(encoding="utf-8"))
                self.assertEqual(arm["movement_condition"], comparison["movement_condition"])
                self.assertNotIn("local_path", json.dumps(arm))
                self.assertFalse(arm["movement"]["fresh_frame_verified"])
