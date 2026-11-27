import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from travel_photo_assistant.adapters.providers import GeminiVisionAdapter
from travel_photo_assistant.adapters.mock import RuleBasedOfflineVisionAdapter
from travel_photo_assistant.config import AssetPaths
from travel_photo_assistant.evaluation import _base_runtime_dict
from travel_photo_assistant.field_pilot import main, placement_evidence, prepare_photo, run_comparison
from travel_photo_assistant.render import render_overlay
from travel_photo_assistant.models import RuntimeInput


class FieldPilotTest(unittest.TestCase):
    def test_blind_even_when_name_collides_and_rotation_is_normalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            named = Path(tmp) / "coast_person.jpg"
            neutral = Path(tmp) / "neutral.jpg"
            exif = Image.Exif()
            exif[274] = 6
            exif[270] = "ground truth that must not leave the machine"
            Image.new("RGB", (32, 12), "blue").save(named, exif=exif)
            neutral.write_bytes(named.read_bytes())
            a, data_a = prepare_photo(named, "人景兼顾")
            b, data_b = prepare_photo(neutral, "人景兼顾")
            self.assertEqual(data_a, data_b)
            self.assertEqual(a.frame.upright_input_width_px, 12)
            self.assertEqual(a.frame.upright_input_height_px, 32)
            with Image.open(io.BytesIO(data_a)) as im:
                self.assertEqual(im.size, (12, 32))
                self.assertFalse(im.getexif())
            for req in (a, b):
                self.assertEqual(req.local_vision_hints, {})
                self.assertIsNone(req.user_goal.subject_id)
                self.assertIsNone(req.user_goal.subject_count)
                self.assertFalse(req.camera_capabilities.metadata_available)
                self.assertEqual(req.camera_capabilities.available_lenses, [])
                self.assertFalse(req.movement_constraints.standing_in_place_confirmed)

            schema = {"type": "object", "properties": {"result": {"type": "string"}}}
            def check_transport(req, **_):
                self.assertNotIn("mock-key", req.full_url)
                self.assertEqual(req.get_header("X-goog-api-key"), "mock-key")
                contents = json.loads(req.data)["contents"][0]["parts"]
                self.assertEqual(len(contents), 2)
                for banned in ("coast_person", "local_path", "image_attachments", str(named)):
                    self.assertNotIn(banned, contents[0]["text"])
                self.assertEqual(base64.b64decode(contents[1]["inline_data"]["data"]), data_a)
                return io.BytesIO(json.dumps({"candidates": [{"content": {"parts": [{"text": '{"result":"stub"}'}]}}]}).encode())
            adapter = GeminiVisionAdapter(api_key="mock-key")
            with patch("urllib.request.urlopen", side_effect=check_transport) as transport:
                self.assertEqual(adapter.observe(a, "observe", schema), {"result": "stub"})
                self.assertEqual(adapter.plan(a, {}, [], "plan", schema), {"result": "stub"})
                self.assertEqual(transport.call_count, 2)

    def test_no_network_without_both_acknowledgments(self):
        args = ["--photo", "does-not-exist.jpg", "--goal", "人景兼顾", "--out", "unused",
                "--model", "gemini-2.5-flash", "--allow-photo-to-gemini"]
        with patch("urllib.request.urlopen") as transport, patch("sys.stderr"):
            with self.assertRaises(SystemExit) as ex:
                main(args)
            self.assertEqual(ex.exception.code, 2)
            transport.assert_not_called()

    def test_invalid_observation_coordinates_do_not_break_fallback_overlay(self):
        adapter = RuleBasedOfflineVisionAdapter()
        adapter.api_key = "mock-only"
        observe = adapter.observe

        def malformed(*args, **kwargs):
            result = observe(*args, **kwargs)
            result["anchors"].append({
                "anchor_id": "bad_ground", "kind": "possible_ground",
                "description": "Mixed coordinate units are unsafe to render",
                "bbox": [0, 688, 1, 1], "points": [[500, 850]], "fact_ids": [],
            })
            return result

        with tempfile.TemporaryDirectory() as tmp:
            photo, out = Path(tmp) / "scene.jpg", Path(tmp) / "out"
            Image.new("RGB", (300, 400), (32, 48, 64)).save(photo)
            with patch("travel_photo_assistant.field_pilot.load_dotenv"), \
                 patch("travel_photo_assistant.field_pilot.GeminiVisionAdapter", return_value=adapter), \
                 patch.object(adapter, "observe", side_effect=malformed), \
                 patch.object(adapter, "plan") as plan:
                result = main(["--photo", str(photo), "--out", str(out), "--goal", "Travel portrait",
                               "--model", "gemini-test", "--mode", "prompt_only",
                               "--confirm-free-project", "--allow-photo-to-gemini"])
                plan.assert_not_called()
            self.assertEqual(result, 1)
            run = json.loads((out / "run.json").read_text(encoding="utf-8"))
            observation = json.loads((out / "observation.json").read_text(encoding="utf-8"))
            fallback = json.loads((out / "plan.json").read_text(encoding="utf-8"))
            self.assertFalse(run["accepted"])
            self.assertTrue(run["fallback"])
            self.assertEqual(observation["anchors"][-1]["bbox"], [0, 688, 1, 1])
            self.assertEqual(fallback["status"], "hold")
            self.assertIsNone(fallback["subject_placement_guide"])
            with Image.open(out / "overlay.jpg") as preview:
                self.assertEqual(preview.size, (300, 400))

    def test_shared_observation_is_sent_to_three_planners_once(self):
        req = RuntimeInput.from_dict(_base_runtime_dict(hints={
            "scene_tags": ["coast", "leading_line"],
            "anchors": [{"anchor_id": "coast_path", "kind": "leading_line", "description": "海岸线",
                         "bbox": [0.05, 0.4, 0.4, 0.9], "points": [[0.05, 0.9], [0.4, 0.4]]}],
        }))
        adapter = RuleBasedOfflineVisionAdapter()
        with patch.object(adapter, "observe", wraps=adapter.observe) as obs, \
             patch.object(adapter, "plan", wraps=adapter.plan) as plan:
            arms, report = run_comparison(adapter, AssetPaths.default(), req)
            self.assertEqual(obs.call_count, 1)
            self.assertEqual(plan.call_count, 3)
            for call in plan.call_args_list[1:]:
                self.assertEqual(call.args[1], plan.call_args_list[0].args[1])
                self.assertIsNot(call.args[1], plan.call_args_list[0].args[1])
            self.assertEqual([len(call.args[2]) for call in plan.call_args_list], [0, 3, 1])
        self.assertEqual(report["status"], "needs_human_review")
        self.assertEqual((report["observer_requests"], report["planner_requests"]), (1, 3))
        self.assertTrue(all(result.accepted for result, _ in arms.values()))

    def test_hold_baseline_skips_later_arms(self):
        req = RuntimeInput.from_dict(_base_runtime_dict(
            subject_id=None, subject_count=2, hints={"detected_person_count": 2, "scene_tags": ["city_street"]},
        ))
        adapter = RuleBasedOfflineVisionAdapter()
        arms, report = run_comparison(adapter, AssetPaths.default(), req)
        self.assertEqual(list(arms), ["prompt_only"])
        self.assertEqual(arms["prompt_only"][0].plan["status"], "hold")
        self.assertEqual(report["status"], "comparison_not_scored")
        self.assertEqual((report["observer_requests"], report["planner_requests"]), (1, 1))
        self.assertEqual(list(report["skipped_modes"]), ["fixed_cards", "retrieved_cards"])

    def test_empty_scene_skips_duplicate_and_renders_distinct_conditions(self):
        req = RuntimeInput.from_dict(_base_runtime_dict(
            subject_id=None, subject_count=None,
            hints={"detected_person_count": 0, "scene_tags": ["forest"],
                   "anchors": [{"anchor_id": "ground", "kind": "possible_ground",
                                "description": "Visible level area in fixture",
                                "bbox": [0.1, 0.65, 0.9, 0.98], "points": []}]},
        ))
        adapter = RuleBasedOfflineVisionAdapter()
        arms, report = run_comparison(adapter, AssetPaths.default(), req, empty_scene=True)
        self.assertEqual(report["comparison_target"], "empty_scene_placement")
        self.assertEqual((report["observer_requests"], report["planner_requests"]), (1, 2))
        self.assertEqual(report["skipped_modes"], {"retrieved_cards": "duplicate_planner_input"})
        self.assertEqual(report["duplicate_input_of"], {"retrieved_cards": "prompt_only"})
        self.assertFalse(report["retrieved_card_effect_testable"])
        self.assertEqual(report["effective_card_ids_by_mode"]["prompt_only"], [])
        self.assertEqual(report["effective_card_ids_by_mode"]["retrieved_cards"], [])
        self.assertEqual(len(report["effective_card_ids_by_mode"]["fixed_cards"]), 3)
        self.assertEqual(report["status"], "needs_human_review")
        with tempfile.TemporaryDirectory() as tmp:
            photo = Path(tmp) / "fixture.jpg"
            Image.new("RGB", (300, 400)).save(photo)
            for mode, (arm, _) in arms.items():
                self.assertTrue(arm.accepted, arm.plan_validation)
                self.assertIsNone(arm.observation["selected_subject"])
                self.assertEqual(arm.plan["status"], "no_subject")
                guide = report["arm_placement"][mode]
                self.assertEqual(guide["status"], "tentative_composition_only")
                self.assertFalse(guide["standing_safety_verified"])
                self.assertFalse(guide["new_frame_verified"])
                if arm.plan["subject_placement_guide"]["framing"] == "full_body":
                    self.assertIsNotNone(guide["foot_point"])
                rendered = render_overlay(photo, arm.plan, Path(tmp) / (mode + ".jpg"),
                                          observation=arm.observation)
                self.assertIsNotNone(rendered.ghost_bbox_px)
                self.assertIsNone(rendered.subject_bbox_px)
                self.assertIsNone(rendered.target_bbox_px)

    def test_observed_empty_scene_can_hold_and_ask_to_see_ground(self):
        req = RuntimeInput.from_dict(_base_runtime_dict(
            subject_id=None, subject_count=None,
            hints={"detected_person_count": 0, "scene_tags": ["forest"]},
        ))
        assets = AssetPaths.default()
        adapter = RuleBasedOfflineVisionAdapter()
        offline_plan = adapter.plan

        def hold(*args, **kwargs):
            observation = args[1]
            self.assertEqual(observation["status"], "observed")
            self.assertIsNone(observation["selected_subject"])
            self.assertFalse(any(a["kind"] == "possible_ground" for a in observation["anchors"]))
            plan = offline_plan(*args, **kwargs)
            plan.update(
                status="hold", subject_placement_guide=None, crop_recommendation=None,
                primary_plan=None, alternative=None, actions=[], active_action_id=None,
                fallback_instruction="请重新取景，把前景地面拍进画面。",
            )
            return plan

        with patch.object(adapter, "plan", side_effect=hold):
            arms, report = run_comparison(adapter, assets, req, empty_scene=True)

        self.assertEqual(report["framing_shortlist"]["status"], "no_supported_candidate")
        self.assertEqual(report["framing_shortlist"]["groups"], [])
        with tempfile.TemporaryDirectory() as tmp:
            photo = Path(tmp) / "fixture.jpg"
            Image.new("RGB", (300, 400)).save(photo)
            for mode, (arm, _) in arms.items():
                self.assertTrue(arm.accepted, arm.reason)
                self.assertFalse(arm.downgraded_to_fallback)
                self.assertEqual(arm.plan["status"], "hold")
                self.assertEqual(arm.plan["fallback_instruction"], "请重新取景，把前景地面拍进画面。")
                self.assertEqual(arm.plan["actions"], [])
                self.assertEqual(report["arm_placement"][mode]["status"], "no_accepted_person_guide")
                rendered = render_overlay(photo, arm.plan, Path(tmp) / (mode + ".jpg"),
                                          observation=arm.observation)
                self.assertIsNone(rendered.ghost_bbox_px)
                self.assertIsNone(rendered.crop_rect_px)

    def test_empty_scene_hold_does_not_allow_geometry_or_a_plan(self):
        from travel_photo_assistant.controller import TravelPhotoController
        req = RuntimeInput.from_dict(_base_runtime_dict(
            subject_id=None, subject_count=None,
            hints={"detected_person_count": 0, "scene_tags": ["forest"]},
        ))
        ctrl = TravelPhotoController(RuleBasedOfflineVisionAdapter(), assets=AssetPaths.default())
        result = ctrl.run_slow_loop(req)
        self.assertTrue(result.accepted)
        checks = ctrl.validator.validate_shot_plan(
            dict(result.plan, status="hold", fallback_instruction=" "),
            result.observation, req, result.retrieval.selected_cards,
        )
        self.assertLessEqual(
            {"PLACEMENT_GUIDE_NOT_ALLOWED", "CROP_NOT_ALLOWED",
             "EMPTY_SCENE_HOLD_HAS_GUIDANCE", "EMPTY_SCENE_HOLD_MISSING_FALLBACK"},
            {error.code for error in checks.errors},
        )
        checks = ctrl.validator.validate_shot_plan(
            dict(result.plan, subject_placement_guide=None, crop_recommendation=None),
            result.observation, req, result.retrieval.selected_cards,
        )
        self.assertLessEqual({"MISSING_PLACEMENT_GUIDE", "MISSING_CROP_RECOMMENDATION"},
                             {error.code for error in checks.errors})

    def test_empty_scene_excludes_unselected_people_and_existing_subject(self):
        for count in (1, 2):
            req = RuntimeInput.from_dict(_base_runtime_dict(
                subject_id=None, subject_count=count, hints={"detected_person_count": count},
            ))
            arms, report = run_comparison(RuleBasedOfflineVisionAdapter(), AssetPaths.default(),
                                          req, empty_scene=True)
            self.assertEqual(list(arms), ["prompt_only"])
            self.assertEqual(report["skipped_modes"]["fixed_cards"], "not_an_observed_empty_scene")
            self.assertEqual(placement_evidence(arms["prompt_only"][0])["status"],
                             "no_accepted_person_guide")

    def test_empty_scene_keeps_a_distinct_retrieved_condition(self):
        from travel_photo_assistant.knowledge import RetrievalMode, ShotCardRetriever
        req = RuntimeInput.from_dict(_base_runtime_dict(
            subject_id=None, subject_count=None,
            hints={"detected_person_count": 0, "scene_tags": ["forest"]},
        ))
        real_retrieve = ShotCardRetriever.retrieve
        def simulated_empty_card(retriever, observation, runtime_input, *, mode, max_cards):
            if mode == RetrievalMode.RETRIEVED_CARDS:
                result = real_retrieve(retriever, observation, runtime_input,
                                       mode=RetrievalMode.FIXED_CARDS, max_cards=max_cards)
                result.selected_cards = result.selected_cards[:1]
                return result
            return real_retrieve(retriever, observation, runtime_input,
                                 mode=mode, max_cards=max_cards)
        with patch.object(ShotCardRetriever, "retrieve", new=simulated_empty_card):
            arms, report = run_comparison(RuleBasedOfflineVisionAdapter(), AssetPaths.default(),
                                          req, empty_scene=True)
        self.assertIn("retrieved_cards", arms)
        self.assertEqual(report["planner_requests"], 3)
        self.assertTrue(report["retrieved_card_effect_testable"])
        self.assertEqual(len(report["effective_card_ids_by_mode"]["retrieved_cards"]), 1)
