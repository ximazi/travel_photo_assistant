import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from travel_photo_assistant.adapters.mock import RuleBasedOfflineVisionAdapter
from travel_photo_assistant.evaluation import _base_runtime_dict
from travel_photo_assistant.field_pilot import main
from travel_photo_assistant.knowledge import ShotCardRepository, ShotCardRetriever
from travel_photo_assistant.models import RuntimeInput

CARD = "FOREST_DAPPLED_LIGHT_04"
TAG = "dappled_face_light"
POSITIVE = "\u9009\u5b9a\u4eba\u7269\u8138\u90e8\u53ef\u89c1\u5f3a\u70c8\u6591\u9a73\u5149\u6591"
FLOOR_ONLY = "\u5730\u9762\u6709\u6591\u9a73\u5149\u6591\uff0c\u9009\u5b9a\u4eba\u7269\u8138\u90e8\u5149\u7ebf\u5747\u5300"


def observe(req, *, positive=False):
    obs = RuleBasedOfflineVisionAdapter().observe(req, "", {})
    obs["scene_tags"] = ["forest"] + ([] if positive else [TAG])
    obs["light"]["appearance"] = "daylight" if positive else FLOOR_ONLY
    fid = "face-light" if positive else "floor-light"
    obs["facts"].append({"fact_id": fid, "kind": "light", "source": "vlm_visual", "strength": "high",
                         "statement": POSITIVE if positive else FLOOR_ONLY,
                         "region": [.46, .33, .53, .41] if positive else [.10, .75, .38, .93]})
    obs["light"]["fact_ids"].append(fid)
    if positive:
        obs["selected_subject"]["fact_ids"].append(fid)
    else:
        obs["issues"].append({"issue_id": "ground-light", "description": FLOOR_ONLY,
                              "priority": "low", "fact_ids": [fid]})
    return obs


class DappleCardEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.req = RuntimeInput.from_dict(_base_runtime_dict())
        self.retriever = ShotCardRetriever(ShotCardRepository())

    def receipt(self, obs):
        return next(e for e in self.retriever.retrieve(obs, self.req).evaluations if e.card_id == CARD)

    def test_floor_only_with_selected_face_and_raw_tag_is_rejected(self):
        obs = observe(self.req)
        for tags in (["forest", TAG], ["forest"]):
            with self.subTest(tags=tags):
                obs["scene_tags"] = tags
                receipt = self.receipt(obs)
                self.assertFalse(receipt.accepted)
                self.assertNotIn(TAG, receipt.matched_visual_tags)
                self.assertTrue(any("selected visible face" in r for r in receipt.rejection_reasons))

    def test_linked_local_strong_selected_face_evidence_selects_card(self):
        obs = observe(self.req, positive=True)
        self.assertIn(CARD, [c.id for c in self.retriever.retrieve(obs, self.req).selected_cards])
        self.assertIn(TAG, self.receipt(obs).matched_visual_tags)

    def test_other_subject_or_location_negation_and_low_certainty_fail(self):
        positive = observe(self.req, positive=True)
        changes = {
            "other_subject": ("region", [.12, .34, .20, .40]),
            "torso": ("region", [.46, .66, .53, .74]),
            "whole_frame": ("region", [0, 0, 1, 1]),
            "no_region": ("region", None),
            "negated": ("statement", POSITIVE + " not confirmed"),
            "uncertain": ("strength", "medium"),
            "wrong_source": ("source", "user_input"),
        }
        for name in (*changes, "occluded", "subject_link", "light_link", "no_selected_subject"):
            with self.subTest(name=name):
                obs = copy.deepcopy(positive)
                obs["scene_tags"].append(TAG)
                if name in changes:
                    key, value = changes[name]
                    obs["facts"][-1][key] = value
                elif name == "occluded":
                    obs["selected_subject"]["face_visibility"] = "occluded"
                elif name == "subject_link":
                    obs["selected_subject"]["fact_ids"].remove("face-light")
                elif name == "light_link":
                    obs["light"]["fact_ids"].remove("face-light")
                else:
                    obs["selected_subject"] = None
                self.assertFalse(self.receipt(obs).accepted)

    def test_field_cli_keeps_frozen_observation_and_a_b_c_receipts(self):
        for eligible in (False, True):
            with self.subTest(eligible=eligible), tempfile.TemporaryDirectory() as td:
                photo, out = Path(td) / "synthetic.jpg", Path(td) / "output"
                Image.new("RGB", (64, 64), "gray").save(photo)
                adapter = RuleBasedOfflineVisionAdapter()
                adapter.api_key = "local-stub-only"
                def fixture(req, *_):
                    return {**observe(self.req, positive=eligible), "frame_id": req.frame.frame_id}
                with patch("travel_photo_assistant.field_pilot.load_dotenv"), \
                     patch("travel_photo_assistant.field_pilot.GeminiVisionAdapter", return_value=adapter), \
                     patch.object(adapter, "observe", side_effect=fixture), \
                     patch("urllib.request.urlopen") as transport:
                    rc = main(["--photo", str(photo), "--goal", "natural portrait", "--out", str(out),
                               "--model", "gemini-stub", "--confirm-free-project", "--allow-photo-to-gemini", "--compare"])
                    self.assertEqual(rc, 0)
                    transport.assert_not_called()
                def read(path):
                    return json.loads((out / path).read_text(encoding="utf-8"))
                a, b, c = (read(f"{mode}/run.json")
                           for mode in ("prompt_only", "fixed_cards", "retrieved_cards"))
                comparison = read("comparison.json")
                self.assertEqual((comparison["observer_requests"], comparison["planner_requests"]), (1, 3))
                self.assertEqual(len({arm["observation_sha256"] for arm in (a, b, c)}), 1)
                self.assertEqual(a["selected_card_ids"], [])
                self.assertEqual(set(b["selected_card_ids"]), set(ShotCardRetriever.DEFAULT_FIXED_CARD_IDS))
                self.assertEqual(CARD in c["selected_card_ids"], eligible)
                receipt = next(e for e in read("retrieved_cards/retrieval.json") if e["card_id"] == CARD)
                self.assertEqual(receipt["accepted"], eligible)
                self.assertFalse(c["movement"]["active_photographer_step_accepted"])
                self.assertFalse(c["movement"]["fresh_frame_verified"])
