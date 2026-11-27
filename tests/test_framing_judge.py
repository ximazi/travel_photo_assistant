import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
import unittest

from travel_photo_assistant.framing_judge import (
    AXES, EMPTY_SCENE_PROMPT, PLACEMENT_PROMPT, PROMPT, main, review_pair,
)


def output(winner):
    a, b = (1, 2) if winner == "B" else (2, 1)
    return {"winner": winner, "A": dict.fromkeys(AXES, a), "B": dict.fromkeys(AXES, b),
            "evidence_A": "Tower at the top left", "evidence_B": "Tower near the right edge"}


class FramingJudgeTest(unittest.TestCase):
    def test_same_real_winner_on_reversal(self):
        judge = Mock(side_effect=[output("B"), output("A")])
        result = review_pair(judge, (b"left", b"right"), flip=True)
        self.assertEqual([c.args for c in judge.call_args_list], [(b"right", b"left"), (b"left", b"right")])
        self.assertEqual(result["winner"], 0)
        self.assertFalse(result["human_verified"])

    def test_preference_for_display_slot_abstains(self):
        result = review_pair(Mock(return_value=output("A")), (b"first", b"second"), flip=False)
        self.assertEqual(result["status"], "abstain_order_disagreement")
        self.assertIsNone(result["winner"])

    def test_tie_and_invalid_are_never_winners(self):
        for value, status in [(output("tie"), "tie"), (output("C"), "abstain_invalid_response")]:
            result = review_pair(Mock(return_value=value), (b"first", b"second"), flip=False)
            self.assertEqual(result["status"], status)
            self.assertIsNone(result["winner"])

    def test_cli_uses_target_specific_rubric_and_records_it(self):
        for flags, target, rubric in (
            ((), "portrait", PROMPT),
            (("--review-target", "empty_scene"), "empty_scene", EMPTY_SCENE_PROMPT),
            (("--review-target", "placement", "--confirm-matched-placement-pair"), "placement", PLACEMENT_PROMPT),
        ):
            with TemporaryDirectory() as tmp, \
                 patch("travel_photo_assistant.framing_judge.prepare_pair", return_value=(b"a", b"b")), \
                 patch("travel_photo_assistant.framing_judge.load_dotenv"), \
                 patch("travel_photo_assistant.framing_judge.GeminiVisionAdapter") as adapter:
                out = Path(tmp) / "receipt.json"
                judge = adapter.return_value._generate
                judge.return_value = output("tie")
                rc = main(["--a", "a", "--b", "b", "--intent", "Person and scenery",
                           "--model", "gemini-test", "--out", str(out), "--confirm-free-project",
                           "--allow-both-images-to-gemini", *flags])
                self.assertEqual(rc, 0)
                self.assertEqual([c.args[0] for c in judge.call_args_list], [rubric, rubric])
                for call in judge.call_args_list:
                    self.assertEqual(call.args[1], {"intent": "Person and scenery"})
                report = json.loads(out.read_text(encoding="utf-8"))
                self.assertEqual(report["review_target"], target)
                self.assertEqual(report["rubric_sha256"], hashlib.sha256(rubric.encode()).hexdigest())
                self.assertIsNone(report["winner"])

    def test_placement_requires_operator_to_confirm_matched_previews(self):
        with patch("travel_photo_assistant.framing_judge.GeminiVisionAdapter") as provider, patch("sys.stderr"):
            with self.assertRaises(SystemExit) as caught:
                main(["--a", "a", "--b", "b", "--intent", "position", "--model", "gemini-test",
                      "--out", "x", "--confirm-free-project", "--allow-both-images-to-gemini",
                      "--review-target", "placement"])
            self.assertEqual(caught.exception.code, 2)
            provider.assert_not_called()

    def test_neither_image_leaves_without_both_authorizations(self):
        with patch("urllib.request.urlopen") as transport, patch("sys.stderr"):
            with self.assertRaises(SystemExit) as caught:
                main(["--a", "a", "--b", "b", "--intent", "portrait", "--model", "gemini-test",
                      "--out", "x", "--confirm-free-project"])
            self.assertEqual(caught.exception.code, 2)
            transport.assert_not_called()


    def test_consistent_winner_dominated_in_one_round_abstains_and_keeps_both(self):
        first, second = output("A"), output("B")
        first["A"] = dict.fromkeys(AXES, 0)
        first["B"] = dict.fromkeys(AXES, 2)
        judge = Mock(side_effect=[first, second])
        result = review_pair(judge, (b"first", b"second"), flip=False)
        self.assertEqual(result["status"], "abstain_score_contradiction")
        self.assertIsNone(result["winner"])
        self.assertEqual(result["score_contradiction_rounds"], [1])
        self.assertEqual([r["judge"] for r in result["rounds"]], [first, second])
        self.assertEqual(judge.call_count, 2)

    def test_axis_tradeoff_is_not_a_contradiction(self):
        first, second = output("A"), output("B")
        first["A"], first["B"] = dict.fromkeys(AXES, 0), dict.fromkeys(AXES, 1)
        first["A"]["intent"] = 2
        result = review_pair(Mock(side_effect=[first, second]), (b"first", b"second"), flip=False)
        self.assertEqual(result["status"], "model_preference_only")
        self.assertEqual(result["winner"], 0)
        self.assertEqual(result["score_contradiction_rounds"], [])
