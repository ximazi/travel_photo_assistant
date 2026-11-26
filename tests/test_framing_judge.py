from unittest.mock import Mock, patch
import unittest

from travel_photo_assistant.framing_judge import AXES, main, review_pair


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
