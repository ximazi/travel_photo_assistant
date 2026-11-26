from unittest.mock import Mock, patch
import unittest

from travel_photo_assistant.framing_judge import AXES, main, review_pair


def output(winner):
    return {"winner": winner, "A": dict.fromkeys(AXES, 2), "B": dict.fromkeys(AXES, 1),
            "evidence_A": "Clear tower at the top left", "evidence_B": "Tower is clipped"}


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

