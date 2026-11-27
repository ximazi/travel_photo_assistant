"""Test locked-panel matching and uncertainty denominators offline."""
import hashlib
import json
import unittest

from test_portrait_people_enumeration import VERSION, frame, instance
from compare_people_panels import scorecard


class PanelTests(unittest.TestCase):
    def setUp(self):
        self.a, self.b = "a" * 64, "b" * 64
        self.panel = json.dumps({"version": "people-panel-v0.1", "pairs": {
            "P1": {"A": {"sha256": self.a, "votes": ["one", "one"]},
                   "B": {"sha256": self.b, "votes": ["multiple", "multiple"]}}}}).encode()
        self.one = frame(instance([1, 1, 10, 10]))
        self.two = frame(instance([1, 1, 10, 10]), instance([20, 20, 30, 30]))
        self.unknown = frame(scan="incomplete")

    def run_for(self, f, r):
        def arm(hashes, values):
            if values is None:
                return None
            return {"image_sha256": hashes,
                    "receipt": {"version": VERSION, "A": values[0], "B": values[1]}}
        return {"Qwen": {"panel_sha256": hashlib.sha256(self.panel).hexdigest(),
                         "pairs": {"P1": {"forward": arm([self.a, self.b], f),
                                          "reverse": arm([self.b, self.a], r)}}}}

    def test_normalizes_reversed_attachments(self):
        m = scorecard(self.panel, self.run_for((self.one, self.two), (self.two, self.one)))
        self.assertEqual(m["models"]["Qwen"]["one_correct_stable_pass"], {"n": 1, "d": 1})
        self.assertEqual(m["models"]["Qwen"]["negative_false_pass_either_order"], {"n": 0, "d": 1})

    def test_one_wrong_order_cannot_hide_behind_aggregation(self):
        m = scorecard(self.panel, self.run_for((self.one, self.one), (self.two, self.unknown)))
        m = m["models"]["Qwen"]
        self.assertEqual(m["negative_false_pass_either_order"], {"n": 1, "d": 1})
        self.assertEqual(m["negative_false_stable_pass"], {"n": 0, "d": 1})
        self.assertEqual(m["order_disagreement"], {"n": 2, "d": 2})

    def test_abstention_and_invalid_are_distinct(self):
        test = self.run_for((self.unknown, self.unknown), (self.unknown, self.unknown))
        m = scorecard(self.panel, test)["models"]["Qwen"]
        self.assertEqual(m["stable_unknown"], {"n": 2, "d": 2})
        test["Qwen"]["pairs"]["P1"]["reverse"] = None
        m = scorecard(self.panel, test)["models"]["Qwen"]
        self.assertEqual(m["valid_orders"], {"n": 1, "d": 2})
        self.assertEqual(m["invalid_photo_pairs"], {"n": 2, "d": 2})

    def test_disputed_labels_do_not_enter_accuracy_denominators(self):
        self.panel = self.panel.replace(b'["one", "one"]', b'["one", "uncertain"]')
        m = scorecard(self.panel, self.run_for((self.one, self.two), (self.two, self.one)))
        self.assertEqual(m["human_labels"], {"disagreement": 1, "multiple": 1})
        self.assertEqual(m["models"]["Qwen"]["one_correct_stable_pass"], {"n": 0, "d": 0})

    def test_panel_and_image_order_mismatches_fail_closed(self):
        test = self.run_for((self.one, self.two), (self.two, self.one))
        test["Qwen"]["pairs"]["P1"]["reverse"]["image_sha256"].reverse()
        with self.assertRaises(ValueError):
            scorecard(self.panel, test)
        with self.assertRaises(ValueError):
            scorecard(self.panel + b" ", self.run_for((self.one, self.two), None))
