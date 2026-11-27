"""Tests for reporting boundary sensitivity without changing winners."""
import unittest

from test_portrait_observation_verdict import BRIEF, image, receipt
from portrait_beauty_sensitivity import diagnostics


class SensitivityTests(unittest.TestCase):
    def test_inclusive_tie_and_feasible_step_across_boundary(self):
        a, b = image(), image()
        a["scores"]["light_color"]["score"] = 4
        r = diagnostics(BRIEF, receipt(a, b), receipt(b, a))["orders"]
        for row in r.values():
            self.assertEqual(row["signed_gap_A_minus_B"], 5)
            self.assertEqual(row["verdict"], "tie")
            self.assertIn({"source": "A", "axis": "technical", "delta_grade": 1,
                           "verdict": "A"}, row["single_grade_changes_verdict"])
            self.assertNotIn({"source": "A", "axis": "light_color", "delta_grade": 1,
                              "verdict": "A"}, row["single_grade_changes_verdict"])

    def test_drift_follows_source_not_slot(self):
        a, b, a2 = image(), image(), image()
        a2["scores"]["composition"]["score"] = 4
        r = diagnostics(BRIEF, receipt(a, b), receipt(b, a2))
        self.assertEqual(r["orders"]["reverse"]["signed_gap_A_minus_B"], 6.25)
        self.assertEqual(r["observed_grade_drift"], {"A": {"composition": [3, 4]}, "B": {}})

    def test_unknown_is_not_zero_and_bad_order_stays_invalid(self):
        a = image()
        a["scores"]["technical"].update(score=None, region=None)
        r = diagnostics(BRIEF, receipt(a, image()), None)
        self.assertIsNone(r["orders"]["forward"]["signed_gap_A_minus_B"])
        self.assertEqual(r["orders"]["reverse"]["status"], "invalid")
        self.assertIsNone(r["observed_grade_drift"])
