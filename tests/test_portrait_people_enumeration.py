"""Unit tests for the opt-in exactly-one observation diagnostic."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from portrait_people_enumeration import VERSION, audit_pair, count_requirement


def instance(box, state="confirmed"):
    return {"box": box, "state": state, "evidence": "visible localized cue"}


def frame(*items, scan="complete"):
    return {"scan": scan, "scan_evidence": "region checked", "instances": list(items)}


class PeopleEnumerationTests(unittest.TestCase):
    def setUp(self):
        self.one = instance([100, 100, 180, 180])
        self.two = instance([460, 450, 500, 500])
        self.maybe = instance([600, 100, 620, 140], "possible")

    def test_one_complete_and_zero(self):
        self.assertEqual(count_requirement(frame(self.one)), "pass")
        self.assertEqual(count_requirement(frame()), "fail")

    def test_unresolved_shape_and_incomplete_scan(self):
        self.assertEqual(count_requirement(frame(self.one, self.maybe)), "unknown")
        self.assertEqual(count_requirement(frame(self.one, scan="incomplete")), "unknown")
        self.assertEqual(count_requirement(frame(scan="incomplete")), "unknown")

    def test_two_distinct_override_uncertainty(self):
        self.assertEqual(count_requirement(frame(self.one, self.two, self.maybe,
                                                 scan="incomplete")), "fail")

    def test_overlap_does_not_prove_two_or_one(self):
        duplicate = instance([100, 100, 180, 180])
        contained = instance([120, 120, 140, 140])
        for x in (duplicate, contained):
            self.assertEqual(count_requirement(frame(self.one, x)), "unknown")
            self.assertEqual(count_requirement(frame(self.one, x, self.two)), "fail")

    def test_invalid_boxes_and_empty_evidence_do_not_get_verdict(self):
        for bad in ([100, 100, 100, 140], [0, 0, 4, 1001], [False, 1, 3, 4]):
            with self.assertRaises(ValueError):
                count_requirement(frame(self.one, self.two, instance(bad)))
        with self.assertRaises(ValueError):
            count_requirement(frame(dict(self.one, evidence=" ")))

    def test_pair_uses_presentation_order(self):
        r = {"version": VERSION, "A": frame(self.one), "B": frame(self.one, self.two)}
        self.assertEqual(audit_pair(r), {"A": "pass", "B": "fail"})
        with self.assertRaises(ValueError):
            audit_pair(dict(r, version="wrong"))
