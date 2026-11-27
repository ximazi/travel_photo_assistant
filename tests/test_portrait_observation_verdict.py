"""Offline v0.4 contract tests; no provider dependencies."""
import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_portrait_judges import WEIGHTS
from portrait_observation_verdict import (
    VERSION, adjudicate, adjudicate_swap, no_duplicate_keys,
)

BRIEF = {"goal": "Recognizable main traveler with context", "required_elements": [
    {"id": "x17", "criterion": "A single visible person"},
    {"id": "q2", "criterion": "Visible rainbow"},
]}


def observation(key, value):
    return {"id": key, "status": value, "region": "left mist" if value != "unknown" else None,
            "evidence": "Visible pixels checked; uncertainty recorded when needed."}


def image(first="pass", second="pass", goal="met", score=3):
    return {"scores": {k: {"score": score, "region": "center", "evidence": "visible anchor"}
                       for k in WEIGHTS},
            "requirements": [observation("x17", first), observation("q2", second)],
            "goal": {"status": goal, "evidence": "larger person's visual role"}}


def receipt(a, b):
    return {"version": VERSION, "status": "ok", "A": a, "B": b}


class ObservationVerdictTests(unittest.TestCase):
    def test_hard_failure_dominates_known_and_unknown_requirements(self):
        r = adjudicate(BRIEF, receipt(image("fail", "unknown"), image()))
        self.assertEqual(r["A"]["intent_score"], 0)
        self.assertEqual(r["winner"]["intent"], "B")

    def test_unknown_is_distinct_from_fail_and_tie(self):
        r = adjudicate(BRIEF, receipt(image("pass", "unknown"), image()))
        self.assertIsNone(r["A"]["intent_score"])
        self.assertEqual(r["winner"]["intent"], "uncertain")

    def test_all_pass_does_not_promote_partly_met_to_met(self):
        r = adjudicate(BRIEF, receipt(image(goal="partly_met"), image()))
        self.assertEqual(r["A"]["intent_score"], 1)
        self.assertEqual(r["B"]["intent_score"], 2)
        self.assertEqual(r["winner"], {"beauty": "tie", "intent": "B"})

    def test_malformed_sibling_still_invalid_after_hard_failure(self):
        for corrupt in ("omitted", "duplicate", "unasked", "empty_evidence", "bad_status"):
            a = image("fail", "unknown")
            if corrupt == "omitted":
                a["requirements"].pop()
            else:
                replacement = a["requirements"][1]
                field, value = {"duplicate": ("id", "x17"), "unasked": ("id", "z9"),
                                "empty_evidence": ("evidence", " "), "bad_status": ("status", "maybe")}[corrupt]
                replacement[field] = value
            with self.subTest(corrupt=corrupt), self.assertRaises(ValueError):
                adjudicate(BRIEF, receipt(a, image()))

    def test_requirement_reordering_is_allowed_by_exact_id(self):
        a = image("pass", "fail")
        a["requirements"].reverse()
        self.assertEqual(adjudicate(BRIEF, receipt(a, image()))["A"]["intent_score"], 0)

    def test_bare_legacy_axis_without_evidence_is_invalid(self):
        a = image("fail", "pass")
        a["scores"]["technical"] = 4
        with self.assertRaises(ValueError):
            adjudicate(BRIEF, receipt(a, image()))

    def test_beauty_exact_tie_boundary_and_unknown(self):
        a, b = image(), image()
        b["scores"]["light_color"]["score"] = 4
        self.assertEqual(adjudicate(BRIEF, receipt(a, b))["winner"]["beauty"], "tie")
        b["scores"]["technical"]["score"] = 4
        self.assertEqual(adjudicate(BRIEF, receipt(a, b))["winner"]["beauty"], "B")
        b["scores"]["technical"].update(score=None, region=None)
        self.assertEqual(adjudicate(BRIEF, receipt(a, b))["winner"]["beauty"], "uncertain")

    def test_swapped_summary_and_missing_legacy_or_claimed_winner(self):
        a, b = image("fail", "pass"), image(score=4)
        f, rev = receipt(a, b), receipt(b, a)
        r = adjudicate_swap(BRIEF, f, rev)
        self.assertEqual(r["stable_source_order_winner"], {"beauty": "B", "intent": "B"})
        for bad in (None, {**rev, "version": "v0.2"}, {**rev, "winner": "A"}):
            r = adjudicate_swap(BRIEF, f, bad)
            self.assertEqual(r["orders"]["reverse"]["status"], "invalid")
            self.assertEqual(r["stable_source_order_winner"], {"beauty": "uncertain", "intent": "uncertain"})
        same_order = adjudicate_swap(BRIEF, f, deepcopy(f))
        self.assertEqual(same_order["stable_source_order_winner"]["intent"], "uncertain")

    def test_duplicate_json_key_is_rejected_before_scoring(self):
        with self.assertRaises(ValueError):
            json.loads('{"status":"pass","status":"fail"}', object_pairs_hook=no_duplicate_keys)


if __name__ == "__main__":
    unittest.main()
