"""Offline contract tests for the standalone comparison runner."""
import importlib.util
import unittest
from copy import deepcopy
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "portrait_judge", Path(__file__).resolve().parents[1] / "scripts" / "compare_portrait_judges.py")
judge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge)


def image(score, intent, status):
    return {
        "scores": {k: {"score": score, "region": "center", "evidence": "visible anchor"}
                   for k in judge.WEIGHTS},
        "intent": {"score": intent, "requirements": [{"status": status, "evidence": "present or missing"}],
                   "reason": "requirement check"},
    }


def response(reverse=False):
    a, b = image(4, 0, "missing"), image(2, 2, "present")
    return {"status": "ok", "A": b if reverse else a, "B": a if reverse else b,
            "beauty": {"winner": "B" if reverse else "A", "reason": "composition"},
            "intent": {"winner": "A" if reverse else "B", "reason": "landmark"}}


class PortraitJudgeTests(unittest.TestCase):
    def test_beauty_and_intent_can_disagree(self):
        self.assertEqual(judge.checked_vote(response(), 1), {"beauty": "A", "intent": "B"})

    def test_five_point_inclusive_tie(self):
        self.assertEqual(judge.result(300, 280, 20), "tie")
        self.assertEqual(judge.result(300, 279, 20), "A")

    def test_booleans_are_not_numeric_scores(self):
        data = response()
        data["A"]["scores"]["technical"]["score"] = True
        with self.assertRaises(ValueError):
            judge.checked_vote(data, 1)

    def test_missing_hard_requirement_cannot_pass(self):
        data = response()
        data["A"]["intent"]["score"] = 2
        with self.assertRaises(ValueError):
            judge.checked_vote(data, 1)

    def test_false_vote_rejected_not_adjusted(self):
        data = response()
        data["beauty"]["winner"] = "B"
        with self.assertRaises(ValueError):
            judge.checked_vote(data, 1)

    def test_unknown_axis_suppresses_beauty_not_intent(self):
        data = response()
        data["A"]["scores"]["technical"].update(score=None, region=None)
        data["beauty"]["winner"] = "uncertain"
        self.assertEqual(judge.checked_vote(data, 1), {"beauty": "uncertain", "intent": "B"})

    def test_nonportrait_explicit_abstain(self):
        self.assertEqual(judge.checked_vote({"status": "abstain", "reason": "empty"}, 0),
                         {"beauty": "uncertain", "intent": "uncertain"})

    def test_two_models_comparator_counts_invalid_and_order_bias(self):
        data = {"pair_id": "p1", "scene": "coast", "requirement_count": 1,
                "human": {"beauty": "A", "intent": "B"},
                "models": {"provider/m1": {"forward": response(), "reverse": response(True)},
                           "provider/m2": {"forward": response(), "reverse": response()}}}
        missing = deepcopy(data)
        missing.update(pair_id="p2", models={})
        report = judge.summarize([data, missing], ["provider/m1", "provider/m2"])
        votes = report["stable_votes_by_pair"]["p1"]
        self.assertEqual(votes["provider/m1"], {"beauty": "A", "intent": "B"})
        self.assertEqual(votes["provider/m2"], {"beauty": "uncertain", "intent": "uncertain"})
        metric = next(x for x in report["metrics"] if
                      (x["model"], x["scene"], x["target"]) == ("provider/m1", "all", "beauty"))
        self.assertEqual((metric["matches_human"], metric["human_decided"],
                          metric["invalid_or_missing_orders"]), (1, 2, 2))


    def test_always_tie_is_not_selection_coverage(self):
        flat = response()
        flat["A"] = deepcopy(flat["B"])
        flat["beauty"]["winner"] = flat["intent"]["winner"] = "tie"
        pairs = [
            {"pair_id": f"p{i}", "scene": "coast", "requirement_count": 1,
             "human": {"beauty": "A" if i < 8 else "tie", "intent": "uncertain"},
             "models": {"model/a": {"forward": flat, "reverse": flat}}}
            for i in range(10)
        ]
        row = next(r for r in judge.summarize(pairs, ["model/a"])["metrics"]
                   if r["scene"] == "all" and r["target"] == "beauty")
        self.assertEqual(row["swap_stable"], 10)
        self.assertEqual(row["human_decided_model_covered"], 10)
        self.assertEqual(row["stable_model_picks"], 0)
        self.assertEqual(row["stable_model_ties"], 10)
        self.assertEqual(row["human_picks"], 8)
        self.assertEqual(row["human_pick_model_picked"], 0)
        self.assertEqual(row["human_pick_model_tied"], 8)
        self.assertEqual(row["matches_human_picks"], 0)
        self.assertEqual(row["matches_human_ties"], 2)
        self.assertEqual(row["matches_human"], 2)

    def test_real_pick_and_invalid_flat_vote_are_distinct(self):
        bad = response()
        bad["A"] = deepcopy(bad["B"])  # Equal axes, but receipt still claims A/B.
        data = {"pair_id": "p1", "scene": "coast", "requirement_count": 1,
                "human": {"beauty": "A", "intent": "tie"},
                "models": {"good": {"forward": response(), "reverse": response(True)},
                           "bad": {"forward": bad, "reverse": bad}}}
        metrics = {(r["model"], r["target"]): r
                   for r in judge.summarize([data], ["good", "bad"])["metrics"]
                   if r["scene"] == "all"}
        self.assertEqual(metrics["good", "beauty"]["matches_human_picks"], 1)
        self.assertEqual(metrics["good", "intent"]["human_tie_model_picked"], 1)
        self.assertEqual(metrics["bad", "beauty"]["invalid_or_missing_orders"], 2)
        self.assertEqual(metrics["bad", "beauty"]["stable_model_ties"], 0)


if __name__ == "__main__":
    unittest.main()
