import csv
import itertools
import tempfile
import unittest
from pathlib import Path

from tally_aesthetic_anchors import FIELDS, GRADES, category, tally


def vote(grade, rid="a", photo="P01", axis="composition", **extra):
    return dict(reviewer_id=rid, image_id=photo, axis=axis, grade=grade,
                region="left edge", pixel_evidence="two separated shapes",
                extra_effect="keeps gestures legible") | extra


class CoverageTests(unittest.TestCase):
    def test_all_36_rating_combinations_preserve_disagreement(self):
        for a, b in itertools.product(("NA",) + GRADES[2:], repeat=2):
            with self.subTest(a=a, b=b):
                expected = ("unavailable" if "NA" in (a, b)
                            else "boundary_split" if {a, b} == {"3", "4"}
                            else "other_disagreement" if a != b
                            else {"3": "consensus_3", "4": "consensus_4"}.get(a, "agreed_0_to_2"))
                self.assertEqual(category(vote(a), vote(b)), expected)

    def test_missing_evidence_is_not_a_reference(self):
        for field in ("region", "pixel_evidence", "extra_effect"):
            self.assertEqual(category(vote("4", **{field: " "}), vote("4")), "insufficient_4_evidence")
        self.assertEqual(category(vote("3", region=""), vote("3")), "insufficient_evidence")
        self.assertEqual(category(None, vote("4")), "unavailable")

    def write(self, p, rows):
        with p.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def test_photo_axis_denominator_and_blank_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = (Path(tmp) / name for name in ("a.csv", "b.csv"))
            self.write(a, [vote("3")])
            self.write(b, [vote("4", "b")])
            r = tally({"P01", "P02", "P03"}, (a, b))
            self.assertEqual(r["ballot_reviewers_present"], 2)
            comp = r["axes"]["composition"]
            self.assertEqual((comp["eligible_photo_rows"], comp["boundary_split"], comp["unavailable"]), (3, 1, 2))
            self.assertEqual(r["axes"]["pose_story"]["unavailable"], 3)

    def test_bad_inputs_never_produce_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = (Path(tmp) / name for name in ("a.csv", "b.csv"))
            self.write(b, [vote("4", "a")])
            for rows in ([vote("3","A")], [vote("3.5","b")], [vote("4","b",photo="P99")],
                         [vote("4","b"), vote("4","b")], [vote("4","")]):
                self.write(a, rows)
                with self.subTest(rows=rows), self.assertRaises(ValueError):
                    tally({"P01"}, (a, b))


if __name__ == "__main__":
    unittest.main()
