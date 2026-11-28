import unittest

from portrait_people_typed import VERSION, audit_pair, inspect

HEAD1 = [120, 110, 170, 180]
HEAD2 = [190, 105, 250, 175]
BODY1 = [100, 100, 300, 500]
BODY2 = [150, 90, 350, 510]


def row(kind, box, *heads):
    return {"kind": kind, "box": box, "evidence": "fixture",
            "head_cues": [{"box": h, "evidence": "visible head"} for h in heads]}


def inspect_rows(*rows, scan="complete"):
    return inspect({"scan": scan, "scan_evidence": "fixture", "observations": list(rows)})


class TypedPeople(unittest.TestCase):
    def test_solo(self):
        self.assertEqual(inspect_rows(row("person", BODY1, HEAD1)),
                         {"verdict": "pass", "proven_at_least": 1})

    def test_solo_head_outside_frame(self):
        self.assertEqual(inspect_rows(row("person", BODY1))["verdict"], "pass")

    def test_crowd_without_head_proof(self):
        self.assertEqual(inspect_rows(row("group", BODY1)),
                         {"verdict": "unknown", "proven_at_least": 0})

    def test_unresolved_group_with_one_cue_blocks_solo(self):
        self.assertEqual(inspect_rows(row("person", BODY1, HEAD1), row("group", BODY1, HEAD1)),
                         {"verdict": "unknown", "proven_at_least": 1})

    def test_group_with_pair_despite_incomplete_scan(self):
        self.assertEqual(inspect_rows(row("group", BODY1, HEAD1, HEAD2), scan="incomplete"),
                         {"verdict": "fail", "proven_at_least": 2})

    def test_group_member_duplicates_do_not_increase_bound(self):
        g = row("group", BODY1, HEAD1, HEAD2)
        self.assertEqual(inspect_rows(g, g, row("person", BODY1, HEAD1), row("person", BODY2, HEAD2)),
                         {"verdict": "fail", "proven_at_least": 2})

    def test_duplicate_head_not_pair(self):
        for head in (HEAD1, [130, 120, 160, 175]):
            with self.subTest(head=head):
                self.assertEqual(inspect_rows(row("group", BODY1, HEAD1, head)),
                                 {"verdict": "unknown", "proven_at_least": 1})

    def test_overlapping_bodies_with_distinct_heads(self):
        self.assertEqual(inspect_rows(row("person", BODY1, HEAD1), row("person", BODY2, HEAD2))["verdict"],
                         "fail")

    def test_disjoint_body_fragments_without_second_head(self):
        self.assertEqual(inspect_rows(row("person", BODY1, HEAD1),
                                      row("person", [400, 300, 500, 500]))["verdict"], "unknown")

    def test_no_people_requires_complete_scan(self):
        self.assertEqual(inspect_rows(), {"verdict": "fail", "proven_at_least": 0})
        self.assertEqual(inspect_rows(scan="incomplete"), {"verdict": "unknown", "proven_at_least": 0})

    def test_malformed_evidence_after_pair_invalidates_receipt(self):
        for invalid in (row("group", BODY1, [0, 10, 80, 90]),
                        row("person", BODY1, HEAD1, HEAD2),
                        row("uncertain", BODY1, HEAD1), row("unexpected", BODY1)):
            with self.subTest(row=invalid), self.assertRaises(ValueError):
                inspect_rows(row("group", BODY1, HEAD1, HEAD2), invalid)

    def test_version_and_both_images_are_validated(self):
        image = {"scan": "complete", "scan_evidence": "clear", "observations": []}
        self.assertEqual(audit_pair({"version": VERSION, "A": image, "B": image})["B"]["verdict"], "fail")
        for receipt in ({"version": "people-enumeration-v0.1", "A": image, "B": image},
                        {"version": VERSION, "A": image, "B": {}}):
            with self.assertRaises(ValueError):
                audit_pair(receipt)
