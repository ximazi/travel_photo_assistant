"""Document known geometry failure modes; not model or human results."""
import unittest

from test_portrait_people_enumeration import count_requirement, frame, instance


class GeometryLimitTests(unittest.TestCase):
    def test_v01_disjoint_fragments_can_false_fail(self):
        # Fixture semantics: head and torso are from the same person.
        img = frame(instance([450, 100, 550, 180]), instance([420, 181, 580, 700]))
        self.assertEqual(count_requirement(img), "fail")  # Actual; expected one.

    def test_v01_heavily_overlapping_people_can_abstain(self):
        # Fixture semantics: these are two distinct people.
        img = frame(instance([400, 100, 550, 800]), instance([440, 150, 580, 790]))
        self.assertEqual(count_requirement(img), "unknown")  # Actual; expected multiple.
