"""Tests for local JSON schema transport; no model or photo required."""
import unittest

from test_portrait_observation_verdict import BRIEF, VERSION, adjudicate, image, receipt
from portrait_observation_schema import observation_schema

try:
    from jsonschema import Draft202012Validator
except ImportError:
    Draft202012Validator = None


@unittest.skipUnless(Draft202012Validator, "requires optional jsonschema")
class ObservationSchemaTests(unittest.TestCase):
    def setUp(self):
        schema = observation_schema(BRIEF)
        Draft202012Validator.check_schema(schema)
        self.validator = Draft202012Validator(schema)

    def test_keeps_abstention_and_unknown(self):
        self.assertTrue(self.validator.is_valid(
            {"version": VERSION, "status": "abstain", "reason": "unreadable"}))
        a = image(second="unknown", goal="unknown")
        a["scores"]["technical"].update(score=None, region=None)
        self.assertTrue(self.validator.is_valid(receipt(a, image())))
        self.assertIsNone(adjudicate(BRIEF, receipt(a, image()))["A"]["intent_score"])

    def test_rejects_bare_goal_and_extra_keys(self):
        a = image()
        a["goal"] = "met"
        self.assertFalse(self.validator.is_valid(receipt(a, image())))
        a = image()
        a["winner"] = "A"
        self.assertFalse(self.validator.is_valid(receipt(a, image())))

    def test_rejects_missing_requirements_and_unasked_id(self):
        a = image()
        a["requirements"].pop()
        self.assertFalse(self.validator.is_valid(receipt(a, image())))
        a = image()
        a["requirements"][0]["id"] = "other"
        self.assertFalse(self.validator.is_valid(receipt(a, image())))

    def test_duplicate_opaque_id_still_needs_adjudicator(self):
        a = image()
        a["requirements"][1]["id"] = "x17"
        raw = receipt(a, image())
        self.assertTrue(self.validator.is_valid(raw))
        with self.assertRaises(ValueError):
            adjudicate(BRIEF, raw)

    def test_empty_requirements(self):
        brief = {"goal": "Read the setting", "required_elements": []}
        validator = Draft202012Validator(observation_schema(brief))
        a = image()
        a["requirements"] = []
        self.assertTrue(validator.is_valid(receipt(a, a)))
        self.assertFalse(validator.is_valid(receipt(a, image())))
