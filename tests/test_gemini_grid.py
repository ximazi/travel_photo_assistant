import unittest
from unittest.mock import patch

from travel_photo_assistant.adapters.gemini_grid import CANONICAL, WIRE, GeminiGridPilotAdapter, _geometry
from travel_photo_assistant.adapters.providers import GeminiVisionAdapter
from travel_photo_assistant.config import AssetPaths
from travel_photo_assistant.evaluation import _base_runtime_dict
from travel_photo_assistant.models import RuntimeInput
from travel_photo_assistant.schema_validator import ProviderSchemaAdapter, SchemaValidator


class GeminiGridTest(unittest.TestCase):
    def test_two_stage_adapter_preserves_box_point_and_ratio_units(self):
        req = RuntimeInput.from_dict(_base_runtime_dict())
        schemas = SchemaValidator(AssetPaths.default())
        observation = {"coordinate_space": WIRE, "facts": [{"region": [671,125,864,760]}],
                       "anchors": [{"bbox": [671,125,864,760], "points": [[125,671]]}],
                       "selected_subject": None}
        plan = {"coordinate_space": WIRE,
                "subject_placement_guide": {"bbox": [671,125,864,760], "target_height_ratio": [.24,.46]},
                "crop_recommendation": {"crop_rect": [100,50,1000,980]},
                "primary_plan": {"composition": {"target_subject_bbox": None}},
                "actions": [{"completion_check": {"target_range": [.24,.46]}}]}
        adapter = GeminiGridPilotAdapter(api_key="mock")
        with patch.object(GeminiVisionAdapter, "_generate", side_effect=[observation,plan]) as transport:
            got = adapter.observe(req, "observe", schemas.observation_schema)
            result = adapter.plan(req, got, [], "plan", schemas.shot_plan_schema)
        self.assertEqual(got["coordinate_space"], CANONICAL)
        self.assertEqual(got["facts"][0]["region"], [.125,.671,.760,.864])
        self.assertEqual(got["anchors"][0]["points"], [[.125,.671]])
        self.assertEqual(result["subject_placement_guide"]["bbox"], [.125,.671,.760,.864])
        self.assertEqual(result["subject_placement_guide"]["target_height_ratio"], [.24,.46])
        self.assertEqual(result["actions"][0]["completion_check"]["target_range"], [.24,.46])
        self.assertEqual(result["crop_recommendation"]["crop_rect"], [.05,.1,.98,1])
        self.assertEqual(transport.call_args_list[1].args[1]["scene_observation"], observation)
        self.assertEqual(transport.call_args_list[0].args[0].count(WIRE), 2)
        schema = ProviderSchemaAdapter.adapt(transport.call_args_list[0].args[3], provider="gemini")
        anchors = schema["properties"]["anchors"]["items"]["properties"]
        self.assertEqual(anchors["bbox"]["items"]["type"], "integer")
        self.assertEqual(anchors["points"]["items"]["items"]["type"], "integer")
        self.assertEqual(schema["properties"]["coordinate_space"]["enum"], [WIRE])
        self.assertEqual(schemas.observation_schema["properties"]["coordinate_space"]["const"], CANONICAL)

    def test_mixed_fractional_reversed_and_out_of_range_are_rejected(self):
        for box in ([0,688,1,1], [671,.125,864,760], [0,0,1001,1000],
                    [False,125,864,760], [671,125,864], [671,125,671,760]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                _geometry({"coordinate_space": WIRE, "facts": [{"region": box}]}, observation=True)
        with self.assertRaises(ValueError):
            _geometry({"coordinate_space": CANONICAL}, observation=True)
