"""Tests for coordinate mapping, preview mirroring, and shot card retrieval."""

from __future__ import annotations

import unittest

from travel_photo_assistant.config import AssetPaths
from travel_photo_assistant.coordinates import (
    PreviewMapper,
    PreviewTransformConfig,
    TransformVersionMismatchError,
)
from travel_photo_assistant.knowledge import RetrievalMode, ShotCardRepository, ShotCardRetriever
from travel_photo_assistant.models import RuntimeInput


class TestCoordinatesAndKnowledge(unittest.TestCase):
    def test_contain_and_cover_transform_with_mirroring(self) -> None:
        # 1080x1440 (3:4) fitted into 540x960 (9:16) with contain -> drawn 540x720, y offset=120
        mapper = PreviewMapper(
            PreviewTransformConfig(
                input_width_px=1080,
                input_height_px=1440,
                viewport_width_px=540,
                viewport_height_px=960,
                viewport_fit="contain",
                preview_mirrored=False,
                transform_version=2,
            )
        )
        rect = mapper.map_bbox([0.25, 0.5, 0.75, 1.0], transform_version=2)
        self.assertAlmostEqual(rect.x_min, 135.0, places=1)
        self.assertAlmostEqual(rect.x_max, 405.0, places=1)
        self.assertAlmostEqual(rect.y_min, 480.0, places=1)
        self.assertAlmostEqual(rect.y_max, 840.0, places=1)

        with self.assertRaises(TransformVersionMismatchError):
            mapper.map_bbox([0.25, 0.5, 0.75, 1.0], transform_version=1)

        # Cover mode crops horizontally when fitting 3:4 into 9:16
        cover_mapper = PreviewMapper(
            PreviewTransformConfig(
                input_width_px=1080,
                input_height_px=1440,
                viewport_width_px=540,
                viewport_height_px=960,
                viewport_fit="cover",
                preview_mirrored=True,
                transform_version=1,
            )
        )
        cover_rect = cover_mapper.map_bbox([0.0, 0.2, 0.2, 0.8], transform_version=1)
        self.assertTrue(cover_rect.clipped)
        self.assertEqual(cover_mapper.map_horizontal_direction("left", "input_image"), "right")
        self.assertEqual(cover_mapper.map_horizontal_direction("left", "photographer_body"), "left")
        self.assertEqual(cover_mapper.map_horizontal_direction("right", "subject_body"), "right")

    def test_shot_card_retriever_modes_and_constraint_gating(self) -> None:
        repo = ShotCardRepository(AssetPaths.default().shot_cards)
        self.assertEqual(len(repo.all_cards()), 9)
        retriever = ShotCardRetriever(repo)

        obs_coast_with_line = {
            "status": "observed",
            "scene_tags": ["coast", "single_person", "leading_line"],
            "selected_subject": {"subject_id": "person_1"},
            "anchors": [{
                "anchor_id": "c1", "kind": "leading_line", "description": "岸线",
                "points": [[0.1, 0.4], [0.8, 0.8]],
            }],
            "issues": [],
            "light": {"appearance": "柔光"},
        }
        rt = RuntimeInput.from_dict(
            {
                "frame": {"frame_id": "f1", "scene_version": 1},
                "user_goal": {"priority": "自然旅行人像，保留海岸环境"},
            }
        )

        # Mode A: prompt_only -> 0 cards
        res_a = retriever.retrieve(obs_coast_with_line, rt, mode=RetrievalMode.PROMPT_ONLY)
        self.assertEqual(res_a.selected_cards, [])

        # Mode B: fixed_cards -> curated subset
        res_b = retriever.retrieve(obs_coast_with_line, rt, mode=RetrievalMode.FIXED_CARDS)
        self.assertEqual(len(res_b.selected_cards), 3)

        # Mode C: retrieved_cards -> selects COAST_LINE_ENV_01
        res_c = retriever.retrieve(obs_coast_with_line, rt, mode=RetrievalMode.RETRIEVED_CARDS)
        self.assertIn("COAST_LINE_ENV_01", [c.id for c in res_c.selected_cards])

        # Mode C without leading_line evidence -> MUST reject COAST_LINE_ENV_01 (E11)
        obs_coast_no_line = {
            "status": "observed",
            "scene_tags": ["coast", "single_person", "leading_line"],
            "selected_subject": {"subject_id": "person_1"},
            "anchors": [],
            "issues": [],
            "light": {"appearance": "柔光"},
        }
        res_c_missing = retriever.retrieve(obs_coast_no_line, rt, mode=RetrievalMode.RETRIEVED_CARDS)
        self.assertNotIn("COAST_LINE_ENV_01", [c.id for c in res_c_missing.selected_cards])
        for anchor in (
            {"kind": "leading_line"},
            {"kind": "leading_line", "points": [[0.2, 0.3], [0.2, 0.3]]},
            {"kind": "leading_line", "bbox": [0.5, 0.1, 0.5, 0.9]},
            {"kind": "leading_line", "bbox": [0.1, 0.1, 1.5, 0.9]},
            {"kind": "leading_line", "points": [[0.2, 0.3], [True, 0.8]]},
            {"kind": "leading_line", "points": [[0.2, 0.3], [float("nan"), 0.8]]},
            {"kind": "landmark", "bbox": [0.2, 0.3, 0.6, 0.7]},
        ):
            with self.subTest(anchor=anchor):
                result = retriever.retrieve({**obs_coast_no_line, "anchors": [anchor]}, rt)
                self.assertNotIn("COAST_LINE_ENV_01", [c.id for c in result.selected_cards])

        bbox_only = {"kind": "leading_line", "bbox": [0.2, 0.3, 0.6, 0.7]}
        result = retriever.retrieve({**obs_coast_no_line, "anchors": [bbox_only]}, rt)
        self.assertIn("COAST_LINE_ENV_01", [c.id for c in result.selected_cards])

        for scene, card in (
            ("forest", "FOREST_PATH_DEPTH_03"),
            ("city_street", "STREET_DEPTH_05"),
        ):
            with self.subTest(scene=scene):
                tagged = {**obs_coast_no_line, "scene_tags": [scene, "single_person", "leading_line"]}
                self.assertNotIn(card, [c.id for c in retriever.retrieve(tagged, rt).selected_cards])
                anchored = {**tagged, "anchors": [bbox_only]}
                self.assertIn(card, [c.id for c in retriever.retrieve(anchored, rt).selected_cards])


if __name__ == "__main__":
    unittest.main()
