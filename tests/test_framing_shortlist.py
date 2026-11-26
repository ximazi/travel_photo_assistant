import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image
from travel_photo_assistant.framing_shortlist import Proposal, export_clean_crops, shortlist

ORIGINAL = (0, 0, 1, 1)
ANCHORS = [
    {"kind": "landmark", "bbox": [.45, .16, .62, .58]},
    {"kind": "possible_ground", "bbox": [.12, .68, .84, .96]},
]


class ShortlistTest(unittest.TestCase):
    def test_same_pixels_two_safe_boxes_and_one_bad_box(self):
        report = shortlist([
            Proposal("left", ORIGINAL, (.18, .42, .28, .84), baseline=True),
            Proposal("right", ORIGINAL, (.68, .42, .78, .84), baseline=True),
            Proposal("repeat left", ORIGINAL, (.18, .42, .28, .84), baseline=True),
            Proposal("blocks tower", ORIGINAL, (.48, .32, .58, .83), baseline=True),
            Proposal("foot not on ground", ORIGINAL, (.03, .35, .11, .63), baseline=True),
        ], ANCHORS, (1200, 900))
        self.assertEqual(report["clean_crop_count"], 1)
        self.assertEqual(len(report["groups"][0]["placements"]), 2)
        self.assertEqual(report["rejected"][0]["reasons"], ["silhouette_overlaps_landmark"])
        self.assertEqual(report["rejected"][1]["reasons"], ["footpoint_not_on_visible_ground"])
        self.assertTrue(all(not p["standing_safety_verified"] for p in report["groups"][0]["placements"]))
        self.assertIsNone(report["placement_winner"])

    def test_crop_uses_pixel_aspect_and_cannot_cut_landmark(self):
        fig = (.68, .42, .78, .84)
        report = shortlist([
            Proposal("valid 4:3", (.10, .10, .95, .95), fig, aspect="4:3"),
            Proposal("normalized square is not actual square", (.10, .10, .95, .95), fig, aspect="1:1"),
            Proposal("cuts tower", (.54, .02, .99, .92), fig),
            Proposal("missing local ground", ORIGINAL, (.68, .1, .78, .5), baseline=True),
        ], ANCHORS, (1200, 900))
        self.assertEqual(report["clean_crop_count"], 1)
        self.assertIn("pixel_aspect_mismatch", report["rejected"][0]["reasons"])
        self.assertIn("landmark_cut_or_too_close_to_edge", report["rejected"][1]["reasons"])
        self.assertIn("too_few_original_pixels", report["rejected"][1]["reasons"])
        self.assertIn("footpoint_not_on_visible_ground", report["rejected"][2]["reasons"])


    def test_export_deduped_unmarked_crop(self):
        report = shortlist([
            Proposal("first", ORIGINAL, (.18, .42, .28, .84), baseline=True),
            Proposal("second", ORIGINAL, (.68, .42, .78, .84), baseline=True),
        ], ANCHORS, (1200, 900))
        image = BytesIO()
        Image.new("RGB", (1200, 900), (63, 63, 63)).save(image, format="JPEG")
        with tempfile.TemporaryDirectory() as folder:
            dest = Path(folder) / "review"
            export_clean_crops(image.getvalue(), report, dest)
            self.assertEqual([p.name for p in dest.iterdir()], ["frame-01.jpg"])
            with Image.open(dest / report["groups"][0]["clean_preview"]) as img:
                self.assertEqual(img.size, (1200, 900))
                self.assertEqual(img.getextrema(), ((63, 63), (63, 63), (63, 63)))
