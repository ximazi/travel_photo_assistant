import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from travel_photo_assistant.adapters.providers import GeminiVisionAdapter
from travel_photo_assistant.field_pilot import main, prepare_photo


class FieldPilotTest(unittest.TestCase):
    def test_blind_even_when_name_collides_and_rotation_is_normalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            named = Path(tmp) / "coast_person.jpg"
            neutral = Path(tmp) / "neutral.jpg"
            exif = Image.Exif()
            exif[274] = 6
            exif[270] = "ground truth that must not leave the machine"
            Image.new("RGB", (32, 12), "blue").save(named, exif=exif)
            neutral.write_bytes(named.read_bytes())
            a, data_a = prepare_photo(named, "人景兼顾")
            b, data_b = prepare_photo(neutral, "人景兼顾")
            self.assertEqual(data_a, data_b)
            self.assertEqual(a.frame.upright_input_width_px, 12)
            self.assertEqual(a.frame.upright_input_height_px, 32)
            with Image.open(io.BytesIO(data_a)) as im:
                self.assertEqual(im.size, (12, 32))
                self.assertFalse(im.getexif())
            for req in (a, b):
                self.assertEqual(req.local_vision_hints, {})
                self.assertIsNone(req.user_goal.subject_id)
                self.assertIsNone(req.user_goal.subject_count)
                self.assertFalse(req.camera_capabilities.metadata_available)
                self.assertEqual(req.camera_capabilities.available_lenses, [])
                self.assertFalse(req.movement_constraints.standing_in_place_confirmed)

            schema = {"type": "object", "properties": {"result": {"type": "string"}}}
            def check_transport(req, **_):
                self.assertNotIn("mock-key", req.full_url)
                self.assertEqual(req.get_header("X-goog-api-key"), "mock-key")
                contents = json.loads(req.data)["contents"][0]["parts"]
                self.assertEqual(len(contents), 2)
                for banned in ("coast_person", "local_path", "image_attachments", str(named)):
                    self.assertNotIn(banned, contents[0]["text"])
                self.assertEqual(base64.b64decode(contents[1]["inline_data"]["data"]), data_a)
                return io.BytesIO(json.dumps({"candidates": [{"content": {"parts": [{"text": '{"result":"stub"}'}]}}]}).encode())
            adapter = GeminiVisionAdapter(api_key="mock-key")
            with patch("urllib.request.urlopen", side_effect=check_transport) as transport:
                self.assertEqual(adapter.observe(a, "observe", schema), {"result": "stub"})
                self.assertEqual(adapter.plan(a, {}, [], "plan", schema), {"result": "stub"})
                self.assertEqual(transport.call_count, 2)

    def test_no_network_without_both_acknowledgments(self):
        args = ["--photo", "does-not-exist.jpg", "--goal", "人景兼顾", "--out", "unused",
                "--model", "gemini-2.5-flash", "--allow-photo-to-gemini"]
        with patch("urllib.request.urlopen") as transport, patch("sys.stderr"):
            with self.assertRaises(SystemExit) as ex:
                main(args)
            self.assertEqual(ex.exception.code, 2)
            transport.assert_not_called()
