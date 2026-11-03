import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from src import media
from src.llm import generate_chat_answer, generate_rag_answer
from tests.helpers import FakeClient


class VisionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.patches = [patch.object(media, "DATA_DIR", root),
                        patch.object(media, "IMAGE_DIR", root / "uploads" / "images"),
                        patch.dict("os.environ", {"DEEPSEEK_TEXT_MODEL": "text-test",
                                                  "DEEPSEEK_VISION_MODEL": "deepseek-flash"})]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def image(self, fmt="PNG"):
        data = io.BytesIO()
        Image.new("RGB", (40, 30), "blue").save(data, fmt)
        return media.save_image(data.getvalue(), "sample." + fmt.lower())

    def test_all_supported_formats_and_small_descriptors(self):
        for fmt in ("JPEG", "PNG", "GIF", "WEBP"):
            with self.subTest(fmt=fmt):
                image = self.image(fmt)
                self.assertTrue(media.attachment_path(image).is_file())
                self.assertNotIn("base64", json.dumps(image))
                self.assertLess(len(json.dumps(image)), 600)

    def test_real_multimodal_request_and_followup(self):
        image = self.image()
        client = FakeClient()
        generate_chat_answer("explain chart", [], client, images=[image])
        generate_chat_answer("what about the blue line?", [
            {"role": "user", "content": "explain chart", "attachments": [image]},
            {"role": "assistant", "content": "answer", "attachments": [image]}], client)
        for call in client.calls:
            self.assertEqual(call["model"], "deepseek-flash")
            for msg in call["messages"]:
                if isinstance(msg["content"], list):
                    self.assertEqual(msg["role"], "user")
                    self.assertEqual(msg["content"][0]["type"], "text")
                    self.assertTrue(msg["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_text_only_and_rag_models(self):
        client = FakeClient()
        generate_chat_answer("hello", [], client)
        self.assertEqual(client.calls[-1]["model"], "text-test")
        generate_rag_answer("Figure 1?", [{"source": "paper.pdf", "text": "Fig 1", "page": 2}],
                            [], client, images=[self.image()])
        self.assertEqual(client.calls[-1]["model"], "deepseek-flash")
        self.assertIn("p. 2", client.calls[-1]["messages"][1]["content"][0]["text"])

    def test_validation_and_path_escape(self):
        for raw in (b"bad-image", b"x" * (media.MAX_UPLOAD_BYTES + 1)):
            with self.assertRaises(media.MediaError):
                media.save_image(raw)
        with self.assertRaises(media.MediaError):
            media.attachment_path({"path": "../../secrets.toml"})
        with self.assertRaises(media.MediaError):
            media.user_content("x", [self.image()] * 5)

    def test_history_image_budget(self):
        history = [{"role": "user", "content": str(i), "attachments": [self.image()]} for i in range(10)]
        messages = media.chat_messages("system", history, "followup")
        count = sum(sum(b.get("type") == "image_url" for b in m["content"])
                    for m in messages if isinstance(m["content"], list))
        self.assertEqual(count, 1)  # Identical pixels are sent once across history.
