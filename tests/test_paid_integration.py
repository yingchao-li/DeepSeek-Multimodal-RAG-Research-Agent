"""Never runs in the default suite; explicitly billed when opted in."""
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from src import media
from src.api import ManagedClient
from src.llm import generate_chat_answer


@unittest.skipUnless(os.environ.get('RUN_PAID_API_TESTS') == '1',
                     'Paid API integration is disabled; set RUN_PAID_API_TESTS=1 to opt in')
class PaidIntegrationTests(unittest.TestCase):
    def test_native_vision_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(media, 'DATA_DIR', root), patch.object(media, 'IMAGE_DIR', root/'uploads'/'images'):
                buffer = io.BytesIO()
                Image.new('RGB', (64,64), 'red').save(buffer, 'PNG')
                image = media.save_image(buffer.getvalue(), 'red.png')
                answer = generate_chat_answer('What is the dominant color? One word.', [],
                                              ManagedClient(), images=[image], lang_code='en')
                self.assertTrue(answer.strip())
