"""Real Streamlit execution; persistence and API are isolated from user data."""
import unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from src.api import ManagedClient
from tests.helpers import FakeClient


class UITests(unittest.TestCase):
    def setUp(self):
        self.client = ManagedClient(FakeClient())
        self.patches = [patch('src.chat.load_chats_from_disk', return_value={}),
                        patch('src.chat.save_chats_to_disk'),
                        patch('src.config.load_llm_client', return_value=self.client),
                        patch('src.web_search.fetch_web_snippets', return_value=[])]
        for p in self.patches:
            p.start()
        self.app = AppTest.from_file('app.py').run(timeout=30)

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()

    def test_all_modes_and_languages_start(self):
        self.assertEqual(len(self.app.exception), 0)
        for mode in ('paper', 'data', 'eval', 'doc'):
            self.app.sidebar.radio[0].set_value(mode).run(timeout=30)
            self.assertEqual(len(self.app.exception), 0, mode)
        for language in ('en', 'de', 'fr', 'zh'):
            self.app.selectbox(key='ui_language_select').set_value(language).run(timeout=30)
            self.assertEqual(len(self.app.exception), 0, language)

    def test_mock_text_chat_runs_in_existing_app(self):
        self.app.chat_input[0].set_value('hello').run(timeout=30)
        self.assertEqual(len(self.app.exception), 0)
        messages = self.app.session_state['chats'][self.app.session_state['current_chat_id']]['messages']
        self.assertEqual(messages[-1]['role'], 'assistant')
        self.assertEqual(messages[-1]['content'], 'mock answer')
        self.assertEqual(len(self.client._sdk.calls), 2)  # Query rewrite + answer.
        answer_call = self.client._sdk.calls[-1]
        self.assertEqual(sum(m.get('content') == 'hello' for m in answer_call['messages']), 1)

    def test_native_vision_upload_and_followup_in_existing_app(self):
        import io
        import tempfile
        from pathlib import Path
        from PIL import Image
        from src import media
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(media, 'DATA_DIR', root), patch.object(media, 'IMAGE_DIR', root/'uploads'/'images'):
                image_buffer = io.BytesIO()
                Image.new('RGB', (64, 64), 'blue').save(image_buffer, 'PNG')
                attachment = media.save_image(image_buffer.getvalue(), 'chart.png')
                with patch('src.ui.image_upload', return_value=([attachment], True, False, '_test_epoch')):
                    self.app.run(timeout=30)
                self.assertEqual(len(self.app.exception), 0)
                self.assertEqual(self.client._sdk.calls[-1]['model'], 'deepseek-flash')
                messages = self.app.session_state['chats'][self.app.session_state['current_chat_id']]['messages']
                self.assertEqual(messages[-2]['attachments'][0]['name'], 'chart.png')
                with patch('src.ui.image_upload', return_value=([], False, False, '_test_epoch')):
                    self.app.chat_input[0].set_value('What is the dominant color?').run(timeout=30)
                self.assertEqual(len(self.app.exception), 0)
                last = self.client._sdk.calls[-1]
                self.assertEqual(last['model'], 'deepseek-flash')
                self.assertTrue(any(isinstance(m['content'], list) for m in last['messages']))

    def test_paper_figure_qa_uses_vision_and_page_evidence(self):
        self.client._sdk.replies = iter(['Figure 3 shows the architecture [S1].'] * 30)
        import tempfile
        from pathlib import Path
        import fitz
        from src import media
        from src.paper_structure import parse_paper
        from src.vector_store import build_retrieval_index
        from src.i18n import TRANSLATIONS
        from tests.test_retrieval import ToyEmbedding
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'papers').mkdir()
            pdf_path = root/'papers'/'architecture.pdf'
            with fitz.open() as pdf:
                page = pdf.new_page()
                page.insert_text((72,72), 'Figure 3. Neural network architecture.')
                page.draw_rect(fitz.Rect(72,120,260,200))
                pdf.save(pdf_path)
            parsed = parse_paper(pdf_path, 'architecture.pdf')
            model = ToyEmbedding()
            index = build_retrieval_index(parsed['chunks'], model)
            with patch('src.ui.load_embedding_model', return_value=model), \
                 patch('src.paper_structure.DATA_DIR', root), patch.object(media, 'DATA_DIR', root):
                self.app.sidebar.radio[0].set_value('paper').run(timeout=30)
                for checkbox in self.app.checkbox:
                    if checkbox.label == TRANSLATIONS['zh']['agentic_checkbox']:
                        checkbox.set_value(False)
                state = {'paper_ready':True, 'paper_vector_ready':True, 'paper_index':index,
                         'paper_chunks':parsed['chunks'], 'paper_text':parsed['text']}
                for key, value in state.items():
                    self.app.session_state[key] = value
                self.app.chat_input[0].set_value('Explain Figure 3.').run(timeout=30)
                self.assertEqual(len(self.app.exception), 0)
                self.assertEqual(self.client._sdk.calls[-1]['model'], 'deepseek-flash')
                messages = self.app.session_state['chats'][self.app.session_state['current_chat_id']]['messages']
                self.assertEqual(messages[-1]['evidence'][0]['source'], 'architecture.pdf')
                self.assertEqual(messages[-1]['evidence'][0]['page'], 1)
