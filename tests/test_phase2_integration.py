"""Actual app + real widget deserialization + loopback OpenAI HTTP API.

RUN_LOCAL_MODEL_TESTS=1 uses the real cached MiniLM and neural reranker.
Otherwise deterministic test vectors avoid mandatory model downloads.
Neither mode calls DeepSeek or consumes paid tokens.
"""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
from openai import OpenAI
from streamlit.testing.v1 import AppTest
from src.api import ManagedClient
from src.embeddings import SentenceEmbeddingAdapter
from src.i18n import TRANSLATIONS
from src.evaluation import save_report as actual_save_report
from scripts.validation_fixtures import create_validation_fixtures
from tests.local_mock_api import LocalMockAPI
from tests.real_uploads import UploadDriver
from tests.test_retrieval import ToyEmbedding


class RealAppIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.real_models = os.environ.get('RUN_LOCAL_MODEL_TESTS') == '1'
        cls.model = SentenceEmbeddingAdapter() if cls.real_models else ToyEmbedding()

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.files=create_validation_fixtures(self.root/'fixtures')
        self.mock=LocalMockAPI().__enter__()
        self.client=ManagedClient(OpenAI(api_key='local-validation-only',base_url=self.mock.base_url,
                                         max_retries=0,timeout=10))
        self.driver=UploadDriver()
        self.patches=[
            patch('streamlit.testing.v1.local_script_runner.MemoryUploadedFileManager',return_value=self.driver.manager),
            patch('src.config.load_llm_client',return_value=self.client),
            patch('src.ui.load_embedding_model',return_value=self.model),
            patch('src.storage.DATA_DIR',self.root),patch('src.storage.CHAT_FILE',self.root/'chats.json'),
            patch('src.paper_reader.DATA_DIR',self.root),patch('src.paper_reader.PAPER_DIR',self.root/'papers'),
            patch('src.paper_reader.IMAGE_DIR',self.root/'paper_images'),
            patch('src.paper_structure.DATA_DIR',self.root),patch('src.media.DATA_DIR',self.root),
            patch('src.media.IMAGE_DIR',self.root/'uploads'/'images'),
            patch('src.web_search.fetch_web_snippets',return_value=[]),
            patch('src.evaluation.save_report',side_effect=lambda report:actual_save_report(report,self.root/'evaluation')),
        ]
        if not self.real_models:
            self.patches.append(patch('src.reranking.load_reranker',side_effect=ImportError))
        for p in self.patches:
            p.start()
        self.app=AppTest.from_file('app.py').run(timeout=90)
        self.app.selectbox(key='ui_language_select').set_value('zh').run(timeout=90)
        self.driver.bind(self.app)

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.mock.__exit__(None,None,None)
        self.temp.cleanup()

    def button(self,key):
        return next(b for b in self.app.button if b.label==TRANSLATIONS['zh'][key])

    def assert_ok(self):
        self.assertEqual([e.message for e in self.app.exception],[])
        self.assertEqual(len(self.app.error),0,[e.value for e in self.app.error])

    def test_real_upload_image_preview_followup_and_disk_persistence(self):
        self.driver.upload(self.app,TRANSLATIONS['zh']['image_upload'],[self.files['image']])
        self.assertGreater(len(self.app.get('imgs')),0)
        self.button('image_send').click().run(timeout=90)
        self.assert_ok()
        self.assertEqual(self.mock.summaries[-1]['model'],'deepseek-flash')
        self.assertEqual(len(self.mock.summaries[-1]['images']),1)
        self.assertGreater(self.mock.summaries[-1]['images'][0]['blue_pixels_sample'],100)
        self.app.chat_input[0].set_value('继续解释刚才的图').run(timeout=90)
        self.assert_ok()
        self.assertEqual(len(self.mock.summaries[-1]['images']),1)
        self.assertFalse(any(r['invalid_image_roles'] for r in self.mock.summaries))
        disk=(self.root/'chats.json').read_text()
        self.assertNotIn('base64',disk)
        self.assertIn('attachments',disk)
        restored=AppTest.from_file('app.py').run(timeout=90)
        current=restored.session_state['chats'][restored.session_state['current_chat_id']]
        self.assertEqual(len(current['messages']),4)
        self.assertTrue(current['messages'][0]['attachments'])

    def test_pdf_upload_visual_qa_multidoc_agent_and_strategy_switch(self):
        self.app.sidebar.radio[0].set_value('paper').run(timeout=90)
        self.driver.upload(self.app,TRANSLATIONS['zh']['upload_paper_pdf'],[self.files['alpha'],self.files['beta']])
        self.button('btn_parse_paper').click().run(timeout=90)
        self.assert_ok()
        chunks=self.app.session_state['paper_chunks']
        self.assertEqual({c['source'] for c in chunks},{self.files['alpha'].name,self.files['beta'].name})
        self.assertTrue(any(c['kind']=='table' and '81.0' in c['text'] for c in chunks))
        calls=len(self.mock.requests)
        index=self.app.session_state['paper_index']
        self.button('btn_parse_paper').click().run(timeout=90)
        self.assertEqual(self.mock.requests.__len__(),calls,'identical import must reuse summary')
        self.assertIs(self.app.session_state['paper_index'],index)
        self.app.chat_input[0].set_value('Which color has the tallest bar in Figure 3?').run(timeout=90)
        self.assert_ok()
        visual_requests=[r for r in self.mock.summaries if r['images']]
        self.assertEqual(len(visual_requests[-1]['images']),1)
        self.assertGreater(visual_requests[-1]['images'][0]['blue_pixels_sample'],100)
        current=self.app.session_state['chats'][self.app.session_state['current_chat_id']]
        evidence=current['messages'][-1]['evidence']
        self.assertTrue(any(e.get('label')=='Figure 3' and e['page']==2 and e['source']==self.files['alpha'].name for e in evidence))
        content=self.mock.requests[-1]['messages'][-1]['content']
        image_index=next(i for i,b in enumerate(content) if b.get('type')=='image_url')
        self.assertIn('synthetic-aster.pdf',content[image_index-1].get('text',''))
        self.assertIn('p. 2',content[image_index-1].get('text',''))
        for strategy in ('dense','hybrid','hybrid_rerank'):
            self.app.selectbox(key='retrieval_strategy').set_value(strategy).run(timeout=90)
            self.app.chat_input[0].set_value('Compare Aster and Boreal controlled results.').run(timeout=90)
            self.assert_ok()
            current=self.app.session_state['chats'][self.app.session_state['current_chat_id']]
            self.assertEqual({e['source'] for e in current['messages'][-1]['evidence']},
                             {self.files['alpha'].name,self.files['beta'].name})
            self.assertGreater(len(self.app.get('json')),0)

    def test_csv_excel_loading_invalidates_stale_dictionary_and_report(self):
        self.app.sidebar.radio[0].set_value('data').run(timeout=90)
        self.driver.upload(self.app,TRANSLATIONS['zh']['data_upload'],[self.files['csv']])
        self.app.button(key='da_btn_load').click().run(timeout=90)
        self.assert_ok()
        self.assertEqual(self.app.session_state['da_df'].shape,(3,3))
        self.app.button(key='da_dict_build').click().run(timeout=90)
        self.assertTrue(self.app.session_state['da_dict_chunks'])
        self.app.button(key='da_report_btn').click().run(timeout=90)
        self.assertTrue(self.app.session_state['da_last_report'])
        newer=self.root/'new.xlsx'
        pd.DataFrame({'product':['A','B','C'],'profit':[4,6,8]}).to_excel(newer,index=False)
        self.driver.upload(self.app,TRANSLATIONS['zh']['data_upload'],[newer])
        self.app.button(key='da_btn_load').click().run(timeout=90)
        self.assert_ok()
        self.assertEqual(list(self.app.session_state['da_df'].columns),['product','profit'])
        self.assertEqual(self.app.session_state['da_dict_chunks'],[])
        self.assertEqual(self.app.session_state['da_last_report'],'')
        self.app.chat_input[0].set_value('显示前两行').run(timeout=90)
        self.assert_ok()
        self.assertGreater(len(self.app.dataframe),0)

    def test_evaluation_dashboard_and_four_languages(self):
        self.app.sidebar.radio[0].set_value('eval').run(timeout=90)
        self.button('eval_run').click().run(timeout=90)
        self.assert_ok()
        report=self.app.session_state['_evaluation_report']
        self.assertEqual(report['api_calls'],0)
        self.assertEqual(report['conditions']['k'],5)
        self.assertEqual(report['conditions']['mrr_cutoff'],None)
        self.assertGreater(len(self.app.get('plotly_chart')),0)
        for language in ('en','de','fr','zh'):
            self.app.selectbox(key='ui_language_select').set_value(language).run(timeout=90)
            self.assert_ok()
