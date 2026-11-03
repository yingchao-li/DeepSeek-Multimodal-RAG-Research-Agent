"""Regressions discovered by the second-stage source/fixture review."""
import tempfile
import unittest
import fitz
from pathlib import Path
from unittest.mock import patch
from scripts.validation_fixtures import create_validation_fixtures
from src import media
from src.paper_structure import parse_paper, select_visual_evidence, reference_keys
from src.embeddings import query_embedder
from tests.test_retrieval import ToyEmbedding


class Phase2Regressions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.files = create_validation_fixtures(self.root/'papers')
        self.parsed = parse_paper(self.files['alpha'], self.files['alpha'].name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_figure_keeps_the_section_at_its_own_position(self):
        figure = next(c for c in self.parsed['visuals'] if c.get('label') == 'Figure 3')
        self.assertEqual(figure['page'], 2)
        self.assertEqual(figure['section'], '2 Visual Results')

    def test_explicit_figure_does_not_send_unrelated_logo(self):
        with patch('src.paper_structure.DATA_DIR', self.root):
            selected, images = select_visual_evidence('Which color has the tallest bar in Figure 3?',
                                                       self.parsed['chunks'], self.parsed['chunks'], 3)
        self.assertEqual(len(images), 1)
        self.assertEqual({d.get('label') for d in selected}, {'Figure 3'})

    def test_chinese_adjacent_figure_table_references(self):
        self.assertEqual(reference_keys('解释图3的颜色，并对比表2和4的结果。'),
                         {('figure','3'),('table','2'),('table','4')})

    def test_same_page_image_dedup_retains_both_citations(self):
        docs = [{**self.parsed['chunks'][0], 'kind':'table', 'label':f'Table {n}',
                 'text':f'Table {n}. synthetic caption', 'chunk_id':f'table-{n}',
                 'bbox':[0,0,612,792]} for n in (2,4)]
        with patch('src.paper_structure.DATA_DIR', self.root):
            selected, images = select_visual_evidence('Compare Tables 2 and 4', docs, docs, 1)
        self.assertEqual(len(images), 1)
        self.assertEqual({d['label'] for d in selected}, {'Table 2','Table 4'})

    def test_same_query_is_encoded_once_across_source_searches(self):
        model = ToyEmbedding()
        with patch.object(model, 'encode', wraps=model.encode) as encode:
            embed = query_embedder(model)
            embed(['same question'])
            embed(['same question'])
            self.assertEqual(encode.call_count, 1)

    def test_duplicate_historical_image_sent_only_once(self):
        with patch.object(media,'DATA_DIR',self.root), patch.object(media,'IMAGE_DIR',self.root/'uploads'/'images'):
            attachment=media.save_image(self.files['image'].read_bytes(), 'chart.png')
            history=[{'role':'user','content':str(i),'attachments':[attachment]} for i in range(3)]
            messages=media.chat_messages('system',history,'followup')
            count=sum(sum(b.get('type')=='image_url' for b in m['content'])
                      for m in messages if isinstance(m['content'],list))
        self.assertEqual(count,1)

    def test_pdf_cache_invalidates_after_content_change(self):
        from src import paper_structure
        with patch('src.paper_structure._pymupdf_blocks',wraps=paper_structure._pymupdf_blocks) as parse:
            again=parse_paper(self.files['alpha'],self.files['alpha'].name)
            self.assertEqual(parse.call_count,0)
            with fitz.open(self.files['alpha']) as pdf:
                pdf[0].insert_text((55,390),'New controlled observation.')
                pdf.saveIncr()
            updated=parse_paper(self.files['alpha'],self.files['alpha'].name)
            self.assertEqual(parse.call_count,1)
            self.assertNotEqual(again['chunks'][0]['document_sha256'],updated['chunks'][0]['document_sha256'])
        old=next(c for c in again['visuals'] if c.get('label')=='Figure 3')
        with patch('src.paper_structure.DATA_DIR',self.root),self.assertRaisesRegex(ValueError,'pdf_changed_reparse'):
            paper_structure.render_evidence_image(old)

    def test_download_policy_does_not_duplicate_loaded_reranker(self):
        from src.reranking import load_reranker
        load_reranker.cache_clear()
        with patch('src.reranking.CrossEncoderReranker') as factory:
            first=load_reranker('test-reranker','cpu',False)
            second=load_reranker('test-reranker','cpu',True)
            self.assertIs(first,second)
            self.assertEqual(factory.call_count,1)
        load_reranker.cache_clear()

    def test_cached_image_is_not_decoded_on_every_rerun(self):
        with patch.object(media,'DATA_DIR',self.root),patch.object(media,'IMAGE_DIR',self.root/'uploads'/'images'):
            raw=self.files['image'].read_bytes()
            first=media.save_image(raw,'chart.png')
            with patch('src.media.Image.open',side_effect=AssertionError('unnecessary decoding')):
                second=media.save_image(raw,'chart.png')
            self.assertEqual(first,second)
