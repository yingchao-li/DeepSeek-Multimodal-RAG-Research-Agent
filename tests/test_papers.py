import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import fitz
from src.paper_structure import parse_paper, select_visual_evidence, reference_keys
from src.splitter import chunk_text
from src.citations import validate_citations


class PaperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        folder = self.root / 'papers'
        folder.mkdir()
        self.path = folder / 'research.pdf'
        with fitz.open() as pdf:
            page = pdf.new_page()
            page.insert_text((72, 60), '1 Architecture', fontsize=18)
            page.insert_text((72, 100), 'Our network uses a small object detector.', fontsize=11)
            page.draw_rect(fitz.Rect(80, 150, 300, 240))
            page.insert_text((72, 270), 'Figure 3. Neural network architecture.', fontsize=11)
            page = pdf.new_page()
            page.insert_text((72, 60), '2 Experiments', fontsize=18)
            page.insert_text((72, 100), 'Table 2. Small object accuracy is 81 percent.', fontsize=11)
            pdf.save(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_structure_pages_and_stable_ids(self):
        result = parse_paper(self.path, 'research.pdf')
        again = parse_paper(self.path, 'research.pdf')
        self.assertEqual([c['chunk_id'] for c in result['chunks']], [c['chunk_id'] for c in again['chunks']])
        self.assertEqual({c['page'] for c in result['chunks']}, {1, 2})
        self.assertTrue(any(c.get('label') == 'Figure 3' for c in result['visuals']))
        self.assertTrue(any(c['kind'] == 'heading' for c in result['chunks']))

    def test_select_only_relevant_visual_and_crop(self):
        result = parse_paper(self.path, 'research.pdf')
        with patch('src.paper_structure.DATA_DIR', self.root):
            selected, images = select_visual_evidence('Explain Figure 3', [], result['chunks'], max_visuals=1)
        self.assertEqual(selected[0]['label'], 'Figure 3')
        self.assertEqual(len(images), 1)
        self.assertTrue((self.root / images[0]['path']).is_file())
        self.assertEqual(select_visual_evidence('What is the title?', [], result['chunks']), ([], []))

    def test_docling_failure_falls_back(self):
        with patch('src.paper_structure._docling_blocks', side_effect=ImportError):
            result = parse_paper(self.path, parser='docling')
        self.assertEqual(result['parser'], 'pymupdf')
        self.assertIn('docling_fallback', result['warnings'])

    def test_invalid_chunk_settings_and_citations(self):
        with self.assertRaises(ValueError):
            chunk_text('long text', 3, 3)
        answer, invalid = validate_citations('Supported [S1]; invented [S9]', [{'source': 'a'}])
        self.assertIn('[S1]', answer)
        self.assertNotIn('[S9]', answer)
        self.assertEqual(invalid, ['[S9]'])

    def test_plural_and_abbreviated_figure_table_references(self):
        self.assertEqual(reference_keys('Compare Tables 2 and 4.'), {('table', '2'), ('table', '4')})
        self.assertEqual(reference_keys('Explain Fig. 3'), {('figure', '3')})
