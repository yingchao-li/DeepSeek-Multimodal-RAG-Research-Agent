import unittest
from unittest.mock import patch
import pandas as pd
from src.data_analysis import auto_clean_dataframe, dataframe_context_for_llm
from src.data_analysis_ext import build_data_dictionary_chunks, build_dict_index, answer_schema_rag, train_simple_model
from src.chat import get_recent_history
from tests.helpers import FakeClient
from tests.test_retrieval import ToyEmbedding


class DataCompatibilityTests(unittest.TestCase):
    def test_text_only_table_and_dictionary(self):
        cleaned, _ = auto_clean_dataframe(pd.DataFrame({'city':['Berlin','Paris',None]}))
        self.assertIn('city', dataframe_context_for_llm(cleaned))
        chunks = build_data_dictionary_chunks(cleaned)
        index, chunks = build_dict_index(chunks, ToyEmbedding())
        answer = answer_schema_rag(FakeClient(), 'city?', index, chunks, ToyEmbedding(), [], 'en')
        self.assertEqual(answer, 'mock answer')

    def test_history_compatibility_excludes_current_user_and_keeps_image(self):
        messages = [{'role':'user', 'content':'picture', 'attachments':[{'path':'old.jpg'}]}]
        messages += [{'role':'assistant', 'content':str(i)} for i in range(12)]
        messages += [{'role':'user', 'content':'current'}]
        with patch('src.chat.get_current_messages', return_value=messages):
            history = get_recent_history(2)
        self.assertNotIn('current', [m['content'] for m in history])
        self.assertEqual(history[0]['content'], 'picture')

    def test_random_forest_baseline_still_runs(self):
        df = pd.DataFrame({'feature':range(40), 'target':[float(i*2+1) for i in range(40)]})
        result = train_simple_model(df, 'target')
        self.assertTrue(result['ok'])
        self.assertIn('r2', result['metrics'])
