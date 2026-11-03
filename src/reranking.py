"""Optional lazy neural rerankers. Never download weights without opt-in."""
from __future__ import annotations
from collections import OrderedDict
import threading

QWEN_RERANKER = 'Qwen/Qwen3-Reranker-0.6B'


class CrossEncoderReranker:
    def __init__(self, model_name, device='cpu', allow_download=False):
        from sentence_transformers import CrossEncoder
        self.model_name = model_name
        self.inference_calls = self.pairs_scored = 0
        self.model = CrossEncoder(model_name, device=device, local_files_only=not allow_download,
                                  max_length=512)
        self.model.max_length = min(1024, getattr(self.model.model.config, 'max_position_embeddings', 512))

    def score(self, query, documents):
        scores = self.model.predict([(query, doc) for doc in documents], batch_size=8,
                                    show_progress_bar=False).reshape(-1).tolist()
        self.inference_calls += 1
        self.pairs_scored += len(documents)
        return scores

    def diagnostics(self):
        model = self.model.model
        return {'model':self.model_name, 'class':type(model).__name__,
                'revision':getattr(model.config, '_commit_hash', None),
                'parameters':sum(p.numel() for p in model.parameters()),
                'dtype':str(next(model.parameters()).dtype), 'device':str(model.device),
                'max_length':self.model.max_length, 'inference_calls':self.inference_calls,
                'pairs_scored':self.pairs_scored}


class QwenReranker:
    """Official yes/no causal-LM scoring; avoids treating Qwen as a classifier."""
    def __init__(self, device='cpu', allow_download=False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(QWEN_RERANKER, padding_side='left',
                                                       local_files_only=not allow_download)
        self.inference_calls = self.pairs_scored = 0
        dtype = torch.float32 if device == 'cpu' else 'auto'
        self.model = AutoModelForCausalLM.from_pretrained(QWEN_RERANKER, torch_dtype=dtype,
                                                         local_files_only=not allow_download).to(device).eval()
        self.no = self.tokenizer.convert_tokens_to_ids('no')
        self.yes = self.tokenizer.convert_tokens_to_ids('yes')
        prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        self.prefix = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix = self.tokenizer.encode(suffix, add_special_tokens=False)

    def score(self, query, documents):
        import torch
        scores = []
        # Small batches and bounded context keep the CPU implementation usable.
        with torch.inference_mode():
            for offset in range(0, len(documents), 2):
                pairs = [f'<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: {query}\n<Document>: {doc}'
                         for doc in documents[offset:offset+2]]
                encoded = self.tokenizer(pairs, add_special_tokens=False, truncation=True,
                                         max_length=2048-len(self.prefix)-len(self.suffix))
                tokens = [{'input_ids': self.prefix+ids+self.suffix} for ids in encoded['input_ids']]
                batch = self.tokenizer.pad(tokens, padding=True, return_tensors='pt').to(self.model.device)
                logits = self.model(**batch, logits_to_keep=1).logits[:, -1, :]
                probabilities = torch.softmax(logits[:, [self.no, self.yes]], dim=1)[:, 1]
                scores.extend(probabilities.cpu().tolist())
                self.inference_calls += 1
                self.pairs_scored += len(pairs)
        return scores

    def diagnostics(self):
        return {'model':QWEN_RERANKER, 'class':type(self.model).__name__,
                'revision':getattr(self.model.config, '_commit_hash', None),
                'parameters':sum(p.numel() for p in self.model.parameters()),
                'dtype':str(next(self.model.parameters()).dtype), 'device':str(self.model.device),
                'max_length':2048, 'logits_to_keep':1, 'inference_calls':self.inference_calls,
                'pairs_scored':self.pairs_scored}


_RERANKERS = OrderedDict()
_RERANKER_LOCK = threading.RLock()


def load_reranker(model_name, device='cpu', allow_download=False):
    key = (model_name,device)
    with _RERANKER_LOCK:
        if key in _RERANKERS:
            _RERANKERS.move_to_end(key)
            return _RERANKERS[key]
        model = QwenReranker(device,allow_download) if model_name == QWEN_RERANKER else CrossEncoderReranker(model_name,device,allow_download)
        _RERANKERS[key] = model
        while len(_RERANKERS) > 2:
            _RERANKERS.popitem(last=False)
        return model


def _clear_reranker_cache():
    with _RERANKER_LOCK:
        _RERANKERS.clear()


load_reranker.cache_clear = _clear_reranker_cache
