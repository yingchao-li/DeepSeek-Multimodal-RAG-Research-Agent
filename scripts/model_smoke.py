"""Explicit real local inference smoke test; no DeepSeek/API calls."""
from pathlib import Path
import argparse
import json
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.embeddings import SentenceEmbeddingAdapter, QWEN_MODEL
from src.evaluation import MemoryProbe
from src.vector_store import build_retrieval_index, retrieve, IndexMismatch
from src.embeddings import query_embedder
from src.reranking import load_reranker, QWEN_RERANKER

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=['embedding','reranker'],required=True)
    parser.add_argument('--model')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result={'kind':args.kind,'api_calls':0,'status':'failed'}
    started=time.perf_counter()
    try:
        with MemoryProbe() as memory:
            if args.kind=='embedding':
                model=SentenceEmbeddingAdapter(args.model or QWEN_MODEL)
                documents=model.encode(['Paris is the capital of France.','Bananas are yellow.'])
                queries=model.encode(['What is the capital of France?'],is_query=True)
                similarities=(queries@documents.T).ravel().tolist()
                actual=model._load()
                transformer=actual[0].auto_model
                docs=[{'source':'paris','text':'Paris is the capital of France.'},
                      {'source':'banana','text':'Bananas are yellow.'}]
                index=build_retrieval_index(docs,model)
                hits=retrieve('What is the capital of France?',index,docs,1,query_embedder(model))
                original_fingerprint=index.embedding_fingerprint
                index.embedding_fingerprint='different-model-same-dimension'
                rejected=False
                try:
                    retrieve('What is the capital of France?',index,docs,1,query_embedder(model))
                except IndexMismatch:
                    rejected=True
                index.embedding_fingerprint=original_fingerprint
                result.update(faiss_top_source=hits[0]['source'],index_mismatch_rejected=rejected)
                result.update(model=model.model_name,document_shape=list(documents.shape),
                              query_shape=list(queries.shape),similarities=similarities,
                              class_name=type(transformer).__name__,
                              dtype=str(next(transformer.parameters()).dtype),
                              parameters=sum(p.numel() for p in transformer.parameters()),
                              revision=getattr(transformer.config,'_commit_hash',None),
                              expected_order_passed=similarities[0]>similarities[1])
            else:
                model=load_reranker(args.model or QWEN_RERANKER,'cpu',False)
                scores=model.score('What is the capital of France?',
                                   ['Paris is the capital of France.','Bananas are yellow.'])
                if not (args.model or QWEN_RERANKER).startswith('Qwen/'):
                    long_scores=model.score('What is the capital of France?',
                        ['Paris is the capital of France. '+'context '*1500,'Bananas are yellow.'])
                    result['long_input_scores']=long_scores
                    result['long_input_completed']=len(long_scores)==2
                result.update(model=model.diagnostics(),scores=scores,
                              expected_order_passed=scores[0]>scores[1])
            result['status']='passed' if result['expected_order_passed'] else 'failed_order'
            result['peak_rss_mb']=memory.peak
    except Exception as exc:
        result.update(error_type=type(exc).__name__,error=str(exc)[:300])
    result['elapsed_seconds']=time.perf_counter()-started
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
