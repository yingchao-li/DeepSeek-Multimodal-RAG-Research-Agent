"""Prepare a deterministic slice of official BEIR SciFact test queries/qrels."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import urllib.request
import zipfile

URL='https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip'
ARCHIVE_SHA='536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165'


def prepare(folder, count=20):
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    archive=folder/'scifact.zip'
    if not archive.exists():
        urllib.request.urlretrieve(URL,archive)
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != ARCHIVE_SHA:
        raise ValueError('SciFact archive checksum changed; review before benchmarking')
    with zipfile.ZipFile(archive) as z:
        corpus=[json.loads(line) for line in z.read('scifact/corpus.jsonl').splitlines()]
        queries={d['_id']:d for d in (json.loads(line) for line in z.read('scifact/queries.jsonl').splitlines())}
        qrels_bytes=z.read('scifact/qrels/test.tsv')
    qrels={}
    for row in csv.DictReader(qrels_bytes.decode().splitlines(),delimiter='\t'):
        if int(row['score']) > 0:
            qrels.setdefault(row['query-id'],{})['scifact-'+row['corpus-id']]=int(row['score'])
    chosen=sorted(qrels,key=lambda x:int(x))[:count]
    chunks=[{'chunk_id':'scifact-'+d['_id'],'source':'SciFact/'+d['_id'],'section':d['title'],
             'kind':'abstract','text':d['title']+'\n'+d['text']} for d in corpus]
    questions=[{'id':qid,'question':queries[qid]['text'], 'relevant_chunk_ids':list(qrels[qid]),
                'relevance_grades':qrels[qid]} for qid in chosen]
    manifest={'source':URL,'archive_sha256':digest,'qrels_sha256':hashlib.sha256(qrels_bytes).hexdigest(),
              'selection':'first N numeric-sorted official test query IDs; full corpus',
              'query_ids':chosen,'corpus_documents':len(chunks),'questions':len(questions),
              'judgments':'original official test.tsv; no LLM labels; unjudged documents treated as nonrelevant',
              'text_representation':'one corpus abstract per chunk: title + newline + text; original corpus IDs preserved'}
    dataset={'description':'BEIR SciFact full corpus, deterministic official test-query slice.',
             'provenance':manifest,'chunks':chunks,'questions':questions}
    output=folder/f'scifact-{count}.json'
    output.write_text(json.dumps(dataset,ensure_ascii=False),encoding='utf-8')
    (folder/f'manifest-{count}.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return output,manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='reports/phase2/dataset')
    parser.add_argument('--queries',type=int,default=20)
    args=parser.parse_args()
    output,manifest=prepare(args.output,args.queries)
    print(json.dumps({'dataset':str(output.resolve()),**manifest},indent=2))
