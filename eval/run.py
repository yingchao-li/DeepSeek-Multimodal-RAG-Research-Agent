"""CLI runner for the real Plain RAG versus Agentic RAG pilot."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

import yaml

from eval.cache import ResultCache
from eval.cost import estimated_cost, usage_totals
from eval.judge import judge_answer
from eval.metrics import summarize
from eval.pipelines import EvalContext, answer_agentic, answer_baseline
from src.agentic_rag import AgentLimits
from src.api import APIError, ManagedClient, setting
from src.embeddings import LIGHTWEIGHT_MODEL, SentenceEmbeddingAdapter, query_embedder
from src.paper_reader import save_uploaded_pdf
from src.paper_structure import parse_paper
from src.vector_store import RetrievalConfig, build_retrieval_index

ROOT = Path(__file__).resolve().parents[1]


class CappedClient(ManagedClient):
    def __init__(self, max_calls):
        super().__init__()
        self.max_calls = max_calls

    def create(self, **kwargs):
        if self.total_calls >= self.max_calls:
            raise APIError("evaluation_api_call_limit")
        return super().create(**kwargs)


class Upload:
    def __init__(self, path):
        self.name, self.raw = Path(path).name, Path(path).read_bytes()

    def getvalue(self):
        return self.raw


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_dataset(rows):
    required = {"id", "question", "reference_answer", "gold_doc", "gold_evidence",
                "type", "needs_human_review", "unanswerable"}
    if not 6 <= len(rows) <= 50 or any(required - set(row) for row in rows):
        raise ValueError("invalid_evaluation_dataset")
    if sum(bool(row["unanswerable"]) for row in rows) / len(rows) < 0.25:
        raise ValueError("dataset_requires_25_percent_unanswerable")


def build_context(config, client):
    chunks, documents = [], []
    for name in ("synthetic-aster.pdf", "synthetic-boreal.pdf"):
        original = ROOT / "output" / "pdf" / name
        if not original.exists():
            raise FileNotFoundError(f"Missing Phase 3 fixture: {original}")
        saved = save_uploaded_pdf(Upload(original))
        parsed = parse_paper(saved, name, parser="pymupdf")
        chunks.extend(parsed["chunks"])
        documents.append({"name": name, "sha256": hashlib.sha256(original.read_bytes()).hexdigest()})
    model = SentenceEmbeddingAdapter(config.get("embedding_model", LIGHTWEIGHT_MODEL),
                                     device="cpu", allow_download=False)
    retrieval_index = build_retrieval_index(chunks, model)
    retrieval = RetrievalConfig(config.get("retrieval_strategy", "hybrid"),
                                int(config.get("candidate_k", 20)), 12,
                                rrf_k=int(config.get("rrf_k", 60)))
    agent_cfg = config["agent"]
    limits = AgentLimits(max_iterations=int(agent_cfg["max_iterations"]),
                         max_tool_calls=int(agent_cfg["max_tool_calls"]),
                         max_api_calls=int(agent_cfg["max_api_calls"]),
                         max_output_tokens=int(agent_cfg["max_output_tokens"]),
                         max_evidence=int(agent_cfg["max_evidence"]), max_visuals=0)
    return EvalContext(retrieval_index, chunks, query_embedder(model), retrieval, client, limits), documents


def git_state():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def write_outputs(folder, rows, config_snapshot, k, seed):
    summaries = [summarize(rows, pipeline, k) for pipeline in ("baseline", "agentic")
                 if any(r.get("pipeline") == pipeline for r in rows)]
    for summary in summaries:
        records = [record for row in rows if row.get("pipeline") == summary["pipeline"]
                   for record in row.get("api_records", [])]
        judge_records = [record for row in rows if row.get("pipeline") == summary["pipeline"]
                         for record in row.get("judge_api_records", [])]
        summary["generator_cost_usd"] = estimated_cost(records, config_snapshot.get("prices_per_million_tokens"))
        summary["judge_usage"] = usage_totals(judge_records)
        summary["judge_cost_usd"] = estimated_cost(judge_records, config_snapshot.get("prices_per_million_tokens"))
    with (folder / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = list(summaries[0]) if summaries else ["pipeline"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(summaries)
    lines = ["# Real DeepSeek API Pilot Evaluation — Synthetic Dataset", "",
             "> This is an unreviewed synthetic pilot, not a formal benchmark or CV claim.", "",
             f"Questions: {len(set(r['question_id'] for r in rows))}; K={k}; seed={seed}.", "",
             "| Pipeline | Correctness | Faithfulness | Abstention accuracy | Hit@K | Unsupported claims | Mean / Median / P95 latency | API calls | Prompt / Completion tokens |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        lines.append(f"| {s['pipeline']} | {s['answer_correctness']:.3f} | {s['faithfulness']:.3f} | "
                     f"{s['abstention_accuracy']:.3f} | {s['retrieval_hit_at_k']:.3f} | "
                     f"{s['unsupported_claim_rate']:.3f} | {s['mean_ms']:.0f} / {s['median_ms']:.0f} / {s['p95_ms']:.0f} ms | "
                     f"{s['api_calls']} | {s['prompt_tokens']} / {s['completion_tokens']} |")
    lines += ["", "Generator and Judge usage are stored separately in `summary.csv`. Cost is N/A unless reliable prices are configured.", "",
              "Every candidate and pilot item is marked `needs_human_review=true`. LLM-as-Judge may favor outputs from a related model family and requires human review."]
    (folder / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    rng = random.Random(seed)
    sample = rng.sample(rows, min(10, len(rows)))
    with (folder / "manual_check.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["question_id", "pipeline", "question", "answer", "reference_answer", "correctness",
                  "faithfulness", "abstention_correct", "unsupported_claim_rate", "judge_reason"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in sample:
            judge = row.get("judge", {})
            writer.writerow({"question_id": row["question_id"], "pipeline": row["pipeline"],
                             "question": row["dataset"]["question"], "answer": row["answer"],
                             "reference_answer": row["dataset"]["reference_answer"],
                             "correctness": judge.get("correctness"), "faithfulness": judge.get("faithfulness"),
                             "abstention_correct": judge.get("abstention_correct"),
                             "unsupported_claim_rate": judge.get("unsupported_claim_rate"),
                             "judge_reason": judge.get("reason")})
    (folder / "config_snapshot.json").write_text(json.dumps(config_snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "agentic", "both"), default="both")
    parser.add_argument("--dataset", default="eval/dataset.jsonl")
    parser.add_argument("--config", default="eval/config.yaml")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-api-calls", type=int, default=None)
    parser.add_argument("--output-dir")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--retry-key", action="append", default=[],
                        help="Force only one cached stage, e.g. p08:agentic")
    parser.add_argument("--prior-api-calls", type=int, default=0,
                        help="Paid calls already consumed by a resumed run; enforces the global cap")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    dataset_path = ROOT / args.dataset
    dataset = load_jsonl(dataset_path); validate_dataset(dataset)
    if args.dry_run:
        print(json.dumps({"status": "dry_run_ok", "questions": len(dataset), "mode": args.mode}))
        return 0
    if not setting("DEEPSEEK_API_KEY"):
        raise SystemExit("DEEPSEEK_API_KEY is not configured")
    os.environ["DEEPSEEK_MAX_RETRIES"] = "0"
    total_cap = min(80, args.max_api_calls or int(config.get("max_api_calls", 80)))
    remaining_calls = max(0, total_cap - args.prior_api_calls)
    if remaining_calls < 1:
        raise SystemExit("No API calls remain under the global cap")
    client = CappedClient(remaining_calls)
    context, documents = build_context(config, client)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = ROOT / (args.output_dir or f"eval/results/{stamp}")
    folder.mkdir(parents=True, exist_ok=True)
    cache = ResultCache(folder / "cache.jsonl")
    per_path = folder / "per_question.jsonl"
    rows = []
    modes = ("baseline", "agentic") if args.mode == "both" else (args.mode,)
    pipeline_fn = {"baseline": answer_baseline, "agentic": answer_agentic}
    for item in dataset:
        for mode in modes:
            pipeline_key = f"{item['id']}:{mode}:pipeline:k{args.k}"
            result = cache.get(pipeline_key)
            forced = f"{item['id']}:{mode}" in args.retry_key
            if forced or (args.retry_failed and result is not None and result.get("status") != "ok"):
                result = None
            if result is None:
                result = pipeline_fn[mode](item["question"], context, args.k)
                cache.put(pipeline_key, result)
            judge_key = f"{item['id']}:{mode}:judge:v1"
            judged = cache.get(judge_key)
            if forced:
                judged = None
            if result["status"] == "ok" and judged is None:
                sequence = client.total_calls
                try:
                    judged = judge_answer(question=item["question"], reference_answer=item["reference_answer"],
                                          evidence=result["evidence"], answer=result["answer"],
                                          unanswerable=item["unanswerable"], client=client,
                                          model=config["judge_model"])
                except Exception as exc:
                    judged = {"status": "error", "error": str(exc)}
                judged["api_records"] = [dict(r) for r in client.records if r.get("sequence", 0) > sequence]
                judged["usage"] = usage_totals(judged["api_records"])
                cache.put(judge_key, judged)
            row = {**result, "question_id": item["id"], "dataset": item,
                   "judge": {k: v for k, v in (judged or {}).items() if k not in ("api_records", "usage")},
                   "judge_api_records": (judged or {}).get("api_records", []),
                   "judge_usage": (judged or {}).get("usage", {})}
            rows.append(row)
            print(f"{item['id']} {mode}: {result['status']} judge={row['judge'].get('status')} calls={client.total_calls}", flush=True)
    with per_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    snapshot = {**config, "run_utc": datetime.now(timezone.utc).isoformat(), "mode": args.mode,
                "dataset": str(dataset_path.relative_to(ROOT)), "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
                "dataset_needs_human_review": True, "documents": documents, "k": args.k, "seed": args.seed,
                "api_base_url": setting("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                "max_api_calls": total_cap, "prior_api_calls": args.prior_api_calls,
                "resume_api_calls": client.total_calls,
                "actual_api_calls": args.prior_api_calls + client.total_calls, "sdk_retries": 0,
                "git": git_state()}
    summaries = write_outputs(folder, rows, snapshot, args.k, args.seed)
    print(json.dumps({"output": str(folder), "actual_api_calls": args.prior_api_calls + client.total_calls,
                      "summaries": summaries}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
