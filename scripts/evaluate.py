#!/usr/bin/env python3
"""Evaluate BookRAG's retrieval (and, when Ollama is reachable, generation)
quality against a small hand-built question set (data/eval/eval_dataset.json).

Retrieval metrics (always computed, no LLM required):
    Recall@K    - did any retrieved chunk's page range overlap an expected page?
    Precision@K - what fraction of the top-K retrieved chunks were relevant?
    MRR         - reciprocal rank of the first relevant chunk.

Generation metrics (need a running Ollama; skipped otherwise):
    citation correctness - fraction of returned source pages that fall inside
                            the question's expected page range.
    faithfulness          - word-overlap between the answer and the retrieved
                             source snippets (a *heuristic proxy*: how much of
                             the answer's vocabulary actually came from the
                             retrieved text, not a rigorous NLI-based check).
    answer relevance      - word-overlap between the answer and the question's
                             expected_topic label (also a heuristic proxy).
    refusal accuracy       - for questions marked answerable=false, did the
                              system correctly say the evidence was insufficient?

These proxies are intentionally simple/explainable rather than another LLM
judge call; they're good enough to catch regressions, not a publishable eval.

Usage:
    python scripts/evaluate.py [--dataset PATH] [--top-k N] [--no-generation] [--report PATH]

Requires the book to already be ingested (see scripts/ingest.py).
"""
import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.generation import llm  # noqa: E402
from app.models.schemas import QueryFilters, QueryRequest  # noqa: E402
from app.retrieval.hybrid import hybrid_search  # noqa: E402
from app.retrieval.reranker import rerank as rerank_chunks  # noqa: E402
from app.services.rag_pipeline import answer_question  # noqa: E402

DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "data" / "eval" / "eval_dataset.json"
_WORD_RE = re.compile(r"[a-z]+")


def _word_set(text: str) -> set:
    return set(_WORD_RE.findall(text.lower()))


def _page_overlap(chunk, expected_pages) -> bool:
    return any(chunk.start_page <= p <= chunk.end_page for p in expected_pages)


def evaluate_retrieval(item: dict, settings, top_k: int) -> dict:
    candidates = hybrid_search(item["question"], QueryFilters(), settings)
    if settings.rerank_enabled and candidates:
        candidates = rerank_chunks(item["question"], candidates, settings, top_k=top_k)
    else:
        candidates = candidates[:top_k]

    expected = item["expected_pages"]
    hits = [_page_overlap(c, expected) for c in candidates]
    recall = 1.0 if any(hits) else 0.0
    precision = (sum(hits) / len(hits)) if hits else 0.0
    mrr = 0.0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            mrr = 1.0 / rank
            break
    return {"recall": recall, "precision": precision, "mrr": mrr, "retrieved": len(candidates)}


def evaluate_generation(item: dict, settings) -> dict:
    response = answer_question(QueryRequest(query=item["question"]), settings)

    if not item["answerable"]:
        return {"refusal_correct": not response.sufficient_evidence}

    if not response.sufficient_evidence:
        return {"citation_correctness": 0.0, "faithfulness": 0.0, "relevance": 0.0, "answered": False}

    expected_pages = set(item["expected_pages"])
    citation_hits = [
        any(s.start_page <= p <= s.end_page for p in expected_pages) for s in response.sources
    ]
    citation_correctness = (sum(citation_hits) / len(citation_hits)) if citation_hits else 0.0

    answer_words = _word_set(response.answer)
    context_words = set()
    for s in response.sources:
        context_words |= _word_set(s.snippet)
    faithfulness = (len(answer_words & context_words) / len(answer_words)) if answer_words else 0.0

    topic_words = _word_set(item["expected_topic"])
    relevance = (len(answer_words & topic_words) / len(topic_words)) if topic_words else 0.0

    return {
        "citation_correctness": citation_correctness,
        "faithfulness": faithfulness,
        "relevance": relevance,
        "answered": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BookRAG retrieval and generation quality.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--top-k", type=int, default=None, help="override RERANK_TOP_K for retrieval scoring")
    parser.add_argument("--no-generation", action="store_true", help="skip LLM-based generation metrics")
    parser.add_argument("--report", default=None, help="optional path to write a JSON report")
    args = parser.parse_args()

    settings = get_settings()
    top_k = args.top_k or settings.rerank_top_k
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    print(f"Loaded {len(dataset)} evaluation questions from {args.dataset}\n")

    run_generation = not args.no_generation
    if run_generation and not llm.is_available():
        print("Ollama is not reachable at the configured OLLAMA_BASE_URL -- skipping generation metrics")
        print("(citation correctness / faithfulness / relevance / refusal accuracy).")
        print("Retrieval metrics (Recall@K, Precision@K, MRR) do not need the LLM and still run.\n")
        run_generation = False

    retrieval_rows, generation_rows = [], []
    for item in dataset:
        retrieval = evaluate_retrieval(item, settings, top_k)
        print(
            f"[retrieval] recall={retrieval['recall']:.0f} precision={retrieval['precision']:.2f} "
            f"mrr={retrieval['mrr']:.2f}  {'(answerable)' if item['answerable'] else '(unanswerable)':<14} {item['question']}"
        )
        if item["answerable"]:
            retrieval_rows.append(retrieval)

        if run_generation:
            gen = evaluate_generation(item, settings)
            generation_rows.append({"question": item["question"], "answerable": item["answerable"], **gen})

    print(f"\n=== Retrieval summary (answerable questions only, top_k={top_k}) ===")
    if retrieval_rows:
        print(f"Recall@K:    {mean(r['recall'] for r in retrieval_rows):.2f}")
        print(f"Precision@K: {mean(r['precision'] for r in retrieval_rows):.2f}")
        print(f"MRR:         {mean(r['mrr'] for r in retrieval_rows):.2f}")

    if generation_rows:
        answerable_gen = [g for g in generation_rows if g["answerable"]]
        unanswerable_gen = [g for g in generation_rows if not g["answerable"]]
        print("\n=== Generation summary (heuristic proxies, see module docstring) ===")
        if answerable_gen:
            print(f"Citation correctness: {mean(g['citation_correctness'] for g in answerable_gen):.2f}")
            print(f"Faithfulness (proxy): {mean(g['faithfulness'] for g in answerable_gen):.2f}")
            print(f"Answer relevance (proxy): {mean(g['relevance'] for g in answerable_gen):.2f}")
        if unanswerable_gen:
            refusal_acc = mean(1.0 if g["refusal_correct"] else 0.0 for g in unanswerable_gen)
            print(f"Correct refusal on unanswerable questions: {refusal_acc:.2f}")

    if args.report:
        Path(args.report).write_text(
            json.dumps({"retrieval": retrieval_rows, "generation": generation_rows}, indent=2), encoding="utf-8"
        )
        print(f"\nWrote report to {args.report}")


if __name__ == "__main__":
    main()
