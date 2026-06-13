import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from app.engine import InferenceEngine
from Rag.answer_evaluation import (
    SUPPORTED_ANSWER_MODES,
    AnswerModeEvaluator,
    write_answer_summary_csv,
)
from Rag.evaluation import load_dataset, write_json_report
from Rag.QueryRewriter import QueryRewriter
from Rag.retriever import Retriever
from Rag.service import RAGService
from Rag.vector_store import collection_supports_hybrid_search, configure_client


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "docs" / "evaluation" / "questions.json"
DEFAULT_REPORT = (
    PROJECT_ROOT / "docs" / "evaluation" / "results" / "answer_modes.json"
)
DEFAULT_CSV_REPORT = (
    PROJECT_ROOT / "docs" / "evaluation" / "results" / "answer_modes_summary.csv"
)


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Compare no-RAG, RAG and RAG with query rewriting."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV_REPORT)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=SUPPORTED_ANSWER_MODES,
        default=list(SUPPORTED_ANSWER_MODES),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-questions", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--max-context-chars", type=int, default=2000)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    configure_client(settings.qdrant_url)
    try:
        collection_ready = collection_supports_hybrid_search(
            settings.rag_collection_name
        )
    except Exception as exc:
        raise RuntimeError(
            f"Не удалось подключиться к Qdrant по адресу {settings.qdrant_url!r}"
        ) from exc
    if not collection_ready:
        raise RuntimeError(
            f"Hybrid collection {settings.rag_collection_name!r} не готова."
        )

    engine = InferenceEngine(settings)
    engine.load()
    retriever = Retriever(
        collection_name=settings.rag_collection_name,
        model_name=settings.rag_embedding_model,
        device=settings.rag_embedding_device,
        use_prefixes=settings.rag_use_prefixes,
    )
    service = RAGService(
        retriever=retriever,
        inference_engine=engine,
        query_rewriter=QueryRewriter(engine),
        query_rewriting_enabled=settings.query_rewriting_enabled,
        query_rewriting_temperature=settings.query_rewriting_temperature,
        query_rewriting_max_tokens=settings.query_rewriting_max_tokens,
        retrieval_candidate_multiplier=settings.rag_candidate_multiplier,
        rrf_k=settings.rag_rrf_k,
        disable_thinking=settings.rag_disable_thinking,
    )
    evaluator = AnswerModeEvaluator(service=service, seed=args.seed)
    dataset = load_dataset(args.dataset)
    request_options = {
        "top_k": args.top_k,
        "score_threshold": args.score_threshold,
        "max_context_chars": args.max_context_chars,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    evaluator.warmup(
        dataset,
        modes=args.modes,
        questions=args.warmup_questions,
        **request_options,
    )
    report = evaluator.evaluate(
        dataset,
        modes=args.modes,
        **request_options,
    )
    report["run"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": settings.model_name,
        "collection": settings.rag_collection_name,
        "qdrant_url": settings.qdrant_url,
        "warmup_questions": args.warmup_questions,
    }
    write_json_report(report, args.output)
    write_answer_summary_csv(report, args.csv_output)

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Detailed report: {args.output}")
    print(f"Summary CSV: {args.csv_output}")


if __name__ == "__main__":
    main()
