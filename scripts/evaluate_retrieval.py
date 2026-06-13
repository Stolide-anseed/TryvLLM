import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from Rag.evaluation import (
    SUPPORTED_MODES,
    RetrievalEvaluator,
    load_dataset,
    write_json_report,
    write_summary_csv,
)
from Rag.retriever import Retriever
from Rag.vector_store import collection_supports_hybrid_search, configure_client


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "docs" / "evaluation" / "questions.json"
DEFAULT_JSON_REPORT = (
    PROJECT_ROOT / "docs" / "evaluation" / "results" / "retrieval_detailed.json"
)
DEFAULT_CSV_REPORT = (
    PROJECT_ROOT / "docs" / "evaluation" / "results" / "retrieval_summary.csv"
)


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Evaluate dense, BM25 and hybrid retrieval quality."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV_REPORT)
    parser.add_argument("--collection-name", default=settings.rag_collection_name)
    parser.add_argument("--model-name", default=settings.rag_embedding_model)
    parser.add_argument("--device", default=settings.rag_embedding_device)
    parser.add_argument("--qdrant-url", default=settings.qdrant_url)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=SUPPORTED_MODES,
        default=list(SUPPORTED_MODES),
    )
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=settings.rag_candidate_multiplier,
    )
    parser.add_argument("--rrf-k", type=int, default=settings.rag_rrf_k)
    parser.add_argument(
        "--warmup-queries",
        type=int,
        default=1,
        help="Number of answerable questions to run before latency measurement.",
    )
    parser.add_argument("--disable-prefixes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_client(args.qdrant_url)
    try:
        collection_ready = collection_supports_hybrid_search(args.collection_name)
    except Exception as exc:
        raise RuntimeError(
            f"Не удалось подключиться к Qdrant по адресу {args.qdrant_url!r}"
        ) from exc

    if not collection_ready:
        raise RuntimeError(
            f"Hybrid collection {args.collection_name!r} не готова. "
            "Запустите Qdrant и выполните ingestion с --recreate."
        )

    retriever = Retriever(
        collection_name=args.collection_name,
        model_name=args.model_name,
        device=args.device,
        use_prefixes=False if args.disable_prefixes else None,
    )
    evaluator = RetrievalEvaluator(
        retriever=retriever,
        candidate_multiplier=args.candidate_multiplier,
        rrf_k=args.rrf_k,
    )
    if args.warmup_queries < 0:
        raise ValueError("warmup_queries не может быть отрицательным")
    if not args.ks or any(k <= 0 for k in args.ks):
        raise ValueError("Все значения K должны быть больше нуля")

    dataset = load_dataset(args.dataset)
    candidate_top_k = max(args.ks) * args.candidate_multiplier
    if candidate_top_k > 100:
        raise ValueError("max(K) * candidate_multiplier не может превышать 100")
    for question in [
        item
        for item in dataset.get("questions", [])
        if item.get("answerable")
    ][: args.warmup_queries]:
        retriever.retrieve(
            query=question["question"],
            sparse_query=question["question"],
            top_k=candidate_top_k,
        )

    report = evaluator.evaluate(
        dataset=dataset,
        modes=args.modes,
        ks=args.ks,
    )
    report["run"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "collection": args.collection_name,
        "embedding_model": args.model_name,
        "embedding_device": args.device,
        "qdrant_url": args.qdrant_url,
        "warmup_queries": args.warmup_queries,
    }
    write_json_report(report, args.json_output)
    write_summary_csv(report, args.csv_output)

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Detailed report: {args.json_output}")
    print(f"Summary CSV: {args.csv_output}")


if __name__ == "__main__":
    main()
