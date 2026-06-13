import csv
import json
import time
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from Rag.service import RAGService


SUPPORTED_MODES = ("dense", "sparse", "hybrid")


def recall_at_k(
    results: list[dict],
    expected_ids: set[str],
    k: int,
    relevance_level: str = "document",
) -> float:
    if not expected_ids:
        return 0.0

    found_ids = {
        result_id(result, relevance_level)
        for result in results[:k]
    }
    return len(expected_ids & found_ids) / len(expected_ids)


def reciprocal_rank(
    results: list[dict],
    expected_ids: set[str],
    relevance_level: str = "document",
) -> tuple[float, int | None]:
    for rank, result in enumerate(results, start=1):
        if result_id(result, relevance_level) in expected_ids:
            return 1 / rank, rank
    return 0.0, None


def result_id(result: dict, relevance_level: str) -> str | None:
    if relevance_level == "chunk":
        return result.get("chunk_id")
    if relevance_level == "document":
        return (result.get("metadata") or {}).get("document_id")
    raise ValueError(f"Неизвестный relevance_level: {relevance_level}")


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    if not 0 <= percent <= 100:
        raise ValueError("percent должен находиться в диапазоне 0..100")

    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * percent / 100
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index
    return (
        sorted_values[lower_index]
        + (sorted_values[upper_index] - sorted_values[lower_index]) * fraction
    )


class RetrievalEvaluator:
    def __init__(
        self,
        retriever: Any,
        candidate_multiplier: int = 3,
        rrf_k: int = 60,
    ):
        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier должен быть больше нуля")
        if rrf_k < 0:
            raise ValueError("rrf_k не может быть отрицательным")

        self.retriever = retriever
        self.candidate_multiplier = candidate_multiplier
        self.rrf_k = rrf_k

    def evaluate(
        self,
        dataset: dict,
        modes: Iterable[str] = SUPPORTED_MODES,
        ks: Iterable[int] = (1, 3, 5),
    ) -> dict:
        modes = self._validate_modes(modes)
        ks = self._validate_ks(ks)
        max_k = max(ks)
        candidate_top_k = max_k * self.candidate_multiplier
        if candidate_top_k > 100:
            raise ValueError(
                "max(K) * candidate_multiplier не может превышать 100"
            )
        details: list[dict] = []

        for question in dataset.get("questions", []):
            details.append(self._evaluate_question(question, modes, ks, max_k))

        return {
            "dataset_name": dataset.get("dataset_name"),
            "dataset_version": dataset.get("version"),
            "relevance_note": (
                "expected_chunk_ids используются при наличии, иначе "
                "оценка выполняется по expected_document_ids"
            ),
            "config": {
                "modes": modes,
                "ks": ks,
                "candidate_multiplier": self.candidate_multiplier,
                "rrf_k": self.rrf_k,
            },
            "summary": self._build_summary(details, modes, ks),
            "questions": details,
        }

    def _evaluate_question(
        self,
        question: dict,
        modes: list[str],
        ks: list[int],
        max_k: int,
    ) -> dict:
        base_result = {
            "id": question.get("id"),
            "question": question.get("question"),
            "category": question.get("category"),
            "answerable": bool(question.get("answerable")),
        }
        if not question.get("answerable"):
            return {
                **base_result,
                "evaluated": False,
                "skip_reason": "unanswerable_question",
                "modes": {},
            }

        expected_chunk_ids = set(question.get("expected_chunk_ids") or [])
        expected_document_ids = set(question.get("expected_document_ids") or [])
        if expected_chunk_ids:
            relevance_level = "chunk"
            expected_ids = expected_chunk_ids
        else:
            relevance_level = "document"
            expected_ids = expected_document_ids

        if not expected_ids:
            return {
                **base_result,
                "evaluated": False,
                "skip_reason": "missing_expected_ids",
                "modes": {},
            }

        response = self.retriever.retrieve(
            query=question["question"],
            sparse_query=question["question"],
            top_k=max_k * self.candidate_multiplier,
        )
        dense_results = response.get("dense_results", [])
        sparse_results = response.get("sparse_results", [])

        fusion_started_at = time.perf_counter()
        hybrid_results = RAGService.reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            top_k=max_k,
            rrf_k=self.rrf_k,
        )
        fusion_latency = time.perf_counter() - fusion_started_at
        rankings = {
            "dense": dense_results[:max_k],
            "sparse": sparse_results[:max_k],
            "hybrid": hybrid_results,
        }
        latencies = self._mode_latencies(
            response.get("latencies") or {},
            fusion_latency=fusion_latency,
        )

        mode_results = {
            mode: self._evaluate_ranking(
                rankings[mode],
                expected_ids=expected_ids,
                relevance_level=relevance_level,
                ks=ks,
                latency_seconds=latencies[mode],
            )
            for mode in modes
        }

        return {
            **base_result,
            "evaluated": True,
            "relevance_level": relevance_level,
            "expected_ids": sorted(expected_ids),
            "modes": mode_results,
        }

    @staticmethod
    def _evaluate_ranking(
        results: list[dict],
        expected_ids: set[str],
        relevance_level: str,
        ks: list[int],
        latency_seconds: float,
    ) -> dict:
        rr, first_relevant_rank = reciprocal_rank(
            results,
            expected_ids,
            relevance_level,
        )
        recalls = {
            str(k): recall_at_k(results, expected_ids, k, relevance_level)
            for k in ks
        }
        return {
            "recall_at_k": recalls,
            "hit_at_k": {
                str(k): float(recalls[str(k)] > 0.0)
                for k in ks
            },
            "reciprocal_rank": rr,
            "first_relevant_rank": first_relevant_rank,
            "latency_seconds": latency_seconds,
            "results": [
                {
                    "rank": rank,
                    "chunk_id": result.get("chunk_id"),
                    "document_id": (result.get("metadata") or {}).get("document_id"),
                    "score": result.get("score"),
                    "dense_score": result.get("dense_score"),
                    "sparse_score": result.get("sparse_score"),
                    "rrf_score": result.get("rrf_score"),
                }
                for rank, result in enumerate(results, start=1)
            ],
        }

    @staticmethod
    def _mode_latencies(latencies: dict, fusion_latency: float) -> dict[str, float]:
        embedding = float(latencies.get("embedding_seconds", 0.0))
        dense_search = float(latencies.get("dense_search_seconds", 0.0))
        sparse_search = float(latencies.get("sparse_search_seconds", 0.0))
        return {
            "dense": embedding + dense_search,
            "sparse": sparse_search,
            "hybrid": embedding + dense_search + sparse_search + fusion_latency,
        }

    @staticmethod
    def _build_summary(
        details: list[dict],
        modes: list[str],
        ks: list[int],
    ) -> dict:
        evaluated = [detail for detail in details if detail["evaluated"]]
        categories = sorted({
            detail.get("category") or "uncategorized"
            for detail in evaluated
        })
        category_summaries = {}
        for category in categories:
            category_details = [
                detail
                for detail in evaluated
                if (detail.get("category") or "uncategorized") == category
            ]
            category_summaries[category] = {
                "evaluated_questions": len(category_details),
                "modes": RetrievalEvaluator._summarize_modes(
                    category_details,
                    modes,
                    ks,
                ),
            }

        return {
            "total_questions": len(details),
            "evaluated_questions": len(evaluated),
            "skipped_questions": len(details) - len(evaluated),
            "modes": RetrievalEvaluator._summarize_modes(evaluated, modes, ks),
            "categories": category_summaries,
        }

    @staticmethod
    def _summarize_modes(
        details: list[dict],
        modes: list[str],
        ks: list[int],
    ) -> dict:
        summary: dict[str, dict] = {}

        for mode in modes:
            mode_results = [detail["modes"][mode] for detail in details]
            latencies = [result["latency_seconds"] for result in mode_results]
            summary[mode] = {
                "evaluated_questions": len(mode_results),
                "recall_at_k": {
                    str(k): mean(
                        result["recall_at_k"][str(k)]
                        for result in mode_results
                    )
                    if mode_results
                    else 0.0
                    for k in ks
                },
                "hit_rate_at_k": {
                    str(k): mean(
                        result["hit_at_k"][str(k)]
                        for result in mode_results
                    )
                    if mode_results
                    else 0.0
                    for k in ks
                },
                "mrr": mean(
                    result["reciprocal_rank"]
                    for result in mode_results
                )
                if mode_results
                else 0.0,
                "latency_seconds": {
                    "mean": mean(latencies) if latencies else 0.0,
                    "p50": percentile(latencies, 50),
                    "p95": percentile(latencies, 95),
                    "p99": percentile(latencies, 99),
                },
            }

        return summary

    @staticmethod
    def _validate_modes(modes: Iterable[str]) -> list[str]:
        normalized = list(dict.fromkeys(modes))
        invalid = set(normalized) - set(SUPPORTED_MODES)
        if invalid:
            raise ValueError(f"Неизвестные retrieval modes: {sorted(invalid)}")
        if not normalized:
            raise ValueError("Нужно указать хотя бы один retrieval mode")
        return normalized

    @staticmethod
    def _validate_ks(ks: Iterable[int]) -> list[int]:
        normalized = sorted(set(ks))
        if not normalized or any(
            isinstance(k, bool) or not isinstance(k, int) or k <= 0
            for k in normalized
        ):
            raise ValueError("Все значения K должны быть положительными целыми числами")
        return normalized


def load_dataset(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_report(report: dict, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_summary_csv(report: dict, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ks = report["config"]["ks"]

    fieldnames = ["mode", "evaluated_questions", "mrr"]
    fieldnames.extend(f"recall@{k}" for k in ks)
    fieldnames.extend(f"hit_rate@{k}" for k in ks)
    fieldnames.extend(("latency_mean", "latency_p50", "latency_p95", "latency_p99"))

    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for mode, metrics in report["summary"]["modes"].items():
            row = {
                "mode": mode,
                "evaluated_questions": metrics["evaluated_questions"],
                "mrr": metrics["mrr"],
                "latency_mean": metrics["latency_seconds"]["mean"],
                "latency_p50": metrics["latency_seconds"]["p50"],
                "latency_p95": metrics["latency_seconds"]["p95"],
                "latency_p99": metrics["latency_seconds"]["p99"],
            }
            row.update({
                f"recall@{k}": metrics["recall_at_k"][str(k)]
                for k in ks
            })
            row.update({
                f"hit_rate@{k}": metrics["hit_rate_at_k"][str(k)]
                for k in ks
            })
            writer.writerow(row)
