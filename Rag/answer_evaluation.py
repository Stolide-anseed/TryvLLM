import csv
import random
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from app.schemas import RAGRequest
from Rag.evaluation import percentile, reciprocal_rank


SUPPORTED_ANSWER_MODES = ("no_rag", "rag", "rag_rewrite")


class AnswerModeEvaluator:
    def __init__(self, service: Any, seed: int = 42):
        self.service = service
        self.seed = seed

    def warmup(
        self,
        dataset: dict,
        modes: Iterable[str] = SUPPORTED_ANSWER_MODES,
        questions: int = 1,
        **request_options: Any,
    ) -> None:
        if questions < 0:
            raise ValueError("questions не может быть отрицательным")

        modes = self._validate_modes(modes)
        for item in dataset.get("questions", [])[:questions]:
            for mode in modes:
                self.service.answer(
                    RAGRequest(
                        question=item["question"],
                        mode=mode,
                        **request_options,
                    )
                )

    def evaluate(
        self,
        dataset: dict,
        modes: Iterable[str] = SUPPORTED_ANSWER_MODES,
        **request_options: Any,
    ) -> dict:
        modes = self._validate_modes(modes)
        rng = random.Random(self.seed)
        details = []

        for question in dataset.get("questions", []):
            mode_order = modes.copy()
            rng.shuffle(mode_order)
            details.append(
                self._evaluate_question(question, mode_order, request_options)
            )

        return {
            "dataset_name": dataset.get("dataset_name"),
            "dataset_version": dataset.get("version"),
            "config": {
                "modes": modes,
                "seed": self.seed,
                "request": request_options,
            },
            "summary": self._build_summary(details, modes),
            "questions": details,
        }

    def _evaluate_question(
        self,
        question: dict,
        mode_order: list[str],
        request_options: dict,
    ) -> dict:
        expected_ids = set(question.get("expected_document_ids") or [])
        mode_results = {}

        for mode in mode_order:
            response = self.service.answer(
                RAGRequest(
                    question=question["question"],
                    mode=mode,
                    **request_options,
                )
            )
            response_data = response.model_dump(mode="json")
            response_data["retrieval_recall"] = None
            response_data["retrieval_reciprocal_rank"] = None
            response_data["first_relevant_rank"] = None

            if mode != "no_rag" and expected_ids:
                ranked_sources = [
                    {
                        "metadata": {
                            "document_id": source.document_id,
                        }
                    }
                    for source in response.sources
                ]
                response_data["retrieval_recall"] = (
                    len({
                        source.document_id
                        for source in response.sources
                        if source.document_id in expected_ids
                    })
                    / len(expected_ids)
                )
                rr, rank = reciprocal_rank(ranked_sources, expected_ids)
                response_data["retrieval_reciprocal_rank"] = rr
                response_data["first_relevant_rank"] = rank

            mode_results[mode] = response_data

        return {
            "id": question.get("id"),
            "question": question.get("question"),
            "category": question.get("category"),
            "answerable": bool(question.get("answerable")),
            "expected_answer": question.get("expected_answer"),
            "expected_document_ids": sorted(expected_ids),
            "execution_order": mode_order,
            "modes": mode_results,
        }

    @staticmethod
    def _build_summary(details: list[dict], modes: list[str]) -> dict:
        return {
            "total_questions": len(details),
            "modes": {
                mode: AnswerModeEvaluator._summarize_mode(details, mode)
                for mode in modes
            },
        }

    @staticmethod
    def _summarize_mode(details: list[dict], mode: str) -> dict:
        results = [detail["modes"][mode] for detail in details]
        metrics = [result["metrics"] for result in results]
        return {
            "questions": len(results),
            "retrieval_recall": AnswerModeEvaluator._mean_nullable(
                result["retrieval_recall"] for result in results
            ),
            "retrieval_mrr": AnswerModeEvaluator._mean_nullable(
                result["retrieval_reciprocal_rank"] for result in results
            ),
            "mean_sources": AnswerModeEvaluator._mean_nullable(
                None if mode == "no_rag" else len(result["sources"])
                for result in results
            ),
            "latency_seconds": {
                "generation_mean": mean(
                    metric["generation_latency_seconds"] for metric in metrics
                )
                if metrics
                else 0.0,
                "retrieval_mean": AnswerModeEvaluator._mean_nullable(
                    metric["retrieval_latency_seconds"] for metric in metrics
                ),
                "rewrite_mean": AnswerModeEvaluator._mean_nullable(
                    metric["query_rewrite_latency_seconds"] for metric in metrics
                ),
                "total_mean": mean(
                    metric["total_latency_seconds"] for metric in metrics
                )
                if metrics
                else 0.0,
                "total_p50": percentile(
                    [metric["total_latency_seconds"] for metric in metrics],
                    50,
                ),
                "total_p95": percentile(
                    [metric["total_latency_seconds"] for metric in metrics],
                    95,
                ),
            },
        }

    @staticmethod
    def _mean_nullable(values: Iterable[float | int | None]) -> float | None:
        present = [float(value) for value in values if value is not None]
        return mean(present) if present else None

    @staticmethod
    def _validate_modes(modes: Iterable[str]) -> list[str]:
        normalized = list(dict.fromkeys(modes))
        invalid = set(normalized) - set(SUPPORTED_ANSWER_MODES)
        if invalid:
            raise ValueError(f"Неизвестные answer modes: {sorted(invalid)}")
        if not normalized:
            raise ValueError("Нужно указать хотя бы один answer mode")
        return normalized


def write_answer_summary_csv(report: dict, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode",
        "questions",
        "retrieval_recall",
        "retrieval_mrr",
        "mean_sources",
        "rewrite_latency_mean",
        "retrieval_latency_mean",
        "generation_latency_mean",
        "total_latency_mean",
        "total_latency_p50",
        "total_latency_p95",
    ]

    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for mode, metrics in report["summary"]["modes"].items():
            latencies = metrics["latency_seconds"]
            writer.writerow({
                "mode": mode,
                "questions": metrics["questions"],
                "retrieval_recall": metrics["retrieval_recall"],
                "retrieval_mrr": metrics["retrieval_mrr"],
                "mean_sources": metrics["mean_sources"],
                "rewrite_latency_mean": latencies["rewrite_mean"],
                "retrieval_latency_mean": latencies["retrieval_mean"],
                "generation_latency_mean": latencies["generation_mean"],
                "total_latency_mean": latencies["total_mean"],
                "total_latency_p50": latencies["total_p50"],
                "total_latency_p95": latencies["total_p95"],
            })
