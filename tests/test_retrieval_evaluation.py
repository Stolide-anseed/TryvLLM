import csv
import json
import tempfile
import unittest
from pathlib import Path

from Rag.evaluation import (
    RetrievalEvaluator,
    percentile,
    recall_at_k,
    reciprocal_rank,
    write_json_report,
    write_summary_csv,
)


def result(chunk_id: str, document_id: str, score: float) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": f"text for {chunk_id}",
        "score": score,
        "metadata": {"document_id": document_id},
    }


DENSE_RESULTS = [
    result("wrong-1", "wrong-document", 0.95),
    result("po-1", "kung-fu-panda-2008", 0.90),
    result("shrek-1", "shrek-2001", 0.80),
]
SPARSE_RESULTS = [
    result("po-1", "kung-fu-panda-2008", 10.0),
    result("wrong-1", "wrong-document", 8.0),
    result("shrek-1", "shrek-2001", 6.0),
]


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def retrieve(self, query: str, sparse_query: str, top_k: int) -> dict:
        self.calls.append((query, sparse_query, top_k))
        return {
            "dense_results": DENSE_RESULTS[:top_k],
            "sparse_results": SPARSE_RESULTS[:top_k],
            "latencies": {
                "embedding_seconds": 0.01,
                "dense_search_seconds": 0.02,
                "sparse_search_seconds": 0.04,
                "total_seconds": 0.07,
            },
        }


DATASET = {
    "dataset_name": "test-dataset",
    "version": "1",
    "questions": [
        {
            "id": "q1",
            "question": "Кто победил Тай Лунга?",
            "category": "single_fact",
            "answerable": True,
            "expected_document_ids": ["kung-fu-panda-2008"],
        },
        {
            "id": "q2",
            "question": "Сравни фильмы",
            "category": "multi_document",
            "answerable": True,
            "expected_document_ids": ["kung-fu-panda-2008", "shrek-2001"],
        },
        {
            "id": "q3",
            "question": "Неизвестный факт",
            "category": "unanswerable",
            "answerable": False,
            "expected_document_ids": [],
        },
    ],
}


class RetrievalMetricTests(unittest.TestCase):
    def test_recall_at_k_and_reciprocal_rank(self) -> None:
        expected = {"kung-fu-panda-2008"}

        self.assertEqual(recall_at_k(DENSE_RESULTS, expected, 1), 0.0)
        self.assertEqual(recall_at_k(DENSE_RESULTS, expected, 2), 1.0)
        self.assertEqual(reciprocal_rank(DENSE_RESULTS, expected), (0.5, 2))

    def test_percentile_uses_linear_interpolation(self) -> None:
        self.assertEqual(percentile([1.0, 2.0, 3.0], 50), 2.0)
        self.assertAlmostEqual(percentile([1.0, 2.0], 95), 1.95)

    def test_evaluator_compares_modes_and_skips_unanswerable_questions(self) -> None:
        retriever = FakeRetriever()
        evaluator = RetrievalEvaluator(
            retriever=retriever,
            candidate_multiplier=2,
            rrf_k=60,
        )

        report = evaluator.evaluate(DATASET, ks=[1, 3])

        self.assertEqual(report["summary"]["total_questions"], 3)
        self.assertEqual(report["summary"]["evaluated_questions"], 2)
        self.assertEqual(report["summary"]["skipped_questions"], 1)
        self.assertEqual(report["questions"][2]["skip_reason"], "unanswerable_question")
        self.assertEqual(retriever.calls[0][2], 6)
        self.assertEqual(
            report["summary"]["categories"]["single_fact"]["evaluated_questions"],
            1,
        )
        self.assertEqual(
            report["summary"]["categories"]["multi_document"]["evaluated_questions"],
            1,
        )

        dense = report["summary"]["modes"]["dense"]
        sparse = report["summary"]["modes"]["sparse"]
        hybrid = report["summary"]["modes"]["hybrid"]
        self.assertEqual(dense["mrr"], 0.5)
        self.assertEqual(sparse["mrr"], 1.0)
        self.assertEqual(dense["latency_seconds"]["mean"], 0.03)
        self.assertEqual(sparse["latency_seconds"]["mean"], 0.04)
        self.assertGreater(hybrid["latency_seconds"]["mean"], 0.07)

    def test_evaluator_prefers_expected_chunk_ids(self) -> None:
        dataset = {
            "questions": [
                {
                    "id": "q1",
                    "question": "Вопрос",
                    "answerable": True,
                    "expected_chunk_ids": ["po-1"],
                    "expected_document_ids": ["wrong-document"],
                }
            ]
        }

        report = RetrievalEvaluator(FakeRetriever()).evaluate(dataset, ks=[1])

        question = report["questions"][0]
        self.assertEqual(question["relevance_level"], "chunk")
        self.assertEqual(question["modes"]["sparse"]["recall_at_k"]["1"], 1.0)

    def test_evaluator_rejects_candidate_pool_over_retriever_limit(self) -> None:
        evaluator = RetrievalEvaluator(FakeRetriever(), candidate_multiplier=3)

        with self.assertRaisesRegex(ValueError, "не может превышать 100"):
            evaluator.evaluate(DATASET, ks=[50])

    def test_report_writers_create_json_and_csv(self) -> None:
        report = RetrievalEvaluator(FakeRetriever()).evaluate(DATASET, ks=[1, 3])

        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "details.json"
            csv_path = Path(temp_dir) / "summary.csv"
            write_json_report(report, json_path)
            write_summary_csv(report, csv_path)

            loaded_json = json.loads(json_path.read_text(encoding="utf-8"))
            with csv_path.open(encoding="utf-8-sig", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(loaded_json["dataset_name"], "test-dataset")
        self.assertEqual({row["mode"] for row in rows}, {"dense", "sparse", "hybrid"})
        self.assertIn("recall@3", rows[0])
        self.assertIn("latency_p95", rows[0])


if __name__ == "__main__":
    unittest.main()
