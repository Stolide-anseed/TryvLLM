import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.schemas import (
    RAGMetrics,
    RAGResponse,
    RAGSource,
    TokenUsage,
)
from Rag.answer_evaluation import AnswerModeEvaluator, write_answer_summary_csv


DATASET = {
    "dataset_name": "test",
    "version": "1",
    "questions": [
        {
            "id": "q1",
            "question": "Вопрос 1",
            "category": "fact",
            "answerable": True,
            "expected_answer": "Ответ 1",
            "expected_document_ids": ["doc-1"],
        },
        {
            "id": "q2",
            "question": "Вопрос 2",
            "category": "unknown",
            "answerable": False,
            "expected_answer": None,
            "expected_document_ids": [],
        },
    ],
}


class FakeService:
    def __init__(self):
        self.calls = []

    def answer(self, request):
        self.calls.append((request.question, request.mode))
        sources = []
        retrieval_latency = None
        retrieved_chunks = None
        used_context_chars = None
        rewrite_latency = None
        rewritten_query = None
        if request.mode != "no_rag":
            sources = [
                RAGSource(
                    citation=1,
                    chunk_id="chunk-1",
                    text="Текст",
                    score=1.0,
                    document_id="doc-1",
                    title="Фильм",
                    section="Сюжет",
                    subsection=None,
                )
            ]
            retrieval_latency = 0.1
            retrieved_chunks = 1
            used_context_chars = 5
        if request.mode == "rag_rewrite":
            rewrite_latency = 0.05
            rewritten_query = "Переписанный вопрос"

        return RAGResponse(
            mode=request.mode,
            model="test-model",
            answer=f"{request.mode}: ответ",
            finish_reason="stop",
            rewritten_query=rewritten_query,
            sources=sources,
            usage=TokenUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
            metrics=RAGMetrics(
                query_rewrite_latency_seconds=rewrite_latency,
                retrieval_latency_seconds=retrieval_latency,
                generation_latency_seconds=0.2,
                total_latency_seconds=0.3,
                retrieved_chunks=retrieved_chunks,
                used_context_chars=used_context_chars,
                top_score=1.0 if sources else None,
            ),
        )


class AnswerModeEvaluatorTests(unittest.TestCase):
    def test_evaluator_compares_modes_and_uses_randomized_order(self) -> None:
        evaluator = AnswerModeEvaluator(FakeService(), seed=42)

        report = evaluator.evaluate(DATASET)

        self.assertEqual(report["summary"]["total_questions"], 2)
        self.assertEqual(
            set(report["questions"][0]["execution_order"]),
            {"no_rag", "rag", "rag_rewrite"},
        )
        no_rag = report["summary"]["modes"]["no_rag"]
        rag = report["summary"]["modes"]["rag"]
        self.assertIsNone(no_rag["retrieval_recall"])
        self.assertIsNone(no_rag["retrieval_mrr"])
        self.assertIsNone(no_rag["mean_sources"])
        self.assertIsNone(no_rag["latency_seconds"]["retrieval_mean"])
        self.assertEqual(rag["retrieval_recall"], 1.0)
        self.assertEqual(rag["retrieval_mrr"], 1.0)

    def test_warmup_runs_each_mode_without_adding_report(self) -> None:
        service = FakeService()
        evaluator = AnswerModeEvaluator(service)

        evaluator.warmup(DATASET, questions=1)

        self.assertEqual(len(service.calls), 3)
        self.assertEqual({mode for _, mode in service.calls}, {
            "no_rag",
            "rag",
            "rag_rewrite",
        })

    def test_rejects_unknown_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "Неизвестные answer modes"):
            AnswerModeEvaluator(FakeService()).evaluate(DATASET, modes=["bad"])

    def test_writes_summary_csv(self) -> None:
        report = AnswerModeEvaluator(FakeService()).evaluate(DATASET)

        with TemporaryDirectory() as directory:
            output = Path(directory) / "summary.csv"
            write_answer_summary_csv(report, output)
            content = output.read_text(encoding="utf-8-sig")

        self.assertIn("mode,questions,retrieval_recall", content)
        self.assertIn("no_rag,2,,,", content)


if __name__ == "__main__":
    unittest.main()
