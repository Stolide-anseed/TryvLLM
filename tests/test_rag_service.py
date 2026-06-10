import unittest
from types import SimpleNamespace

from Rag.service import NO_CONTEXT_ANSWER, RAGService
from app.schemas import (
    InferenceResponse,
    Metrics,
    RAGRequest,
    TokenUsage,
)


RESULTS = [
    {
        "chunk_id": "kung-fu-panda-2008-0001",
        "text": "По стал Воином Дракона и победил Тай Лунга.",
        "score": 0.91,
        "metadata": {
            "document_id": "kung-fu-panda-2008",
            "title": "Кунг-фу Панда",
            "section": "Сюжет",
            "subsection": None,
        },
    },
    {
        "chunk_id": "shrek-2001-0001",
        "text": "Шрэк живёт на болоте.",
        "score": 0.40,
        "metadata": {
            "document_id": "shrek-2001",
            "title": "Шрэк",
            "section": "Сюжет",
            "subsection": None,
        },
    },
]


class FakeRetriever:
    is_ready = True

    def retrieve(self, query: str, top_k: int) -> dict:
        return {"query": query, "results": RESULTS[:top_k]}


class FakeInferenceEngine:
    def __init__(self):
        self.settings = SimpleNamespace(model_name="test-model")
        self.last_request = None

    def chat(self, request):
        self.last_request = request
        return InferenceResponse(
            model="test-model",
            text="По победил Тай Лунга [1].",
            finish_reason="stop",
            usage=TokenUsage(
                prompt_tokens=50,
                completion_tokens=10,
                total_tokens=60,
            ),
            metrics=Metrics(
                latency_seconds=0.2,
                tokens_per_second=50.0,
            ),
        )


class RAGServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = FakeInferenceEngine()
        self.service = RAGService(FakeRetriever(), self.engine)

    def test_answer_builds_context_and_calls_engine(self) -> None:
        response = self.service.answer(
            RAGRequest(
                question="Кто победил Тай Лунга?",
                top_k=2,
                score_threshold=0.8,
            )
        )

        self.assertEqual(response.answer, "По победил Тай Лунга [1].")
        self.assertEqual(len(response.sources), 1)
        self.assertEqual(response.sources[0].citation, 1)
        self.assertIn("[Источник 1]", self.engine.last_request.messages[1].content)
        self.assertIn("Кто победил Тай Лунга?", self.engine.last_request.messages[1].content)
        self.assertIn("/no_think", self.engine.last_request.messages[1].content)

    def test_answer_returns_no_context_without_calling_engine(self) -> None:
        response = self.service.answer(
            RAGRequest(
                question="Неизвестный вопрос",
                score_threshold=0.99,
            )
        )

        self.assertEqual(response.answer, NO_CONTEXT_ANSWER)
        self.assertEqual(response.finish_reason, "no_context")
        self.assertEqual(response.sources, [])
        self.assertIsNone(self.engine.last_request)

    def test_build_context_respects_character_limit(self) -> None:
        context, sources = self.service.build_context(
            RESULTS,
            max_context_chars=100,
        )

        self.assertLessEqual(len(context), 100)
        self.assertEqual(sources[0].text, RESULTS[0]["text"][: len(sources[0].text)])
        self.assertIn("[Источник 1]", context)


if __name__ == "__main__":
    unittest.main()
