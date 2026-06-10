import unittest
from types import SimpleNamespace

from app.main import get_rag_service, rag_chat
from app.schemas import RAGRequest, RAGResponse, RAGMetrics, TokenUsage
from Rag.service import RAGNotReadyError


class FakeRAGService:
    is_ready = True

    def answer(self, request: RAGRequest) -> RAGResponse:
        return RAGResponse(
            model="test-model",
            answer=f"Ответ на вопрос: {request.question}",
            finish_reason="stop",
            sources=[],
            usage=TokenUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
            metrics=RAGMetrics(
                retrieval_latency_seconds=0.01,
                generation_latency_seconds=0.02,
                total_latency_seconds=0.03,
                retrieved_chunks=0,
                used_context_chars=0,
                top_score=None,
            ),
        )


class RAGEndpointTests(unittest.TestCase):
    def test_rag_chat_uses_service_from_app_state(self) -> None:
        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(rag_service=FakeRAGService()))
        )

        response = rag_chat(RAGRequest(question="Тестовый вопрос"), request)

        self.assertEqual(response.answer, "Ответ на вопрос: Тестовый вопрос")

    def test_get_rag_service_rejects_missing_service(self) -> None:
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))

        with self.assertRaises(RAGNotReadyError):
            get_rag_service(request)

    def test_get_rag_service_rejects_unavailable_service(self) -> None:
        unavailable_service = SimpleNamespace(is_ready=False)
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(rag_service=unavailable_service)
            )
        )

        with self.assertRaises(RAGNotReadyError):
            get_rag_service(request)


if __name__ == "__main__":
    unittest.main()
