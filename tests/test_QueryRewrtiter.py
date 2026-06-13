import unittest

from app.config import get_settings
from app.engine import InferenceEngine
from app.schemas import InferenceResponse, Metrics, TokenUsage
from Rag.QueryRewriter import QueryRewriter


'''
docker run --rm --gpus all --ipc=host `
  --env-file .env `
  -e PYTHONPATH=/app `
  -v "${PWD}:/app" `
  -v hf-cache:/root/.cache/huggingface `
  -v vllm-cache:/root/.cache/vllm `
  -w /app `
  --entrypoint python3 `
  tryvllm:dev `
  tests/test_QueryRewrtiter.py
'''


class FakeInferenceEngine:
    def __init__(self, text: str):
        self.text = text
        self.last_request = None

    def chat(self, request):
        self.last_request = request
        return InferenceResponse(
            model="test-model",
            text=self.text,
            finish_reason="stop",
            usage=TokenUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
            metrics=Metrics(
                latency_seconds=0.1,
                tokens_per_second=50.0,
            ),
        )


class QueryRewriterTests(unittest.TestCase):
    def test_rewrite_returns_clean_query_without_thinking(self) -> None:
        engine = FakeInferenceEngine(
            "<think>Сначала проанализирую вопрос.</think>\n"
            "\"Причина расследования Бенуа Блана смерти Харлана\""
        )
        rewriter = QueryRewriter(engine)

        result = rewriter.rewrite(
            query="Кто такой Бенуа Блан и почему он расследует смерть Харлана?"
        )

        self.assertEqual(
            result,
            "Причина расследования Бенуа Блана смерти Харлана",
        )
        self.assertIn("/no_think", engine.last_request.messages[1].content)

    def test_rewrite_returns_original_query_when_disabled(self) -> None:
        engine = FakeInferenceEngine("Не должен использоваться")
        rewriter = QueryRewriter(engine)

        result = rewriter.rewrite(query="  Исходный вопрос  ", enabled=False)

        self.assertEqual(result, "Исходный вопрос")
        self.assertIsNone(engine.last_request)

    def test_rewrite_rejects_empty_model_response(self) -> None:
        rewriter = QueryRewriter(FakeInferenceEngine("<think>Анализ</think>"))

        with self.assertRaisesRegex(RuntimeError, "empty query"):
            rewriter.rewrite(query="Исходный вопрос")


def main():
    engine = InferenceEngine(get_settings())
    engine.load()

    rewriter = QueryRewriter(
        inference_engine=engine,
    )

    result = rewriter.rewrite(
        query="Кто такой Бенуа Блан и почему он расследует смерть Харлана?",
        temperature=0.0,
        max_tokens=128,
    )

    print(result)


if __name__ == "__main__":
    unittest.main()
