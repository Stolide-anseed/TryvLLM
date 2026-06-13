from app.engine import InferenceEngine
from app.schemas import ChatRequest, ChatMessage

SYSTEM_PROMPT = """
Ты переформулируешь вопрос пользователя для семантического поиска в базе документов о фильмах.

Правила:
- Сохраняй имена персонажей, названия мест и фильмов.
- Убирай разговорные и лишние формулировки.
- Заменяй местоимения конкретными сущностями, только если они явно известны из вопроса.
- Не отвечай на вопрос.
- Не добавляй факты, отсутствующие в исходном вопросе.
- Не изменяй смысл вопроса.
- Возвращай только один поисковый запрос без пояснений и кавычек.
"""


class QueryRewriter:
    def __init__(self, inference_engine: InferenceEngine):
        self.engine = inference_engine

    def rewrite(
        self,
        query: str,
        temperature: float = 0.0,
        max_tokens: int = 128,
        enabled: bool = True,
    ) -> str:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query не может быть пустым")
        if not enabled:
            return normalized_query

        request = ChatRequest(
            messages=[
                ChatMessage(role="system", content=SYSTEM_PROMPT),
                ChatMessage(role="user", content=f"{normalized_query}\n/no_think"),
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response = self.engine.chat(request)
        rewritten_query = self._remove_thinking(response.text).strip().strip("\"'")
        if not rewritten_query:
            raise RuntimeError("Query rewriter returned an empty query")

        return rewritten_query

    @staticmethod
    def _remove_thinking(text: str) -> str:
        if "</think>" in text:
            return text.rsplit("</think>", maxsplit=1)[-1]
        return text
