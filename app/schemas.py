from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# Общая родительский класс для всех схем
class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

# схема для настройки параметров
class SamplingRequest(APIModel):
    max_tokens: int | None = Field(default=400)
    temperature: float | None = Field(default=0.0, )
    top_p: float | None = Field(default=0.9)

# схема для промта
class GenerateRequest(SamplingRequest):
    prompt: str = Field(min_length=1)

# схема для разделения на system, user, assistant
class ChatMessage(APIModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)

# соединение в один messages
class ChatRequest(SamplingRequest):
    messages: list[ChatMessage] = Field(min_length=1)

# статистика токенов
class TokenUsage(APIModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class Metrics(APIModel):
    latency_seconds: float
    tokens_per_second: float

# полный и собранный инференс
class InferenceResponse(APIModel):
    model: str
    text: str
    finish_reason: str | None
    usage: TokenUsage
    metrics: Metrics


class HealthResponse(APIModel):
    status: Literal["ok"]
    ready: bool
    model: str
