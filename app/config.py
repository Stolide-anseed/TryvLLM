from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_name: str = "Qwen/Qwen3-0.6B"
    dtype: str = "auto"
    max_model_len: int = Field(default=1024, ge=1)
    gpu_memory_utilization: float = Field(default=0.7, gt=0.0, le=1.0)

    default_max_tokens: int = Field(default=256, ge=1)
    default_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    default_top_p: float = Field(default=0.9, gt=0.0, le=1.0)

    rag_enabled: bool = True
    rag_collection_name: str = "movies"
    rag_embedding_model: str = "intfloat/multilingual-e5-small"
    rag_embedding_device: str = "cpu"
    rag_use_prefixes: bool | None = None
    rag_disable_thinking: bool = True
    query_rewriting_enabled: bool = True
    query_rewriting_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    query_rewriting_max_tokens: int = Field(default=128, ge=1, le=512)
    qdrant_url: str = "http://127.0.0.1:6333"

    model_config = SettingsConfigDict(env_prefix="LLM_", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
