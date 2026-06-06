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

    model_config = SettingsConfigDict(env_prefix="LLM_", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
