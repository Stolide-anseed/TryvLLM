from threading import Lock

from vllm import LLM, SamplingParams

from app.config import Settings
from app.schemas import ChatRequest, GenerateRequest, InferenceResponse, TokenUsage


class InferenceError(RuntimeError):
    pass


class EngineNotReadyError(InferenceError):
    pass


class InvalidInferenceRequest(InferenceError):
    pass

# Соединяет между собой vLLM и FastAPI
class InferenceEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm: LLM | None = None
        self._lock = Lock()

    @property
    def is_ready(self) -> bool:
        return self.llm is not None
    # загрузка llm
    def load(self) -> None:
        if self.llm is not None:
            return

        self.llm = LLM(
            model=self.settings.model_name,
            dtype=self.settings.dtype,
            max_model_len=self.settings.max_model_len,
            gpu_memory_utilization=self.settings.gpu_memory_utilization,
        )
    # генерации llm
    def generate(self, request: GenerateRequest) -> InferenceResponse:
        self._ensure_loaded()
        sampling_params = self._sampling_params(request)

        try:
            with self._lock:
                outputs = self.llm.generate(
                    [request.prompt],
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
        except (TypeError, ValueError) as exc:
            raise InvalidInferenceRequest(str(exc)) from exc
        except Exception as exc:
            raise InferenceError(f"vLLM generate failed: {exc}") from exc

        return self._to_response(outputs[0])
    # чат с LLM
    def chat(self, request: ChatRequest) -> InferenceResponse:
        self._ensure_loaded()
        sampling_params = self._sampling_params(request)
        messages = [message.model_dump() for message in request.messages]

        try:
            with self._lock:
                outputs = self.llm.chat(
                    messages,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
        except (TypeError, ValueError) as exc:
            raise InvalidInferenceRequest(str(exc)) from exc
        except Exception as exc:
            raise InferenceError(f"vLLM chat failed: {exc}") from exc

        return self._to_response(outputs[0])

    def _ensure_loaded(self) -> None:
        if self.llm is None:
            raise EngineNotReadyError("Inference engine is not loaded")

    def _sampling_params(self, request: GenerateRequest | ChatRequest) -> SamplingParams:
        return SamplingParams(
            max_tokens=request.max_tokens or self.settings.default_max_tokens,
            temperature=(
                self.settings.default_temperature
                if request.temperature is None
                else request.temperature
            ),
            top_p=request.top_p or self.settings.default_top_p,
        )

    def _to_response(self, request_output) -> InferenceResponse:
        completion = request_output.outputs[0]
        prompt_tokens = len(request_output.prompt_token_ids or [])
        completion_tokens = len(completion.token_ids or [])

        return InferenceResponse(
            model=self.settings.model_name,
            text=completion.text,
            finish_reason=completion.finish_reason,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
