from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.engine import (
    EngineNotReadyError,
    InferenceEngine,
    InferenceError,
    InvalidInferenceRequest,
)
from app.schemas import ChatRequest, GenerateRequest, HealthResponse, InferenceResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    engine = InferenceEngine(settings)
    engine.load()
    app.state.engine = engine
    yield


app = FastAPI(title="Try vLLM API", version="0.1.0", lifespan=lifespan)


def get_engine(request: Request) -> InferenceEngine:
    engine = getattr(request.app.state, "engine", None)
    if engine is None or not engine.is_ready:
        raise EngineNotReadyError("Inference engine is not ready")
    return engine


@app.exception_handler(InvalidInferenceRequest)
async def invalid_inference_request_handler(
    _request: Request,
    exc: InvalidInferenceRequest,
) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(EngineNotReadyError)
async def engine_not_ready_handler(
    _request: Request,
    exc: EngineNotReadyError,
) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(InferenceError)
async def inference_error_handler(
    _request: Request,
    exc: InferenceError,
) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    settings = get_settings()
    engine = getattr(request.app.state, "engine", None)

    return HealthResponse(
        status="ok",
        ready=engine is not None and engine.is_ready,
        model=settings.model_name,
    )


@app.post("/generate", response_model=InferenceResponse)
def generate(payload: GenerateRequest, request: Request) -> InferenceResponse:
    return get_engine(request).generate(payload)


@app.post("/chat", response_model=InferenceResponse)
def chat(payload: ChatRequest, request: Request) -> InferenceResponse:
    return get_engine(request).chat(payload)
