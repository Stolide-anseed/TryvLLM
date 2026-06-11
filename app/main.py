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
from app.schemas import (
    ChatRequest,
    GenerateRequest,
    HealthResponse,
    InferenceResponse,
    RAGRequest,
    RAGResponse,
)
from Rag.retriever import Retriever
from Rag.QueryRewriter import QueryRewriter
from Rag.service import RAGNotReadyError, RAGService, RAGServiceError
from Rag.vector_store import configure_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    engine = InferenceEngine(settings)
    engine.load()
    app.state.engine = engine

    app.state.rag_service = None
    if settings.rag_enabled:
        configure_client(settings.qdrant_url)
        retriever = Retriever(
            collection_name=settings.rag_collection_name,
            model_name=settings.rag_embedding_model,
            device=settings.rag_embedding_device,
            use_prefixes=settings.rag_use_prefixes,
        )
        app.state.rag_service = RAGService(
            retriever=retriever,
            inference_engine=engine,
            query_rewriter=QueryRewriter(engine),
            query_rewriting_enabled=settings.query_rewriting_enabled,
            query_rewriting_temperature=settings.query_rewriting_temperature,
            query_rewriting_max_tokens=settings.query_rewriting_max_tokens,
            retrieval_candidate_multiplier=settings.rag_candidate_multiplier,
            rrf_k=settings.rag_rrf_k,
            disable_thinking=settings.rag_disable_thinking,
        )
    yield


app = FastAPI(title="Try vLLM API", version="0.1.0", lifespan=lifespan)


def get_engine(request: Request) -> InferenceEngine:
    engine = getattr(request.app.state, "engine", None)
    if engine is None or not engine.is_ready:
        raise EngineNotReadyError("Inference engine is not ready")
    return engine


def get_rag_service(request: Request) -> RAGService:
    rag_service = getattr(request.app.state, "rag_service", None)
    if rag_service is None or not rag_service.is_ready:
        raise RAGNotReadyError("RAG service is not ready")
    return rag_service


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


@app.exception_handler(RAGNotReadyError)
async def rag_not_ready_handler(
    _request: Request,
    exc: RAGNotReadyError,
) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(RAGServiceError)
async def rag_service_error_handler(
    _request: Request,
    exc: RAGServiceError,
) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    settings = get_settings()
    engine = getattr(request.app.state, "engine", None)
    rag_service = getattr(request.app.state, "rag_service", None)

    return HealthResponse(
        status="ok",
        ready=engine is not None and engine.is_ready,
        model=settings.model_name,
        rag_ready=rag_service is not None and rag_service.is_ready,
    )


@app.post("/generate", response_model=InferenceResponse)
def generate(payload: GenerateRequest, request: Request) -> InferenceResponse:
    return get_engine(request).generate(payload)


@app.post("/chat", response_model=InferenceResponse)
def chat(payload: ChatRequest, request: Request) -> InferenceResponse:
    return get_engine(request).chat(payload)


@app.post("/rag/chat", response_model=RAGResponse)
def rag_chat(payload: RAGRequest, request: Request) -> RAGResponse:
    return get_rag_service(request).answer(payload)
