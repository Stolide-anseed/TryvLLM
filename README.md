# Local-llm-rag-server

Учебный локальный inference-сервис на базе vLLM с FastAPI и RAG по сюжетам
фильмов.

Проект загружает LLM один раз при старте, предоставляет HTTP API для обычной
генерации и чата, а также выполняет semantic search по Qdrant и формирует
ответы с указанием использованных источников.

## Возможности

- локальный запуск `Qwen/Qwen3-0.6B` через vLLM;
- endpoints `/generate`, `/chat`, `/rag/chat` и `/health`;
- Pydantic-валидация запросов и ответов;
- метрики latency, tokens/sec и использования токенов;
- preprocessing Markdown-документов с YAML-метаданными;
- embeddings через `intfloat/multilingual-e5-small`;
- vector search в Qdrant;
- RAG-ответы с citations и отдельными retrieval-метриками;
- Docker-образ с поддержкой NVIDIA GPU.

## Архитектура

```text
Markdown-документы
    -> preprocessor
    -> chunks
    -> multilingual-e5-small
    -> Qdrant

HTTP-запрос
    -> FastAPI
    -> Retriever
    -> Qdrant
    -> RAGService
    -> InferenceEngine
    -> vLLM
    -> ответ + источники + метрики
```

## Стек

- Python 3.12
- FastAPI
- Pydantic
- vLLM
- Sentence Transformers
- Qdrant
- Docker
- Qwen3

## Требования

- Docker Desktop с Linux containers;
- NVIDIA GPU;
- NVIDIA Container Toolkit / поддержка `docker run --gpus all`;
- запущенный Qdrant;
- Python-окружение для выполнения ingestion-скрипта.

vLLM нативно не запускается на Windows. Основной inference-сервис запускается
в Linux Docker-контейнере.

## Быстрый запуск

### 1. Запустить Qdrant

```powershell
docker run -d --name qdrant `
  -p 6333:6333 `
  -p 6334:6334 `
  qdrant/qdrant
```

Dashboard Qdrant:

```text
http://localhost:6333/dashboard
```

### 2. Установить зависимости для ingestion

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install sentence-transformers qdrant-client "PyYAML>=6.0,<7.0"
```

Полный `requirements.txt` включает vLLM и предназначен для Linux
Docker-образа.

### 3. Подготовить и загрузить документы

Команда читает Markdown-файлы из `docs/data`, пересоздаёт
`docs/documents.json`, создаёт embeddings и загружает points в collection
`movies`.

```powershell
python -m scripts.ingest --recreate
```

Основные параметры ingestion:

```powershell
python -m scripts.ingest `
  --collection-name movies `
  --max-char 384 `
  --overlap-char 50 `
  --batch-size 32 `
  --recreate
```

Для E5-моделей префиксы `passage:` и `query:` включаются автоматически.
Принудительное отключение:

```powershell
python -m scripts.ingest --recreate --disable-prefixes
```

### 4. Собрать API-образ

```powershell
docker build -t tryvllm:dev .
```

### 5. Запустить API

```powershell
docker run --rm --gpus all --ipc=host -p 8000:8000 `
  -e LLM_QDRANT_URL=http://host.docker.internal:6333 `
  -v hf-cache:/root/.cache/huggingface `
  -v vllm-cache:/root/.cache/vllm `
  -v fastembed-cache:/root/.cache/fastembed `
  tryvllm:dev
```

Swagger UI:

```text
http://localhost:8000/docs
```

## API

### `GET /health`

Показывает готовность inference-движка и RAG.

```powershell
curl.exe http://localhost:8000/health
```

Пример ответа:

```json
{
  "status": "ok",
  "ready": true,
  "model": "Qwen/Qwen3-0.6B",
  "rag_ready": true
}
```

### `POST /generate`

Продолжает обычный текстовый prompt без chat template.

```powershell
$json = @{
  prompt = "Once upon a time"
  max_tokens = 100
  temperature = 0.7
  top_p = 0.9
} | ConvertTo-Json

curl.exe -X POST "http://localhost:8000/generate" `
  -H "Content-Type: application/json" `
  -d $json
```

### `POST /chat`

Принимает сообщения с ролями `system`, `user`, `assistant`.

```powershell
$json = @{
  messages = @(
    @{
      role = "system"
      content = "Отвечай кратко."
    },
    @{
      role = "user"
      content = "Что такое KV-cache?"
    }
  )
  max_tokens = 150
  temperature = 0.0
} | ConvertTo-Json -Depth 5

curl.exe -X POST "http://localhost:8000/chat" `
  -H "Content-Type: application/json" `
  -d $json
```

### `POST /rag/chat`

Переформулирует вопрос для dense-поиска, выполняет dense и BM25 поиск в Qdrant,
объединяет результаты через Reciprocal Rank Fusion, формирует контекст и
генерирует ответ со ссылками на источники. Для BM25 используется исходный
вопрос, чтобы сохранить точные слова и имена.

```powershell
$json = @{
  question = "Кто победил Тай Лунга?"
  top_k = 3
  score_threshold = 0.5
  max_context_chars = 2000
  max_tokens = 200
  temperature = 0.0
} | ConvertTo-Json

curl.exe -X POST "http://localhost:8000/rag/chat" `
  -H "Content-Type: application/json" `
  -d $json
```

RAG-ответ содержит:

- ответ модели;
- поисковый запрос после переформулирования;
- использованные chunks и нормализованный RRF score;
- исходные `dense_score` и `sparse_score` каждого источника;
- citations;
- prompt/completion token usage;
- retrieval, generation и total latency;
- количество использованных chunks и символов контекста.

Если подходящего контекста нет, LLM не вызывается и сервис сообщает о
недостатке информации.

## Документы и ingestion

Исходные документы находятся в `docs/data`. Каждый файл содержит YAML-блок и
структурированный Markdown:

```yaml
---
id: kung-fu-panda-2008
title: Кунг-фу Панда
original_title: Kung Fu Panda
year: 2008
genres:
  - animation
  - comedy
---
```

Ingestion pipeline:

1. извлекает YAML-метаданные;
2. разделяет текст по заголовкам и подзаголовкам;
3. создаёт chunks с overlap;
4. сохраняет промежуточный `docs/documents.json`;
5. создаёт embeddings;
6. загружает named dense vectors, BM25 sparse vectors, текст и metadata в Qdrant;
7. проверяет количество загруженных points.

Повторный upsert без удаления collection:

```powershell
python -m scripts.ingest
```

Пересоздание collection:

```powershell
python -m scripts.ingest --recreate
```

После перехода с обычного dense-поиска на hybrid search существующую collection
необходимо пересоздать, чтобы добавить named vectors `dense` и `bm25`.
Проверка готовности RAG также проверяет наличие обоих named vectors.
Если collection отсутствует, ingestion создаст её автоматически. Если найдена
старая несовместимая collection, ingestion сообщит о необходимости `--recreate`.

При первом ingestion или BM25-запросе FastEmbed скачивает модель
`Qdrant/bm25`. Её последующие запуски используют из локального кэша.

## Конфигурация

Настройки читаются из переменных окружения с префиксом `LLM_`.

| Переменная | Значение по умолчанию | Назначение |
|---|---:|---|
| `LLM_MODEL_NAME` | `Qwen/Qwen3-0.6B` | LLM-модель |
| `LLM_DTYPE` | `auto` | Тип данных модели |
| `LLM_MAX_MODEL_LEN` | `1024` | Максимальная длина контекста |
| `LLM_GPU_MEMORY_UTILIZATION` | `0.7` | Доля VRAM для vLLM |
| `LLM_DEFAULT_MAX_TOKENS` | `256` | Лимит ответа по умолчанию |
| `LLM_DEFAULT_TEMPERATURE` | `0.0` | Temperature по умолчанию |
| `LLM_DEFAULT_TOP_P` | `0.9` | Top-p по умолчанию |
| `LLM_RAG_ENABLED` | `true` | Включить RAG |
| `LLM_RAG_COLLECTION_NAME` | `movies` | Qdrant collection |
| `LLM_RAG_EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | Embedding-модель |
| `LLM_RAG_EMBEDDING_DEVICE` | `cpu` | Устройство embedding-модели |
| `LLM_RAG_USE_PREFIXES` | auto | Использовать `query:` / `passage:` |
| `LLM_RAG_DISABLE_THINKING` | `true` | Добавлять `/no_think` для Qwen3 |
| `LLM_RAG_CANDIDATE_MULTIPLIER` | `3` | Во сколько раз расширять пул кандидатов до RRF |
| `LLM_RAG_RRF_K` | `60` | Константа сглаживания Reciprocal Rank Fusion |
| `LLM_QUERY_REWRITING_ENABLED` | `true` | Переформулировать вопрос перед retrieval |
| `LLM_QUERY_REWRITING_TEMPERATURE` | `0.0` | Temperature для query rewriting |
| `LLM_QUERY_REWRITING_MAX_TOKENS` | `128` | Лимит токенов для query rewriting |
| `LLM_QDRANT_URL` | `http://127.0.0.1:6333` | Адрес Qdrant |

API запускается с одним Uvicorn worker. Несколько workers загрузили бы несколько
копий LLM в VRAM.

## Метрики

Обычные `/generate` и `/chat` возвращают:

- `latency_seconds`;
- `tokens_per_second`;
- prompt, completion и total tokens.

`/rag/chat` дополнительно возвращает:

- `query_rewrite_latency_seconds`;
- `retrieval_latency_seconds`;
- `generation_latency_seconds`;
- `total_latency_seconds`;
- `retrieved_chunks`;
- `used_context_chars`;
- `top_score`.

## Тестирование

Изолированные тесты RAG не требуют запущенных vLLM и Qdrant:

```powershell
.\.venv\Scripts\python.exe -m unittest `
  .\tests\test_rag_service.py `
  .\tests\test_rag_endpoint.py `
  .\tests\test_hybrid_retriever.py `
  .\tests\test_vector_store.py `
  .\tests\test_ingest.py `
  -v
```

Проверка API вручную удобнее всего через Swagger:

```text
http://localhost:8000/docs
```

## Структура проекта

```text
TryvLLM/
├── app/
│   ├── config.py          # настройки приложения
│   ├── engine.py          # wrapper над vLLM
│   ├── main.py            # FastAPI endpoints и lifespan
│   └── schemas.py         # Pydantic-схемы
├── Rag/
│   ├── embedder.py        # embeddings и E5-префиксы
│   ├── preprocessor.py    # Markdown -> chunks
│   ├── retriever.py       # dense + BM25 retrieval
│   ├── service.py         # RRF -> context -> vLLM
│   └── vector_store.py    # работа с Qdrant
├── scripts/
│   ├── generator.py       # offline inference
│   └── ingest.py          # ingestion pipeline
├── docs/
│   ├── data/              # исходные Markdown-документы
│   └── documents.json     # промежуточные chunks
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```

## Текущее состояние и ограничения

- RAG работает по небольшому набору пересказов сюжетов фильмов.
- Retrieval использует hybrid search: dense embeddings + BM25 + RRF.
- RRF score нормализован в диапазон `0..1`; исходные dense/BM25 scores
  возвращаются отдельно.
- Размер контекста пока ограничивается символами, а не токенами.
- Некоторые исходные chunks содержат буквальные `/n`; данные требуют очистки и
  повторного ingestion.
- Синхронный `InferenceEngine` защищён `Lock`, поэтому запросы к модели
  обрабатываются последовательно.
- Streaming, память диалога, reranking и автоматическая оценка RAG пока не
  реализованы.

## Следующие шаги

1. очистить документы и повторно создать collection;
2. добавить metadata filters;
3. оценивать retrieval на фиксированном наборе вопросов;
4. добавить reranker и сравнить его с hybrid search по качеству/latency;
5. ограничивать контекст по токенам;
6. добавить streaming и память диалога.
