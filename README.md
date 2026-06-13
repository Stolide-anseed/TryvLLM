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

Поддерживает три режима с единым форматом ответа:

- `no_rag` — модель отвечает из собственных знаний без retrieval;
- `rag` — hybrid search выполняется по исходному вопросу;
- `rag_rewrite` — вопрос переформулируется перед dense-поиском.

По умолчанию используется `rag_rewrite`. В режимах с RAG выполняются dense и
BM25 поиск в Qdrant, объединение через Reciprocal Rank Fusion и генерация ответа
со ссылками на источники. Для BM25 всегда используется исходный вопрос, чтобы
сохранить точные слова и имена. Ошибка QueryRewriter в `rag_rewrite` завершает
запрос ошибкой и не маскируется поиском по исходному вопросу.

```powershell
$json = @{
  mode = "rag_rewrite"
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

- фактически использованный режим;
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

Для `no_rag` используются пустые `sources`, `rewritten_query: null` и `null`
для неприменимых retrieval-метрик. Endpoint всё равно требует готовый
RAG-сервис: для обычного ответа без зависимости от Qdrant используйте `/chat`.

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
| `LLM_QUERY_REWRITING_ENABLED` | `true` | Разрешить `rag_rewrite`; при `false` режим вернёт ошибку |
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

## Оценка retrieval

Скрипт `scripts/evaluate_retrieval.py` сравнивает `dense`, `BM25` и `hybrid`
по фиксированному набору `docs/evaluation/questions.json`.

Он рассчитывает:

- document-level `Recall@1`, `Recall@3`, `Recall@5`;
- Hit Rate@K;
- MRR;
- mean, p50, p95 и p99 latency для каждого режима.
- отдельные агрегаты по категориям вопросов.

Если у вопроса присутствует `expected_chunk_ids`, оценка выполняется по chunks.
Иначе используются `expected_document_ids`. Вопросы с `answerable=false`
сохраняются в подробном отчёте, но не участвуют в Recall и MRR.
Dense и sparse latency измеряются как длительности соответствующих частей
одного retrieval-запроса, hybrid latency дополнительно включает RRF.
Перед измерением по умолчанию выполняется один warmup-запрос. Количество можно
изменить через `--warmup-queries`.

Запуск:

```powershell
python -m scripts.evaluate_retrieval
```

Выбор режимов и значений K:

```powershell
python -m scripts.evaluate_retrieval `
  --modes dense sparse hybrid `
  --ks 1 3 5 10
```

Результаты сохраняются в:

```text
docs/evaluation/results/retrieval_detailed.json
docs/evaluation/results/retrieval_summary.csv
```

## Сравнение режимов ответа

Скрипт `scripts/evaluate_answers.py` запускает каждый вопрос в режимах
`no_rag`, `rag` и `rag_rewrite`. Порядок режимов перемешивается отдельно для
каждого вопроса с фиксированным `seed`, а перед измерениями выполняется warmup.

В подробный отчёт сохраняются ответы, источники, rewritten query, token usage,
retrieval Recall/MRR и latency. Для `no_rag` неприменимые retrieval-метрики
сохраняются как `null`.

Ответы также были вручную оценены LLM-as-judge относительно `expected_answer`:

- `1.0` — полностью правильный ответ;
- `0.5` — частично правильный или неполный ответ;
- `0.0` — неправильный, выдуманный ответ или отсутствие ответа.

`Answer Score` — среднее значение оценок `0.0`, `0.5` и `1.0`, поэтому он
учитывает частично правильные ответы. `Strict Accuracy` — доля полностью
правильных ответов с оценкой `1.0`; частично правильные ответы считаются
ошибками. Для `answerable=false` правильным считается явный отказ без
выдуманных фактов. Эта оценка проверяет корректность ответа, но не faithfulness
относительно использованных источников.

На Windows evaluator запускается внутри Docker-контейнера с vLLM. Перед первым
запуском соберите образ командой `docker build -t tryvllm:dev .` и запустите
Qdrant. PowerShell-обёртка монтирует проект в контейнер, поэтому отчёты
сохраняются на хосте.

Запуск всех трёх режимов:

```powershell
.\scripts\evaluate_answers_docker.ps1 `
  --modes no_rag rag rag_rewrite `
  --seed 42 `
  --warmup-questions 1 `
  --top-k 5
```

Запуск одного режима:

```powershell
.\scripts\evaluate_answers_docker.ps1 --modes rag
```

Для запуска с другим образом передайте параметр обёртки перед аргументами
evaluation:

```powershell
.\scripts\evaluate_answers_docker.ps1 -Image tryvllm:dev --modes no_rag
```

Результаты сохраняются в:

```text
docs/evaluation/results/answer_modes.json
docs/evaluation/results/answer_modes_summary.csv
docs/evaluation/results/answer_judgements.json
docs/evaluation/results/answer_judgements_summary.csv
```

### Текущие результаты

Оценка выполнена на 40 вопросах при `top_k=5`, `temperature=0.0` и
`max_tokens=200`.

| Режим | Answer Score | Strict Accuracy | Retrieval Recall | Retrieval MRR | Среднее число источников | Generation latency | Total latency | Total latency p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `no_rag` | 1.3% | 0.0% | None | None | None | 251 ms | 251 ms | 402 ms |
| `rag` | 62.5% | 52.5% | 97.1% | 97.1% | 4.68 | 355 ms | 431 ms | 816 ms |
| `rag_rewrite` | 52.5% | 37.5% | 97.1% | 97.1% | 4.70 | 405 ms | 630 ms | 1074 ms |

Выводы:

- `Answer Score` показывает общую полезность ответов с учётом частично
  правильных результатов: `rag` набрал `62.5%`, `rag_rewrite` — `52.5%`, а
  `no_rag` — только `1.3%`.
- `rag` и `rag_rewrite` показали одинаковые Retrieval Recall и MRR.
- При одинаковом retrieval-качестве обычный `rag` дал более высокий Answer
  Score (`62.5%` против `52.5%`) и Strict Accuracy (`52.5%` против `37.5%`).
- `rag_rewrite` увеличил среднюю total latency примерно на 46% относительно
  `rag`.
- На текущем наборе вопросов query rewriting не улучшил качество и увеличил
  latency, поэтому обычный `rag` обеспечивает лучший баланс качества и
  скорости.

## Тестирование

Изолированные тесты RAG не требуют запущенных vLLM и Qdrant:

```powershell
.\.venv\Scripts\python.exe -m unittest `
  .\tests\test_rag_service.py `
  .\tests\test_rag_endpoint.py `
  .\tests\test_hybrid_retriever.py `
  .\tests\test_vector_store.py `
  .\tests\test_ingest.py `
  .\tests\test_retrieval_evaluation.py `
  .\tests\test_answer_evaluation.py `
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
- Синхронный `InferenceEngine` защищён `Lock`, поэтому запросы к модели
  обрабатываются последовательно.
- Streaming, память диалога, reranking и автоматическая оценка RAG пока не
  реализованы.

## Следующие шаги

1. добавить reranker и сравнить его с hybrid search по качеству/latency;
2. добавить streaming и память диалога.
