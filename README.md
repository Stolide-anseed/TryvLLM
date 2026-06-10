# TryvLLM

Минимальный учебный API-wrapper вокруг vLLM.

## Запуск Docker

```powershell
docker build -t tryvllm:dev .
docker run --rm --gpus all --ipc=host -p 8000:8000 `
  -v hf-cache:/root/.cache/huggingface `
  -v vllm-cache:/root/.cache/vllm `
  tryvllm:dev
```

Swagger UI будет доступен на `http://localhost:8000/docs`.

## Health check

```powershell
curl http://localhost:8000/health
```

## Generate

```powershell
curl -X POST http://localhost:8000/generate `
  -H "Content-Type: application/json" `
  -d "{\"prompt\":\"Explain KV-cache in simple words.\",\"max_tokens\":128,\"temperature\":0.0}"
```

## Chat

```powershell
curl -X POST http://localhost:8000/chat `
  -H "Content-Type: application/json" `
  -d "{\"messages\":[{\"role\":\"system\",\"content\":\"Отвечай кратко.\"},{\"role\":\"user\",\"content\":\"Что такое vLLM?\"}],\"max_tokens\":128}"
```

## Настройки

Настройки читаются из переменных окружения с префиксом `LLM_`:

- `LLM_MODEL_NAME`
- `LLM_DTYPE`
- `LLM_MAX_MODEL_LEN`
- `LLM_GPU_MEMORY_UTILIZATION`
- `LLM_DEFAULT_MAX_TOKENS`
- `LLM_DEFAULT_TEMPERATURE`
- `LLM_DEFAULT_TOP_P`

Пример:

```powershell
docker run --rm --gpus all --ipc=host -p 8000:8000 `
  -e LLM_MODEL_NAME=Qwen/Qwen3-0.6B `
  -e LLM_MAX_MODEL_LEN=1024 `
  tryvllm:dev
```

## Подготовка документов для RAG

Markdown-документы с YAML-метаданными находятся в `docs/data`. Для разбиения
всех документов на chunks запусти:

```powershell
python .\Rag\preprocessor.py
```

Результат сохраняется в `docs/documents.json`. Размер и overlap можно изменить:

```powershell
python .\Rag\preprocessor.py --max-chars 1000 --overlap-chars 150
```
