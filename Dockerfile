FROM vllm/vllm-openai:v0.21.0-cu129-ubuntu2404

WORKDIR /app

ENTRYPOINT []

COPY requirements.txt ./requirements.txt
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY Rag ./Rag
COPY scripts ./scripts

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
