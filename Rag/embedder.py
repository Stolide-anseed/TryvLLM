from sentence_transformers import SentenceTransformer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


class Embedder:
    def __init__(
        self,
        model_name: str,
        device: str = device,
        use_prefixes: bool | None = None,
        document_prefix: str = "passage: ",
        query_prefix: str = "query: ",
    ):
        self.model_name = model_name
        self.use_prefixes = (
            self._model_requires_prefixes(model_name)
            if use_prefixes is None
            else use_prefixes
        )
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix
        self.embedder = SentenceTransformer(model_name, device=device)
        self.vector_size = self.embedder.get_sentence_embedding_dimension()

    @staticmethod
    def _model_requires_prefixes(model_name: str) -> bool:
        return "e5" in model_name.lower()

    def encode_documents(self, texts: list[str], batch_size: int) -> list[list[float]]:
        prepared_texts = self._add_prefix(texts, self.document_prefix)
        try:
            embeddings = self.embedder.encode(
                prepared_texts,
                batch_size=batch_size,
                convert_to_numpy=False,
                normalize_embeddings=True,
            )
            return embeddings

        except Exception as exc:
            raise RuntimeError("Не удалось создать embeddings") from exc

    def encode_query(self, query: str) -> list[float]:
        prepared_query = self._add_prefix([query], self.query_prefix)[0]
        try:
            embeddding_query = self.embedder.encode(
                prepared_query,
                convert_to_numpy=False,
                normalize_embeddings=True,
            )
            return embeddding_query
        except Exception as exc:
            raise RuntimeError("Не удалось создать embeddings") from exc

    def _add_prefix(self, texts: list[str], prefix: str) -> list[str]:
        if not self.use_prefixes:
            return texts
        return [
            text if text.startswith(prefix) else f"{prefix}{text}"
            for text in texts
        ]
