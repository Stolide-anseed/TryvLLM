import time

from Rag.embedder import Embedder
from Rag.vector_store import collection_supports_hybrid_search, search


class Retriever:
    def __init__(
        self,
        collection_name: str,
        model_name: str,
        device: str = "cpu",
        use_prefixes: bool | None = None,
    ):
        self.coll_name = collection_name
        self.models = Embedder(
            model_name=model_name,
            device=device,
            use_prefixes=use_prefixes,
        )

    @property
    def is_ready(self) -> bool:
        try:
            return collection_supports_hybrid_search(self.coll_name)
        except Exception:
            return False

    def retrieve(
        self,
        query: str,
        top_k: int,
        sparse_query: str | None = None,
    ) -> dict:
        query, top_k = self._validate_query(query, top_k)
        sparse_query = self._validate_sparse_query(sparse_query, fallback=query)

        total_started_at = time.perf_counter()
        embedding_started_at = time.perf_counter()
        query_vector = self.models.encode_query(query=query)
        embedding_latency = time.perf_counter() - embedding_started_at

        if hasattr(query_vector, "tolist"):
            query_vector = query_vector.tolist()

        response = search(
            query_vector=query_vector,
            sparse_query=sparse_query,
            top_k=top_k,
            collection_name=self.coll_name,
        )
        total_latency = time.perf_counter() - total_started_at

        return {
            "dense_query": query,
            "sparse_query": sparse_query,
            "collection": self.coll_name,
            "dense_results": self._format_results(response.get("dense_response", [])),
            "sparse_results": self._format_results(response.get("sparse_response", [])),
            "latencies": {
                "embedding_seconds": embedding_latency,
                "dense_search_seconds": float(
                    response.get("dense_latency_seconds", 0.0)
                ),
                "sparse_search_seconds": float(
                    response.get("sparse_latency_seconds", 0.0)
                ),
                "total_seconds": total_latency,
            },
        }

    def _validate_query(self, query, top_k) -> tuple[str, int]:
        if not isinstance(query, str):
            raise TypeError("query должен быть строкой")

        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query не может быть пустым")

        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k должен быть целым числом")
        if top_k <= 0:
            raise ValueError("top_k должен быть больше нуля")
        if top_k > 100:
            raise ValueError("top_k не может быть больше 100")

        return normalized_query, top_k

    @staticmethod
    def _validate_sparse_query(sparse_query: str | None, fallback: str) -> str:
        if sparse_query is None:
            return fallback
        if not isinstance(sparse_query, str):
            raise TypeError("sparse_query должен быть строкой")

        normalized_query = sparse_query.strip()
        if not normalized_query:
            raise ValueError("sparse_query не может быть пустым")
        return normalized_query

    def _format_results(self, points) -> list[dict]:
        formatted_results = []

        for point in points:
            payload = point.payload or {}
            metadata = payload.get("metadata") or {}
            chunk_id = payload.get("chunk_id") or str(point.id)

            formatted_results.append({
                "chunk_id": chunk_id,
                "text": payload.get("text", ""),
                "score": float(point.score),
                "metadata": metadata,
            })

        return sorted(
            formatted_results,
            key=lambda result: result["score"],
            reverse=True,
        )
