from Rag.embedder import Embedder
from Rag.vector_store import collection_exists, search

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
            return collection_exists(self.coll_name)
        except Exception:
            return False

    def retrieve(self, query, top_k):
        query, top_k = self._validate_query(query, top_k)

        query_vector = self.models.encode_query(query=query)

        if hasattr(query_vector, "tolist"):
            query_vector = query_vector.tolist()

        response = search(query_vector=query_vector, top_k=top_k,collection_name=self.coll_name)

        return {
            "query": query,
            "collection": self.coll_name,
            "results": self._format_results(response.get("results", [])),
        }

    def retieve(self, query, top_k):
        return self.retrieve(query, top_k)

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

    def _format_results(self, points) -> list[dict]:
        formatted_results = []

        for point in points:
            payload = point.payload or {}
            metadata = payload.get("metadata") or {}

            formatted_results.append({
                "chunk_id": payload.get("chunk_id"),
                "text": payload.get("text", ""),
                "score": float(point.score),
                "metadata": metadata,
            })

        return sorted(
            formatted_results,
            key=lambda result: result["score"],
            reverse=True,
        )
