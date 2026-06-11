from qdrant_client import QdrantClient, models
from uuid import NAMESPACE_URL, uuid5


client = QdrantClient(host="127.0.0.1", port=6333)


def configure_client(url: str) -> None:
    global client
    client = QdrantClient(url=url)


def collection_exists(collection_name: str) -> bool:
    return client.collection_exists(collection_name)


def collection_supports_hybrid_search(collection_name: str) -> bool:
    if not collection_exists(collection_name):
        return False

    params = client.get_collection(collection_name).config.params
    dense_vectors = params.vectors
    sparse_vectors = params.sparse_vectors

    return (
        isinstance(dense_vectors, dict)
        and "dense" in dense_vectors
        and isinstance(sparse_vectors, dict)
        and "bm25" in sparse_vectors
    )


def create_collection(vector_size: int, collection_name: str) -> dict:
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        },
    )
    return {
        "success": True,
        "Build collection": collection_name,
    }


def upsert_chunks(
    chunks: list[dict],
    vectors: list[list[float]],
    collection_name: str,
) -> dict:
    if len(chunks) != len(vectors):
        raise ValueError("Не совпадает количество chunks и vectors")

    points = []

    for chunk, vector in zip(chunks, vectors):
        chunk_id = chunk.get("chunk_id")

        if not chunk_id:
            raise ValueError("У chunk отсутствует chunk_id")

        points.append(
            models.PointStruct(
                # Создаём стабильный UUID из строкового chunk_id.
                # Также преимущество, что результат детерминированный
                id=str(uuid5(NAMESPACE_URL, chunk_id)),
                vector={
                    "dense": vector,
                    "bm25": models.Document(
                        text=chunk["text"],
                        model="Qdrant/bm25",
                    ),
                },
                payload=chunk,
            )
        )

    client.upsert(
        collection_name=collection_name,
        wait=True,
        points=points,
    )

    return {
        "success": True,
        "inserted": len(points),
        "collection": collection_name,
    }


def search(
    query_vector: list[float],
    top_k: int,
    collection_name: str,
    sparse_query: str,
) -> dict:
    dense_response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        using="dense",
        limit=top_k,
        with_payload=True,
    )
    sparse_response = client.query_points(
        collection_name=collection_name,
        query=models.Document(
            text=sparse_query,
            model="Qdrant/bm25",
        ),
        using="bm25",
        limit=top_k,
        with_payload=True,
    )

    return {
        "success": True,
        "collection": collection_name,
        "dense_response": dense_response.points,
        "sparse_response": sparse_response.points,
    }


def count_points(collection_name: str) -> int:
    result = client.count(
        collection_name=collection_name,
        exact=True,
    )
    return result.count


def delete_collection(collection_name: str) -> dict:
    try:
        client.delete_collection(collection_name=collection_name)
    except Exception as exc:
        raise RuntimeError(f'Программа не смогла удалить {collection_name} из за {exc}') from exc
    return {
        "success": True,
        "collection": collection_name,
    }
