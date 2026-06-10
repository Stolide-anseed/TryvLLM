from qdrant_client import QdrantClient, models
from uuid import NAMESPACE_URL, uuid5


client = QdrantClient(host="127.0.0.1", port=6333)


def configure_client(url: str) -> None:
    global client
    client = QdrantClient(url=url)


def collection_exists(collection_name: str) -> bool:
    return client.collection_exists(collection_name)


def create_collection(vector_size: int, collection_name:str) -> dict:
    client.create_collection(
        collection_name=collection_name,
        vectors_config = models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE
        )
    )
    return {
        "success": True,
        "Build collection": collection_name,
    }

def upsert_chunks(chunks:list[dict], vectors:list[list[float]], collection_name:str) -> dict:
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
                vector=vector,
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
) -> dict:
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    return {
        "success": True,
        "collection": collection_name,
        "results": response.points,
    }

def count_points(collection_name: str) -> int:
    result = client.count(
        collection_name=collection_name,
        exact=True,
    )
    return result.count

def delete_collection(collection_name:str)->dict:
    try:
        client.delete_collection(collection_name=collection_name)
    except Exception as exc:
        raise RuntimeError(f'Программа не смогла удалить {collection_name} из за {exc}') from exc
    return {
        "success": True,
        "collection": collection_name,
    }
