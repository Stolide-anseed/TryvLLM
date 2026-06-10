import argparse
from pathlib import Path

from Rag.preprocessor import preprocess_documents
from Rag.embedder import Embedder
from Rag.vector_store import (
    configure_client,
    count_points,
    create_collection,
    delete_collection,
    upsert_chunks,
)


E5_MODEL = "intfloat/multilingual-e5-small"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "docs" / "data"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "docs" / "documents.json"
DEFAULT_COLLECTION_NAME = "movies"
DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"


def ingest(
    collection_name: str,
    input_dir: str | Path,
    output_file: str | Path,
    max_char: int,
    overlap_char: int,
    model_name: str = E5_MODEL,
    recreate: bool = False,
    batch_size: int = 32,
    use_prefixes: bool | None = None,
    qdrant_url: str = DEFAULT_QDRANT_URL,
) -> dict:
    if not collection_name.strip():
        raise ValueError("collection_name не может быть пустым")
    if not model_name.strip():
        raise ValueError("model_name не может быть пустым")
    if not qdrant_url.strip():
        raise ValueError("qdrant_url не может быть пустым")
    if max_char <= 0:
        raise ValueError("max_char должен быть больше нуля")
    if overlap_char < 0 or overlap_char >= max_char:
        raise ValueError("overlap_char должен быть от 0 до max_char - 1")
    if batch_size <= 0:
        raise ValueError("batch_size должен быть больше нуля")

    input_path = Path(input_dir)
    output_path = Path(output_file)
    if not input_path.is_dir():
        raise ValueError(f"Папка с документами не найдена: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    configure_client(qdrant_url)
    embedder = Embedder(model_name, use_prefixes=use_prefixes)

    chunks = preprocess_documents(
        input_dir=str(input_path),
        output_file=str(output_path),
        max_char=max_char,
        overlap_char=overlap_char,
    )
    if not chunks:
        raise ValueError(f"В папке {input_path} не найдено документов для загрузки")

    texts = [chunk["text"] for chunk in chunks]
    vectors = embedder.encode_documents(texts=texts, batch_size=batch_size)
    if hasattr(vectors, "tolist"):
        vectors = vectors.tolist()
    else:
        vectors = [
            vector.tolist() if hasattr(vector, "tolist") else vector
            for vector in vectors
        ]
    if len(vectors) != len(chunks):
        raise RuntimeError(
            f"Количество vectors ({len(vectors)}) не совпадает "
            f"с количеством chunks ({len(chunks)})"
        )

    if recreate:
        try:
            delete_collection(collection_name)
        except RuntimeError:
            pass
        create_collection(
            vector_size=embedder.vector_size,
            collection_name=collection_name,
        )

    upsert_result = upsert_chunks(
        chunks=chunks,
        vectors=vectors,
        collection_name=collection_name,
    )
    collection_points = count_points(collection_name)
    if recreate and collection_points != len(chunks):
        raise RuntimeError(
            f"Проверка загрузки не пройдена: ожидалось {len(chunks)} points, "
            f"в collection находится {collection_points}"
        )
    if not recreate and collection_points < len(chunks):
        raise RuntimeError(
            f"Проверка загрузки не пройдена: загружено {len(chunks)} chunks, "
            f"но в collection находится только {collection_points} points"
        )

    return {
        "success": True,
        "collection": collection_name,
        "qdrant_url": qdrant_url,
        "model": model_name,
        "prefixes_enabled": embedder.use_prefixes,
        "vector_size": embedder.vector_size,
        "chunks_processed": len(chunks),
        "vectors_created": len(vectors),
        "collection_points": collection_points,
        "upserted": upsert_result["inserted"],
        "documents_file": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess movie documents and upload their embeddings to Qdrant."
    )
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--max-char", type=int, default=384)
    parser.add_argument("--overlap-char", type=int, default=50)
    parser.add_argument("--model-name", default=E5_MODEL)
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument(
        "--disable-prefixes",
        action="store_true",
        help="Disable query/passage prefixes even when the model normally uses them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = ingest(
        collection_name=args.collection_name,
        input_dir=args.input_dir,
        output_file=args.output_file,
        max_char=args.max_char,
        overlap_char=args.overlap_char,
        model_name=args.model_name,
        recreate=args.recreate,
        batch_size=args.batch_size,
        use_prefixes=False if args.disable_prefixes else None,
        qdrant_url=args.qdrant_url,
    )
    print(result)


if __name__ == "__main__":
    main()
