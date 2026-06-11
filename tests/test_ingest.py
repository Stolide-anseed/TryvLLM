import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.ingest import ingest


CHUNKS = [{"chunk_id": "chunk-1", "text": "текст", "metadata": {}}]


class FakeEmbedder:
    vector_size = 2
    use_prefixes = True

    def encode_documents(self, texts, batch_size):
        return [[0.1, 0.2] for _text in texts]


class IngestTests(unittest.TestCase):
    def call_ingest(self, input_dir: str, output_file: str):
        return ingest(
            collection_name="movies",
            input_dir=input_dir,
            output_file=output_file,
            max_char=384,
            overlap_char=50,
        )

    @patch("scripts.ingest.count_points", return_value=1)
    @patch("scripts.ingest.upsert_chunks", return_value={"inserted": 1})
    @patch("scripts.ingest.create_collection")
    @patch("scripts.ingest.collection_exists", return_value=False)
    @patch("scripts.ingest.configure_client")
    @patch("scripts.ingest.preprocess_documents", return_value=CHUNKS)
    @patch("scripts.ingest.Embedder", return_value=FakeEmbedder())
    def test_ingest_creates_missing_collection(
        self,
        _embedder,
        _preprocess,
        _configure,
        _collection_exists,
        create_collection,
        _upsert,
        _count,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.call_ingest(
                input_dir=temp_dir,
                output_file=str(Path(temp_dir) / "documents.json"),
            )

        create_collection.assert_called_once_with(
            vector_size=2,
            collection_name="movies",
        )
        self.assertEqual(result["collection_points"], 1)

    @patch("scripts.ingest.collection_supports_hybrid_search", return_value=False)
    @patch("scripts.ingest.collection_exists", return_value=True)
    @patch("scripts.ingest.configure_client")
    @patch("scripts.ingest.preprocess_documents", return_value=CHUNKS)
    @patch("scripts.ingest.Embedder", return_value=FakeEmbedder())
    def test_ingest_rejects_incompatible_collection(
        self,
        _embedder,
        _preprocess,
        _configure,
        _collection_exists,
        _supports_hybrid,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(RuntimeError, "--recreate"):
                self.call_ingest(
                    input_dir=temp_dir,
                    output_file=str(Path(temp_dir) / "documents.json"),
                )


if __name__ == "__main__":
    unittest.main()
