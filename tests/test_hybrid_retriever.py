import unittest
from types import SimpleNamespace
from unittest.mock import patch

from Rag.retriever import Retriever


class FakeEmbedder:
    def __init__(self):
        self.last_query = None

    def encode_query(self, query: str):
        self.last_query = query
        return [0.1, 0.2]


class HybridRetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.retriever = Retriever.__new__(Retriever)
        self.retriever.coll_name = "movies"
        self.retriever.models = FakeEmbedder()

    @patch("Rag.retriever.collection_supports_hybrid_search")
    def test_is_ready_requires_hybrid_collection(self, collection_mock) -> None:
        collection_mock.return_value = True
        self.assertTrue(self.retriever.is_ready)

        collection_mock.return_value = False
        self.assertFalse(self.retriever.is_ready)

    @patch("Rag.retriever.search")
    def test_retrieve_uses_rewritten_dense_and_original_sparse_query(
        self,
        search_mock,
    ) -> None:
        search_mock.return_value = {
            "dense_response": [
                SimpleNamespace(
                    id="dense-id",
                    score=0.9,
                    payload={"text": "dense result", "metadata": {}},
                )
            ],
            "sparse_response": [
                SimpleNamespace(
                    id="sparse-id",
                    score=5.0,
                    payload={"chunk_id": "sparse-chunk", "text": "sparse result"},
                )
            ],
            "dense_latency_seconds": 0.02,
            "sparse_latency_seconds": 0.03,
        }

        response = self.retriever.retrieve(
            query="переписанный вопрос",
            sparse_query="исходный вопрос",
            top_k=6,
        )

        search_mock.assert_called_once_with(
            query_vector=[0.1, 0.2],
            sparse_query="исходный вопрос",
            top_k=6,
            collection_name="movies",
        )
        self.assertEqual(self.retriever.models.last_query, "переписанный вопрос")
        self.assertEqual(response["dense_query"], "переписанный вопрос")
        self.assertEqual(response["sparse_query"], "исходный вопрос")
        self.assertEqual(response["dense_results"][0]["chunk_id"], "dense-id")
        self.assertEqual(response["sparse_results"][0]["chunk_id"], "sparse-chunk")
        self.assertGreaterEqual(response["latencies"]["embedding_seconds"], 0.0)
        self.assertEqual(response["latencies"]["dense_search_seconds"], 0.02)
        self.assertEqual(response["latencies"]["sparse_search_seconds"], 0.03)

    @patch("Rag.retriever.search")
    def test_retrieve_uses_dense_query_as_sparse_fallback(self, search_mock) -> None:
        search_mock.return_value = {"dense_response": [], "sparse_response": []}

        self.retriever.retrieve(query="один вопрос", top_k=3)

        self.assertEqual(search_mock.call_args.kwargs["sparse_query"], "один вопрос")

    def test_retrieve_rejects_empty_sparse_query(self) -> None:
        with self.assertRaises(ValueError):
            self.retriever.retrieve(query="вопрос", sparse_query=" ", top_k=3)


if __name__ == "__main__":
    unittest.main()
