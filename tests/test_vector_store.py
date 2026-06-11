import unittest
from types import SimpleNamespace
from unittest.mock import patch

from qdrant_client import models

from Rag.vector_store import collection_supports_hybrid_search, search, upsert_chunks


class VectorStoreTests(unittest.TestCase):
    @patch("Rag.vector_store.client")
    def test_collection_supports_hybrid_search_checks_named_vectors(
        self,
        client_mock,
    ) -> None:
        client_mock.collection_exists.return_value = True
        client_mock.get_collection.return_value = SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors={"dense": object()},
                    sparse_vectors={"bm25": object()},
                )
            )
        )

        self.assertTrue(collection_supports_hybrid_search("movies"))

        client_mock.get_collection.return_value.config.params.sparse_vectors = None
        self.assertFalse(collection_supports_hybrid_search("movies"))

    @patch("Rag.vector_store.client")
    def test_search_queries_named_dense_and_sparse_vectors(self, client_mock) -> None:
        client_mock.query_points.side_effect = [
            SimpleNamespace(points=["dense"]),
            SimpleNamespace(points=["sparse"]),
        ]

        result = search(
            query_vector=[0.1, 0.2],
            sparse_query="исходный вопрос",
            top_k=5,
            collection_name="movies",
        )

        dense_call, sparse_call = client_mock.query_points.call_args_list
        self.assertEqual(dense_call.kwargs["using"], "dense")
        self.assertEqual(dense_call.kwargs["query"], [0.1, 0.2])
        self.assertEqual(sparse_call.kwargs["using"], "bm25")
        self.assertIsInstance(sparse_call.kwargs["query"], models.Document)
        self.assertEqual(sparse_call.kwargs["query"].text, "исходный вопрос")
        self.assertEqual(result["dense_response"], ["dense"])
        self.assertEqual(result["sparse_response"], ["sparse"])

    @patch("Rag.vector_store.client")
    def test_upsert_stores_dense_and_bm25_vectors(self, client_mock) -> None:
        result = upsert_chunks(
            chunks=[{"chunk_id": "chunk-1", "text": "текст документа"}],
            vectors=[[0.1, 0.2]],
            collection_name="movies",
        )

        point = client_mock.upsert.call_args.kwargs["points"][0]
        self.assertEqual(point.vector["dense"], [0.1, 0.2])
        self.assertIsInstance(point.vector["bm25"], models.Document)
        self.assertEqual(point.vector["bm25"].text, "текст документа")
        self.assertEqual(result["inserted"], 1)


if __name__ == "__main__":
    unittest.main()
