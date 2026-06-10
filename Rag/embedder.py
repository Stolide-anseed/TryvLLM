from sentence_transformers import SentenceTransformer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Embedder:
    def __init__(self, model_name: str, device:str = device):
        self.embedder = SentenceTransformer(model_name, device = device)

    def encode_documents(self, texts: list[str], batch_size:int) -> list[list[float]]:
        try:
            embeddings = self.embedder.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=False,
                normalize_embeddings=True
            )
            return embeddings

        except Exception as exc:
            raise RuntimeError("Не удалось создать embeddings") from exc

    def encode_query(self, query: str)-> list[float]:
        try:
            embeddding_query = self.embedder.encode(
                query,
                convert_to_numpy=False,
                normalize_embeddings=True
            )
            return embeddding_query
        except Exception as exc:
            raise RuntimeError("Не удалось создать embeddings") from exc