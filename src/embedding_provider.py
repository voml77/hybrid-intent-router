from __future__ import annotations

from sentence_transformers import SentenceTransformer


class EmbeddingProvider:
    _models: dict[str, SentenceTransformer] = {}

    @classmethod
    def get_model(
        cls,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        if model_name not in cls._models:
            print(f"🧠 Lade Embedding-Modell: {model_name}")
            cls._models[model_name] = SentenceTransformer(model_name)

        return cls._models[model_name]

    @classmethod
    def embed(
        cls,
        texts,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        model = cls.get_model(model_name)
        return model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )