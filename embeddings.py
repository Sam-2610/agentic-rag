"""Sentence embeddings (loaded lazily so importing this file is fast)."""
from config import EMBED_MODEL

_model = None


def embed(texts):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBED_MODEL)
    return _model.encode(texts, normalize_embeddings=True).tolist()
