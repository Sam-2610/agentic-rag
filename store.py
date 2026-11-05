"""ChromaDB vector store: add chunks and search them."""
import chromadb

import embeddings
from config import CHROMA_DIR, COLLECTION, MIN_SCORE, TOP_K

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _client


def reset_collection():
    client = _get_client()
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    return client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def get_collection():
    return _get_client().get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def add_chunks(col, chunks, batch=64):
    """chunks: list of dicts with id, text, source, page."""
    for i in range(0, len(chunks), batch):
        part = chunks[i : i + batch]
        texts = [c["text"] for c in part]
        col.add(
            ids=[c["id"] for c in part],
            embeddings=embeddings.embed(texts),
            documents=texts,
            metadatas=[{"source": c["source"], "page": c["page"]} for c in part],
        )


def search(query, k=TOP_K, min_score=MIN_SCORE):
    col = get_collection()
    total = col.count()
    if total == 0:
        return []
    vec = embeddings.embed([query])[0]
    res = col.query(query_embeddings=[vec], n_results=min(k, total))
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        score = 1 - dist  # cosine distance -> similarity
        if score >= min_score:
            hits.append(
                {"text": doc, "source": meta["source"], "page": int(meta["page"]), "score": round(score, 3)}
            )
    return hits
