import os
import sys
import tempfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Throw-away folders so tests never touch your real index or artifacts
_tmp = tempfile.mkdtemp(prefix="agentic_tests_")
os.environ["CHROMA_DIR"] = os.path.join(_tmp, "chroma")
os.environ["ARTIFACT_DIR"] = os.path.join(_tmp, "artifacts")
os.environ["MPL_CACHE_DIR"] = os.path.join(_tmp, "mpl")

import pytest  # noqa: E402


def fake_embed(texts):
    """Tiny deterministic bag-of-words embedding: tests need no model download."""
    out = []
    for t in texts:
        v = [0.0] * 128
        for w in "".join(c.lower() if c.isalnum() else " " for c in t).split():
            v[zlib.crc32(w.encode()) % 128] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        out.append([x / norm for x in v])
    return out


@pytest.fixture(scope="session", autouse=True)
def indexed_sample():
    import embeddings
    import ingest
    import store

    embeddings.embed = fake_embed
    col = store.reset_collection()
    store.add_chunks(col, ingest.build_chunks(ROOT / "data" / "sample_company_report.md"))
