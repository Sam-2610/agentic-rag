FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/.hf MPL_CACHE_DIR=/app/.mplcache
WORKDIR /app

# CPU-only PyTorch keeps the image far smaller than the default CUDA build
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu || pip install torch --index-url https://download.pytorch.org/whl/cpu || pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Download the embedding model at build time so the first request is fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY . .

# Run as a normal user, not root: the code sandbox executes generated Python
RUN useradd -m app && mkdir -p chroma_db artifacts .mplcache && chown -R app:app /app
USER app

EXPOSE 8000 8001
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
