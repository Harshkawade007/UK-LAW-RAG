# Always-on server image (Render web service). Not built for Vercel/serverless -
# see the module docstrings in api/main.py and retrieval/store.py for why: the
# embedding + cross-encoder models are loaded once at startup and kept warm in
# memory, which only makes sense on a long-running process.

FROM python:3.12-slim

WORKDIR /app

# System deps for sentence-transformers / torch wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Plain `pip install torch` on Linux pulls the CUDA build by default - several
# GB of GPU libraries (cublas, cudnn, nccl, ...) this CPU-only container never
# uses. Installing the CPU-only wheel first satisfies sentence-transformers'
# `torch>=2.2` requirement, so the later install doesn't replace it.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# laws/ is committed; cleaned/ and chunks/ are gitignored and normally built
# locally. Do that here at image-build time so the container has
# chunks/parents.jsonl on disk without needing a live Qdrant write - the
# vector index itself already lives in Qdrant Cloud (QDRANT_URL/QDRANT_API_KEY
# at runtime), so this only runs the free, local, no-LLM half of the pipeline.
RUN cd ingestion && python clean.py && python chunk.py

# Bake the embedding + cross-encoder weights into the image. Without this, the
# lifespan warm-up in api/main.py downloads both from Hugging Face on first
# boot - fine on a laptop with a warm HF cache, but it turned a measured ~12s
# local warm-up into several minutes on a cold container (verified while
# testing this image), and makes every deploy depend on HF Hub being up.
# Model names must match store.py's MODEL_NAME and rerank.py's MODEL_NAME.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# The weights are already in the image, so never reach out to the Hub at
# runtime - startup no longer depends on Hugging Face being reachable.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

EXPOSE 8000

# Render sets $PORT; default to 8000 for local `docker run`. Never --reload or
# --workers > 1 here even though QDRANT_URL removes the folder-lock reason -
# --reload would re-run the model warm-up on every file change, which makes no
# sense in a container that never changes at runtime.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
