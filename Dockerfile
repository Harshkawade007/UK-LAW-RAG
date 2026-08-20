# Always-on server image, built for Hugging Face Spaces (Docker SDK). Not
# built for Vercel/serverless - see the module docstrings in api/main.py and
# retrieval/store.py for why: the embedding + cross-encoder models are loaded
# once at startup and kept warm in memory, which only makes sense on a
# long-running process.

FROM python:3.12-slim

# HF Spaces runs Docker containers as UID 1000 regardless of what the
# Dockerfile specifies, so create that user up front and own everything
# under it from the first COPY - avoids permission errors reading app files
# at runtime. See https://huggingface.co/docs/hub/spaces-sdks-docker.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# System deps for sentence-transformers / torch wheels (root, before USER).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# WORKDIR above created $HOME/app while still root, so it's root-owned even
# though `useradd -m` made $HOME itself user-owned. COPY --chown=user below
# fixes ownership on the files it copies in, but not on this pre-existing
# directory - without this, `clean.py` fails to mkdir cleaned/ inside it.
RUN chown -R user:user $HOME/app

USER user

# Plain `pip install torch` on Linux pulls the CUDA build by default - several
# GB of GPU libraries (cublas, cudnn, nccl, ...) this CPU-only container never
# uses. Installing the CPU-only wheel first satisfies sentence-transformers'
# `torch>=2.2` requirement, so the later install doesn't replace it.
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Bake the embedding + cross-encoder weights into the image now, before the
# app code is copied in. Without this, the lifespan warm-up in api/main.py
# downloads both from Hugging Face on first boot - fine on a laptop with a
# warm HF cache, but it turned a measured ~12s local warm-up into several
# minutes on a cold container (verified while testing this image), and makes
# every deploy depend on HF Hub being up. Model names must match store.py's
# MODEL_NAME and rerank.py's MODEL_NAME.
#
# Deliberately placed BEFORE `COPY . .`: Docker caches each layer, and once
# one layer's inputs change, every layer after it reruns too. A model
# download only depends on requirements.txt (already installed above), not on
# any application code - putting it after the code copy meant a one-line
# HTML edit forced a ~300MB re-download on every rebuild for no reason.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# The weights are already in the image, so never reach out to the Hub at
# runtime - startup no longer depends on Hugging Face being reachable.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

COPY --chown=user . .

# laws/ is committed; cleaned/ and chunks/ are gitignored and normally built
# locally. Do that here at image-build time so the container has
# chunks/parents.jsonl on disk without needing a live Qdrant write - the
# vector index itself already lives in Qdrant Cloud (QDRANT_URL/QDRANT_API_KEY
# at runtime), so this only runs the free, local, no-LLM half of the pipeline.
# This step (and everything below it) DOES rerun on every code change, since
# it needs whatever just got copied in - but it's pure local CPU text
# processing, a few seconds, not a network download.
RUN cd ingestion && python clean.py && python chunk.py

# Must match app_port in the README.md Spaces config block at the top of the
# repo. Spaces doesn't set a $PORT env var the way Render does, so this just
# defaults straight to 8000.
EXPOSE 8000

# Never --reload or --workers > 1 here even though QDRANT_URL removes the
# folder-lock reason - --reload would re-run the model warm-up on every file
# change, which makes no sense in a container that never changes at runtime.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
