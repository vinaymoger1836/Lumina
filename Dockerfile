# Lumina FastAPI API — CPU-only image sized for a small AWS EC2 box (t3.micro).
#
# torch is installed from the PyTorch CPU index FIRST so we get the ~200 MB
# CPU wheel instead of the ~800 MB CUDA build (which is useless on a CPU host
# and would blow the free-tier disk/RAM). The subsequent `-r requirements.txt`
# then sees torch as already satisfied (PEP 440: `==2.12.1` matches `2.12.1+cpu`).
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Cache the sentence-transformers model on a mounted volume across restarts.
    HF_HOME=/models

WORKDIR /app

# If a pinned package ever lacks a manylinux wheel and needs compiling, add:
#   RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY app ./app

EXPOSE 8000
# Single worker: the embedding model + torch already push RAM on 1 GB; more
# workers would multiply that footprint. Scale vertically if needed.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
