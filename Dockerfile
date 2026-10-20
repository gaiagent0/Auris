# Auris in a container — CPU-only ONNX engines (no GPU runtime needed).
#   docker compose up -d          # then open http://localhost:7860
# The image targets ARM64 hosts (Snapdragon X Elite); build on ARM64.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    AURIS_HOST=0.0.0.0 \
    AURIS_PORT=7860 \
    AURIS_ALLOWED_HOSTS=localhost,127.0.0.1 \
    HF_HOME=/data/huggingface

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg tesseract-ocr tesseract-ocr-hun \
      libsndfile1 git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app
# The installer creates /app/reader/.venv, exactly as on Windows/Linux.
RUN python -m venv /app/reader/.venv \
 && /app/reader/.venv/bin/python /app/reader/setup.py

# Library, models, exports and cache live on volumes (see docker-compose.yml).
VOLUME ["/app/reader/data", "/app/reader/models", "/app/reader/exports", "/app/reader/audio_cache", "/app/reader/uploads", "/app/model_backup", "/data/huggingface"]
EXPOSE 7860
WORKDIR /app/reader
CMD [".venv/bin/python", "app.py"]
