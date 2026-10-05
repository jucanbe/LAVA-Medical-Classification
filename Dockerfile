# ──────────────────────────────────────────────────────────────
#  Medical Entity Classifier – Docker image
# ──────────────────────────────────────────────────────────────
#  KnowledgeGraph/ and BERT_models/ are NOT baked into the image.
#  Mount them at runtime via docker-compose volumes or -v flags:
#    -v /host/path/KnowledgeGraph:/app/KnowledgeGraph
#    -v /host/path/BERT_models:/app/BERT_models
# ──────────────────────────────────────────────────────────────

FROM python:3.12-slim

# System deps needed by some Python packages (lxml, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libxml2-dev \
        libxslt1-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies first (layer cache) ──────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ────────────────────────────────────
COPY config.py main.py ./
COPY database/ database/
COPY models/ models/
COPY routers/ routers/
COPY services/ services/
COPY frontend/ frontend/

# ── Create mount-point directories so the app never crashes
#    even if the user forgets to mount them ────────────────────
RUN mkdir -p /app/KnowledgeGraph /app/BERT_models/Entities /app/BERT_models/Relations /app/data

# ── Runtime configuration ────────────────────────────────────
ENV DATABASE_URL="sqlite+aiosqlite:////app/data/entity_classifier.db"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
