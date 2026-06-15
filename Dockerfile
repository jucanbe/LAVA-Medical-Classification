FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libxml2-dev \
        libxslt1-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py main.py ./
COPY database/ database/
COPY models/ models/
COPY routers/ routers/
COPY services/ services/
COPY frontend/ frontend/

RUN mkdir -p /app/KnowledgeGraph /app/BERT_models/Entities /app/BERT_models/Relations /app/data

ENV DATABASE_URL="sqlite+aiosqlite:////app/data/entity_classifier.db"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
