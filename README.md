# LAVA Medical NER & RE

System for extracting medical entities (NER) and relations (RE) from clinical texts using LLM and BERT-based models, with Knowledge Graph validation.

## Pre-trained Models

BERT models are **not included** in the repository. Download them from HuggingFace:

| Task | Model |
|------|-------|
| NER (Entity Recognition) | [jucanbe/LAVA_BERT_Entity](https://huggingface.co/jucanbe/LAVA_BERT_Entity) |

Place downloaded models under `BERT_models/Entities/<model-name>/`.

## Entity Types

14 medical entity types:

| Type | Examples |
|------|----------|
| Disease | Pulmonary embolism, diabetes |
| Symptom | Dyspnea, fever, headache |
| Finding | Filling defect, elevated troponin |
| Organ | Pulmonary artery, lung, heart |
| ImagingProcedure | CT angiography, MRI, X-ray |
| ExaminationProcedure | Blood test, physical exam |
| TherapeuticProcedure | Thrombectomy, surgery, biopsy |
| ImagingResult | CT shows embolus, MRI reveals lesion |
| ExaminationMeasure | Blood count, urinalysis |
| Parameter | D-dimer level, blood pressure |
| Score | Wells score, APACHE score |
| Therapy | Anticoagulation therapy, chemotherapy |
| Substance | Heparin, insulin, contrast agent |
| AdverseEvent | Bleeding, allergic reaction |

## Quick Start

### Docker (recommended)

```bash
# Copy and edit environment variables
cp .env.example .env

# Start the service
docker compose up -d
```

The app will be available at http://localhost:8080/app/

To connect to an LLM running on the host machine, `docker-compose.yml` already sets `DEFAULT_VLLM_BASE_URL=http://host.docker.internal:1234/v1`. Adjust the model name as needed:

```yaml
environment:
  - DEFAULT_VLLM_BASE_URL=http://host.docker.internal:1234/v1
  - DEFAULT_MODEL_NAME=meta-llama-3.1-8b-instruct
```

Persistent data is stored in mounted volumes:

| Volume | Purpose |
|--------|---------|
| `./KnowledgeGraph` | Medical ontology (TTL files) |
| `./BERT_models` | Fine-tuned BERT models |
| `./data` | SQLite database |

### Local (Python)

```bash
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8080
```

## Configuration

Configure LLM servers via **Configuration → LLM Servers** in the UI, or set defaults in `.env`:

```env
DATABASE_URL=sqlite+aiosqlite:///./entity_classifier.db
DEFAULT_VLLM_BASE_URL=http://localhost:1234/v1
DEFAULT_MODEL_NAME=meta-llama-3.1-8b-instruct
```

Supported servers: LM Studio, vLLM, OpenAI, any OpenAI-compatible endpoint.

## API Overview

### NER (LLM)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/classify/` | Classify text |
| POST | `/classify/document` | Classify uploaded document |
| GET | `/classify/kg/stats` | Knowledge Graph statistics |
| POST | `/classify/kg/add-entity` | Add entity to KG |

### RE (Relation Extraction)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/relations/classify` | Extract relations (LLM) |
| POST | `/relations/bert/classify` | Extract relations (BERT) |
| GET | `/relations/pending` | List pending relations for review |

### BERT NER
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/bert/models` | List available models |
| POST | `/bert/classify` | Classify text |
| POST | `/bert/train/files` | Train model (file upload) |

### Configuration
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET/POST | `/llm-config/` | List / create LLM configs |
| PUT/DELETE | `/llm-config/{id}` | Update / delete config |
| POST | `/llm-config/{id}/set-default` | Set as default |

Full interactive docs: http://localhost:8080/docs

## Knowledge Graph Validation

| Status | Similarity | Color |
|--------|-----------|-------|
| Exact Match | ≥95% | Green |
| Similar Match | 70–94% | Orange |
| Low Match | 60–69% | Yellow |
| Not Found | <60% | Gray |

Unmatched entities and relations can be added to the KG from the review UI.

## Project Structure

```
EntityClass/
├── main.py                        # FastAPI entry point
├── config.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── routers/                       # API endpoints
│   ├── classification.py          # LLM NER
│   ├── bert_classification.py     # BERT NER
│   ├── relation_classification.py # LLM & BERT RE
│   ├── entity_review.py
│   ├── relation_review.py
│   ├── triple_store.py
│   ├── sparql.py
│   └── llm_config.py
├── services/                      # Business logic
│   ├── entity_classifier.py
│   ├── relation_classifier.py
│   ├── bert_ner.py
│   ├── knowledge_graph.py
│   ├── triple_store.py
│   └── document_processor.py
├── models/                        # Pydantic models
├── database/                      # SQLAlchemy models
├── frontend/templates/            # Jinja2 templates
├── KnowledgeGraph/                # TTL ontology files
└── BERT_models/                   # Downloaded BERT models (not in repo)
    ├── Entities/
    └── Relations/
```