# Medical Entity Classification System

A comprehensive system for extracting and classifying medical entities from clinical texts using LLM and BERT-based NER models, with Knowledge Graph validation.

## Features

- **LLM-based Classification**: Extract medical entities using Large Language Models (compatible with LM Studio, vLLM, OpenAI API)
- **BERT NER**: Train and use custom BERT models for Named Entity Recognition
- **Knowledge Graph Validation**: Validate extracted entities against a medical ontology
- **Document Processing**: Support for PDF, DOCX, DOC, and TXT files
- **Web Interface**: Bootstrap-based UI for classification, training, and configuration
- **REST API**: Full FastAPI backend with OpenAPI documentation

## Entity Types

The system recognizes 14 types of medical entities:

| Type | Description | Examples |
|------|-------------|----------|
| **Disease** | Pathological conditions | Pulmonary embolism, diabetes, pneumonia |
| **Symptom** | Symptoms | Dyspnea, fever, headache, nausea |
| **Finding** | Clinical or imaging-based findings | Filling defect, elevated troponin |
| **Organ** | Anatomical structures | Pulmonary artery, lung, heart |
| **ImagingProcedure** | Imaging procedures (subclass of Procedure) | CT angiography, MRI, X-ray |
| **ExaminationProcedure** | Clinical exams or lab tests (subclass of Procedure) | Blood test, physical exam |
| **TherapeuticProcedure** | Therapeutic procedures (subclass of Procedure) | Thrombectomy, surgery, biopsy |
| **ImagingResult** | Imaging results | CT shows embolus, MRI reveals lesion |
| **ExaminationMeasure** | Examination measures (subclass of QuantitativeMeasure) | Blood count, urinalysis |
| **Parameter** | Quantitative parameters (subclass of QuantitativeMeasure) | D-dimer level, Heart rate, blood pressure |
| **Score** | Clinical scores (subclass of QuantitativeMeasure) | Wells score, APACHE score |
| **Therapy** | Therapies | Anticoagulation therapy, chemotherapy |
| **Substance** | Drugs / Substances | Heparin, insulin, Contrast agent |
| **AdverseEvent** | Adverse events | Bleeding, allergic reaction |

## Project Structure

```
EntityClass/
├── main.py                 # FastAPI application entry point
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
├── start.bat              # Quick start script (Windows)
│
├── database/              # Database models and connection
│   ├── connection.py
│   └── models.py
│
├── models/                # Pydantic models
│   ├── entities.py        # Entity classification models
│   └── llm_config.py      # LLM configuration models
│
├── routers/               # API endpoints
│   ├── classification.py  # LLM classification endpoints
│   ├── bert_classification.py  # BERT NER endpoints
│   └── llm_config.py      # LLM server configuration
│
├── services/              # Business logic
│   ├── llm_client.py      # LLM API client
│   ├── entity_classifier.py  # Entity classification logic
│   ├── knowledge_graph.py # Knowledge Graph operations
│   ├── bert_ner.py        # BERT NER service
│   └── document_processor.py  # Document parsing
│
├── frontend/              # Web interface
│   └── templates/         # Jinja2 HTML templates
│
├── KnowledgeGraph/        # Medical ontology files
│   ├── medical_ontology.ttl   # Entity type definitions
│   └── medical_entities.ttl   # Custom entities
│
└── BERT_models/           # Trained BERT models (not in the repository)
```

BERT models are not included. Download them from HuggingFace, for example [jucanbe/LAVA_BERT_Entity](https://huggingface.co/jucanbe/LAVA_BERT_Entity), and place them under `BERT_models/Entities/<model-name>/`.

## Installation

### Prerequisites

- Python 3.10+
- CUDA (optional, for GPU acceleration with BERT)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd EntityClass
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install optional dependencies**
   ```bash
   # For document processing
   pip install pypdf python-docx
   
   # For BERT NER
   pip install transformers torch datasets
   ```

5. **Run the application**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

   Or use the quick start script (Windows):
   ```bash
   start.bat
   ```

6. **Access the application**
   - Web Interface: http://localhost:8000/app/
   - API Documentation: http://localhost:8000/docs

## Configuration

### LLM Server

The system supports any OpenAI-compatible API. Configure your LLM server in the web interface:

1. Go to **Configuration → LLM Servers**
2. Add your server (LM Studio, vLLM, OpenAI, etc.)
3. Set as default for classification

**Supported servers:**
- [LM Studio](https://lmstudio.ai/) (recommended for local use)
- [vLLM](https://github.com/vllm-project/vllm)
- OpenAI API
- Any OpenAI-compatible endpoint

### Knowledge Graph

The Knowledge Graph is stored in TTL (Turtle) format in the `KnowledgeGraph/` directory:

- `medical_ontology.ttl`: Base entity definitions and types
- `medical_entities.ttl`: Custom entities added through the UI

## Usage

### Web Interface

#### LLM Classification
1. Navigate to **LLM → Classify**
2. Enter medical text or upload documents
3. View classified entities with KG validation status
4. Click unmatched entities to add them to the Knowledge Graph

#### BERT Training
1. Navigate to **BERT → Train Model**
2. Upload training data files (CSV format):
   - **train.csv** (required)
   - **dev.csv** (optional, for validation)
   - **test.csv** (optional, for final evaluation)
3. Configure training parameters
4. Start training

**CSV Format:**
```csv
words,sentence_id,labels
The,0,O
patient,0,O
has,0,O
diabetes,0,B-Disease
mellitus,0,I-Disease
.,0,O
```

#### BERT Classification
1. Navigate to **BERT → Classify**
2. Select a trained model
3. Enter text or upload documents
4. View classification results

### REST API

#### Classify Text (LLM)
```bash
curl -X POST "http://localhost:8000/classify/" \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient has diabetes and hypertension."}'
```

#### Classify Document (LLM)
```bash
curl -X POST "http://localhost:8000/classify/document" \
  -F "file=@medical_report.pdf" \
  -F "chunk_size=2000" \
  -F "min_confidence=0.5"
```

#### Train BERT Model
```bash
curl -X POST "http://localhost:8000/bert/train/files" \
  -F "train_file=@train.csv" \
  -F "test_file=@test.csv" \
  -F "model_name=my_model" \
  -F "epochs=3"
```

#### Classify Text (BERT)
```bash
curl -X POST "http://localhost:8000/bert/classify" \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient has fever and cough.", "model_name": "my_model"}'
```

## API Endpoints

### Classification (LLM)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/classify/` | Classify text with KG validation |
| POST | `/classify/document` | Classify uploaded document |
| GET | `/classify/kg/stats` | Get Knowledge Graph statistics |
| GET | `/classify/kg/search` | Search entities in KG |
| POST | `/classify/kg/add-entity` | Add entity to KG |

### BERT NER
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/bert/models` | List available models |
| POST | `/bert/classify` | Classify text |
| POST | `/bert/classify/document` | Classify document |
| POST | `/bert/train` | Train model (JSON) |
| POST | `/bert/train/files` | Train model (file upload) |
| GET | `/bert/status` | Service status |

### Configuration
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/llm-config/` | List LLM configurations |
| POST | `/llm-config/` | Create configuration |
| PUT | `/llm-config/{id}` | Update configuration |
| DELETE | `/llm-config/{id}` | Delete configuration |
| POST | `/llm-config/{id}/set-default` | Set as default |

## Knowledge Graph Validation

Entities are validated against the Knowledge Graph with the following status levels:

| Status | Similarity | Color |
|--------|-----------|-------|
| **Exact Match** | ≥95% | 🟢 Green |
| **Similar Match** | 70-94% | 🟠 Orange |
| **Low Match** | 60-69% | 🟡 Yellow |
| **Not Found** | <60% | ⚫ Gray |

Unmatched entities can be added to the KG directly from the classification results.

## Review Scorecard

Every classified entity can be reviewed with a scorecard adapted from the *7 Cs for Synthetic Medical Data Evaluation* (Zamzmi et al., 2025). Five criteria are scored per entity:

| Criterion | What it measures |
|-----------|------------------|
| Congruence | Similarity to the nearest Knowledge Graph entity of a compatible type |
| Coverage | Novelty with respect to the Knowledge Graph |
| Constraint | Rule-based plausibility (type, length, characters, type-specific patterns) |
| Completeness | Presence of the required and optional fields of the prediction record |
| Consistency | Classifier confidence, cross-checked with an independent BERT model |

The overall score is the weighted mean of the assessed criteria. Since scoring version 2.1 the default entity weights are Congruence 0.20, Coverage 0.00, Constraint 0.05, Completeness 0.30 and Consistency 0.45. They were calibrated against gold-standard correctness of LLM entity predictions on BC5CDR, BioRED and MedMentions ([experiments](https://github.com/jucanbe/SeWeBMeDA-Extension)). In that study, novelty (Coverage) was inversely related to correctness, so it receives no weight by default. An entity passes at an overall score of at least 0.75 and needs review at 0.50 or above; both thresholds and all weights can be changed under **Configuration → Entity Configuration**. Relation reviews keep the previous default weights (0.25 / 0.15 / 0.25 / 0.15 / 0.20).

## Token Limit Handling

When processing large documents, the system automatically:
1. Splits documents into chunks
2. Detects token limit errors during LLM processing
3. Recursively splits problematic chunks into smaller pieces
4. Merges and deduplicates results

## Development

### Running Tests
```bash
python -m unittest discover -s tests -t .
```
The suite uses only the standard library (`pytest tests/` also works if pytest is installed). It runs against a temporary database and a temporary Knowledge Graph, so `entity_classifier.db` and `KnowledgeGraph/` are never touched. The LLM server is replaced by a fake at the HTTP-client boundary; a tiny BERT model is built on the fly, and the shipped `BERT_models/Entities/MedMentions` model is used when present.

### Review scoring versions
Reviews store the `scoring_version` that produced them (`NULL` = logic before 2.0; `2.1` = calibrated default entity weights). On startup the database is migrated by adding nullable columns only; existing rows are kept as they are. To rescore old reviews explicitly:
```bash
curl -X POST "http://localhost:8000/entity-reviews/re-evaluate-all?only_legacy=true&dry_run=true"
curl -X POST "http://localhost:8000/entity-reviews/re-evaluate-all?only_legacy=true"
curl -X POST "http://localhost:8000/relation-reviews/re-evaluate-all?only_legacy=true"
```
`dry_run=true` returns old and new scores without saving. A real rescore appends the previous scores to each review's `score_history`.

### Code Structure
- **Routers**: Handle HTTP requests and responses
- **Services**: Contain business logic
- **Models**: Define data structures (Pydantic)
- **Database**: SQLAlchemy models and connection

## Technologies

- **Backend**: FastAPI, SQLAlchemy, Pydantic
- **LLM Integration**: OpenAI Python SDK (compatible with any OpenAI API)
- **NER**: HuggingFace Transformers, PyTorch
- **Knowledge Graph**: RDFLib (Turtle format)
- **Document Processing**: PyPDF, python-docx
- **Frontend**: Bootstrap 5, Jinja2

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Contributing

[Contributing guidelines]
