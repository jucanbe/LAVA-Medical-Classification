import io
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models.entities import (
    MedicalEntity,
    MedicalEntityType,
    EntityClassificationResult,
    BERTClassificationRequest,
    BERTClassificationResponse,
    BERTTrainingRequest,
    BERTTrainingResponse,
    BERTModelInfo,
    ClassificationWithValidationResponse,
)
from services.bert_ner import get_bert_ner_service, BERTNERService
from services.document_processor import get_document_processor
from services.knowledge_graph import get_kg_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/bert", tags=["BERT Medical Entity Classification"])


ENTITY_TYPE_MAPPING = {
    "disease": MedicalEntityType.DISEASE,
    "symptom": MedicalEntityType.SYMPTOM,
    "finding": MedicalEntityType.FINDING,
    "organ": MedicalEntityType.ORGAN,
    "imaging_procedure": MedicalEntityType.IMAGING_PROCEDURE,
    "examination_procedure": MedicalEntityType.EXAMINATION_PROCEDURE,
    "therapeutic_procedure": MedicalEntityType.THERAPEUTIC_PROCEDURE,
    "imaging_result": MedicalEntityType.IMAGING_RESULT,
    "examination_measure": MedicalEntityType.EXAMINATION_MEASURE,
    "parameter": MedicalEntityType.PARAMETER,
    "score": MedicalEntityType.SCORE,
    "therapy": MedicalEntityType.THERAPY,
    "substance": MedicalEntityType.SUBSTANCE,
    "adverse_event": MedicalEntityType.ADVERSE_EVENT,
    
    "quantitative_measure": MedicalEntityType.PARAMETER,
    "procedure": MedicalEntityType.THERAPEUTIC_PROCEDURE,
    
    "anatomicalstructure": MedicalEntityType.ORGAN,
    "bacterium": MedicalEntityType.SUBSTANCE,
    "biologicfunction": MedicalEntityType.FINDING,
    "biomedicaloccupationordiscipline": MedicalEntityType.EXAMINATION_PROCEDURE,
    "bodysubstance": MedicalEntityType.SUBSTANCE,
    "bodysystem": MedicalEntityType.ORGAN,
    "chemical": MedicalEntityType.SUBSTANCE,
    "clinicalattribute": MedicalEntityType.FINDING,
    "eukaryote": MedicalEntityType.SUBSTANCE,
    "food": MedicalEntityType.SUBSTANCE,
    "healthcareactivity": MedicalEntityType.THERAPEUTIC_PROCEDURE,
    "injuryorpoisoning": MedicalEntityType.DISEASE,
    "intellectualproduct": MedicalEntityType.FINDING,
    "medicaldevice": MedicalEntityType.IMAGING_PROCEDURE,
    "organization": MedicalEntityType.FINDING,
    "populationgroup": MedicalEntityType.FINDING,
    "professionaloroccupationalgroup": MedicalEntityType.FINDING,
    "researchactivity": MedicalEntityType.EXAMINATION_PROCEDURE,
    "spatialconcept": MedicalEntityType.ORGAN,
    "virus": MedicalEntityType.SUBSTANCE,
}


def _map_entity_type(entity_type_str: str) -> MedicalEntityType:
    """Map various BERT model label formats to our MedicalEntityType enum."""
    normalized = entity_type_str.lower().replace("_", "").replace("-", "")
    
    if normalized in ENTITY_TYPE_MAPPING:
        return ENTITY_TYPE_MAPPING[normalized]
    
    try:
        return MedicalEntityType(entity_type_str.lower())
    except ValueError:
        pass
    
    return MedicalEntityType.FINDING


def _get_service() -> BERTNERService:
    """Get the BERT NER service."""
    return get_bert_ner_service()


@router.get(
    "/models",
    response_model=List[BERTModelInfo],
    summary="List available BERT models",
    description="Get a list of all BERT models available for classification. Use model_type query param to filter by 'entity' or 'relation'."
)
async def list_models(model_type: Optional[str] = None):
    """List all available BERT models, optionally filtered by type."""
    service = _get_service()
    models = service.get_available_models(model_type=model_type)
    return [BERTModelInfo(**m) for m in models]


@router.post(
    "/models/{model_name}/load",
    summary="Load a BERT model",
    description="Load a BERT model into memory for faster inference."
)
async def load_model(model_name: str):
    """Load a specific model into memory."""
    service = _get_service()
    
    try:
        service.load_model(model_name)
        return {
            "success": True,
            "message": f"Model '{model_name}' loaded successfully",
            "model_name": model_name
        }
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found"
        )
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {str(e)}"
        )


@router.post(
    "/models/{model_name}/unload",
    summary="Unload a BERT model",
    description="Unload a BERT model from memory to free resources."
)
async def unload_model(model_name: str):
    """Unload a model from memory."""
    service = _get_service()
    service.unload_model(model_name)
    
    return {
        "success": True,
        "message": f"Model '{model_name}' unloaded",
        "model_name": model_name
    }


@router.post(
    "/classify",
    response_model=ClassificationWithValidationResponse,
    summary="Classify medical entities using BERT",
    description="""
    Classify medical entities in text using a BERT model.
    
    This endpoint uses a fine-tuned BERT model for Named Entity Recognition.
    The model must be loaded first using the /bert/models/{name}/load endpoint,
    or it will be loaded automatically on first use.
    
    Results are validated against the Knowledge Graph.
    
    **Entity Types:**
    - disease: Pathological conditions
    - symptom: Symptoms
    - finding: Clinical or imaging-based findings
    - organ: Anatomical structures
    - imaging_procedure: Imaging procedures
    - examination_procedure: Clinical exams or lab tests
    - therapeutic_procedure: Therapeutic procedures
    - imaging_result: Imaging results
    - examination_measure: Examination measures
    - parameter: Quantitative parameters
    - score: Clinical scores
    - therapy: Therapies
    - substance: Drugs / Substances
    - adverse_event: Adverse events
    """
)
async def classify_bert(
    request: BERTClassificationRequest,
    kg_backend: str = Query("all", description="KG backend to query: all, internal, external"),
    db: AsyncSession = Depends(get_db)
):
    """Classify entities using BERT with KG validation."""
    service = _get_service()
    
    try:
        entities, inference_time = service.classify(
            text=request.text,
            model_name=request.model_name
        )
        
        medical_entities = []
        for entity in entities:
            entity_type = _map_entity_type(entity["type"])
            
            medical_entities.append(MedicalEntity(
                text=entity["text"],
                entity_type=entity_type,
                confidence=entity.get("confidence", 0.9),
                start_position=entity.get("start_pos"),
                end_position=entity.get("end_pos")
            ))
        
        result = EntityClassificationResult(
            entities=medical_entities,
            raw_text=request.text
        )
        
        kg_service = get_kg_service()
        validation_result = await kg_service.validate_entities_async(result, kg_backend=kg_backend)
        
        try:
            from routers.pending_entities import save_pending_entities
            
            entities_to_save = []
            if validation_result and validation_result.validated_entities:
                for ve in validation_result.validated_entities:
                    if ve.validation_status != "exact_match":
                        entity_data = {
                            "text": ve.entity.text,
                            "entity_type": ve.entity.entity_type.value if hasattr(ve.entity.entity_type, 'value') else ve.entity.entity_type,
                            "confidence": ve.entity.confidence,
                            "validation_status": ve.validation_status,
                        }
                        if ve.kg_matches and len(ve.kg_matches) > 0:
                            entity_data["similarity_score"] = ve.kg_matches[0].similarity_score
                            entity_data["matched_kg_label"] = ve.kg_matches[0].kg_label
                            entity_data["matched_kg_uri"] = str(ve.kg_matches[0].kg_uri) if ve.kg_matches[0].kg_uri else None
                        entities_to_save.append(entity_data)
            
            if entities_to_save:
                await save_pending_entities(
                    entities=entities_to_save,
                    source="bert",
                    source_model=request.model_name or service.default_model,
                    context=request.text[:500],
                    db=db
                )
        except Exception as pe:
            logger.warning(f"Error saving pending entities: {pe}")
        
        return ClassificationWithValidationResponse(
            success=True,
            result=result,
            kg_validation=validation_result,
            model_used=request.model_name or service.default_model
        )
        
    except ValueError as e:
        return ClassificationWithValidationResponse(
            success=False,
            error=str(e)
        )
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e)
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"BERT classification error: {e}")
        return ClassificationWithValidationResponse(
            success=False,
            error=f"Classification failed: {str(e)}"
        )


@router.post(
    "/classify/document",
    response_model=ClassificationWithValidationResponse,
    summary="Classify medical entities from document using BERT",
    description="""
    Upload a document (PDF, DOCX, TXT) and extract medical entities using BERT.
    
    The document is automatically:
    1. Parsed to extract text
    2. Split into manageable chunks
    3. Each chunk is processed by the BERT model
    4. Results are merged and deduplicated
    5. Validated against Knowledge Graph
    6. Non-KG entities saved to Pending Entities
    
    **Supported formats:** PDF, DOCX, DOC, TXT
    """
)
async def classify_document_bert(
    file: UploadFile = File(..., description="Document to analyze (PDF, DOCX, TXT)"),
    model_name: Optional[str] = Form(None, description="BERT model to use"),
    min_confidence: float = Form(0.5, description="Minimum confidence threshold"),
    chunk_size: int = Form(2000, description="Characters per chunk (500-4000)"),
    kg_backend: str = Form("all", description="KG backend to query: all, internal, external"),
    db: AsyncSession = Depends(get_db)
):
    """
    Classify medical entities from an uploaded document using BERT.
    """
    service = _get_service()
    
    try:
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="No file provided"
            )
        
        processor = get_document_processor()
        processor.chunk_size = max(500, min(4000, chunk_size))
        
        content = await file.read()
        file_obj = io.BytesIO(content)
        
        try:
            chunks, doc_info = processor.process_document(file_obj, file.filename)
        except ImportError as e:
            raise HTTPException(
                status_code=503,
                detail=str(e)
            )
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=str(e)
            )
        
        if not chunks:
            return BERTClassificationResponse(
                success=True,
                result=EntityClassificationResult(entities=[], raw_text=""),
                kg_validation=None,
                model_used=model_name or service.default_model
            )
        
        all_entities = []
        seen_entities = set()
        total_inference_time = 0
        
        for chunk in chunks:
            try:
                entities, inference_time = service.classify(
                    text=chunk.text,
                    model_name=model_name
                )
                total_inference_time += inference_time
                
                for entity in entities:
                    entity_key = (entity["text"].lower(), entity["type"])
                    
                    if entity_key not in seen_entities:
                        seen_entities.add(entity_key)
                        
                        entity_type = _map_entity_type(entity["type"])
                        
                        start_pos = entity.get("start_pos")
                        end_pos = entity.get("end_pos")
                        if start_pos is not None:
                            start_pos += chunk.start_char
                        if end_pos is not None:
                            end_pos += chunk.start_char
                        
                        all_entities.append(MedicalEntity(
                            text=entity["text"],
                            entity_type=entity_type,
                            confidence=entity.get("confidence", 0.9),
                            start_position=start_pos,
                            end_position=end_pos
                        ))
                        
            except Exception as e:
                logger.warning(f"Error processing chunk {chunk.chunk_index}: {e}")
        
        filtered_entities = [e for e in all_entities if e.confidence >= min_confidence]
        
        result = EntityClassificationResult(
            entities=filtered_entities,
            raw_text=f"[Document: {doc_info.filename}] ({doc_info.total_chars} chars, {doc_info.total_chunks} chunks)"
        )
        
        kg_service = get_kg_service()
        validation_result = await kg_service.validate_entities_async(result, kg_backend=kg_backend)
        
        try:
            from routers.pending_entities import save_pending_entities
            
            entities_to_save = []
            if validation_result and validation_result.validated_entities:
                for ve in validation_result.validated_entities:
                    if ve.validation_status != "exact_match":
                        entity_data = {
                            "text": ve.entity.text,
                            "entity_type": ve.entity.entity_type.value if hasattr(ve.entity.entity_type, 'value') else ve.entity.entity_type,
                            "confidence": ve.entity.confidence,
                            "validation_status": ve.validation_status,
                        }
                        if ve.kg_matches and len(ve.kg_matches) > 0:
                            entity_data["similarity_score"] = ve.kg_matches[0].similarity_score
                            entity_data["matched_kg_label"] = ve.kg_matches[0].kg_label
                            entity_data["matched_kg_uri"] = str(ve.kg_matches[0].kg_uri) if ve.kg_matches[0].kg_uri else None
                        entities_to_save.append(entity_data)
            
            if entities_to_save:
                await save_pending_entities(
                    entities=entities_to_save,
                    source="bert",
                    source_model=model_name or service.default_model,
                    context=f"[Document: {doc_info.filename}]",
                    db=db
                )
                logger.info(f"Saved {len(entities_to_save)} entities to pending list from document")
        except Exception as pe:
            logger.warning(f"Error saving pending entities from document: {pe}")
        
        return ClassificationWithValidationResponse(
            success=True,
            result=result,
            kg_validation=validation_result,
            model_used=model_name or service.default_model
        )
        
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"BERT document classification error: {e}")
        return ClassificationWithValidationResponse(
            success=False,
            error=f"Document classification failed: {str(e)}"
        )


@router.post(
    "/train",
    response_model=BERTTrainingResponse,
    summary="Train a new BERT model (JSON)",
    description="""
    Train or fine-tune a BERT model for medical entity recognition.
    Use /train/files endpoint for file uploads.
    
    This endpoint supports:
    - Training from a base model (default: BioBERT)
    - Fine-tuning an existing custom model
    
    **Note:** Training can take several minutes depending on data size and epochs.
    """
)
async def train_model(request: BERTTrainingRequest):
    """Train a new BERT model."""
    service = _get_service()
    
    try:
        result = service.train(
            model_name=request.model_name,
            base_model=request.base_model,
            training_data=request.training_data,
            train_data=request.train_data,
            dev_data=request.dev_data,
            test_data=request.test_data,
            training_file=request.training_file,
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            model_type=request.model_type
        )
        
        return BERTTrainingResponse(
            success=True,
            model_name=result["model_name"],
            model_path=result.get("model_path"),
            training_metrics=result.get("training_metrics")
        )
        
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e)
        )
    except ValueError as e:
        return BERTTrainingResponse(
            success=False,
            model_name=request.model_name,
            error=str(e)
        )
    except Exception as e:
        logger.error(f"BERT training error: {e}")
        return BERTTrainingResponse(
            success=False,
            model_name=request.model_name,
            error=f"Training failed: {str(e)}"
        )


@router.post(
    "/train/files",
    response_model=BERTTrainingResponse,
    summary="Train a BERT model with file uploads",
    description="""
    Train or fine-tune a BERT model using CSV file uploads.
    
    **CSV Format (required columns: words, sentence_id, labels):**
    ```csv
    words,sentence_id,labels
    The,0,O
    patient,0,O
    has,0,O
    diabetes,0,B-Disease
    mellitus,0,I-Disease
    .,0,O
    Treatment,1,O
    includes,1,O
    insulin,1,B-Substance
    ```
    
    **Files:**
    - **train_file** (required): Training data CSV
    - **dev_file** (optional): Development/validation data CSV
    - **test_file** (optional): Test data CSV for final evaluation
    
    **Note:** Training can take several minutes depending on data size and epochs.
    """
)
async def train_model_with_files(
    train_file: UploadFile = File(..., description="Training data CSV file (required)"),
    test_file: Optional[UploadFile] = File(None, description="Test data CSV file (optional)"),
    dev_file: Optional[UploadFile] = File(None, description="Development/validation data CSV file (optional)"),
    model_name: str = Form(..., description="Name for the new model"),
    base_model: str = Form("dmis-lab/biobert-base-cased-v1.1", description="Base HuggingFace model"),
    epochs: int = Form(3, description="Number of training epochs"),
    batch_size: int = Form(16, description="Training batch size"),
    learning_rate: float = Form(5e-5, description="Learning rate"),
    model_type: str = Form("entity", description="Type of model: 'entity' or 'relation'")
):
    """Train a BERT model with CSV file uploads."""
    service = _get_service()
    
    try:
        train_content = await train_file.read()
        train_data = train_content.decode('utf-8')
        
        dev_data = None
        if dev_file:
            dev_content = await dev_file.read()
            dev_data = dev_content.decode('utf-8')
        
        test_data = None
        if test_file:
            test_content = await test_file.read()
            test_data = test_content.decode('utf-8')
        
        result = service.train(
            model_name=model_name,
            base_model=base_model,
            train_data=train_data,
            dev_data=dev_data,
            test_data=test_data,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            model_type=model_type
        )
        
        return BERTTrainingResponse(
            success=True,
            model_name=result["model_name"],
            model_path=result.get("model_path"),
            training_metrics=result.get("training_metrics")
        )
        
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e)
        )
    except ValueError as e:
        return BERTTrainingResponse(
            success=False,
            model_name=model_name,
            error=str(e)
        )
    except Exception as e:
        logger.error(f"BERT training error: {e}")
        return BERTTrainingResponse(
            success=False,
            model_name=model_name,
            error=f"Training failed: {str(e)}"
        )


@router.get(
    "/status",
    summary="Get BERT service status",
    description="Get the current status of the BERT NER service."
)
async def get_status():
    """Get service status."""
    service = _get_service()
    
    try:
        service._load_dependencies()
        transformers_available = True
        device = str(service._device)
    except ImportError:
        transformers_available = False
        device = None
    
    return {
        "transformers_available": transformers_available,
        "device": device,
        "models_directory": str(service.models_dir),
        "loaded_models": list(service.loaded_models.keys()),
        "default_model": service.default_model,
        "available_models": [m["name"] for m in service.get_available_models()]
    }
