import io
import logging
from fastapi import APIRouter, Depends, HTTPException, status, Query, UploadFile, File, Form
from fastapi.responses import PlainTextResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional

from database import get_db, db_manager
from models import (
    ClassificationRequest,
    ClassificationResponse,
    MedicalEntityType,
    MedicalEntity,
    EntityClassificationResult,
    KGValidationResult,
    ClassificationWithValidationResponse,
    IOBExportResponse,
    AddEntityToKGRequest,
    AddEntityToKGResponse
)
from services import LLMClient, MedicalEntityClassifier, get_kg_service, reload_kg_service
from services.knowledge_graph import KnowledgeGraphService
from services.document_processor import get_document_processor, TextChunk, DocumentInfo
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/classify", tags=["LLM Medical Entity Classification"])


async def get_llm_client(
    config_id: int = None,
    db: AsyncSession = None,
    verify: bool = True
) -> tuple:
    """
    Get an LLM client and custom prompt based on the specified configuration.
    If config_id is not specified, uses the default configuration.
    
    Args:
        config_id: ID of the LLM configuration to use
        db: Database session
        verify: Whether to verify connection and detect server capabilities
        
    Returns:
        Tuple of (LLMClient, custom_entity_prompt or None)
    """
    custom_prompt = None
    
    if config_id:
        config = await db_manager.get_llm_config_by_id(db, config_id)
        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"LLM configuration with ID {config_id} not found"
            )
        custom_prompt = getattr(config, 'entity_prompt', None)
    else:
        config = await db_manager.get_default_llm_config(db)
        
        if not config:
            client = LLMClient(
                base_url=settings.DEFAULT_VLLM_BASE_URL,
                model_name=settings.DEFAULT_MODEL_NAME,
                api_key="EMPTY",
                temperature=0.1,
                max_tokens=2048
            )
            if verify:
                await client.verify_connection()
            return client, None
        
        custom_prompt = getattr(config, 'entity_prompt', None)
    
    client = LLMClient.from_config(config)
    if verify:
        await client.verify_connection()
    return client, custom_prompt


@router.post(
    "/",
    response_model=ClassificationWithValidationResponse,
    summary="Classify medical entities with KG validation",
    description="Analyze text, classify entities, and validate against Knowledge Graph"
)
async def classify_medical_entities(
    request: ClassificationRequest,
    kg_backend: str = Query("all", description="KG backend to query: all, internal, external"),
    db: AsyncSession = Depends(get_db)
):
    """
    Main endpoint to classify medical entities in a text with KG validation.
    
    The text is analyzed by an LLM model that identifies and classifies
    medical entities. Each entity is then validated against the Knowledge Graph.
    
    The response includes each entity found with:
    - The exact text of the entity
    - The type of medical entity
    - Classification confidence level
    - Position in the original text
    - KG validation status (exact_match, similar_match, low_match, not_found)
    - Similar entities found in KG
    """
    try:
        llm_client, custom_prompt = await get_llm_client(request.config_id, db)
        
        classifier = MedicalEntityClassifier(llm_client, custom_system_prompt=custom_prompt)
        
        result = await classifier.classify(
            text=request.text,
            include_context=request.include_context,
            min_confidence=request.min_confidence
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
                    source="llm",
                    source_model=llm_client.model_name,
                    context=request.text[:500],
                    db=db
                )
        except Exception as pe:
            logger.warning(f"Error saving pending entities: {pe}")
        
        return ClassificationWithValidationResponse(
            success=True,
            result=result,
            kg_validation=validation_result,
            model_used=llm_client.model_name
        )
        
    except Exception as e:
        return ClassificationWithValidationResponse(
            success=False,
            error=str(e),
            model_used=None
        )


@router.post(
    "/document",
    response_model=ClassificationWithValidationResponse,
    summary="Classify medical entities from document",
    description="""
    Upload a document (PDF, DOCX, TXT) and extract medical entities.
    
    The document is automatically:
    1. Parsed to extract text
    2. Split into manageable chunks
    3. Each chunk is processed by the LLM
    4. Results are merged and deduplicated
    5. Validated against Knowledge Graph
    
    **Supported formats:** PDF, DOCX, DOC, TXT
    """
)
async def classify_document(
    file: UploadFile = File(..., description="Document to analyze (PDF, DOCX, TXT)"),
    config_id: Optional[int] = Form(None, description="LLM configuration ID"),
    include_context: bool = Form(True, description="Include surrounding context"),
    min_confidence: float = Form(0.5, description="Minimum confidence threshold"),
    chunk_size: int = Form(2000, description="Characters per chunk (500-4000)"),
    kg_backend: str = Form("all", description="KG backend to query: all, internal, external"),
    db: AsyncSession = Depends(get_db)
):
    """
    Classify medical entities from an uploaded document.
    Automatically handles large documents by splitting into chunks.
    """
    try:
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
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
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(e)
            )
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        
        if not chunks:
            return ClassificationResponse(
                success=True,
                result=EntityClassificationResult(entities=[], raw_text=""),
                model_used=None
            )
        
        llm_client, custom_prompt = await get_llm_client(config_id, db)
        classifier = MedicalEntityClassifier(llm_client, custom_system_prompt=custom_prompt)
        
        all_entities = []
        seen_entities = set()
        chunks_processed = 0
        chunks_failed = 0
        
        for chunk in chunks:
            try:
                logger.info(f"Processing chunk {chunk.chunk_index + 1}/{len(chunks)} ({len(chunk.text)} chars)")
                
                result = await classifier.classify_with_auto_split(
                    text=chunk.text,
                    include_context=include_context,
                    min_confidence=min_confidence
                )
                
                logger.info(f"Chunk {chunk.chunk_index + 1} returned {len(result.entities)} entities")
                
                for entity in result.entities:
                    entity_key = (entity.text.lower(), entity.entity_type.value)
                    
                    if entity_key not in seen_entities:
                        seen_entities.add(entity_key)
                        
                        if entity.start_position is not None:
                            entity.start_position += chunk.start_char
                        if entity.end_position is not None:
                            entity.end_position += chunk.start_char
                        
                        all_entities.append(entity)
                
                chunks_processed += 1
                        
            except Exception as e:
                chunks_failed += 1
                logger.error(f"Error processing chunk {chunk.chunk_index}: {e}")
        
        logger.info(f"Document processing complete: {chunks_processed} chunks OK, {chunks_failed} failed, {len(all_entities)} total entities")
        
        combined_result = EntityClassificationResult(
            entities=all_entities,
            raw_text=f"[Document: {doc_info.filename}] ({doc_info.total_chars} chars, {doc_info.total_chunks} chunks)"
        )
        
        kg_service = get_kg_service()
        validation_result = await kg_service.validate_entities_async(combined_result, kg_backend=kg_backend)
        
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
                    source="llm",
                    source_model=llm_client.model_name,
                    context=f"[Document: {doc_info.filename}]",
                    db=db
                )
                logger.info(f"Saved {len(entities_to_save)} entities to pending list from document")
        except Exception as pe:
            logger.warning(f"Error saving pending entities from document: {pe}")
        
        return ClassificationWithValidationResponse(
            success=True,
            result=combined_result,
            kg_validation=validation_result,
            model_used=llm_client.model_name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        return ClassificationWithValidationResponse(
            success=False,
            error=str(e),
            model_used=None
        )


@router.post(
    "/batch",
    response_model=List[ClassificationResponse],
    summary="Classify multiple texts",
    description="Analyze multiple texts and classify the medical entities"
)
async def classify_batch(
    texts: List[str],
    config_id: int = None,
    include_context: bool = True,
    min_confidence: float = 0.5,
    db: AsyncSession = Depends(get_db)
):
    """
    Classify medical entities in multiple texts.
    """
    try:
        llm_client, custom_prompt = await get_llm_client(config_id, db)
        classifier = MedicalEntityClassifier(llm_client, custom_system_prompt=custom_prompt)
        
        results = await classifier.classify_batch(
            texts=texts,
            include_context=include_context,
            min_confidence=min_confidence
        )
        
        return [
            ClassificationResponse(
                success=True,
                result=result,
                model_used=llm_client.model_name
            )
            for result in results
        ]
        
    except Exception as e:
        return [
            ClassificationResponse(
                success=False,
                error=str(e),
                model_used=None
            )
            for _ in texts
        ]


@router.get(
    "/entity-types",
    response_model=List[str],
    summary="Get entity types",
    description="Returns the list of supported medical entity types"
)
async def get_entity_types():
    """Returns the types of medical entities that the system can classify."""
    return [e.value for e in MedicalEntityType]


@router.post(
    "/validate",
    response_model=ClassificationWithValidationResponse,
    summary="Classify and validate against Knowledge Graph",
    description="Classify medical entities and validate them against the Knowledge Graph to detect exact matches, similar entities, new discoveries, or possible hallucinations"
)
async def classify_and_validate(
    request: ClassificationRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Classify medical entities and validate against Knowledge Graph.
    
    This endpoint performs two operations:
    1. Extracts and classifies medical entities from the text using LLM
    2. Validates each entity against the Knowledge Graph (RDF/TTL) to determine:
       - exact_match: Entity found exactly in KG
       - similar_match: Similar entity found in KG
       - low_match: Low similarity match
       - not_found: Entity not in KG
    """
    try:
        llm_client, custom_prompt = await get_llm_client(request.config_id, db)
        
        classifier = MedicalEntityClassifier(llm_client, custom_system_prompt=custom_prompt)
        
        classification_result, validation_result = await classifier.classify_and_validate(
            text=request.text,
            include_context=request.include_context,
            min_confidence=request.min_confidence
        )
        
        return ClassificationWithValidationResponse(
            success=True,
            result=classification_result,
            kg_validation=validation_result,
            model_used=llm_client.model_name
        )
        
    except Exception as e:
        return ClassificationWithValidationResponse(
            success=False,
            error=str(e),
            model_used=None
        )


@router.post(
    "/validate/document",
    response_model=ClassificationWithValidationResponse,
    summary="Classify and validate entities from document",
    description="""
    Upload a document (PDF, DOCX, TXT) and extract medical entities,
    then validate them against the Knowledge Graph.
    
    The document is automatically:
    1. Parsed to extract text
    2. Split into manageable chunks
    3. Each chunk is processed by the LLM
    4. Results are merged and deduplicated
    5. Validated against the Knowledge Graph
    """
)
async def classify_and_validate_document(
    file: UploadFile = File(..., description="Document to analyze (PDF, DOCX, TXT)"),
    config_id: Optional[int] = Form(None, description="LLM configuration ID"),
    min_confidence: float = Form(0.5, description="Minimum confidence threshold"),
    chunk_size: int = Form(2000, description="Characters per chunk (500-4000)"),
    db: AsyncSession = Depends(get_db)
):
    """
    Classify medical entities from an uploaded document and validate against KG.
    """
    try:
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
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
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(e)
            )
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        
        if not chunks:
            return ClassificationWithValidationResponse(
                success=True,
                result=EntityClassificationResult(entities=[], raw_text=""),
                kg_validation=None,
                model_used=None
            )
        
        llm_client, custom_prompt = await get_llm_client(config_id, db)
        classifier = MedicalEntityClassifier(llm_client, custom_system_prompt=custom_prompt)
        
        all_entities = []
        seen_entities = set()
        
        for chunk in chunks:
            try:
                result, validation = await classifier.classify_and_validate(
                    text=chunk.text,
                    include_context=True,
                    min_confidence=min_confidence
                )
                
                for entity in result.entities:
                    entity_key = (entity.text.lower(), entity.entity_type.value)
                    
                    if entity_key not in seen_entities:
                        seen_entities.add(entity_key)
                        all_entities.append(entity)
                        
            except Exception as e:
                import logging
                logging.warning(f"Error processing chunk {chunk.chunk_index}: {e}")
        
        combined_classification = EntityClassificationResult(
            entities=all_entities,
            raw_text=f"[Document: {doc_info.filename}]"
        )
        
        kg_service = get_kg_service()
        validated_entities = []
        stats = {
            "exact_matches": 0,
            "similar_matches": 0,
            "low_matches": 0,
            "not_found": 0
        }
        
        for entity in all_entities:
            validated = await kg_service.validate_entity_async(entity)
            validated_entities.append(validated)
            
            if validated.kg_status.value == "exact_match":
                stats["exact_matches"] += 1
            elif validated.kg_status.value == "similar_match":
                stats["similar_matches"] += 1
            elif validated.kg_status.value == "low_match":
                stats["low_matches"] += 1
            else:
                stats["not_found"] += 1
        
        validation_result = KGValidationResult(
            validated_entities=validated_entities,
            summary=stats,
            total_entities=len(all_entities),
            exact_matches=stats["exact_matches"],
            similar_matches=stats["similar_matches"],
            low_matches=stats["low_matches"],
            not_found=stats["not_found"]
        )
        
        return ClassificationWithValidationResponse(
            success=True,
            result=combined_classification,
            kg_validation=validation_result,
            model_used=llm_client.model_name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        return ClassificationWithValidationResponse(
            success=False,
            error=str(e),
            model_used=None
        )


@router.post(
    "/kg/upload",
    summary="Upload TTL file to Knowledge Graph",
    description="Upload a new TTL file to the Knowledge Graph directory"
)
async def upload_ttl_file(
    file: UploadFile = File(..., description="TTL file to upload"),
    reload: bool = Query(True, description="Reload KG after upload")
):
    """
    Upload a TTL file to the Knowledge Graph directory.
    Optionally reloads the KG to include the new file.
    """
    from pathlib import Path
    import rdflib
    
    if not file.filename.endswith('.ttl'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .ttl files are allowed"
        )
    
    try:
        content = await file.read()
        
        test_graph = rdflib.Graph()
        try:
            test_graph.parse(data=content.decode('utf-8'), format='turtle')
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid TTL format: {str(e)}"
            )
        
        kg_service = get_kg_service()
        kg_dir = kg_service.kg_directory
        
        file_path = kg_dir / file.filename
        with open(file_path, 'wb') as f:
            f.write(content)
        
        logger.info(f"Uploaded TTL file: {file_path}")
        
        stats = None
        if reload:
            kg_service = reload_kg_service()
            stats = kg_service.get_kg_stats()
        
        return {
            "success": True,
            "message": f"File '{file.filename}' uploaded successfully",
            "file_path": str(file_path),
            "triples_in_file": len(test_graph),
            "stats": stats
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading file: {str(e)}"
        )


@router.delete(
    "/kg/file/{filename}",
    summary="Delete TTL file from Knowledge Graph",
    description="Delete a TTL file from the Knowledge Graph directory"
)
async def delete_ttl_file(
    filename: str,
    reload: bool = Query(True, description="Reload KG after deletion")
):
    """Delete a TTL file from the Knowledge Graph directory."""
    from pathlib import Path
    
    if not filename.endswith('.ttl'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .ttl files can be deleted"
        )
    
    if '..' in filename or '/' in filename or '\\' in filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename"
        )
    
    try:
        kg_service = get_kg_service()
        kg_dir = kg_service.kg_directory
        file_path = kg_dir / filename
        
        if not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File '{filename}' not found"
            )
        
        file_path.unlink()
        logger.info(f"Deleted TTL file: {file_path}")
        
        stats = None
        if reload:
            kg_service = reload_kg_service()
            stats = kg_service.get_kg_stats()
        
        return {
            "success": True,
            "message": f"File '{filename}' deleted successfully",
            "stats": stats
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting file: {str(e)}"
        )


@router.get(
    "/kg/files",
    summary="List TTL files in Knowledge Graph directory",
    description="Get a list of all TTL files in the Knowledge Graph directory"
)
async def list_ttl_files():
    """List all TTL files in the Knowledge Graph directory."""
    from pathlib import Path
    import os
    
    try:
        kg_service = get_kg_service()
        kg_dir = kg_service.kg_directory
        
        files = []
        for file_path in kg_dir.glob("*.ttl"):
            stat = file_path.stat()
            files.append({
                "name": file_path.name,
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "loaded": file_path.name in [Path(f).name for f in kg_service._loaded_files]
            })
        
        return {
            "directory": str(kg_dir),
            "files": files,
            "total_files": len(files)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing files: {str(e)}"
        )


@router.post(
    "/kg/reload",
    summary="Reload Knowledge Graph",
    description="Reload the Knowledge Graph from TTL files"
)
async def reload_knowledge_graph(
    kg_directory: Optional[str] = Query(
        None,
        description="Optional path to KG directory. Uses default if not specified."
    )
):
    """Reload the Knowledge Graph from disk."""
    try:
        kg_service = reload_kg_service(kg_directory)
        stats = kg_service.get_kg_stats()
        return {
            "success": True,
            "message": "Knowledge Graph reloaded successfully",
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reloading Knowledge Graph: {str(e)}"
        )


@router.get(
    "/kg/stats",
    summary="Get Knowledge Graph statistics",
    description="Get statistics about the loaded Knowledge Graph"
)
async def get_kg_stats():
    """Get statistics about the loaded Knowledge Graph."""
    try:
        kg_service = get_kg_service()
        return kg_service.get_kg_stats()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting KG stats: {str(e)}"
        )


@router.get(
    "/kg/search",
    summary="Search entities in Knowledge Graph",
    description="Search for entities in the Knowledge Graph by text"
)
async def search_kg_entities(q: str, limit: int = 10):
    """Search for entities in the Knowledge Graph."""
    try:
        kg_service = get_kg_service()
        results = kg_service.search_entities(q, max_results=limit)
        return {"query": q, "results": results}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching KG: {str(e)}"
        )


@router.get(
    "/example",
    response_model=ClassificationResponse,
    summary="Classification example",
    description="Execute a classification example with a predefined medical text"
)
async def classification_example(db: AsyncSession = Depends(get_db)):
    """
    Execute a classification example to demonstrate the functionality.
    """
    example_text = """
    The 65-year-old patient presents with chest pain and dyspnea for 3 days.
    Aspirin 100mg was administered orally every 24 hours.
    Laboratory tests show elevated troponin.
    A coronary angiography will be performed at General Hospital.
    Cardiologist Dr. Garcia recommends complete rest.
    """
    
    request = ClassificationRequest(
        text=example_text,
        include_context=True,
        min_confidence=0.5
    )
    
    return await classify_medical_entities(request, db)


@router.post(
    "/exportText",
    response_model=IOBExportResponse,
    summary="Export classification as IOB CSV format",
    description="Classify medical entities and export the result in IOB CSV format (words,sentence_id,labels) for BERT training"
)
async def export_text_iob(
    request: ClassificationRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Classify medical entities and export in IOB CSV format for BERT training.
    
    CSV format example:
    ```
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
    
    This output can be used to train or fine-tune BERT NER models.
    """
    try:
        llm_client, custom_prompt = await get_llm_client(request.config_id, db)
        classifier = MedicalEntityClassifier(llm_client, custom_system_prompt=custom_prompt)
        
        result = await classifier.classify(
            text=request.text,
            include_context=False,
            min_confidence=request.min_confidence
        )
        
        iob_text = _convert_to_iob(request.text, result.entities)
        
        return IOBExportResponse(
            success=True,
            original_text=request.text,
            iob_format=iob_text,
            entities_count=result.total_entities,
            model_used=llm_client.model_name
        )
        
    except Exception as e:
        return IOBExportResponse(
            success=False,
            original_text=request.text,
            iob_format="",
            entities_count=0,
            error=str(e)
        )


@router.post(
    "/exportText/raw",
    response_class=PlainTextResponse,
    summary="Export classification as raw IOB CSV text",
    description="Returns plain text IOB CSV format ready to save as training file"
)
async def export_text_iob_raw(
    request: ClassificationRequest,
    db: AsyncSession = Depends(get_db)
):
    """Export classification as plain text IOB CSV format."""
    response = await export_text_iob(request, db)
    if not response.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=response.error
        )
    return response.iob_format


def _format_entity_type_for_iob(entity_type: str) -> str:
    """Format entity type for IOB labels (CamelCase)."""
    type_mapping = {
        'disease': 'Disease',
        'symptom': 'Symptom',
        'finding': 'Finding',
        'organ': 'Organ',
        'imaging_procedure': 'ImagingProcedure',
        'examination_procedure': 'ExaminationProcedure',
        'therapeutic_procedure': 'TherapeuticProcedure',
        'imaging_result': 'ImagingResult',
        'examination_measure': 'ExaminationMeasure',
        'parameter': 'Parameter',
        'score': 'Score',
        'therapy': 'Therapy',
        'substance': 'Substance',
        'adverse_event': 'AdverseEvent',
    }
    return type_mapping.get(entity_type.lower(), entity_type.capitalize())


def _convert_to_iob(text: str, entities: list) -> str:
    """
    Convert text and entities to IOB CSV format for BERT training.
    
    Args:
        text: Original text
        entities: List of MedicalEntity objects
        
    Returns:
        CSV formatted string with columns: words,sentence_id,labels
    """
    import re
    
    sentence_pattern = r'(?<=[.!?])\s+|\n+'
    sentences = re.split(sentence_pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    sorted_entities = sorted(
        [e for e in entities if e.start_position is not None],
        key=lambda x: x.start_position
    )
    
    csv_lines = ["words,sentence_id,labels"]
    
    current_position = 0
    for sentence_id, sentence in enumerate(sentences):
        sentence_start = text.find(sentence, current_position)
        if sentence_start == -1:
            sentence_start = current_position
        sentence_end = sentence_start + len(sentence)
        current_position = sentence_end
        
        for match in re.finditer(r'\S+', sentence):
            token_text = match.group()
            token_start = sentence_start + match.start()
            token_end = sentence_start + match.end()
            tag = 'O'
            
            for entity in sorted_entities:
                entity_start = entity.start_position
                entity_end = entity.end_position
                entity_type = entity.entity_type.value
                formatted_type = _format_entity_type_for_iob(entity_type)
                
                if token_start >= entity_start and token_end <= entity_end:
                    is_first = True
                    for prev_match in re.finditer(r'\S+', text[entity_start:token_start]):
                        is_first = False
                        break
                    tag = f'B-{formatted_type}' if is_first else f'I-{formatted_type}'
                    break
                elif token_start < entity_end and token_end > entity_start:
                    is_first = token_start <= entity_start
                    tag = f'B-{formatted_type}' if is_first else f'I-{formatted_type}'
                    break
            
            if ',' in token_text or '"' in token_text:
                token_text = f'"{token_text}"'
            
            csv_lines.append(f"{token_text},{sentence_id},{tag}")
    
    return '\n'.join(csv_lines)


@router.post(
    "/kg/add-entity",
    response_model=AddEntityToKGResponse,
    summary="Add entity to Knowledge Graph",
    description="Adds a new entity to the Knowledge Graph with optional linking to existing similar entities"
)
async def add_entity_to_kg(
    request: AddEntityToKGRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Add a new entity to the Knowledge Graph.
    
    If link_to_uri and link_type are provided, the new entity will be linked
    to an existing entity using the specified relationship type.
    
    Link types supported:
    - sameAs: Entities are identical
    - exactMatch: Exact semantic match (SKOS)
    - closeMatch: Close semantic match (SKOS)
    - relatedTo: General relation
    - narrowerThan: New entity is more specific
    - broaderThan: New entity is more general
    """
    try:
        kg_service = get_kg_service()
        
        entity = MedicalEntity(
            text=request.text,
            entity_type=MedicalEntityType(request.entity_type),
            normalized_form=request.normalized_form or request.text,
            confidence=1.0
        )
        
        entity_uri = kg_service.add_entity_to_kg(
            entity=entity,
            entity_uri=None,
            link_to_uri=request.link_to_uri,
            link_type=request.link_type
        )
        
        kg_service.save_kg()
        
        message = f"Entity '{request.text}' added to Knowledge Graph"
        if request.link_to_uri and request.link_type:
            message += f" and linked to existing entity via {request.link_type}"
        
        return AddEntityToKGResponse(
            success=True,
            entity_uri=entity_uri,
            message=message
        )
        
    except Exception as e:
        logger.error(f"Error adding entity to KG: {str(e)}")
        return AddEntityToKGResponse(
            success=False,
            entity_uri=None,
            message=f"Error: {str(e)}"
        )


@router.get(
    "/kg/find-similar",
    summary="Find similar entities in KG",
    description="Find entities in the KG similar to the given text"
)
async def find_similar_entities(
    text: str,
    entity_type: str,
    min_score: float = 0.5,
    db: AsyncSession = Depends(get_db)
):
    """
    Find entities in the Knowledge Graph that are similar to the given text.
    Used to suggest linking options when adding a new entity.
    """
    try:
        kg_service = get_kg_service()
        matches = await kg_service.find_matches_async(text, min_score=min_score)
        
        return {
            "success": True,
            "matches": [
                {
                    "uri": str(match.kg_uri),
                    "label": match.kg_label,
                    "similarity_score": match.similarity_score,
                    "match_method": "text_similarity"
                }
                for match in matches
            ]
        }
        
    except Exception as e:
        logger.error(f"Error finding similar entities: {str(e)}")
        return {
            "success": False,
            "matches": [],
            "error": str(e)
        }
