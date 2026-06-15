import io
import json
import logging
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from database import get_db, db_manager
from database.models import PendingEntityDB, PendingRelationDB, EntityStatus
from services.knowledge_graph import get_kg_service
from models.entities import MedicalEntity, MedicalEntityType
from services.llm_client import LLMClient
from services.classify_all import (
    get_classify_all_service,
    get_visualization_html,
    HAS_LANGEXTRACT,
    ENTITY_TYPES,
    RELATION_TYPES,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/classify-all", tags=["Classify All (LangExtract)"])



class ClassifyAllRequest(BaseModel):
    """Request body for text extraction."""
    text: str = Field(..., min_length=1, description="Text to extract entities/relations from")
    config_id: Optional[int] = Field(None, description="LLM configuration ID")
    use_random_kg: bool = Field(True, description="Include random KG entities as examples")
    use_known_samples: bool = Field(True, description="Include custom KG samples as examples")


class ExtractionEntity(BaseModel):
    text: str
    entity_type: str
    attributes: dict = {}


class ExtractionRelation(BaseModel):
    source_entity: str
    source_type: str
    relation_type: str
    target_entity: str
    target_type: str
    context: Optional[str] = None


class ClassifyAllResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    success: bool
    entities: List[dict] = []
    relations: List[dict] = []
    visualization_id: Optional[str] = None
    total_extractions: int = 0
    entities_saved: int = 0
    relations_saved: int = 0
    model_used: Optional[str] = None
    kg_validation: Optional[dict] = None
    error: Optional[str] = None



async def _get_llm_config(config_id: Optional[int], db: AsyncSession):
    """Resolve LLM configuration from DB (follows same pattern as classification.py)."""
    if config_id:
        config = await db_manager.get_llm_config_by_id(db, config_id)
        if not config:
            raise HTTPException(status_code=404, detail=f"LLM config {config_id} not found")
        return config
    config = await db_manager.get_default_llm_config(db)
    if not config:
        raise HTTPException(
            status_code=404,
            detail="No LLM configuration found. Create one in Configuration → LLM Servers.",
        )
    return config


def _parse_kg_samples(config) -> Optional[dict]:
    """Parse kg_samples JSON string from a DB config into a dict."""
    raw = getattr(config, "kg_samples", None)
    if not raw:
        return None
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return None


async def _validate_extracted_entities(entities: List[dict], kg_service, kg_backend: str = "all") -> Optional[dict]:
    """Run KG validation on extracted entities and return a serialisable dict.

    Converts raw dicts into MedicalEntity objects, runs async validation
    (local + external SPARQL store), and returns a dict matching the format
    used by Entity Classification pages.
    """
    if not entities or not kg_service:
        return None
    try:
        from models.entities import EntityClassificationResult

        medical_entities = []
        for ent in entities:
            etype = ent.get("entity_type", "finding")
            try:
                etype_val = MedicalEntityType(etype)
            except (ValueError, KeyError):
                etype_val = MedicalEntityType.FINDING
            medical_entities.append(MedicalEntity(
                text=ent.get("text", ""),
                entity_type=etype_val,
                confidence=ent.get("confidence", 0.8),
            ))

        ecr = EntityClassificationResult(entities=medical_entities, raw_text="")
        result = await kg_service.validate_entities_async(ecr, kg_backend=kg_backend)

        validated = []
        for ve in result.validated_entities:
            matches_list = []
            for m in (ve.kg_matches or []):
                matches_list.append({
                    "kg_uri": str(m.kg_uri) if m.kg_uri else None,
                    "kg_label": m.kg_label,
                    "kg_type": m.kg_type,
                    "similarity_score": m.similarity_score,
                })
            validated.append({
                "entity": {
                    "text": ve.entity.text,
                    "entity_type": ve.entity.entity_type.value if hasattr(ve.entity.entity_type, "value") else str(ve.entity.entity_type),
                },
                "validation_status": ve.validation_status,
                "kg_matches": matches_list,
                "is_validated": ve.is_validated,
                "validation_notes": ve.validation_notes,
            })

        return {
            "validated_entities": validated,
            "exact_matches": result.exact_matches,
            "similar_matches": result.similar_matches,
            "low_matches": result.low_matches,
            "not_found": result.not_found,
        }
    except Exception as e:
        logger.warning(f"KG validation for Classify All failed: {e}")
        return None


async def _save_entities(entities: List[dict], model_name: str, context: str, db: AsyncSession) -> int:
    """Save extracted entities to pending_entities, skipping duplicates."""
    saved = 0
    for ent in entities:
        text = ent.get("text", "").strip()
        etype = ent.get("entity_type", "").strip()
        if not text or not etype:
            continue
        existing = await db.execute(
            select(PendingEntityDB).where(
                PendingEntityDB.text == text,
                PendingEntityDB.entity_type == etype,
                PendingEntityDB.status == EntityStatus.PENDING.value,
            )
        )
        if existing.scalar_one_or_none():
            continue
        db.add(
            PendingEntityDB(
                text=text,
                entity_type=etype,
                validation_status="not_found",
                source="langextract",
                source_model=model_name,
                confidence=None,
                context=context[:500] if context else None,
                status=EntityStatus.PENDING.value,
            )
        )
        saved += 1
    if saved:
        await db.commit()
    return saved


async def _save_relations(relations: List[dict], model_name: str, context: str, db: AsyncSession) -> int:
    """Save extracted relations to pending_relations, skipping duplicates."""
    saved = 0
    for rel in relations:
        src = rel.get("source_entity", "").strip()
        tgt = rel.get("target_entity", "").strip()
        rtype = rel.get("relation_type", "").strip()
        if not src or not tgt or not rtype:
            continue
        existing = await db.execute(
            select(PendingRelationDB).where(
                PendingRelationDB.source_entity == src,
                PendingRelationDB.relation_type == rtype,
                PendingRelationDB.target_entity == tgt,
            )
        )
        if existing.scalar_one_or_none():
            continue
        db.add(
            PendingRelationDB(
                source_entity=src,
                source_type=rel.get("source_type", "unknown"),
                relation_type=rtype,
                target_entity=tgt,
                target_type=rel.get("target_type", "unknown"),
                confidence=rel.get("confidence"),
                context=(rel.get("context") or context or "")[:500],
                source="langextract",
                source_model=model_name,
                status=EntityStatus.PENDING.value,
            )
        )
        saved += 1
    if saved:
        await db.commit()
    return saved



@router.post(
    "/text",
    response_model=ClassifyAllResponse,
    summary="Extract entities & relations from text",
    description="Uses the configured LLM to extract medical entities and relations, "
                "generates an interactive visualization, and auto-imports "
                "results into pending entities/relations.",
)
async def classify_all_text(
    request: ClassifyAllRequest,
    kg_backend: str = Query("all", description="KG backend to query: all, internal, external"),
    db: AsyncSession = Depends(get_db),
):
    if not HAS_LANGEXTRACT:
        return ClassifyAllResponse(
            success=False,
            error="langextract is not installed. Run: pip install langextract",
        )

    try:
        config = await _get_llm_config(request.config_id, db)

        llm_client = LLMClient.from_config(config)

        svc = get_classify_all_service()
        kg_service = get_kg_service()

        result = await svc.extract(
            text=request.text,
            llm_client=llm_client,
            classify_all_prompt=getattr(config, "classify_all_prompt", None),
            entity_prompt=getattr(config, "entity_prompt", None),
            relation_prompt=getattr(config, "relation_prompt", None),
            kg_service=kg_service,
            kg_samples=_parse_kg_samples(config),
            use_random_kg=request.use_random_kg,
            use_known_samples=request.use_known_samples,
        )

        entities_saved = await _save_entities(
            result["entities"], config.model_name, request.text, db
        )
        relations_saved = await _save_relations(
            result["relations"], config.model_name, request.text, db
        )

        kg_validation_dict = await _validate_extracted_entities(
            result["entities"], kg_service, kg_backend=kg_backend
        )

        return ClassifyAllResponse(
            success=True,
            entities=result["entities"],
            relations=result["relations"],
            visualization_id=result.get("visualization_id"),
            total_extractions=result["total_extractions"],
            entities_saved=entities_saved,
            relations_saved=relations_saved,
            model_used=config.model_name,
            kg_validation=kg_validation_dict,
        )

    except Exception as e:
        logger.error(f"Classify All text error: {e}", exc_info=True)
        return ClassifyAllResponse(success=False, error=str(e))


@router.post(
    "/document",
    response_model=ClassifyAllResponse,
    summary="Extract entities & relations from a document",
    description="Upload a document (PDF, DOCX, TXT), extract text, run extraction, "
                "and auto-import results.",
)
async def classify_all_document(
    file: UploadFile = File(..., description="Document to analyze (PDF, DOCX, TXT)"),
    config_id: Optional[int] = Form(None),
    use_random_kg: bool = Form(True),
    use_known_samples: bool = Form(True),
    kg_backend: str = Form("all", description="KG backend to query: all, internal, external"),
    db: AsyncSession = Depends(get_db),
):
    if not HAS_LANGEXTRACT:
        return ClassifyAllResponse(
            success=False,
            error="langextract is not installed. Run: pip install langextract",
        )

    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        from services.document_processor import get_document_processor

        processor = get_document_processor()
        content = await file.read()
        file_obj = io.BytesIO(content)

        try:
            chunks, doc_info = processor.process_document(file_obj, file.filename)
        except (ImportError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))

        if not chunks:
            return ClassifyAllResponse(success=True, error="No text extracted from document")

        full_text = "\n".join(chunk.text for chunk in chunks)

        config = await _get_llm_config(config_id, db)
        llm_client = LLMClient.from_config(config)

        svc = get_classify_all_service()
        kg_service = get_kg_service()

        result = await svc.extract_chunked(
            chunks=chunks,
            full_text=full_text,
            llm_client=llm_client,
            classify_all_prompt=getattr(config, "classify_all_prompt", None),
            entity_prompt=getattr(config, "entity_prompt", None),
            relation_prompt=getattr(config, "relation_prompt", None),
            kg_service=kg_service,
            kg_samples=_parse_kg_samples(config),
            use_random_kg=use_random_kg,
            use_known_samples=use_known_samples,
        )

        context_label = f"[Document: {doc_info.filename}]"
        entities_saved = await _save_entities(
            result["entities"], config.model_name, context_label, db
        )
        relations_saved = await _save_relations(
            result["relations"], config.model_name, context_label, db
        )

        kg_validation_dict = await _validate_extracted_entities(
            result["entities"], kg_service, kg_backend=kg_backend
        )

        return ClassifyAllResponse(
            success=True,
            entities=result["entities"],
            relations=result["relations"],
            visualization_id=result.get("visualization_id"),
            total_extractions=result["total_extractions"],
            entities_saved=entities_saved,
            relations_saved=relations_saved,
            model_used=config.model_name,
            kg_validation=kg_validation_dict,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Classify All document error: {e}", exc_info=True)
        return ClassifyAllResponse(success=False, error=str(e))


@router.get(
    "/visualization/{viz_id}",
    response_class=HTMLResponse,
    summary="Retrieve langextract interactive visualization",
    description="Returns the self-contained HTML visualization generated by langextract.",
)
async def get_visualization(viz_id: str):
    html = get_visualization_html(viz_id)
    if not html:
        raise HTTPException(status_code=404, detail="Visualization not found or expired")
    return HTMLResponse(content=html)


@router.get(
    "/kg-examples",
    summary="Preview KG few-shot examples",
    description="Returns the random KG entities that would be used as few-shot examples.",
)
async def preview_kg_examples():
    try:
        kg_service = get_kg_service()
        if not HAS_LANGEXTRACT:
            return {"success": False, "error": "langextract not installed"}

        svc = get_classify_all_service()
        examples = svc.build_kg_examples(kg_service)

        result = []
        for ex in examples:
            result.append(
                {
                    "text": ex.text,
                    "extractions": [
                        {
                            "extraction_class": e.extraction_class,
                            "extraction_text": e.extraction_text,
                            "attributes": e.attributes,
                        }
                        for e in ex.extractions
                    ],
                }
            )

        return {"success": True, "examples": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get(
    "/extraction-types",
    summary="Get supported extraction types",
    description="Returns the entity types and relation types supported for extraction.",
)
async def get_extraction_types():
    return {
        "entity_types": ENTITY_TYPES,
        "relation_types": RELATION_TYPES,
    }
