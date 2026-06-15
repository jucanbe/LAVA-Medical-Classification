import logging
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from database import get_db, db_manager
from database.models import PendingRelationDB, EntityStatus
from models.relations import (
    RelationClassificationRequest,
    RelationClassificationResponse,
    RelationClassificationResult,
    BERTRelationClassificationRequest,
    BERTRelationClassificationResponse,
    MedicalRelationType,
    MedicalRelation,
    RELATION_DESCRIPTIONS,
)
from services import LLMClient
from services.relation_classifier import MedicalRelationClassifier
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/relations", tags=["Relation Classification"])



async def _save_relations_pending(
    relations: List[MedicalRelation],
    source: str,
    source_model: str,
    context: str,
    db: AsyncSession,
) -> int:
    """Save extracted relations to pending_relations, skipping duplicates."""
    saved = 0
    for rel in relations:
        src = rel.source_entity.strip()
        tgt = rel.target_entity.strip()
        rtype = rel.relation_type.strip() if isinstance(rel.relation_type, str) else rel.relation_type.value
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
                source_type=rel.source_type if isinstance(rel.source_type, str) else rel.source_type.value,
                relation_type=rtype,
                target_entity=tgt,
                target_type=rel.target_type if isinstance(rel.target_type, str) else rel.target_type.value,
                confidence=rel.confidence,
                context=(rel.context or context or "")[:500],
                source=source,
                source_model=source_model,
                status=EntityStatus.PENDING.value,
            )
        )
        saved += 1

    if saved:
        await db.commit()
    return saved


@router.get(
    "/types",
    summary="Get supported relation types",
    description="Returns all supported medical relation types with descriptions"
)
async def get_relation_types():
    """Get the list of supported relation types."""
    return [
        {"type": rt.value, "description": RELATION_DESCRIPTIONS.get(rt.value, "")}
        for rt in MedicalRelationType
    ]


@router.post(
    "/classify",
    response_model=RelationClassificationResponse,
    summary="Classify medical relations using LLM",
    description="Extract medical relations from text using an LLM"
)
async def classify_relations(
    request: RelationClassificationRequest,
    db: AsyncSession = Depends(get_db)
):
    """Classify medical relations in text using LLM."""
    try:
        config = None
        if request.config_id:
            config = await db_manager.get_llm_config_by_id(db, request.config_id)
            if not config:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"LLM config with id {request.config_id} not found"
                )
        else:
            config = await db_manager.get_default_llm_config(db)
            if not config:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No default LLM configuration found. Please create one first."
                )
        
        llm_client = LLMClient(
            base_url=config.base_url,
            model_name=config.model_name,
            api_key=config.api_key or "EMPTY",
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        
        custom_relation_prompt = getattr(config, 'relation_prompt', None)
        classifier = MedicalRelationClassifier(llm_client=llm_client, custom_system_prompt=custom_relation_prompt)
        
        result = await classifier.classify(
            text=request.text,
            min_confidence=request.min_confidence
        )
        
        relations_saved = 0
        try:
            relations_saved = await _save_relations_pending(
                relations=result.relations,
                source="llm",
                source_model=config.model_name,
                context=request.text[:500],
                db=db
            )
        except Exception as pe:
            logger.warning(f"Error auto-saving pending relations: {pe}")
        
        return RelationClassificationResponse(
            success=True,
            result=result,
            model_used=config.model_name,
            relations_saved=relations_saved
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error classifying relations: {e}")
        return RelationClassificationResponse(
            success=False,
            error=str(e)
        )


@router.post(
    "/classify/bert",
    response_model=BERTRelationClassificationResponse,
    summary="Classify medical relations using BERT",
    description="Extract medical relations from text using BERT (via LLM fallback)"
)
async def classify_relations_bert(
    request: BERTRelationClassificationRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Classify medical relations using BERT.
    Note: BERT NER models are designed for entity extraction. 
    For relation extraction, we use LLM as the backend but present it through the BERT interface.
    """
    import time
    start_time = time.time()
    
    try:
        config = await db_manager.get_default_llm_config(db)
        if not config:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No default LLM configuration found. Relation classification requires an LLM."
            )
        
        llm_client = LLMClient(
            base_url=config.base_url,
            model_name=config.model_name,
            api_key=config.api_key or "EMPTY",
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        
        custom_relation_prompt = getattr(config, 'relation_prompt', None)
        classifier = MedicalRelationClassifier(llm_client=llm_client, custom_system_prompt=custom_relation_prompt)
        result = await classifier.classify(text=request.text)
        
        inference_time = (time.time() - start_time) * 1000
        
        relations_saved = 0
        try:
            relations_saved = await _save_relations_pending(
                relations=result.relations,
                source="bert",
                source_model=config.model_name,
                context=request.text[:500],
                db=db
            )
        except Exception as pe:
            logger.warning(f"Error auto-saving pending relations: {pe}")
        
        return BERTRelationClassificationResponse(
            success=True,
            result=result,
            model_used=config.model_name,
            inference_time_ms=inference_time,
            relations_saved=relations_saved
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error classifying relations with BERT: {e}")
        return BERTRelationClassificationResponse(
            success=False,
            error=str(e)
        )


@router.post(
    "/save",
    summary="Save classified relations as pending",
    description="Save extracted relations to the pending relations table"
)
async def save_relations(
    relations: list,
    source: str = "llm",
    source_model: str = None,
    context: str = None,
    db: AsyncSession = Depends(get_db)
):
    """Save classified relations as pending relations."""
    try:
        saved_count = 0
        for rel in relations:
            pending = PendingRelationDB(
                source_entity=rel.get("source_entity"),
                source_type=rel.get("source_type"),
                relation_type=rel.get("relation_type"),
                target_entity=rel.get("target_entity"),
                target_type=rel.get("target_type"),
                confidence=rel.get("confidence"),
                context=context or rel.get("context"),
                source=source,
                source_model=source_model,
            )
            db.add(pending)
            saved_count += 1
        
        await db.commit()
        
        return {
            "success": True,
            "saved_count": saved_count,
            "message": f"Saved {saved_count} relations"
        }
    except Exception as e:
        await db.rollback()
        logger.error(f"Error saving relations: {e}")
        raise HTTPException(status_code=500, detail=str(e))
