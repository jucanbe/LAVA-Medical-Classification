import json
import logging
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete, asc, desc

from database.connection import get_db
from database.models import EntityReviewDB, PendingEntityDB, ReviewStatus
from models.entities import (
    EntityReviewCreate,
    EntityReviewResponse,
    EntityReviewListResponse,
    EntityReviewStatsResponse,
    BulkReviewRequest,
    CongruenceMetrics,
    CoverageMetrics,
    ConstraintMetrics,
    CompletenessMetrics,
    ConsistencyMetrics,
)
from services.entity_reviewer import EntityReviewService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/entity-reviews", tags=["Entity Reviews"])

_kg_service = None
_bert_service = None
_review_service = None


def get_kg_service():
    """Get or create the Knowledge Graph service instance."""
    global _kg_service
    if _kg_service is None:
        try:
            from services.knowledge_graph import KnowledgeGraphService
            _kg_service = KnowledgeGraphService()
            _kg_service.load_ttl_files()
        except Exception as e:
            logger.warning(f"Could not initialize KG service: {e}")
            _kg_service = None
    return _kg_service


def get_bert_service():
    """Get or create the BERT NER service instance."""
    global _bert_service
    if _bert_service is None:
        try:
            from services.bert_ner import BERTNERService
            _bert_service = BERTNERService()
        except Exception as e:
            logger.warning(f"Could not initialize BERT service: {e}")
            _bert_service = None
    return _bert_service


def get_review_service():
    """Get or create the Entity Review service instance."""
    global _review_service
    if _review_service is None:
        _review_service = EntityReviewService(
            kg_service=get_kg_service(),
            bert_service=get_bert_service()
        )
    return _review_service


def _convert_to_response(review: EntityReviewDB) -> EntityReviewResponse:
    """Convert database model to response model."""
    congruence = None
    if review.congruence_score is not None:
        congruence = CongruenceMetrics(
            score=review.congruence_score,
            nearest_entity=review.congruence_nearest_entity,
            nearest_uri=review.congruence_nearest_uri,
            embedding_distance=review.congruence_embedding_distance,
            method="stored"
        )
    
    coverage = None
    if review.coverage_score is not None:
        coverage = CoverageMetrics(
            score=review.coverage_score,
            similar_entities_count=review.coverage_similar_entities_count or 0,
            is_novel=review.coverage_is_novel or False,
            fills_gap=review.coverage_is_novel or False
        )
    
    constraint = None
    if review.constraint_score is not None:
        violations = []
        if review.constraint_violations:
            try:
                violations = json.loads(review.constraint_violations)
            except:
                violations = []
        constraint = ConstraintMetrics(
            score=review.constraint_score,
            violations=violations,
            ontology_valid=review.constraint_ontology_valid,
            type_valid=review.constraint_type_valid or True
        )
    
    completeness = None
    if review.completeness_score is not None:
        completeness = CompletenessMetrics(
            score=review.completeness_score,
            has_type=review.completeness_has_type or False,
            has_definition=review.completeness_has_definition or False,
            has_normalized_form=review.completeness_has_normalized_form or False,
            has_context=review.completeness_has_context or False,
            has_confidence=True
        )
    
    consistency = None
    if review.consistency_score is not None:
        alternate_types = []
        if review.consistency_alternate_types:
            try:
                alternate_types = json.loads(review.consistency_alternate_types)
            except:
                alternate_types = []
        consistency = ConsistencyMetrics(
            score=review.consistency_score,
            type_confidence=review.consistency_type_confidence or 0.0,
            alternate_types=alternate_types,
            bert_agreement=review.consistency_bert_agreement
        )
    
    return EntityReviewResponse(
        id=review.id,
        pending_entity_id=review.pending_entity_id,
        entity_text=review.entity_text,
        entity_type=review.entity_type,
        congruence=congruence,
        coverage=coverage,
        constraint=constraint,
        completeness=completeness,
        consistency=consistency,
        overall_score=review.overall_score or 0.0,
        review_status=review.review_status or "pending",
        recommendation=review.recommendation,
        review_notes=review.review_notes,
        created_at=review.created_at.isoformat() if review.created_at else "",
        updated_at=review.updated_at.isoformat() if review.updated_at else ""
    )


@router.post("/", response_model=EntityReviewResponse)
async def create_entity_review(
    request: EntityReviewCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new entity review by evaluating an entity using the 5 Cs scorecard.
    
    This endpoint evaluates:
    1. **Congruence** - Semantic alignment with existing KG entities
    2. **Coverage** - Whether the entity fills a gap in the KG
    3. **Constraint** - Clinical validity and constraint adherence
    4. **Completeness** - Presence of all required attributes
    5. **Consistency** - Type coherence and cross-validation agreement
    """
    review_service = get_review_service()
    
    existing_result = await db.execute(
        select(EntityReviewDB).where(
            EntityReviewDB.entity_text == request.entity_text,
            EntityReviewDB.entity_type == request.entity_type
        )
    )
    existing = existing_result.scalar_one_or_none()
    if existing:
        return _convert_to_response(existing)
    
    context = None
    confidence = None
    normalized_form = None
    source = None
    
    if request.pending_entity_id:
        result = await db.execute(
            select(PendingEntityDB).where(PendingEntityDB.id == request.pending_entity_id)
        )
        pending = result.scalar_one_or_none()
        if pending:
            context = pending.context
            confidence = pending.confidence
            normalized_form = pending.normalized_text
            source = pending.source
    
    evaluation = await review_service.evaluate_entity(
        entity_text=request.entity_text,
        entity_type=request.entity_type,
        normalized_form=normalized_form,
        context=context,
        confidence=confidence,
        source=source,
        run_bert_validation=request.run_bert_validation,
        run_llm_validation=request.run_llm_validation
    )
    
    congruence = evaluation["congruence"]
    coverage = evaluation["coverage"]
    constraint = evaluation["constraint"]
    completeness = evaluation["completeness"]
    consistency = evaluation["consistency"]
    
    review_db = EntityReviewDB(
        pending_entity_id=request.pending_entity_id,
        entity_text=request.entity_text,
        entity_type=request.entity_type,
        
        congruence_score=congruence.score,
        congruence_nearest_entity=congruence.nearest_entity,
        congruence_nearest_uri=congruence.nearest_uri,
        congruence_embedding_distance=congruence.embedding_distance,
        
        coverage_score=coverage.score,
        coverage_similar_entities_count=coverage.similar_entities_count,
        coverage_is_novel=coverage.is_novel,
        
        constraint_score=constraint.score,
        constraint_violations=json.dumps(constraint.violations),
        constraint_ontology_valid=constraint.ontology_valid,
        constraint_type_valid=constraint.type_valid,
        
        completeness_score=completeness.score,
        completeness_has_type=completeness.has_type,
        completeness_has_definition=completeness.has_definition,
        completeness_has_normalized_form=completeness.has_normalized_form,
        completeness_has_context=completeness.has_context,
        
        consistency_score=consistency.score,
        consistency_type_confidence=consistency.type_confidence,
        consistency_alternate_types=json.dumps(consistency.alternate_types),
        consistency_bert_agreement=consistency.bert_agreement,
        
        overall_score=evaluation["overall_score"],
        review_status=evaluation["review_status"],
        recommendation=evaluation["recommendation"],
        reviewed_by="auto"
    )
    
    db.add(review_db)
    await db.commit()
    await db.refresh(review_db)
    
    return _convert_to_response(review_db)


@router.get("/", response_model=EntityReviewListResponse)
async def list_entity_reviews(
    db: AsyncSession = Depends(get_db),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page"),
    status: Optional[str] = Query(None, description="Filter by review status"),
    recommendation: Optional[str] = Query(None, description="Filter by recommendation"),
    min_score: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum overall score"),
    max_score: Optional[float] = Query(None, ge=0.0, le=1.0, description="Maximum overall score"),
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    search: Optional[str] = Query(None, description="Search in entity text"),
    sort_by: Optional[str] = Query(None, description="Sort column"),
    sort_dir: Optional[str] = Query("asc", description="Sort direction (asc/desc)"),
):
    """
    List entity reviews with filtering and pagination.
    """
    query = select(EntityReviewDB)
    count_query = select(func.count(EntityReviewDB.id))
    
    if status:
        query = query.where(EntityReviewDB.review_status == status)
        count_query = count_query.where(EntityReviewDB.review_status == status)
    
    if recommendation:
        query = query.where(EntityReviewDB.recommendation == recommendation)
        count_query = count_query.where(EntityReviewDB.recommendation == recommendation)
    
    if min_score is not None:
        query = query.where(EntityReviewDB.overall_score >= min_score)
        count_query = count_query.where(EntityReviewDB.overall_score >= min_score)
    
    if max_score is not None:
        query = query.where(EntityReviewDB.overall_score <= max_score)
        count_query = count_query.where(EntityReviewDB.overall_score <= max_score)
    
    if entity_type:
        query = query.where(EntityReviewDB.entity_type == entity_type)
        count_query = count_query.where(EntityReviewDB.entity_type == entity_type)
    
    if search:
        query = query.where(EntityReviewDB.entity_text.ilike(f"%{search}%"))
        count_query = count_query.where(EntityReviewDB.entity_text.ilike(f"%{search}%"))
    
    total_result = await db.execute(count_query)
    total = total_result.scalar()
    
    offset = (page - 1) * page_size
    sortable_columns = {
        "entity_text": EntityReviewDB.entity_text,
        "entity_type": EntityReviewDB.entity_type,
        "congruence": EntityReviewDB.congruence_score,
        "coverage": EntityReviewDB.coverage_score,
        "constraint": EntityReviewDB.constraint_score,
        "completeness": EntityReviewDB.completeness_score,
        "consistency": EntityReviewDB.consistency_score,
        "overall_score": EntityReviewDB.overall_score,
        "review_status": EntityReviewDB.review_status,
    }
    if sort_by and sort_by in sortable_columns:
        col = sortable_columns[sort_by]
        query = query.order_by(desc(col) if sort_dir == "desc" else asc(col))
    else:
        query = query.order_by(EntityReviewDB.created_at.desc())
    query = query.offset(offset).limit(page_size)
    
    result = await db.execute(query)
    reviews = result.scalars().all()
    
    return EntityReviewListResponse(
        success=True,
        reviews=[_convert_to_response(r) for r in reviews],
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/stats", response_model=EntityReviewStatsResponse)
async def get_review_stats(
    db: AsyncSession = Depends(get_db)
):
    """
    Get statistics about entity reviews.
    """
    total_result = await db.execute(select(func.count(EntityReviewDB.id)))
    total = total_result.scalar() or 0
    
    status_query = select(
        EntityReviewDB.review_status,
        func.count(EntityReviewDB.id)
    ).group_by(EntityReviewDB.review_status)
    status_result = await db.execute(status_query)
    by_status = {row[0]: row[1] for row in status_result.all()}
    
    rec_query = select(
        EntityReviewDB.recommendation,
        func.count(EntityReviewDB.id)
    ).where(EntityReviewDB.recommendation.isnot(None)).group_by(EntityReviewDB.recommendation)
    rec_result = await db.execute(rec_query)
    by_recommendation = {row[0]: row[1] for row in rec_result.all()}
    
    avg_query = select(
        func.avg(EntityReviewDB.congruence_score).label("congruence"),
        func.avg(EntityReviewDB.coverage_score).label("coverage"),
        func.avg(EntityReviewDB.constraint_score).label("constraint"),
        func.avg(EntityReviewDB.completeness_score).label("completeness"),
        func.avg(EntityReviewDB.consistency_score).label("consistency"),
        func.avg(EntityReviewDB.overall_score).label("overall")
    )
    avg_result = await db.execute(avg_query)
    avg_row = avg_result.one()
    
    average_scores = {
        "congruence": round(avg_row.congruence or 0, 3),
        "coverage": round(avg_row.coverage or 0, 3),
        "constraint": round(avg_row.constraint or 0, 3),
        "completeness": round(avg_row.completeness or 0, 3),
        "consistency": round(avg_row.consistency or 0, 3),
        "overall": round(avg_row.overall or 0, 3)
    }
    
    score_distribution = {
        "excellent": 0,
        "good": 0,
        "fair": 0,
        "poor": 0
    }
    
    dist_query = select(EntityReviewDB.overall_score)
    dist_result = await db.execute(dist_query)
    for (score,) in dist_result.all():
        if score is None:
            continue
        if score >= 0.8:
            score_distribution["excellent"] += 1
        elif score >= 0.6:
            score_distribution["good"] += 1
        elif score >= 0.4:
            score_distribution["fair"] += 1
        else:
            score_distribution["poor"] += 1
    
    return EntityReviewStatsResponse(
        total_reviews=total,
        by_status=by_status,
        by_recommendation=by_recommendation,
        average_scores=average_scores,
        score_distribution=score_distribution
    )


@router.get("/{review_id}", response_model=EntityReviewResponse)
async def get_entity_review(
    review_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific entity review by ID.
    """
    result = await db.execute(
        select(EntityReviewDB).where(EntityReviewDB.id == review_id)
    )
    review = result.scalar_one_or_none()
    
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    
    return _convert_to_response(review)


@router.put("/{review_id}")
async def update_entity_review(
    review_id: int,
    text: str = Query(None, description="New entity text"),
    entity_type: str = Query(None, description="New entity type"),
    normalized_text: str = Query(None, description="Normalized text"),
    db: AsyncSession = Depends(get_db)
):
    """
    Update the text and/or type of an entity review.
    Also updates the linked pending entity if exists.
    """
    result = await db.execute(
        select(EntityReviewDB).where(EntityReviewDB.id == review_id)
    )
    review = result.scalar_one_or_none()
    
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    
    if text and text != review.entity_text:
        existing_review = await db.execute(
            select(EntityReviewDB).where(
                EntityReviewDB.entity_text == text,
                EntityReviewDB.entity_type == (entity_type or review.entity_type),
                EntityReviewDB.id != review_id
            )
        )
        if existing_review.scalar_one_or_none():
            raise HTTPException(
                status_code=409,
                detail=f"An entity review with the same text and type already exists"
            )
        
        existing_pending = await db.execute(
            select(PendingEntityDB).where(
                PendingEntityDB.text == text,
                PendingEntityDB.entity_type == (entity_type or review.entity_type),
                PendingEntityDB.id != review.pending_entity_id
            )
        )
        if existing_pending.scalar_one_or_none():
            raise HTTPException(
                status_code=409,
                detail=f"A pending entity with the same text and type already exists"
            )
    
    if text:
        review.entity_text = text
    if entity_type:
        review.entity_type = entity_type
    
    review.updated_at = datetime.utcnow()
    
    if review.pending_entity_id:
        pending_result = await db.execute(
            select(PendingEntityDB).where(PendingEntityDB.id == review.pending_entity_id)
        )
        pending = pending_result.scalar_one_or_none()
        if pending:
            if text:
                pending.text = text
            if entity_type:
                pending.entity_type = entity_type
            if normalized_text is not None:
                pending.normalized_text = normalized_text if normalized_text else None
            pending.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(review)
    
    return {
        "success": True,
        "message": "Entity review updated successfully",
        "review": _convert_to_response(review)
    }


@router.delete("/{review_id}")
async def delete_entity_review(
    review_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete an entity review.
    """
    result = await db.execute(
        select(EntityReviewDB).where(EntityReviewDB.id == review_id)
    )
    review = result.scalar_one_or_none()
    
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    
    await db.delete(review)
    await db.commit()
    
    return {"success": True, "message": "Review deleted"}


@router.put("/{review_id}/status")
async def update_review_status(
    review_id: int,
    status: str = Query(..., description="New status: passed, failed, needs_review"),
    notes: Optional[str] = Query(None, description="Review notes"),
    db: AsyncSession = Depends(get_db)
):
    """
    Update the status of an entity review (manual override).
    """
    valid_statuses = [s.value for s in ReviewStatus]
    if status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Must be one of: {valid_statuses}"
        )
    
    result = await db.execute(
        select(EntityReviewDB).where(EntityReviewDB.id == review_id)
    )
    review = result.scalar_one_or_none()
    
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    
    review.review_status = status
    review.reviewed_by = "manual"
    if notes:
        review.review_notes = notes
    
    if status == ReviewStatus.PASSED.value:
        review.recommendation = "approve"
    elif status == ReviewStatus.FAILED.value:
        review.recommendation = "reject"
    
    await db.commit()
    await db.refresh(review)
    
    return _convert_to_response(review)


@router.post("/bulk-review")
async def bulk_review_entities(
    request: BulkReviewRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Review multiple pending entities at once.
    Skips entities that have already been reviewed.
    """
    review_service = get_review_service()
    results = []
    errors = []
    skipped = []
    
    for entity_id in request.entity_ids:
        try:
            existing_result = await db.execute(
                select(EntityReviewDB).where(EntityReviewDB.pending_entity_id == entity_id)
            )
            if existing_result.scalar_one_or_none():
                skipped.append({"id": entity_id, "reason": "Already reviewed"})
                continue

            entity_result = await db.execute(
                select(PendingEntityDB).where(PendingEntityDB.id == entity_id)
            )
            pending = entity_result.scalar_one_or_none()

            if not pending:
                errors.append({"id": entity_id, "error": "Entity not found"})
                continue

            text_existing = await db.execute(
                select(EntityReviewDB).where(
                    EntityReviewDB.entity_text == pending.text,
                    EntityReviewDB.entity_type == pending.entity_type
                )
            )
            if text_existing.scalar_one_or_none():
                skipped.append({"id": entity_id, "reason": "Duplicate entity text already reviewed"})
                continue

            evaluation = await review_service.evaluate_entity(
                entity_text=pending.text,
                entity_type=pending.entity_type,
                normalized_form=pending.normalized_text,
                context=pending.context,
                confidence=pending.confidence,
                source=pending.source,
                run_bert_validation=request.run_bert_validation,
                run_llm_validation=request.run_llm_validation
            )

            congruence = evaluation["congruence"]
            coverage = evaluation["coverage"]
            constraint = evaluation["constraint"]
            completeness = evaluation["completeness"]
            consistency = evaluation["consistency"]

            review_db = EntityReviewDB(
                pending_entity_id=entity_id,
                entity_text=pending.text,
                entity_type=pending.entity_type,
                congruence_score=congruence.score,
                congruence_nearest_entity=congruence.nearest_entity,
                congruence_nearest_uri=congruence.nearest_uri,
                coverage_score=coverage.score,
                coverage_similar_entities_count=coverage.similar_entities_count,
                coverage_is_novel=coverage.is_novel,
                constraint_score=constraint.score,
                constraint_violations=json.dumps(constraint.violations),
                constraint_type_valid=constraint.type_valid,
                completeness_score=completeness.score,
                completeness_has_type=completeness.has_type,
                completeness_has_context=completeness.has_context,
                consistency_score=consistency.score,
                consistency_bert_agreement=consistency.bert_agreement,
                overall_score=evaluation["overall_score"],
                review_status=evaluation["review_status"],
                recommendation=evaluation["recommendation"],
                reviewed_by="auto"
            )

            db.add(review_db)
            await db.commit()
            results.append({
                "entity_id": entity_id,
                "entity_text": pending.text,
                "overall_score": evaluation["overall_score"],
                "status": evaluation["review_status"],
                "recommendation": evaluation["recommendation"]
            })

        except Exception as e:
            logger.error(f"Error reviewing entity {entity_id}: {e}")
            await db.rollback()
            errors.append({"id": entity_id, "error": str(e)})

    return {
        "success": True,
        "reviewed": len(results),
        "skipped": len(skipped),
        "errors": len(errors),
        "results": results,
        "skipped_details": skipped,
        "error_details": errors
    }


@router.post("/re-evaluate-all")
async def re_evaluate_all_reviews(
    run_bert_validation: bool = True,
    run_llm_validation: bool = False,
    db: AsyncSession = Depends(get_db)
):
    """
    Re-evaluate all existing reviews with the current configuration.
    Updates scores and statuses based on the latest evaluation logic.
    """
    review_service = get_review_service()
    updated = []
    errors = []
    
    try:
        result = await db.execute(select(EntityReviewDB))
        reviews = result.scalars().all()
        
        if not reviews:
            return {
                "success": True,
                "updated": 0,
                "errors": 0,
                "message": "No reviews to re-evaluate"
            }
        
        for review in reviews:
            try:
                context = None
                confidence = None
                normalized_form = None
                source = None

                if review.pending_entity_id:
                    pending_result = await db.execute(
                        select(PendingEntityDB).where(PendingEntityDB.id == review.pending_entity_id)
                    )
                    pending = pending_result.scalar_one_or_none()
                    if pending:
                        context = pending.context
                        confidence = pending.confidence
                        normalized_form = pending.normalized_text
                        source = pending.source

                evaluation = await review_service.evaluate_entity(
                    entity_text=review.entity_text,
                    entity_type=review.entity_type,
                    normalized_form=normalized_form,
                    context=context,
                    confidence=confidence,
                    source=source,
                    run_bert_validation=run_bert_validation,
                    run_llm_validation=run_llm_validation
                )

                congruence = evaluation["congruence"]
                coverage = evaluation["coverage"]
                constraint = evaluation["constraint"]
                completeness = evaluation["completeness"]
                consistency = evaluation["consistency"]

                review.congruence_score = congruence.score
                review.congruence_nearest_entity = congruence.nearest_entity
                review.congruence_nearest_uri = congruence.nearest_uri
                review.coverage_score = coverage.score
                review.coverage_similar_entities_count = coverage.similar_entities_count
                review.coverage_is_novel = coverage.is_novel
                review.constraint_score = constraint.score
                review.constraint_violations = json.dumps(constraint.violations)
                review.constraint_type_valid = constraint.type_valid
                review.completeness_score = completeness.score
                review.completeness_has_type = completeness.has_type
                review.completeness_has_context = completeness.has_context
                review.consistency_score = consistency.score
                review.consistency_bert_agreement = consistency.bert_agreement
                review.overall_score = evaluation["overall_score"]
                review.review_status = evaluation["review_status"]
                review.recommendation = evaluation["recommendation"]
                review.updated_at = datetime.utcnow()

                await db.commit()
                updated.append({
                    "id": review.id,
                    "entity_text": review.entity_text,
                    "old_status": review.review_status,
                    "new_status": evaluation["review_status"],
                    "overall_score": evaluation["overall_score"]
                })

            except Exception as e:
                logger.error(f"Error re-evaluating review {review.id}: {e}")
                await db.rollback()
                errors.append({"id": review.id, "error": str(e)})

        return {
            "success": True,
            "updated": len(updated),
            "errors": len(errors),
            "results": updated,
            "error_details": errors
        }
        
    except Exception as e:
        logger.error(f"Error in re-evaluate-all: {e}")
        return {
            "success": False,
            "error": str(e),
            "updated": 0,
            "errors": 1
        }


@router.post("/{review_id}/re-evaluate")
async def re_evaluate_single_review(
    review_id: int,
    run_bert_validation: bool = True,
    run_llm_validation: bool = False,
    db: AsyncSession = Depends(get_db)
):
    """
    Re-evaluate a single existing review with the current configuration.
    Updates scores and status based on the latest evaluation logic.
    """
    review_service = get_review_service()
    
    try:
        result = await db.execute(
            select(EntityReviewDB).where(EntityReviewDB.id == review_id)
        )
        review = result.scalar_one_or_none()
        
        if not review:
            raise HTTPException(status_code=404, detail="Review not found")
        
        context = None
        confidence = None
        normalized_form = None
        source = None
        
        if review.pending_entity_id:
            pending_result = await db.execute(
                select(PendingEntityDB).where(PendingEntityDB.id == review.pending_entity_id)
            )
            pending = pending_result.scalar_one_or_none()
            if pending:
                context = pending.context
                confidence = pending.confidence
                normalized_form = pending.normalized_text
                source = pending.source
        
        evaluation = await review_service.evaluate_entity(
            entity_text=review.entity_text,
            entity_type=review.entity_type,
            normalized_form=normalized_form,
            context=context,
            confidence=confidence,
            source=source,
            run_bert_validation=run_bert_validation,
            run_llm_validation=run_llm_validation
        )
        
        congruence = evaluation["congruence"]
        coverage = evaluation["coverage"]
        constraint = evaluation["constraint"]
        completeness = evaluation["completeness"]
        consistency = evaluation["consistency"]
        
        old_status = review.review_status
        
        review.congruence_score = congruence.score
        review.congruence_nearest_entity = congruence.nearest_entity
        review.congruence_nearest_uri = congruence.nearest_uri
        review.coverage_score = coverage.score
        review.coverage_similar_entities_count = coverage.similar_entities_count
        review.coverage_is_novel = coverage.is_novel
        review.constraint_score = constraint.score
        review.constraint_violations = json.dumps(constraint.violations)
        review.constraint_type_valid = constraint.type_valid
        review.completeness_score = completeness.score
        review.completeness_has_type = completeness.has_type
        review.completeness_has_context = completeness.has_context
        review.consistency_score = consistency.score
        review.consistency_bert_agreement = consistency.bert_agreement
        review.overall_score = evaluation["overall_score"]
        review.review_status = evaluation["review_status"]
        review.recommendation = evaluation["recommendation"]
        review.updated_at = datetime.utcnow()
        
        await db.commit()
        await db.refresh(review)
        
        return {
            "success": True,
            "id": review.id,
            "entity_text": review.entity_text,
            "old_status": old_status,
            "new_status": evaluation["review_status"],
            "overall_score": evaluation["overall_score"],
            "review": _convert_to_response(review)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error re-evaluating review {review_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pending-entity/{pending_entity_id}", response_model=Optional[EntityReviewResponse])
async def get_review_for_pending_entity(
    pending_entity_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get the review for a specific pending entity.
    """
    result = await db.execute(
        select(EntityReviewDB).where(
            EntityReviewDB.pending_entity_id == pending_entity_id
        ).order_by(EntityReviewDB.created_at.desc())
    )
    review = result.scalars().first()
    
    if not review:
        return None
    
    return _convert_to_response(review)


@router.post("/evaluate-text")
async def evaluate_text_entity(
    entity_text: str = Query(..., description="Entity text to evaluate"),
    entity_type: str = Query(..., description="Entity type"),
    run_bert: bool = Query(True, description="Run BERT cross-validation"),
    db: AsyncSession = Depends(get_db)
):
    """
    Evaluate an entity without saving to database (preview mode).
    
    Useful for testing the evaluation before creating a formal review.
    """
    review_service = get_review_service()
    
    evaluation = await review_service.evaluate_entity(
        entity_text=entity_text,
        entity_type=entity_type,
        run_bert_validation=run_bert
    )
    
    explanation = review_service.get_score_explanation(evaluation)
    
    return {
        "success": True,
        "entity_text": entity_text,
        "entity_type": entity_type,
        "evaluation": {
            "congruence": evaluation["congruence"].model_dump(),
            "coverage": evaluation["coverage"].model_dump(),
            "constraint": evaluation["constraint"].model_dump(),
            "completeness": evaluation["completeness"].model_dump(),
            "consistency": evaluation["consistency"].model_dump(),
            "overall_score": evaluation["overall_score"],
            "review_status": evaluation["review_status"],
            "recommendation": evaluation["recommendation"]
        },
        "explanation": explanation
    }


@router.post("/{review_id}/add-to-kg")
async def add_review_to_kg(
    review_id: int,
    link_to_uri: Optional[str] = Query(None, description="URI to link this entity to"),
    link_type: Optional[str] = Query("closeMatch", description="Type of link"),
    db: AsyncSession = Depends(get_db)
):
    """
    Add a reviewed entity to the Knowledge Graph and remove from both
    entity_reviews and pending_entities tables.
    """
    try:
        result = await db.execute(
            select(EntityReviewDB).where(EntityReviewDB.id == review_id)
        )
        review = result.scalar_one_or_none()
        
        if not review:
            raise HTTPException(status_code=404, detail="Review not found")
        
        kg_service = get_kg_service()
        
        from models.entities import MedicalEntity, MedicalEntityType
        
        type_mapping = {
            "finding": MedicalEntityType.FINDING,
            "disease": MedicalEntityType.DISEASE,
            "quantitative_measure": MedicalEntityType.QUANTITATIVE_MEASURE,
            "substance": MedicalEntityType.SUBSTANCE,
            "procedure": MedicalEntityType.PROCEDURE
        }
        entity_type_enum = type_mapping.get(review.entity_type, MedicalEntityType.FINDING)
        
        medical_entity = MedicalEntity(
            text=review.entity_text,
            entity_type=entity_type_enum,
            normalized_form=review.entity_text,
            confidence=review.overall_score or 0.9
        )
        
        entity_uri = kg_service.add_entity_to_kg(
            entity=medical_entity,
            link_to_uri=link_to_uri,
            link_type=link_type
        )
        
        if entity_uri:
            if review.pending_entity_id:
                pending_result = await db.execute(
                    select(PendingEntityDB).where(PendingEntityDB.id == review.pending_entity_id)
                )
                pending = pending_result.scalar_one_or_none()
                if pending:
                    await db.delete(pending)
                    logger.info(f"Deleted pending entity {review.pending_entity_id}")
            
            await db.delete(review)
            logger.info(f"Deleted review {review_id}")
            
            await db.commit()
            
            return {
                "success": True,
                "message": "Entity added to Knowledge Graph and removed from review/pending lists",
                "entity_uri": str(entity_uri)
            }
        else:
            return {
                "success": False,
                "message": "Failed to add entity to Knowledge Graph"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error adding review to KG: {e}")
        raise HTTPException(status_code=500, detail=str(e))
