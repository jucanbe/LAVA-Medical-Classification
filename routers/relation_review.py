import logging
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, asc, desc

from database.connection import get_db
from database.models import RelationReviewDB, PendingRelationDB, ReviewStatus
from models.relations import (
    RelationReviewCreate,
    RelationReviewResponse,
    RelationReviewListResponse,
)
from services.relation_reviewer import RelationReviewService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/relation-reviews", tags=["Relation Reviews"])

_review_service = None


def get_review_service() -> RelationReviewService:
    """Get or create the review service singleton."""
    global _review_service
    if _review_service is None:
        kg_service = None
        try:
            from services.kg_service import kg_service_instance
            kg_service = kg_service_instance
        except Exception:
            pass
        _review_service = RelationReviewService(kg_service=kg_service)
    return _review_service


def _convert_to_response(review: RelationReviewDB) -> RelationReviewResponse:
    """Convert DB model to response model."""
    return RelationReviewResponse(
        id=review.id,
        pending_relation_id=review.pending_relation_id,
        source_entity=review.source_entity,
        source_type=review.source_type,
        relation_type=review.relation_type,
        target_entity=review.target_entity,
        target_type=review.target_type,
        congruence_score=review.congruence_score,
        coverage_score=review.coverage_score,
        constraint_score=review.constraint_score,
        completeness_score=review.completeness_score,
        consistency_score=review.consistency_score,
        overall_score=review.overall_score or 0.0,
        review_status=review.review_status,
        recommendation=review.recommendation,
        review_notes=review.review_notes,
        created_at=review.created_at.isoformat() if review.created_at else "",
        updated_at=review.updated_at.isoformat() if review.updated_at else "",
    )


async def _score_and_build_review(
    service: RelationReviewService,
    source_entity: str,
    source_type: str,
    relation_type: str,
    target_entity: str,
    target_type: str,
    confidence: Optional[float] = None,
    context: Optional[str] = None,
    source: Optional[str] = None,
    pending_relation_id: Optional[int] = None,
) -> RelationReviewDB:
    """Score a relation using the service and build a DB row."""
    scores = await service.evaluate_relation(
        source_entity=source_entity,
        source_type=source_type,
        relation_type=relation_type,
        target_entity=target_entity,
        target_type=target_type,
        confidence=confidence,
        context=context,
        source=source,
    )
    return RelationReviewDB(
        pending_relation_id=pending_relation_id,
        source_entity=source_entity,
        source_type=source_type,
        relation_type=relation_type,
        target_entity=target_entity,
        target_type=target_type,
        congruence_score=scores["congruence_score"],
        coverage_score=scores["coverage_score"],
        constraint_score=scores["constraint_score"],
        completeness_score=scores["completeness_score"],
        consistency_score=scores["consistency_score"],
        overall_score=scores["overall_score"],
        review_status=scores["review_status"],
        recommendation=scores["recommendation"],
    )


@router.get(
    "/",
    response_model=RelationReviewListResponse,
    summary="List relation reviews",
)
async def list_reviews(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    review_status: Optional[str] = Query(None),
    relation_type: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    min_score: Optional[float] = Query(None, ge=0, le=1),
    sort_by: Optional[str] = Query(None, description="Sort column"),
    sort_dir: Optional[str] = Query("asc", description="Sort direction (asc/desc)"),
    db: AsyncSession = Depends(get_db),
):
    """List all relation reviews with pagination."""
    try:
        query = select(RelationReviewDB)
        count_query = select(func.count(RelationReviewDB.id))

        if review_status:
            query = query.where(RelationReviewDB.review_status == review_status)
            count_query = count_query.where(RelationReviewDB.review_status == review_status)

        if relation_type:
            query = query.where(RelationReviewDB.relation_type == relation_type)
            count_query = count_query.where(RelationReviewDB.relation_type == relation_type)

        if min_score is not None:
            query = query.where(RelationReviewDB.overall_score >= min_score)
            count_query = count_query.where(RelationReviewDB.overall_score >= min_score)

        if search:
            from sqlalchemy import or_
            search_filter = or_(
                RelationReviewDB.source_entity.ilike(f"%{search}%"),
                RelationReviewDB.target_entity.ilike(f"%{search}%"),
            )
            query = query.where(search_filter)
            count_query = count_query.where(search_filter)

        total = (await db.execute(count_query)).scalar() or 0

        offset = (page - 1) * page_size
        sortable_columns = {
            "source_entity": RelationReviewDB.source_entity,
            "congruence_score": RelationReviewDB.congruence_score,
            "coverage_score": RelationReviewDB.coverage_score,
            "constraint_score": RelationReviewDB.constraint_score,
            "completeness_score": RelationReviewDB.completeness_score,
            "consistency_score": RelationReviewDB.consistency_score,
            "overall_score": RelationReviewDB.overall_score,
            "review_status": RelationReviewDB.review_status,
        }
        if sort_by and sort_by in sortable_columns:
            col = sortable_columns[sort_by]
            query = query.order_by(desc(col) if sort_dir == "desc" else asc(col))
        else:
            query = query.order_by(RelationReviewDB.created_at.desc())
        query = query.offset(offset).limit(page_size)
        result = await db.execute(query)
        reviews = result.scalars().all()

        return RelationReviewListResponse(
            success=True,
            reviews=[_convert_to_response(r) for r in reviews],
            total=total,
        )
    except Exception as e:
        logger.error(f"Error listing relation reviews: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/",
    response_model=RelationReviewResponse,
    summary="Create a relation review",
)
async def create_review(
    request: RelationReviewCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new relation review using the 5 Cs scorecard."""
    try:
        dup_query = select(RelationReviewDB).where(
            RelationReviewDB.source_entity == request.source_entity,
            RelationReviewDB.relation_type == request.relation_type,
            RelationReviewDB.target_entity == request.target_entity,
        )
        dup_result = await db.execute(dup_query)
        existing = dup_result.scalar_one_or_none()
        if existing:
            resp = _convert_to_response(existing)
            return JSONResponse(content=resp.dict(), status_code=200, headers={"X-Already-Existed": "true"})

        confidence = None
        context = None
        source = None
        if request.pending_relation_id:
            pr_result = await db.execute(
                select(PendingRelationDB).where(PendingRelationDB.id == request.pending_relation_id)
            )
            pending = pr_result.scalar_one_or_none()
            if pending:
                confidence = pending.confidence
                context = pending.context
                source = pending.source

        service = get_review_service()
        review = await _score_and_build_review(
            service,
            source_entity=request.source_entity,
            source_type=request.source_type,
            relation_type=request.relation_type,
            target_entity=request.target_entity,
            target_type=request.target_type,
            confidence=confidence,
            context=context,
            source=source,
            pending_relation_id=request.pending_relation_id,
        )

        db.add(review)
        await db.commit()
        await db.refresh(review)

        return _convert_to_response(review)
    except Exception as e:
        await db.rollback()
        logger.error(f"Error creating relation review: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put(
    "/{review_id}",
    response_model=RelationReviewResponse,
    summary="Update a relation review",
)
async def update_review(
    review_id: int,
    source_entity: Optional[str] = None,
    source_type: Optional[str] = None,
    relation_type: Optional[str] = None,
    target_entity: Optional[str] = None,
    target_type: Optional[str] = None,
    review_status: Optional[str] = None,
    review_notes: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Update a relation review."""
    try:
        query = select(RelationReviewDB).where(RelationReviewDB.id == review_id)
        result = await db.execute(query)
        review = result.scalar_one_or_none()

        if not review:
            raise HTTPException(status_code=404, detail="Review not found")

        if source_entity is not None:
            review.source_entity = source_entity
        if source_type is not None:
            review.source_type = source_type
        if relation_type is not None:
            review.relation_type = relation_type
        if target_entity is not None:
            review.target_entity = target_entity
        if target_type is not None:
            review.target_type = target_type
        if review_status is not None:
            review.review_status = review_status
        if review_notes is not None:
            review.review_notes = review_notes

        review.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(review)

        return _convert_to_response(review)
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{review_id}", summary="Delete a relation review")
async def delete_review(
    review_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Delete a relation review."""
    try:
        query = select(RelationReviewDB).where(RelationReviewDB.id == review_id)
        result = await db.execute(query)
        review = result.scalar_one_or_none()

        if not review:
            raise HTTPException(status_code=404, detail="Review not found")

        await db.delete(review)
        await db.commit()
        return {"success": True, "message": "Review deleted"}
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk-review", summary="Review all pending relations at once")
async def bulk_review_relations(db: AsyncSession = Depends(get_db)):
    """Review all pending relations that don't have a review yet."""
    try:
        pending_result = await db.execute(
            select(PendingRelationDB).where(PendingRelationDB.status == "pending")
        )
        pending_relations = pending_result.scalars().all()

        if not pending_relations:
            return {"success": True, "reviewed": 0, "skipped": 0, "errors": 0, "message": "No pending relations to review"}

        service = get_review_service()
        reviewed = 0
        skipped = 0
        errors = 0

        for rel in pending_relations:
            try:
                dup_query = select(RelationReviewDB).where(
                    RelationReviewDB.source_entity == rel.source_entity,
                    RelationReviewDB.relation_type == rel.relation_type,
                    RelationReviewDB.target_entity == rel.target_entity,
                )
                dup_result = await db.execute(dup_query)
                if dup_result.scalar_one_or_none():
                        skipped += 1
                        continue

                review = await _score_and_build_review(
                    service,
                    source_entity=rel.source_entity,
                    source_type=rel.source_type,
                    relation_type=rel.relation_type,
                    target_entity=rel.target_entity,
                    target_type=rel.target_type,
                    confidence=rel.confidence,
                    context=rel.context,
                    source=rel.source,
                    pending_relation_id=rel.id,
                )
                db.add(review)
                await db.commit()
                reviewed += 1

            except Exception as e:
                logger.error(f"Error reviewing relation {rel.id}: {e}")
                await db.rollback()
                errors += 1

        return {"success": True, "reviewed": reviewed, "skipped": skipped, "errors": errors}
    except Exception as e:
        await db.rollback()
        logger.error(f"Error in bulk review: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/re-evaluate-all", summary="Re-evaluate all existing relation reviews")
async def re_evaluate_all_reviews(db: AsyncSession = Depends(get_db)):
    """Re-evaluate all existing reviews with the current scoring logic and config."""
    try:
        result = await db.execute(select(RelationReviewDB))
        reviews = result.scalars().all()

        if not reviews:
            return {"success": True, "updated": 0, "errors": 0, "message": "No reviews to re-evaluate"}

        service = get_review_service()
        service._config_cache = None
        service._config_loaded_at = None

        updated = 0
        errors = 0

        for review in reviews:
            try:
                confidence = None
                context = None
                source = None
                if review.pending_relation_id:
                    pr_result = await db.execute(
                        select(PendingRelationDB).where(PendingRelationDB.id == review.pending_relation_id)
                    )
                    pending = pr_result.scalar_one_or_none()
                    if pending:
                        confidence = pending.confidence
                        context = pending.context
                        source = pending.source

                scores = await service.evaluate_relation(
                    source_entity=review.source_entity,
                    source_type=review.source_type,
                    relation_type=review.relation_type,
                    target_entity=review.target_entity,
                    target_type=review.target_type,
                    confidence=confidence,
                    context=context,
                    source=source,
                )

                review.congruence_score = scores["congruence_score"]
                review.coverage_score = scores["coverage_score"]
                review.constraint_score = scores["constraint_score"]
                review.completeness_score = scores["completeness_score"]
                review.consistency_score = scores["consistency_score"]
                review.overall_score = scores["overall_score"]
                review.review_status = scores["review_status"]
                review.recommendation = scores["recommendation"]
                review.updated_at = datetime.utcnow()
                await db.commit()
                updated += 1

            except Exception as e:
                logger.error(f"Error re-evaluating review {review.id}: {e}")
                await db.rollback()
                errors += 1

        return {"success": True, "updated": updated, "errors": errors}
    except Exception as e:
        await db.rollback()
        logger.error(f"Error in re-evaluate-all: {e}")
        return {"success": False, "error": str(e), "updated": 0, "errors": 1}


@router.post("/{review_id}/re-evaluate", summary="Re-evaluate a single relation review")
async def re_evaluate_single_review(
    review_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Re-evaluate a single review with the current scoring logic."""
    try:
        result = await db.execute(
            select(RelationReviewDB).where(RelationReviewDB.id == review_id)
        )
        review = result.scalar_one_or_none()
        if not review:
            raise HTTPException(status_code=404, detail="Review not found")

        confidence = None
        context = None
        source = None
        if review.pending_relation_id:
            pr_result = await db.execute(
                select(PendingRelationDB).where(PendingRelationDB.id == review.pending_relation_id)
            )
            pending = pr_result.scalar_one_or_none()
            if pending:
                confidence = pending.confidence
                context = pending.context
                source = pending.source

        service = get_review_service()
        scores = await service.evaluate_relation(
            source_entity=review.source_entity,
            source_type=review.source_type,
            relation_type=review.relation_type,
            target_entity=review.target_entity,
            target_type=review.target_type,
            confidence=confidence,
            context=context,
            source=source,
        )

        old_status = review.review_status
        review.congruence_score = scores["congruence_score"]
        review.coverage_score = scores["coverage_score"]
        review.constraint_score = scores["constraint_score"]
        review.completeness_score = scores["completeness_score"]
        review.consistency_score = scores["consistency_score"]
        review.overall_score = scores["overall_score"]
        review.review_status = scores["review_status"]
        review.recommendation = scores["recommendation"]
        review.updated_at = datetime.utcnow()

        await db.commit()
        await db.refresh(review)

        return {
            "success": True,
            "id": review.id,
            "old_status": old_status,
            "new_status": scores["review_status"],
            "overall_score": scores["overall_score"],
        }
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error re-evaluating review {review_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", summary="Get relation review statistics")
async def get_review_stats(db: AsyncSession = Depends(get_db)):
    """Get statistics about relation reviews including average per-C scores."""
    try:
        total = (await db.execute(select(func.count(RelationReviewDB.id)))).scalar() or 0
        passed = (await db.execute(
            select(func.count(RelationReviewDB.id)).where(RelationReviewDB.review_status == "passed")
        )).scalar() or 0
        failed = (await db.execute(
            select(func.count(RelationReviewDB.id)).where(RelationReviewDB.review_status == "failed")
        )).scalar() or 0
        needs_review = (await db.execute(
            select(func.count(RelationReviewDB.id)).where(RelationReviewDB.review_status == "needs_review")
        )).scalar() or 0
        pending = (await db.execute(
            select(func.count(RelationReviewDB.id)).where(RelationReviewDB.review_status == "pending")
        )).scalar() or 0

        avg_score = (await db.execute(
            select(func.avg(RelationReviewDB.overall_score))
        )).scalar() or 0.0

        avg_congruence = (await db.execute(
            select(func.avg(RelationReviewDB.congruence_score))
        )).scalar()
        avg_coverage = (await db.execute(
            select(func.avg(RelationReviewDB.coverage_score))
        )).scalar()
        avg_constraint = (await db.execute(
            select(func.avg(RelationReviewDB.constraint_score))
        )).scalar()
        avg_completeness = (await db.execute(
            select(func.avg(RelationReviewDB.completeness_score))
        )).scalar()
        avg_consistency = (await db.execute(
            select(func.avg(RelationReviewDB.consistency_score))
        )).scalar()

        return {
            "success": True,
            "stats": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "needs_review": needs_review,
                "pending": pending,
                "avg_score": round(float(avg_score), 3),
                "avg_congruence": round(float(avg_congruence), 3) if avg_congruence else None,
                "avg_coverage": round(float(avg_coverage), 3) if avg_coverage else None,
                "avg_constraint": round(float(avg_constraint), 3) if avg_constraint else None,
                "avg_completeness": round(float(avg_completeness), 3) if avg_completeness else None,
                "avg_consistency": round(float(avg_consistency), 3) if avg_consistency else None,
            },
        }
    except Exception as e:
        logger.error(f"Error getting relation review stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
