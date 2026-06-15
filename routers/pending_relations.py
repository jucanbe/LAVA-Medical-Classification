import logging
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_, asc, desc

from database import get_db
from database.models import PendingRelationDB, EntityStatus, RelationReviewDB
from models.relations import (
    PendingRelationCreate,
    PendingRelationUpdate,
    PendingRelationResponse,
    PendingRelationsListResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pending-relations", tags=["Pending Relations"])


def _db_to_response(relation: PendingRelationDB, review: RelationReviewDB = None) -> PendingRelationResponse:
    """Convert database model to response model."""
    return PendingRelationResponse(
        id=relation.id,
        source_entity=relation.source_entity,
        source_type=relation.source_type,
        relation_type=relation.relation_type,
        target_entity=relation.target_entity,
        target_type=relation.target_type,
        confidence=relation.confidence,
        context=relation.context,
        source=relation.source,
        source_model=relation.source_model,
        status=relation.status,
        created_at=relation.created_at.isoformat() if relation.created_at else "",
        updated_at=relation.updated_at.isoformat() if relation.updated_at else "",
        review_status=review.review_status if review else None,
        review_score=review.overall_score if review else None
    )


@router.get(
    "/",
    response_model=PendingRelationsListResponse,
    summary="List pending relations",
    description="Get a paginated list of relations pending to be added to Knowledge Graph"
)
async def list_pending_relations(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Items per page"),
    status: Optional[str] = Query(None, description="Filter by status"),
    relation_type: Optional[str] = Query(None, description="Filter by relation type"),
    source: Optional[str] = Query(None, description="Filter by source (llm, bert)"),
    search: Optional[str] = Query(None, description="Search in entity texts"),
    has_review: Optional[str] = Query(None, description="Filter by review status (reviewed, not_reviewed)"),
    sort_by: Optional[str] = Query(None, description="Sort column"),
    sort_dir: Optional[str] = Query("asc", description="Sort direction (asc/desc)"),
    db: AsyncSession = Depends(get_db)
):
    """List all pending relations with pagination and filters."""
    try:
        if has_review:
            if has_review == "reviewed":
                query = select(PendingRelationDB).join(
                    RelationReviewDB,
                    PendingRelationDB.id == RelationReviewDB.pending_relation_id
                ).distinct()
                count_query = select(func.count(func.distinct(PendingRelationDB.id))).select_from(
                    PendingRelationDB
                ).join(
                    RelationReviewDB,
                    PendingRelationDB.id == RelationReviewDB.pending_relation_id
                )
            else:
                reviewed_ids = select(RelationReviewDB.pending_relation_id).where(
                    RelationReviewDB.pending_relation_id.isnot(None)
                )
                query = select(PendingRelationDB).where(
                    PendingRelationDB.id.notin_(reviewed_ids)
                )
                count_query = select(func.count(PendingRelationDB.id)).where(
                    PendingRelationDB.id.notin_(reviewed_ids)
                )
        else:
            query = select(PendingRelationDB)
            count_query = select(func.count(PendingRelationDB.id))
        
        if status:
            query = query.where(PendingRelationDB.status == status)
            count_query = count_query.where(PendingRelationDB.status == status)
        
        if relation_type:
            query = query.where(PendingRelationDB.relation_type == relation_type)
            count_query = count_query.where(PendingRelationDB.relation_type == relation_type)
        
        if source:
            query = query.where(PendingRelationDB.source == source)
            count_query = count_query.where(PendingRelationDB.source == source)
        
        if search:
            search_filter = or_(
                PendingRelationDB.source_entity.ilike(f"%{search}%"),
                PendingRelationDB.target_entity.ilike(f"%{search}%")
            )
            query = query.where(search_filter)
            count_query = count_query.where(search_filter)
        
        total_result = await db.execute(count_query)
        total = total_result.scalar()
        
        offset = (page - 1) * page_size
        sortable_columns = {
            "source_entity": PendingRelationDB.source_entity,
            "relation_type": PendingRelationDB.relation_type,
            "target_entity": PendingRelationDB.target_entity,
            "confidence": PendingRelationDB.confidence,
            "source": PendingRelationDB.source,
            "created_at": PendingRelationDB.created_at,
        }
        if sort_by and sort_by in sortable_columns:
            col = sortable_columns[sort_by]
            query = query.order_by(desc(col) if sort_dir == "desc" else asc(col))
        else:
            query = query.order_by(PendingRelationDB.created_at.desc())
        query = query.offset(offset).limit(page_size)
        result = await db.execute(query)
        relations = result.scalars().all()
        
        response_relations = []
        for rel in relations:
            review_query = select(RelationReviewDB).where(
                RelationReviewDB.pending_relation_id == rel.id
            )
            review_result = await db.execute(review_query)
            review = review_result.scalar_one_or_none()
            response_relations.append(_db_to_response(rel, review))
        
        return PendingRelationsListResponse(
            success=True,
            relations=response_relations,
            total=total,
            page=page,
            page_size=page_size
        )
    except Exception as e:
        logger.error(f"Error listing pending relations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/",
    response_model=PendingRelationResponse,
    summary="Create a pending relation"
)
async def create_pending_relation(
    relation: PendingRelationCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a new pending relation."""
    try:
        dup_query = select(PendingRelationDB).where(
            PendingRelationDB.source_entity == relation.source_entity,
            PendingRelationDB.relation_type == relation.relation_type,
            PendingRelationDB.target_entity == relation.target_entity
        )
        dup_result = await db.execute(dup_query)
        existing = dup_result.scalar_one_or_none()
        
        if existing:
            return _db_to_response(existing)
        
        db_relation = PendingRelationDB(
            source_entity=relation.source_entity,
            source_type=relation.source_type,
            relation_type=relation.relation_type,
            target_entity=relation.target_entity,
            target_type=relation.target_type,
            confidence=relation.confidence,
            context=relation.context,
            source=relation.source,
            source_model=relation.source_model,
        )
        db.add(db_relation)
        await db.commit()
        await db.refresh(db_relation)
        
        return _db_to_response(db_relation)
    except Exception as e:
        await db.rollback()
        logger.error(f"Error creating pending relation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put(
    "/{relation_id}",
    response_model=PendingRelationResponse,
    summary="Update a pending relation"
)
async def update_pending_relation(
    relation_id: int,
    update: PendingRelationUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update a pending relation."""
    try:
        query = select(PendingRelationDB).where(PendingRelationDB.id == relation_id)
        result = await db.execute(query)
        relation = result.scalar_one_or_none()
        
        if not relation:
            raise HTTPException(status_code=404, detail="Relation not found")
        
        update_data = update.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(relation, key, value)
        
        relation.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(relation)
        
        return _db_to_response(relation)
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error updating pending relation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/{relation_id}",
    summary="Delete a pending relation"
)
async def delete_pending_relation(
    relation_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Delete a pending relation."""
    try:
        query = select(PendingRelationDB).where(PendingRelationDB.id == relation_id)
        result = await db.execute(query)
        relation = result.scalar_one_or_none()
        
        if not relation:
            raise HTTPException(status_code=404, detail="Relation not found")
        
        await db.delete(relation)
        await db.commit()
        
        return {"success": True, "message": "Relation deleted"}
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/",
    summary="Bulk delete pending relations"
)
async def bulk_delete_relations(
    ids: list[int] = Query(..., description="List of relation IDs to delete"),
    db: AsyncSession = Depends(get_db)
):
    """Delete multiple pending relations."""
    try:
        deleted = 0
        for rid in ids:
            query = select(PendingRelationDB).where(PendingRelationDB.id == rid)
            result = await db.execute(query)
            relation = result.scalar_one_or_none()
            if relation:
                await db.delete(relation)
                deleted += 1
        
        await db.commit()
        return {"success": True, "deleted": deleted}
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/stats",
    summary="Get pending relations statistics"
)
async def get_stats(db: AsyncSession = Depends(get_db)):
    """Get statistics about pending relations."""
    try:
        total_q = select(func.count(PendingRelationDB.id))
        total = (await db.execute(total_q)).scalar() or 0
        
        pending_q = total_q.where(PendingRelationDB.status == "pending")
        pending = (await db.execute(pending_q)).scalar() or 0
        
        approved_q = select(func.count(PendingRelationDB.id)).where(PendingRelationDB.status == "approved")
        approved = (await db.execute(approved_q)).scalar() or 0
        
        rejected_q = select(func.count(PendingRelationDB.id)).where(PendingRelationDB.status == "rejected")
        rejected = (await db.execute(rejected_q)).scalar() or 0
        
        source_query = select(
            PendingRelationDB.source,
            func.count(PendingRelationDB.id)
        ).group_by(PendingRelationDB.source)
        source_result = await db.execute(source_query)
        by_source = dict(source_result.all())
        
        return {
            "total": total,
            "pending": pending,
            "approved": approved,
            "rejected": rejected,
            "by_source": by_source
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/{relation_id}/add-to-kg",
    summary="Approve relation and add to Knowledge Graph"
)
async def approve_and_add_to_kg(
    relation_id: int,
    source_entity: Optional[str] = Query(None, description="Overridden source entity text"),
    source_type: Optional[str] = Query(None, description="Overridden source entity type"),
    relation_type: Optional[str] = Query(None, description="Overridden relation type"),
    target_entity: Optional[str] = Query(None, description="Overridden target entity text"),
    target_type: Optional[str] = Query(None, description="Overridden target entity type"),
    db: AsyncSession = Depends(get_db)
):
    """Approve a pending relation and add it as a triple to the Knowledge Graph."""
    try:
        query = select(PendingRelationDB).where(PendingRelationDB.id == relation_id)
        result = await db.execute(query)
        relation = result.scalar_one_or_none()

        if not relation:
            raise HTTPException(status_code=404, detail="Relation not found")

        final_source = source_entity or relation.source_entity
        final_source_type = source_type or relation.source_type
        final_relation = relation_type or relation.relation_type
        final_target = target_entity or relation.target_entity
        final_target_type = target_type or relation.target_type

        from services.knowledge_graph import get_kg_service
        kg = get_kg_service()
        uris = kg.add_relation_to_kg(
            source_entity=final_source,
            source_type=final_source_type,
            relation_type=final_relation,
            target_entity=final_target,
            target_type=final_target_type,
        )
        kg.save_kg()

        relation.source_entity = final_source
        relation.source_type = final_source_type
        relation.relation_type = final_relation
        relation.target_entity = final_target
        relation.target_type = final_target_type
        relation.status = "approved"
        relation.updated_at = datetime.utcnow()
        await db.commit()

        return {
            "success": True,
            "message": f"Relation added to KG: {final_source} --[{final_relation}]--> {final_target}",
            "source_uri": uris["source_uri"],
            "relation_uri": uris["relation_uri"],
            "target_uri": uris["target_uri"],
        }
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error adding relation to KG: {e}")
        raise HTTPException(status_code=500, detail=str(e))
