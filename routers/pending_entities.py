import logging
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_, asc, desc

from database import get_db
from database.models import PendingEntityDB, EntityStatus, EntityReviewDB
from models.entities import (
    PendingEntityCreate,
    PendingEntityUpdate,
    PendingEntityResponse,
    PendingEntitiesListResponse,
)
from services.knowledge_graph import get_kg_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pending-entities", tags=["Pending Entities"])


def _db_to_response(entity: PendingEntityDB, review: EntityReviewDB = None) -> PendingEntityResponse:
    """Convert database model to response model."""
    return PendingEntityResponse(
        id=entity.id,
        text=entity.text,
        entity_type=entity.entity_type,
        normalized_text=entity.normalized_text,
        similarity_score=entity.similarity_score,
        matched_kg_label=entity.matched_kg_label,
        matched_kg_uri=entity.matched_kg_uri,
        validation_status=entity.validation_status,
        source=entity.source,
        source_model=entity.source_model,
        confidence=entity.confidence,
        context=entity.context,
        status=entity.status,
        created_at=entity.created_at.isoformat() if entity.created_at else "",
        updated_at=entity.updated_at.isoformat() if entity.updated_at else "",
        review_status=review.review_status if review else None,
        review_score=review.overall_score if review else None
    )


@router.get(
    "/",
    response_model=PendingEntitiesListResponse,
    summary="List pending entities",
    description="Get a paginated list of entities pending to be added to Knowledge Graph"
)
async def list_pending_entities(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Items per page"),
    status: Optional[str] = Query(None, description="Filter by status (pending, approved, rejected)"),
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    source: Optional[str] = Query(None, description="Filter by source (llm, bert)"),
    search: Optional[str] = Query(None, description="Search in text"),
    has_review: Optional[str] = Query(None, description="Filter by review status (reviewed, not_reviewed)"),
    sort_by: Optional[str] = Query(None, description="Sort column"),
    sort_dir: Optional[str] = Query("asc", description="Sort direction (asc/desc)"),
    db: AsyncSession = Depends(get_db)
):
    """List all pending entities with pagination and filters."""
    try:
        if has_review:
            from sqlalchemy.orm import aliased
            review_alias = aliased(EntityReviewDB)
            
            if has_review == "reviewed":
                query = select(PendingEntityDB).join(
                    EntityReviewDB,
                    PendingEntityDB.id == EntityReviewDB.pending_entity_id
                ).distinct()
                count_query = select(func.count(func.distinct(PendingEntityDB.id))).select_from(
                    PendingEntityDB
                ).join(
                    EntityReviewDB,
                    PendingEntityDB.id == EntityReviewDB.pending_entity_id
                )
            else:
                reviewed_ids = select(EntityReviewDB.pending_entity_id).where(
                    EntityReviewDB.pending_entity_id.isnot(None)
                )
                query = select(PendingEntityDB).where(
                    PendingEntityDB.id.notin_(reviewed_ids)
                )
                count_query = select(func.count(PendingEntityDB.id)).where(
                    PendingEntityDB.id.notin_(reviewed_ids)
                )
        else:
            query = select(PendingEntityDB)
            count_query = select(func.count(PendingEntityDB.id))
        
        if status:
            query = query.where(PendingEntityDB.status == status)
            count_query = count_query.where(PendingEntityDB.status == status)
        
        if entity_type:
            query = query.where(PendingEntityDB.entity_type == entity_type)
            count_query = count_query.where(PendingEntityDB.entity_type == entity_type)
        
        if source:
            query = query.where(PendingEntityDB.source == source)
            count_query = count_query.where(PendingEntityDB.source == source)
        
        if search:
            search_filter = or_(
                PendingEntityDB.text.ilike(f"%{search}%"),
                PendingEntityDB.normalized_text.ilike(f"%{search}%")
            )
            query = query.where(search_filter)
            count_query = count_query.where(search_filter)
        
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0
        
        offset = (page - 1) * page_size
        sortable_columns = {
            "text": PendingEntityDB.text,
            "entity_type": PendingEntityDB.entity_type,
            "validation_status": PendingEntityDB.validation_status,
            "similarity_score": PendingEntityDB.similarity_score,
            "source": PendingEntityDB.source,
            "created_at": PendingEntityDB.created_at,
        }
        if sort_by and sort_by in sortable_columns:
            col = sortable_columns[sort_by]
            query = query.order_by(desc(col) if sort_dir == "desc" else asc(col))
        else:
            query = query.order_by(PendingEntityDB.created_at.desc())
        query = query.offset(offset).limit(page_size)
        
        result = await db.execute(query)
        entities = result.scalars().all()
        
        entity_ids = [e.id for e in entities]
        reviews_result = await db.execute(
            select(EntityReviewDB).where(EntityReviewDB.pending_entity_id.in_(entity_ids))
        )
        reviews = {r.pending_entity_id: r for r in reviews_result.scalars().all()}
        
        return PendingEntitiesListResponse(
            success=True,
            entities=[_db_to_response(e, reviews.get(e.id)) for e in entities],
            total=total,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Error listing pending entities: {e}")
        return PendingEntitiesListResponse(
            success=False,
            entities=[],
            total=0,
            page=page,
            page_size=page_size
        )


@router.get(
    "/stats",
    summary="Get pending entities statistics",
    description="Get statistics about pending entities"
)
async def get_pending_stats(db: AsyncSession = Depends(get_db)):
    """Get statistics about pending entities."""
    try:
        status_query = select(
            PendingEntityDB.status,
            func.count(PendingEntityDB.id)
        ).group_by(PendingEntityDB.status)
        status_result = await db.execute(status_query)
        status_counts = dict(status_result.all())
        
        type_query = select(
            PendingEntityDB.entity_type,
            func.count(PendingEntityDB.id)
        ).where(PendingEntityDB.status == EntityStatus.PENDING.value).group_by(PendingEntityDB.entity_type)
        type_result = await db.execute(type_query)
        type_counts = dict(type_result.all())
        
        source_query = select(
            PendingEntityDB.source,
            func.count(PendingEntityDB.id)
        ).where(PendingEntityDB.status == EntityStatus.PENDING.value).group_by(PendingEntityDB.source)
        source_result = await db.execute(source_query)
        source_counts = dict(source_result.all())
        
        return {
            "success": True,
            "by_status": status_counts,
            "by_type": type_counts,
            "by_source": source_counts,
            "total_pending": status_counts.get(EntityStatus.PENDING.value, 0)
        }
        
    except Exception as e:
        logger.error(f"Error getting pending stats: {e}")
        return {"success": False, "error": str(e)}


@router.get(
    "/{entity_id}",
    response_model=PendingEntityResponse,
    summary="Get pending entity",
    description="Get details of a specific pending entity"
)
async def get_pending_entity(
    entity_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific pending entity by ID."""
    result = await db.execute(
        select(PendingEntityDB).where(PendingEntityDB.id == entity_id)
    )
    entity = result.scalar_one_or_none()
    
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    return _db_to_response(entity)


@router.post(
    "/",
    summary="Create pending entity",
    description="Add a new pending entity"
)
async def create_pending_entity(
    entity: PendingEntityCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a new pending entity."""
    try:
        existing = await db.execute(
            select(PendingEntityDB).where(
                PendingEntityDB.text == entity.text,
                PendingEntityDB.entity_type == entity.entity_type,
                PendingEntityDB.status == EntityStatus.PENDING.value
            )
        )
        if existing.scalar_one_or_none():
            return {"success": True, "message": "Entity already exists in pending list", "duplicate": True}
        
        db_entity = PendingEntityDB(
            text=entity.text,
            entity_type=entity.entity_type,
            normalized_text=entity.normalized_text,
            similarity_score=entity.similarity_score,
            matched_kg_label=entity.matched_kg_label,
            matched_kg_uri=entity.matched_kg_uri,
            validation_status=entity.validation_status,
            source=entity.source,
            source_model=entity.source_model,
            confidence=entity.confidence,
            context=entity.context,
            status=EntityStatus.PENDING.value
        )
        
        db.add(db_entity)
        await db.commit()
        await db.refresh(db_entity)
        
        return {
            "success": True,
            "message": "Entity added to pending list",
            "entity_id": db_entity.id
        }
        
    except Exception as e:
        await db.rollback()
        logger.error(f"Error creating pending entity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put(
    "/{entity_id}",
    summary="Update pending entity",
    description="Update a pending entity's text or type"
)
async def update_pending_entity(
    entity_id: int,
    update: PendingEntityUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update a pending entity."""
    try:
        result = await db.execute(
            select(PendingEntityDB).where(PendingEntityDB.id == entity_id)
        )
        entity = result.scalar_one_or_none()
        
        if not entity:
            raise HTTPException(status_code=404, detail="Entity not found")
        
        new_text = update.text if update.text is not None else entity.text
        new_type = update.entity_type if update.entity_type is not None else entity.entity_type
        
        if update.text is not None or update.entity_type is not None:
            existing_pending = await db.execute(
                select(PendingEntityDB).where(
                    PendingEntityDB.text == new_text,
                    PendingEntityDB.entity_type == new_type,
                    PendingEntityDB.id != entity_id
                )
            )
            if existing_pending.scalar_one_or_none():
                raise HTTPException(
                    status_code=409,
                    detail=f"A pending entity with the same text and type already exists"
                )
            
            existing_review = await db.execute(
                select(EntityReviewDB).where(
                    EntityReviewDB.entity_text == new_text,
                    EntityReviewDB.entity_type == new_type,
                    EntityReviewDB.pending_entity_id != entity_id
                )
            )
            if existing_review.scalar_one_or_none():
                raise HTTPException(
                    status_code=409,
                    detail=f"An entity review with the same text and type already exists"
                )
        
        if update.text is not None:
            entity.text = update.text
        if update.entity_type is not None:
            entity.entity_type = update.entity_type
        if update.normalized_text is not None:
            entity.normalized_text = update.normalized_text
        if update.status is not None:
            entity.status = update.status
        
        entity.updated_at = datetime.utcnow()
        
        review_result = await db.execute(
            select(EntityReviewDB).where(EntityReviewDB.pending_entity_id == entity_id)
        )
        review = review_result.scalar_one_or_none()
        if review:
            if update.text is not None:
                review.entity_text = update.text
            if update.entity_type is not None:
                review.entity_type = update.entity_type
            review.updated_at = datetime.utcnow()
        
        await db.commit()
        await db.refresh(entity)
        
        return {
            "success": True,
            "message": "Entity updated successfully",
            "entity": _db_to_response(entity)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error updating pending entity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/{entity_id}",
    summary="Delete pending entity",
    description="Delete a pending entity"
)
async def delete_pending_entity(
    entity_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Delete a pending entity."""
    try:
        result = await db.execute(
            select(PendingEntityDB).where(PendingEntityDB.id == entity_id)
        )
        entity = result.scalar_one_or_none()
        
        if not entity:
            raise HTTPException(status_code=404, detail="Entity not found")
        
        await db.delete(entity)
        await db.commit()
        
        return {
            "success": True,
            "message": "Entity deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error deleting pending entity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/{entity_id}/add-to-kg",
    summary="Add pending entity to Knowledge Graph",
    description="Approve and add a pending entity to the Knowledge Graph"
)
async def add_pending_to_kg(
    entity_id: int,
    link_to_uri: Optional[str] = None,
    link_type: Optional[str] = "closeMatch",
    db: AsyncSession = Depends(get_db)
):
    """Add a pending entity to the Knowledge Graph."""
    try:
        result = await db.execute(
            select(PendingEntityDB).where(PendingEntityDB.id == entity_id)
        )
        entity = result.scalar_one_or_none()
        
        if not entity:
            raise HTTPException(status_code=404, detail="Entity not found")
        
        kg_service = get_kg_service()
        
        from models.entities import MedicalEntity, MedicalEntityType
        
        type_mapping = {
            "finding": MedicalEntityType.FINDING,
            "disease": MedicalEntityType.DISEASE,
            "quantitative_measure": MedicalEntityType.QUANTITATIVE_MEASURE,
            "substance": MedicalEntityType.SUBSTANCE,
            "procedure": MedicalEntityType.PROCEDURE
        }
        entity_type_enum = type_mapping.get(entity.entity_type, MedicalEntityType.FINDING)
        
        medical_entity = MedicalEntity(
            text=entity.text,
            entity_type=entity_type_enum,
            normalized_form=entity.normalized_text or entity.text,
            confidence=entity.confidence or 0.9
        )
        
        entity_uri = kg_service.add_entity_to_kg(
            entity=medical_entity,
            link_to_uri=link_to_uri,
            link_type=link_type
        )
        
        if entity_uri:
            review_result = await db.execute(
                select(EntityReviewDB).where(EntityReviewDB.pending_entity_id == entity_id)
            )
            review = review_result.scalar_one_or_none()
            if review:
                await db.delete(review)
                logger.info(f"Deleted associated review for pending entity {entity_id}")
            
            await db.delete(entity)
            logger.info(f"Deleted pending entity {entity_id} after adding to KG")
            
            await db.commit()
            
            return {
                "success": True,
                "message": "Entity added to Knowledge Graph and removed from pending list",
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
        logger.error(f"Error adding pending to KG: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/bulk-action",
    summary="Perform bulk action on entities",
    description="Delete or reject multiple entities at once"
)
async def bulk_action(
    entity_ids: List[int],
    action: str = Query(..., description="Action: delete, reject, approve"),
    db: AsyncSession = Depends(get_db)
):
    """Perform bulk action on multiple entities."""
    try:
        if action not in ["delete", "reject", "approve"]:
            raise HTTPException(status_code=400, detail="Invalid action")
        
        result = await db.execute(
            select(PendingEntityDB).where(PendingEntityDB.id.in_(entity_ids))
        )
        entities = result.scalars().all()
        
        count = 0
        for entity in entities:
            if action == "delete":
                await db.delete(entity)
                count += 1
            elif action == "reject":
                entity.status = EntityStatus.REJECTED.value
                entity.updated_at = datetime.utcnow()
                count += 1
            elif action == "approve" and entity.status == EntityStatus.PENDING.value:
                entity.status = EntityStatus.APPROVED.value
                entity.updated_at = datetime.utcnow()
                count += 1
        
        await db.commit()
        
        return {
            "success": True,
            "message": f"{action.capitalize()}d {count} entities",
            "affected_count": count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error in bulk action: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def save_pending_entities(
    entities: List[dict],
    source: str,
    source_model: str,
    context: Optional[str],
    db: AsyncSession
):
    """
    Helper function to save entities that are not in the KG.
    Called from classification endpoints.
    """
    saved_count = 0
    
    for entity_data in entities:
        if entity_data.get("validation_status") == "exact_match":
            continue
        
        try:
            existing = await db.execute(
                select(PendingEntityDB).where(
                    PendingEntityDB.text == entity_data["text"],
                    PendingEntityDB.entity_type == entity_data["entity_type"],
                    PendingEntityDB.status == EntityStatus.PENDING.value
                )
            )
            if existing.scalar_one_or_none():
                continue
            
            db_entity = PendingEntityDB(
                text=entity_data["text"],
                entity_type=entity_data["entity_type"],
                normalized_text=entity_data.get("normalized_text"),
                similarity_score=entity_data.get("similarity_score"),
                matched_kg_label=entity_data.get("matched_kg_label"),
                matched_kg_uri=entity_data.get("matched_kg_uri"),
                validation_status=entity_data.get("validation_status"),
                source=source,
                source_model=source_model,
                confidence=entity_data.get("confidence"),
                context=context[:500] if context else None,
                status=EntityStatus.PENDING.value
            )
            
            db.add(db_entity)
            saved_count += 1
            
        except Exception as e:
            logger.warning(f"Error saving pending entity: {e}")
            continue
    
    if saved_count > 0:
        await db.commit()
    
    return saved_count
