import logging
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from database.connection import get_db
from models.review_defaults import RELATION_CONFIG_DEFAULTS, validate_review_config
from services.relation_reviewer import invalidate_config_cache
from database.models import RelationConfigDB

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/relation-config", tags=["Relation Configuration"])

DEFAULT_CONFIG = RELATION_CONFIG_DEFAULTS


async def get_or_create_config(db: AsyncSession) -> RelationConfigDB:
    """Get the active configuration or create default if none exists."""
    result = await db.execute(
        select(RelationConfigDB).where(RelationConfigDB.is_active == True)
    )
    config = result.scalar_one_or_none()
    
    if not config:
        config = RelationConfigDB(**DEFAULT_CONFIG, is_active=True)
        db.add(config)
        await db.commit()
        await db.refresh(config)
        logger.info("Created default relation configuration")
    
    return config


def config_to_dict(config: RelationConfigDB) -> dict:
    """Convert config DB model to dictionary."""
    return {
        "id": config.id,
        "config_name": config.config_name,
        "weight_congruence": config.weight_congruence,
        "weight_coverage": config.weight_coverage,
        "weight_constraint": config.weight_constraint,
        "weight_completeness": config.weight_completeness,
        "weight_consistency": config.weight_consistency,
        "threshold_pass": config.threshold_pass,
        "threshold_review": config.threshold_review,
        "congruence_type_penalty": config.congruence_type_penalty,
        "congruence_kg_bonus": config.congruence_kg_bonus,
        "coverage_novelty_threshold": config.coverage_novelty_threshold,
        "coverage_novelty_bonus": config.coverage_novelty_bonus,
        "constraint_violation_penalty": config.constraint_violation_penalty,
        "constraint_min_confidence": config.constraint_min_confidence,
        "constraint_require_context": config.constraint_require_context,
        "completeness_required_fields": config.completeness_required_fields,
        "completeness_optional_weight": config.completeness_optional_weight,
        "consistency_agreement_bonus": config.consistency_agreement_bonus,
        "consistency_base_score": config.consistency_base_score,
        "is_active": config.is_active,
        "created_at": config.created_at.isoformat() if config.created_at else None,
        "updated_at": config.updated_at.isoformat() if config.updated_at else None,
    }


@router.get("/")
async def get_config(db: AsyncSession = Depends(get_db)):
    """Get the current active relation configuration."""
    config = await get_or_create_config(db)
    return {
        "success": True,
        "config": config_to_dict(config),
        "defaults": DEFAULT_CONFIG,
    }


@router.put("/")
async def update_config(
    weight_congruence: Optional[float] = None,
    weight_coverage: Optional[float] = None,
    weight_constraint: Optional[float] = None,
    weight_completeness: Optional[float] = None,
    weight_consistency: Optional[float] = None,
    threshold_pass: Optional[float] = None,
    threshold_review: Optional[float] = None,
    congruence_type_penalty: Optional[float] = None,
    congruence_kg_bonus: Optional[float] = None,
    coverage_novelty_threshold: Optional[int] = None,
    coverage_novelty_bonus: Optional[float] = None,
    constraint_violation_penalty: Optional[float] = None,
    constraint_min_confidence: Optional[float] = None,
    constraint_require_context: Optional[bool] = None,
    completeness_required_fields: Optional[str] = None,
    completeness_optional_weight: Optional[float] = None,
    consistency_agreement_bonus: Optional[float] = None,
    consistency_base_score: Optional[float] = None,
    db: AsyncSession = Depends(get_db),
):
    """Update the relation configuration. Only provided fields are updated."""
    try:
        config = await get_or_create_config(db)
        
        if weight_congruence is not None:
            config.weight_congruence = weight_congruence
        if weight_coverage is not None:
            config.weight_coverage = weight_coverage
        if weight_constraint is not None:
            config.weight_constraint = weight_constraint
        if weight_completeness is not None:
            config.weight_completeness = weight_completeness
        if weight_consistency is not None:
            config.weight_consistency = weight_consistency
        
        total_weight = (
            config.weight_congruence +
            config.weight_coverage +
            config.weight_constraint +
            config.weight_completeness +
            config.weight_consistency
        )
        if abs(total_weight - 1.0) > 0.01:
            raise HTTPException(
                status_code=400,
                detail=f"Weights must sum to 1.0 (currently {total_weight:.2f})",
            )
        
        if threshold_pass is not None:
            config.threshold_pass = threshold_pass
        if threshold_review is not None:
            config.threshold_review = threshold_review
        
        if config.threshold_pass <= config.threshold_review:
            raise HTTPException(
                status_code=400,
                detail="Pass threshold must be greater than review threshold",
            )
        
        if congruence_type_penalty is not None:
            config.congruence_type_penalty = congruence_type_penalty
        if congruence_kg_bonus is not None:
            config.congruence_kg_bonus = congruence_kg_bonus
        
        if coverage_novelty_threshold is not None:
            config.coverage_novelty_threshold = coverage_novelty_threshold
        if coverage_novelty_bonus is not None:
            config.coverage_novelty_bonus = coverage_novelty_bonus
        
        if constraint_violation_penalty is not None:
            config.constraint_violation_penalty = constraint_violation_penalty
        if constraint_min_confidence is not None:
            config.constraint_min_confidence = constraint_min_confidence
        if constraint_require_context is not None:
            config.constraint_require_context = constraint_require_context
        
        if completeness_required_fields is not None:
            config.completeness_required_fields = completeness_required_fields
        if completeness_optional_weight is not None:
            config.completeness_optional_weight = completeness_optional_weight
        
        if consistency_agreement_bonus is not None:
            config.consistency_agreement_bonus = consistency_agreement_bonus
        if consistency_base_score is not None:
            config.consistency_base_score = consistency_base_score
        
        try:
            validate_review_config({k: getattr(config, k) for k in DEFAULT_CONFIG if k != "config_name"})
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        config.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(config)
        invalidate_config_cache()
        
        return {
            "success": True,
            "message": "Configuration updated successfully",
            "config": config_to_dict(config),
        }
    
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error updating relation config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset")
async def reset_to_defaults(db: AsyncSession = Depends(get_db)):
    """Reset relation configuration to default values."""
    try:
        config = await get_or_create_config(db)
        
        for key, value in DEFAULT_CONFIG.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        config.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(config)
        invalidate_config_cache()
        
        return {
            "success": True,
            "message": "Configuration reset to defaults",
            "config": config_to_dict(config),
        }
    except Exception as e:
        await db.rollback()
        logger.error(f"Error resetting relation config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/defaults")
async def get_defaults():
    """Get the default configuration values."""
    return {
        "success": True,
        "defaults": DEFAULT_CONFIG,
    }
