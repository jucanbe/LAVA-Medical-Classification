from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class MedicalRelationType(str, Enum):
    """Types of medical relations that can be classified."""
    HAS_SYMPTOM = "has_symptom"
    HAS_FINDING = "has_finding"
    SUGGESTS = "suggests"
    LOCATED_IN = "located_in"
    INDICATED_FOR = "indicated_for"
    PRODUCES_RESULT = "produces_result"
    TREATS = "treats"
    FIRST_LINE_FOR = "first_line_for"
    CONTRAINDICATED_IN = "contraindicated_in"
    RULES_OUT = "rules_out"
    ASSESSES = "assesses"
    AFFECTS = "affects"


RELATION_DESCRIPTIONS = {
    "has_symptom": "A disease/condition has a specific symptom",
    "has_finding": "A disease/condition has a specific clinical finding",
    "suggests": "A finding suggests a disease/condition",
    "located_in": "An entity is located in an anatomical location",
    "indicated_for": "A procedure is indicated for a condition",
    "produces_result": "A procedure produces a specific result",
    "treats": "A substance/procedure treats a condition",
    "first_line_for": "A substance is the first-line treatment for a condition",
    "contraindicated_in": "A substance/procedure is contraindicated in a condition",
    "rules_out": "A finding/procedure rules out a condition",
    "assesses": "A tool/score assesses a condition",
    "affects": "A condition affects an organ/system",
}


class MedicalRelation(BaseModel):
    """Represents a medical relation extracted from text."""
    
    source_entity: str = Field(
        ...,
        description="The source entity text (subject of the relation)"
    )
    source_type: str = Field(
        ...,
        description="Type of the source entity (e.g., disease, finding, substance)"
    )
    relation_type: MedicalRelationType = Field(
        ...,
        description="The type of relation"
    )
    target_entity: str = Field(
        ...,
        description="The target entity text (object of the relation)"
    )
    target_type: str = Field(
        ...,
        description="Type of the target entity"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Classification confidence level (0-1)"
    )
    context: Optional[str] = Field(
        None,
        description="Surrounding context of the relation in the text"
    )


class RelationClassificationResult(BaseModel):
    """Result of medical relation classification."""
    
    relations: List[MedicalRelation] = Field(
        default_factory=list,
        description="List of medical relations found"
    )
    raw_text: Optional[str] = Field(
        None,
        description="Original text that was analyzed"
    )
    total_relations: Optional[int] = Field(
        None,
        description="Total number of relations found"
    )
    processing_time_ms: Optional[float] = Field(
        None,
        description="Processing time in milliseconds"
    )
    
    def model_post_init(self, __context):
        if self.total_relations is None:
            object.__setattr__(self, 'total_relations', len(self.relations))


class RelationClassificationRequest(BaseModel):
    """Request to classify medical relations."""
    
    text: str = Field(
        ...,
        min_length=1,
        description="Medical text to analyze for relations"
    )
    config_id: Optional[int] = Field(
        None,
        description="ID of the LLM configuration to use"
    )
    min_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to include a relation"
    )


class RelationClassificationResponse(BaseModel):
    """Response for medical relation classification."""
    
    model_config = {"protected_namespaces": ()}
    
    success: bool = Field(..., description="Indicates if the classification was successful")
    result: Optional[RelationClassificationResult] = Field(
        None,
        description="Classification result"
    )
    error: Optional[str] = Field(None, description="Error message if failed")
    model_used: Optional[str] = Field(None, description="Model used for classification")
    relations_saved: Optional[int] = Field(None, description="Number of relations auto-saved to pending")


class BERTRelationClassificationRequest(BaseModel):
    """Request for BERT-based relation classification."""
    
    model_config = {"protected_namespaces": ()}
    
    text: str = Field(
        ...,
        min_length=1,
        description="Medical text to analyze for relations"
    )
    model_name: Optional[str] = Field(
        None,
        description="Name of the BERT model to use"
    )


class BERTRelationClassificationResponse(BaseModel):
    """Response for BERT-based relation classification."""
    
    model_config = {"protected_namespaces": ()}
    
    success: bool = Field(..., description="Indicates if the classification was successful")
    result: Optional[RelationClassificationResult] = Field(
        None,
        description="Classification result"
    )
    error: Optional[str] = Field(None, description="Error message if failed")
    model_used: Optional[str] = Field(None, description="Model used")
    inference_time_ms: Optional[float] = Field(None, description="Inference time in milliseconds")
    relations_saved: Optional[int] = Field(None, description="Number of relations auto-saved to pending")



class PendingRelationBase(BaseModel):
    """Base model for pending relations."""
    
    source_entity: str = Field(..., description="Source entity text")
    source_type: str = Field(..., description="Source entity type")
    relation_type: str = Field(..., description="Type of relation")
    target_entity: str = Field(..., description="Target entity text")
    target_type: str = Field(..., description="Target entity type")
    confidence: Optional[float] = Field(None, description="Classification confidence")
    context: Optional[str] = Field(None, description="Original text context")
    source: str = Field("llm", description="Source of classification (llm or bert)")
    source_model: Optional[str] = Field(None, description="Model used for classification")


class PendingRelationCreate(PendingRelationBase):
    """Model for creating a pending relation."""
    pass


class PendingRelationUpdate(BaseModel):
    """Model for updating a pending relation."""
    
    source_entity: Optional[str] = Field(None, description="Updated source entity")
    source_type: Optional[str] = Field(None, description="Updated source type")
    relation_type: Optional[str] = Field(None, description="Updated relation type")
    target_entity: Optional[str] = Field(None, description="Updated target entity")
    target_type: Optional[str] = Field(None, description="Updated target type")
    status: Optional[str] = Field(None, description="Updated status")


class PendingRelationResponse(PendingRelationBase):
    """Response model for a pending relation."""
    
    model_config = {"from_attributes": True}
    
    id: int = Field(..., description="Relation ID")
    status: str = Field(..., description="Status (pending, approved, rejected)")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    review_status: Optional[str] = Field(None, description="Review status")
    review_score: Optional[float] = Field(None, description="Review overall score")


class PendingRelationsListResponse(BaseModel):
    """Response model for listing pending relations."""
    
    success: bool = Field(..., description="Operation success")
    relations: List[PendingRelationResponse] = Field(default_factory=list, description="List of pending relations")
    total: int = Field(0, description="Total number of relations")
    page: int = Field(1, description="Current page")
    page_size: int = Field(50, description="Page size")



class RelationReviewCreate(BaseModel):
    """Request to create a relation review."""
    
    pending_relation_id: Optional[int] = Field(None, description="ID of pending relation to review")
    source_entity: str = Field(..., description="Source entity text")
    source_type: str = Field(..., description="Source entity type")
    relation_type: str = Field(..., description="Relation type")
    target_entity: str = Field(..., description="Target entity text")
    target_type: str = Field(..., description="Target entity type")


class RelationReviewResponse(BaseModel):
    """Response model for relation review."""
    
    model_config = {"from_attributes": True}
    
    id: int = Field(..., description="Review ID")
    pending_relation_id: Optional[int] = Field(None, description="Linked pending relation ID")
    source_entity: str = Field(..., description="Source entity text")
    source_type: str = Field(..., description="Source entity type")
    relation_type: str = Field(..., description="Relation type")
    target_entity: str = Field(..., description="Target entity text")
    target_type: str = Field(..., description="Target entity type")
    
    congruence_score: Optional[float] = Field(None, description="Congruence (semantic validity) score")
    coverage_score: Optional[float] = Field(None, description="Coverage (representativeness) score")
    constraint_score: Optional[float] = Field(None, description="Constraint (domain rules) score")
    completeness_score: Optional[float] = Field(None, description="Completeness (required attributes) score")
    consistency_score: Optional[float] = Field(None, description="Consistency (cross-validation) score")
    
    overall_score: float = Field(..., description="Weighted overall score")
    review_status: str = Field(..., description="Review status")
    recommendation: Optional[str] = Field(None, description="Recommendation: approve, reject, modify")
    review_notes: Optional[str] = Field(None, description="Additional review notes")
    
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")


class RelationReviewListResponse(BaseModel):
    """Response model for listing relation reviews."""
    
    success: bool = Field(..., description="Operation success")
    reviews: List[RelationReviewResponse] = Field(default_factory=list, description="List of reviews")
    total: int = Field(0, description="Total number of reviews")
