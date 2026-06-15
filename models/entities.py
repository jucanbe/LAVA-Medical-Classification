from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class MedicalEntityType(str, Enum):
    """Types of medical entities that can be classified."""
    DISEASE = "disease"
    SYMPTOM = "symptom"
    FINDING = "finding"
    ORGAN = "organ"
    IMAGING_PROCEDURE = "imaging_procedure"
    EXAMINATION_PROCEDURE = "examination_procedure"
    THERAPEUTIC_PROCEDURE = "therapeutic_procedure"
    IMAGING_RESULT = "imaging_result"
    EXAMINATION_MEASURE = "examination_measure"
    PARAMETER = "parameter"
    SCORE = "score"
    THERAPY = "therapy"
    SUBSTANCE = "substance"
    ADVERSE_EVENT = "adverse_event"



class MedicalEntity(BaseModel):
    """Represents a medical entity extracted from text."""
    
    text: str = Field(
        ...,
        description="The exact text of the entity found in the document"
    )
    entity_type: MedicalEntityType = Field(
        ...,
        description="The type of medical entity"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Classification confidence level (0-1)"
    )
    start_position: Optional[int] = Field(
        None,
        description="Starting position of the entity in the text"
    )
    end_position: Optional[int] = Field(
        None,
        description="Ending position of the entity in the text"
    )
    context: Optional[str] = Field(
        None,
        description="Surrounding context of the entity"
    )
    normalized_form: Optional[str] = Field(
        None,
        description="Normalized or canonical form of the entity"
    )


class EntityClassificationResult(BaseModel):
    """Result of medical entity classification."""
    
    entities: List[MedicalEntity] = Field(
        default_factory=list,
        description="List of medical entities found"
    )
    raw_text: Optional[str] = Field(
        None,
        description="Original text that was analyzed"
    )
    total_entities: Optional[int] = Field(
        None,
        description="Total number of entities found"
    )
    processing_time_ms: Optional[float] = Field(
        None,
        description="Processing time in milliseconds"
    )
    
    def model_post_init(self, __context):
        """Auto-calculate total_entities if not provided."""
        if self.total_entities is None:
            object.__setattr__(self, 'total_entities', len(self.entities))


class ClassificationRequest(BaseModel):
    """Request to classify medical entities."""
    
    text: str = Field(
        ...,
        min_length=1,
        description="Medical text to analyze"
    )
    config_id: Optional[int] = Field(
        None,
        description="ID of the LLM configuration to use (uses default if not specified)"
    )
    include_context: bool = Field(
        default=True,
        description="Include surrounding context for each entity"
    )
    min_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to include an entity"
    )


class ClassificationResponse(BaseModel):
    """Response for medical entity classification."""
    
    model_config = {"protected_namespaces": ()}
    
    success: bool = Field(..., description="Indicates if the classification was successful")
    result: Optional[EntityClassificationResult] = Field(
        None,
        description="Classification result"
    )
    error: Optional[str] = Field(None, description="Error message if failed")
    model_used: Optional[str] = Field(None, description="LLM model used")


class KGMatchStatus(str, Enum):
    """Status of entity matching against Knowledge Graph."""
    EXACT_MATCH = "exact_match"
    SIMILAR_MATCH = "similar_match"
    LOW_MATCH = "low_match"
    NOT_FOUND = "not_found"


class KGMatch(BaseModel):
    """Represents a match found in the Knowledge Graph."""
    
    kg_uri: str = Field(..., description="URI of the matched entity in the KG")
    kg_label: str = Field(..., description="Label of the matched entity in the KG")
    kg_type: Optional[str] = Field(None, description="Type/class of the entity in the KG")
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Similarity score between extracted entity and KG entity"
    )


class ValidatedEntity(BaseModel):
    """Entity with validation against Knowledge Graph."""
    
    entity: MedicalEntity = Field(..., description="The original extracted entity")
    validation_status: KGMatchStatus = Field(
        ...,
        description="Status of validation against KG"
    )
    kg_matches: List[KGMatch] = Field(
        default_factory=list,
        description="Matching entities found in the KG"
    )
    is_validated: bool = Field(
        ...,
        description="Whether the entity is validated by the KG"
    )
    validation_notes: Optional[str] = Field(
        None,
        description="Additional notes about the validation"
    )


class KGValidationResult(BaseModel):
    """Result of validating entities against Knowledge Graph."""
    
    validated_entities: List[ValidatedEntity] = Field(
        default_factory=list,
        description="List of entities with their validation status"
    )
    total_entities: int = Field(..., description="Total number of entities validated")
    exact_matches: int = Field(..., description="Number of exact matches found")
    similar_matches: int = Field(..., description="Number of similar matches found")
    low_matches: int = Field(0, description="Number of low similarity matches")
    not_found: int = Field(0, description="Number of entities not found in KG")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")


class ClassificationWithValidationResponse(BaseModel):
    """Response for classification with KG validation."""
    
    model_config = {"protected_namespaces": (), "populate_by_name": True}
    
    success: bool = Field(..., description="Indicates if the operation was successful")
    result: Optional[EntityClassificationResult] = Field(
        None,
        serialization_alias="result",
        description="Raw classification result"
    )
    kg_validation: Optional[KGValidationResult] = Field(
        None,
        serialization_alias="kg_validation",
        description="KG validation result"
    )
    error: Optional[str] = Field(None, description="Error message if failed")
    model_used: Optional[str] = Field(None, description="LLM model used")


class IOBExportResponse(BaseModel):
    """Response for IOB CSV format export."""
    
    model_config = {"protected_namespaces": ()}
    
    success: bool = Field(..., description="Indicates if the export was successful")
    original_text: str = Field(..., description="The original input text")
    iob_format: str = Field(..., description="Text in IOB CSV format (words,sentence_id,labels) for BERT training")
    entities_count: int = Field(..., description="Number of entities tagged")
    error: Optional[str] = Field(None, description="Error message if failed")
    model_used: Optional[str] = Field(None, description="LLM model used for classification")


class BERTClassificationRequest(BaseModel):
    """Request for BERT-based classification."""
    
    model_config = {"protected_namespaces": ()}
    
    text: str = Field(
        ...,
        min_length=1,
        description="Medical text to analyze"
    )
    model_name: Optional[str] = Field(
        None,
        description="Name of the BERT model to use (uses default if not specified)"
    )


class BERTClassificationResponse(BaseModel):
    """Response for BERT-based classification."""
    
    model_config = {"protected_namespaces": ()}
    
    success: bool = Field(..., description="Indicates if the classification was successful")
    result: Optional[EntityClassificationResult] = Field(
        None,
        description="Classification result"
    )
    error: Optional[str] = Field(None, description="Error message if failed")
    model_used: Optional[str] = Field(None, description="BERT model used")
    inference_time_ms: Optional[float] = Field(None, description="Model inference time in milliseconds")


class BERTTrainingRequest(BaseModel):
    """Request to train/fine-tune a BERT model."""
    
    model_config = {"protected_namespaces": ()}
    
    model_name: str = Field(
        ...,
        description="Name for the new/updated model"
    )
    model_type: str = Field(
        default="entity",
        description="Type of model: 'entity' or 'relation'"
    )
    base_model: str = Field(
        default="dmis-lab/biobert-base-cased-v1.1",
        description="Base model to fine-tune from HuggingFace"
    )
    training_data: Optional[str] = Field(
        None,
        description="CSV formatted training data (alternative to file uploads)"
    )
    train_data: Optional[str] = Field(
        None,
        description="Training set CSV data (words,sentence_id,labels)"
    )
    dev_data: Optional[str] = Field(
        None,
        description="Development/validation set CSV data (optional)"
    )
    test_data: Optional[str] = Field(
        None,
        description="Test set CSV data"
    )
    training_file: Optional[str] = Field(
        None,
        description="Path to IOB formatted training file"
    )
    epochs: int = Field(default=3, ge=1, le=100, description="Number of training epochs")
    batch_size: int = Field(default=16, ge=1, le=128, description="Training batch size")
    learning_rate: float = Field(default=5e-5, description="Learning rate")


class BERTTrainingResponse(BaseModel):
    """Response for BERT training."""
    
    model_config = {"protected_namespaces": ()}
    
    success: bool = Field(..., description="Indicates if training was successful")
    model_name: str = Field(..., description="Name of the trained model")
    model_path: Optional[str] = Field(None, description="Path to the saved model")
    training_metrics: Optional[dict] = Field(None, description="Training metrics")
    error: Optional[str] = Field(None, description="Error message if failed")


class AddEntityToKGRequest(BaseModel):
    """Request to add an entity to the Knowledge Graph."""
    
    text: str = Field(..., description="Entity text to add")
    entity_type: MedicalEntityType = Field(..., description="Type of medical entity")
    normalized_form: Optional[str] = Field(None, description="Normalized/canonical form")
    link_to_uri: Optional[str] = Field(None, description="URI to link this entity to (for similar entities)")
    link_type: Optional[str] = Field(None, description="Type of link (e.g., 'sameAs', 'relatedTo', 'narrowerThan')")


class AddEntityToKGResponse(BaseModel):
    """Response for adding an entity to the Knowledge Graph."""
    
    success: bool = Field(..., description="Indicates if the entity was added successfully")
    entity_uri: Optional[str] = Field(None, description="URI of the created entity")
    message: Optional[str] = Field(None, description="Success or error message")


class BERTModelInfo(BaseModel):
    """Information about a BERT model."""
    
    model_config = {"protected_namespaces": ()}
    
    name: str = Field(..., description="Model name")
    path: str = Field(..., description="Path to model files")
    base_model: Optional[str] = Field(None, description="Base model used")
    labels: List[str] = Field(default_factory=list, description="Entity labels the model can predict")
    model_type: str = Field(default="entity", description="Type of model: 'entity' or 'relation'")
    created_at: Optional[str] = Field(None, description="When the model was created/fine-tuned")



class PendingEntityBase(BaseModel):
    """Base model for pending entities."""
    
    text: str = Field(..., description="Entity text")
    entity_type: str = Field(..., description="Type of entity")
    normalized_text: Optional[str] = Field(None, description="Normalized text form")
    similarity_score: Optional[float] = Field(None, description="Similarity score with KG match")
    matched_kg_label: Optional[str] = Field(None, description="Matched KG entity label")
    matched_kg_uri: Optional[str] = Field(None, description="Matched KG entity URI")
    validation_status: Optional[str] = Field(None, description="KG validation status")
    source: str = Field("llm", description="Source of classification (llm or bert)")
    source_model: Optional[str] = Field(None, description="Model used for classification")
    confidence: Optional[float] = Field(None, description="Classification confidence")
    context: Optional[str] = Field(None, description="Original text context")


class PendingEntityCreate(PendingEntityBase):
    """Model for creating a pending entity."""
    pass


class PendingEntityUpdate(BaseModel):
    """Model for updating a pending entity."""
    
    text: Optional[str] = Field(None, description="Updated entity text")
    entity_type: Optional[str] = Field(None, description="Updated entity type")
    normalized_text: Optional[str] = Field(None, description="Updated normalized text")
    status: Optional[str] = Field(None, description="Updated status")


class PendingEntityResponse(PendingEntityBase):
    """Response model for a pending entity."""
    
    model_config = {"from_attributes": True}
    
    id: int = Field(..., description="Entity ID")
    status: str = Field(..., description="Entity status (pending, approved, rejected)")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    review_status: Optional[str] = Field(None, description="Entity review status (passed, failed, needs_review)")
    review_score: Optional[float] = Field(None, description="Entity review overall score")


class PendingEntitiesListResponse(BaseModel):
    """Response model for listing pending entities."""
    
    success: bool = Field(..., description="Operation success")
    entities: List[PendingEntityResponse] = Field(default_factory=list, description="List of pending entities")
    total: int = Field(0, description="Total number of entities")
    page: int = Field(1, description="Current page")
    page_size: int = Field(50, description="Page size")



class ReviewStatus(str, Enum):
    """Status of an entity review."""
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"


class CongruenceMetrics(BaseModel):
    """Metrics for Congruence (Semantic Alignment) - 'C1' from Scorecard."""
    
    score: float = Field(..., ge=0.0, le=1.0, description="Overall congruence score")
    nearest_entity: Optional[str] = Field(None, description="Most similar entity in KG")
    nearest_uri: Optional[str] = Field(None, description="URI of nearest entity")
    embedding_distance: Optional[float] = Field(None, description="Cosine distance to nearest")
    method: str = Field("sequence_matcher", description="Method used: 'sequence_matcher', 'bert_embedding', 'hybrid'")


class CoverageMetrics(BaseModel):
    """Metrics for Coverage (Representativeness) - 'C2' from Scorecard."""
    
    score: float = Field(..., ge=0.0, le=1.0, description="Overall coverage score")
    similar_entities_count: int = Field(0, description="Number of similar entities in KG")
    is_novel: bool = Field(False, description="Is this a novel entity type?")
    fills_gap: bool = Field(False, description="Does it fill a gap in the KG?")
    cluster_id: Optional[str] = Field(None, description="Semantic cluster ID")
    coverage_increase: Optional[float] = Field(None, description="% increase in coverage if added")


class ConstraintMetrics(BaseModel):
    """Metrics for Constraint (Clinical Validity) - 'C3' from Scorecard."""
    
    score: float = Field(..., ge=0.0, le=1.0, description="Overall constraint score")
    violations: List[str] = Field(default_factory=list, description="List of constraint violations")
    ontology_valid: Optional[bool] = Field(None, description="Valid in medical ontology?")
    type_valid: bool = Field(True, description="Is entity type valid?")
    format_valid: bool = Field(True, description="Does entity format meet requirements?")
    clinical_plausibility: Optional[float] = Field(None, description="Clinical plausibility score")


class CompletenessMetrics(BaseModel):
    """Metrics for Completeness (Required Attributes) - 'C4' from Scorecard."""
    
    score: float = Field(..., ge=0.0, le=1.0, description="Overall completeness score")
    has_type: bool = Field(False, description="Has entity type")
    has_definition: bool = Field(False, description="Has definition/context")
    has_normalized_form: bool = Field(False, description="Has normalized form")
    has_context: bool = Field(False, description="Has source context")
    has_confidence: bool = Field(False, description="Has confidence score")
    missing_fields: List[str] = Field(default_factory=list, description="List of missing required fields")


class ConsistencyMetrics(BaseModel):
    """Metrics for Consistency (Type Coherence) - 'C5' from Scorecard."""
    
    score: float = Field(..., ge=0.0, le=1.0, description="Overall consistency score")
    type_confidence: float = Field(0.0, description="Confidence in entity type classification")
    alternate_types: List[dict] = Field(default_factory=list, description="Alternate type suggestions with scores")
    bert_agreement: Optional[bool] = Field(None, description="Does BERT classification agree?")
    llm_agreement: Optional[bool] = Field(None, description="Does LLM classification agree?")
    cross_validation_score: Optional[float] = Field(None, description="Cross-validation consistency score")


class EntityReviewCreate(BaseModel):
    """Request to create an entity review."""
    
    pending_entity_id: Optional[int] = Field(None, description="ID of pending entity to review")
    entity_text: str = Field(..., description="Entity text to review")
    entity_type: str = Field(..., description="Proposed entity type")
    run_bert_validation: bool = Field(True, description="Run BERT cross-validation")
    run_llm_validation: bool = Field(False, description="Run LLM cross-validation")


class EntityReviewResponse(BaseModel):
    """Response model for entity review."""
    
    model_config = {"from_attributes": True}
    
    id: int = Field(..., description="Review ID")
    pending_entity_id: Optional[int] = Field(None, description="Linked pending entity ID")
    entity_text: str = Field(..., description="Entity text")
    entity_type: str = Field(..., description="Entity type")
    
    congruence: Optional[CongruenceMetrics] = Field(None, description="Congruence (semantic alignment) metrics")
    coverage: Optional[CoverageMetrics] = Field(None, description="Coverage (representativeness) metrics")
    constraint: Optional[ConstraintMetrics] = Field(None, description="Constraint (clinical validity) metrics")
    completeness: Optional[CompletenessMetrics] = Field(None, description="Completeness (required attributes) metrics")
    consistency: Optional[ConsistencyMetrics] = Field(None, description="Consistency (type coherence) metrics")
    
    overall_score: float = Field(..., description="Weighted overall score")
    review_status: str = Field(..., description="Review status")
    recommendation: Optional[str] = Field(None, description="Recommendation: approve, reject, modify")
    review_notes: Optional[str] = Field(None, description="Additional review notes")
    
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")


class EntityReviewSummary(BaseModel):
    """Summary of entity review scores."""
    
    entity_text: str
    entity_type: str
    congruence_score: float
    coverage_score: float
    constraint_score: float
    completeness_score: float
    consistency_score: float
    overall_score: float
    review_status: str
    recommendation: str


class EntityReviewListResponse(BaseModel):
    """Response model for listing entity reviews."""
    
    success: bool = Field(..., description="Operation success")
    reviews: List[EntityReviewResponse] = Field(default_factory=list, description="List of reviews")
    total: int = Field(0, description="Total number of reviews")
    page: int = Field(1, description="Current page")
    page_size: int = Field(50, description="Page size")


class EntityReviewStatsResponse(BaseModel):
    """Statistics response for entity reviews."""
    
    total_reviews: int = Field(0, description="Total number of reviews")
    by_status: dict = Field(default_factory=dict, description="Count by review status")
    by_recommendation: dict = Field(default_factory=dict, description="Count by recommendation")
    average_scores: dict = Field(default_factory=dict, description="Average scores per C")
    score_distribution: dict = Field(default_factory=dict, description="Score distribution")


class BulkReviewRequest(BaseModel):
    """Request to review multiple entities."""
    
    entity_ids: List[int] = Field(..., description="List of pending entity IDs to review")
    run_bert_validation: bool = Field(True, description="Run BERT cross-validation")
    run_llm_validation: bool = Field(False, description="Run LLM cross-validation")
