from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, Enum as SQLEnum
from sqlalchemy.orm import declarative_base
from datetime import datetime
import enum

Base = declarative_base()


class EntityStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class LLMConfigDB(Base):
    __tablename__ = "llm_configs"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    base_url = Column(String(500), nullable=False)
    model_name = Column(String(200), nullable=False)
    api_key = Column(String(500), default="EMPTY")
    temperature = Column(Float, default=0.1)
    max_tokens = Column(Integer, default=2048)
    entity_prompt = Column(Text, nullable=True)
    relation_prompt = Column(Text, nullable=True)
    classify_all_prompt = Column(Text, nullable=True)
    kg_samples = Column(Text, nullable=True)
    is_default = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<LLMConfig(id={self.id}, name='{self.name}', model='{self.model_name}')>"


class PendingEntityDB(Base):
    __tablename__ = "pending_entities"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    text = Column(String(500), nullable=False, index=True)
    entity_type = Column(String(50), nullable=False, index=True)
    normalized_text = Column(String(500), nullable=True)
    similarity_score = Column(Float, nullable=True)
    matched_kg_label = Column(String(500), nullable=True)
    matched_kg_uri = Column(String(1000), nullable=True)
    validation_status = Column(String(50), nullable=True)
    source = Column(String(50), nullable=False, default="llm")
    source_model = Column(String(200), nullable=True)
    confidence = Column(Float, nullable=True)
    context = Column(Text, nullable=True)
    status = Column(String(20), default=EntityStatus.PENDING.value)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<PendingEntity(id={self.id}, text='{self.text}', type='{self.entity_type}')>"


class ReviewStatus(str, enum.Enum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"


class EntityReviewDB(Base):
    __tablename__ = "entity_reviews"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    pending_entity_id = Column(Integer, nullable=True, index=True)
    entity_text = Column(String(500), nullable=False, index=True)
    entity_type = Column(String(50), nullable=False, index=True)
    
    congruence_score = Column(Float, nullable=True)
    congruence_nearest_entity = Column(String(500), nullable=True)
    congruence_nearest_uri = Column(String(1000), nullable=True)
    congruence_embedding_distance = Column(Float, nullable=True)
    
    coverage_score = Column(Float, nullable=True)
    coverage_similar_entities_count = Column(Integer, nullable=True)
    coverage_is_novel = Column(Boolean, nullable=True)
    coverage_cluster_id = Column(String(100), nullable=True)
    
    constraint_score = Column(Float, nullable=True)
    constraint_violations = Column(Text, nullable=True)
    constraint_ontology_valid = Column(Boolean, nullable=True)
    constraint_type_valid = Column(Boolean, nullable=True)
    
    completeness_score = Column(Float, nullable=True)
    completeness_has_type = Column(Boolean, nullable=True)
    completeness_has_definition = Column(Boolean, nullable=True)
    completeness_has_normalized_form = Column(Boolean, nullable=True)
    completeness_has_context = Column(Boolean, nullable=True)
    
    consistency_score = Column(Float, nullable=True)
    consistency_type_confidence = Column(Float, nullable=True)
    consistency_alternate_types = Column(Text, nullable=True)
    consistency_bert_agreement = Column(Boolean, nullable=True)
    
    overall_score = Column(Float, nullable=True)
    review_status = Column(String(20), default=ReviewStatus.PENDING.value)
    
    reviewed_by = Column(String(100), nullable=True)
    review_notes = Column(Text, nullable=True)
    recommendation = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<EntityReview(id={self.id}, entity='{self.entity_text}', score={self.overall_score})>"


class EntityConfigDB(Base):
    __tablename__ = "entity_config"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    config_name = Column(String(100), unique=True, nullable=False, default="default")
    
    weight_congruence = Column(Float, nullable=False, default=0.25)
    weight_coverage = Column(Float, nullable=False, default=0.15)
    weight_constraint = Column(Float, nullable=False, default=0.25)
    weight_completeness = Column(Float, nullable=False, default=0.15)
    weight_consistency = Column(Float, nullable=False, default=0.20)
    
    threshold_pass = Column(Float, nullable=False, default=0.75)
    threshold_review = Column(Float, nullable=False, default=0.50)
    
    congruence_min_similarity = Column(Float, nullable=False, default=0.7)
    congruence_exact_match_bonus = Column(Float, nullable=False, default=0.2)
    
    coverage_novelty_threshold = Column(Integer, nullable=False, default=3)
    coverage_novelty_bonus = Column(Float, nullable=False, default=0.3)
    
    constraint_min_length = Column(Integer, nullable=False, default=2)
    constraint_max_length = Column(Integer, nullable=False, default=200)
    constraint_violation_penalty = Column(Float, nullable=False, default=0.25)
    
    completeness_required_fields = Column(Text, nullable=False, default="type,text")
    completeness_optional_weight = Column(Float, nullable=False, default=0.15)
    
    consistency_agreement_bonus = Column(Float, nullable=False, default=0.3)
    consistency_base_score = Column(Float, nullable=False, default=0.5)
    
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<EntityConfig(id={self.id}, name='{self.config_name}')>"



class PendingRelationDB(Base):
    __tablename__ = "pending_relations"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    source_entity = Column(String(500), nullable=False, index=True)
    source_type = Column(String(50), nullable=False)
    relation_type = Column(String(50), nullable=False, index=True)
    target_entity = Column(String(500), nullable=False, index=True)
    target_type = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=True)
    context = Column(Text, nullable=True)
    source = Column(String(50), nullable=False, default="llm")
    source_model = Column(String(200), nullable=True)
    status = Column(String(20), default=EntityStatus.PENDING.value)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<PendingRelation(id={self.id}, {self.source_entity} --{self.relation_type}--> {self.target_entity})>"


class RelationConfigDB(Base):
    __tablename__ = "relation_config"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    config_name = Column(String(100), unique=True, nullable=False, default="default")
    
    weight_congruence = Column(Float, nullable=False, default=0.25)
    weight_coverage = Column(Float, nullable=False, default=0.15)
    weight_constraint = Column(Float, nullable=False, default=0.25)
    weight_completeness = Column(Float, nullable=False, default=0.15)
    weight_consistency = Column(Float, nullable=False, default=0.20)
    
    threshold_pass = Column(Float, nullable=False, default=0.75)
    threshold_review = Column(Float, nullable=False, default=0.50)
    
    congruence_type_penalty = Column(Float, nullable=False, default=0.3)
    congruence_kg_bonus = Column(Float, nullable=False, default=0.2)
    
    coverage_novelty_threshold = Column(Integer, nullable=False, default=3)
    coverage_novelty_bonus = Column(Float, nullable=False, default=0.3)
    
    constraint_violation_penalty = Column(Float, nullable=False, default=0.25)
    constraint_min_confidence = Column(Float, nullable=False, default=0.3)
    constraint_require_context = Column(Boolean, nullable=False, default=False)
    
    completeness_required_fields = Column(Text, nullable=False, default="source_entity,target_entity,relation_type,source_type,target_type")
    completeness_optional_weight = Column(Float, nullable=False, default=0.15)
    
    consistency_agreement_bonus = Column(Float, nullable=False, default=0.3)
    consistency_base_score = Column(Float, nullable=False, default=0.5)
    
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<RelationConfig(id={self.id}, name='{self.config_name}')>"


class TripleStoreConfigDB(Base):
    __tablename__ = "triple_store_configs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    store_type = Column(String(20), nullable=False, default="internal")

    ttl_directory = Column(String(500), nullable=True)
    use_cache = Column(Boolean, default=True)

    sparql_query_endpoint = Column(String(500), nullable=True)
    sparql_update_endpoint = Column(String(500), nullable=True)
    sparql_gsp_endpoint = Column(String(500), nullable=True)
    auth_username = Column(String(200), nullable=True)
    auth_password = Column(String(200), nullable=True)
    named_graph = Column(String(500), nullable=True)

    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<TripleStoreConfig(id={self.id}, name='{self.name}', type='{self.store_type}')>"


class RelationReviewDB(Base):
    __tablename__ = "relation_reviews"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    pending_relation_id = Column(Integer, nullable=True, index=True)
    source_entity = Column(String(500), nullable=False, index=True)
    source_type = Column(String(50), nullable=False)
    relation_type = Column(String(50), nullable=False)
    target_entity = Column(String(500), nullable=False)
    target_type = Column(String(50), nullable=False)
    
    congruence_score = Column(Float, nullable=True)
    
    coverage_score = Column(Float, nullable=True)
    
    constraint_score = Column(Float, nullable=True)
    
    completeness_score = Column(Float, nullable=True)
    
    consistency_score = Column(Float, nullable=True)
    
    overall_score = Column(Float, nullable=True)
    review_status = Column(String(20), default=ReviewStatus.PENDING.value)
    recommendation = Column(String(50), nullable=True)
    review_notes = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<RelationReview(id={self.id}, {self.source_entity} --{self.relation_type}--> {self.target_entity})>"

