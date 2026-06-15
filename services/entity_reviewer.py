import logging
import json
import re
from typing import Optional, List, Dict, Tuple
from datetime import datetime

from models.entities import (
    MedicalEntityType,
    CongruenceMetrics,
    CoverageMetrics,
    ConstraintMetrics,
    CompletenessMetrics,
    ConsistencyMetrics,
    EntityReviewResponse,
    ReviewStatus,
)

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "congruence": 0.25,
    "coverage": 0.15,
    "constraint": 0.25,
    "completeness": 0.15,
    "consistency": 0.20,
}

SCORE_WEIGHTS = DEFAULT_WEIGHTS
PASS_THRESHOLD = 0.75
NEEDS_REVIEW_THRESHOLD = 0.50
FAIL_THRESHOLD = 0.25

DEFAULT_THRESHOLDS = {
    "pass_threshold": 0.75,
    "review_threshold": 0.50,
}

DEFAULT_SETTINGS = {
    "congruence_min_similarity": 0.7,
    "congruence_exact_match_bonus": 0.2,
    "coverage_novelty_threshold": 3,
    "coverage_novelty_bonus": 0.3,
    "constraint_min_length": 2,
    "constraint_max_length": 200,
    "constraint_violation_penalty": 0.25,
    "completeness_required_fields": "type,text",
    "completeness_optional_weight": 0.15,
    "consistency_agreement_bonus": 0.3,
    "consistency_base_score": 0.5,
}

VALID_ENTITY_TYPES = [e.value for e in MedicalEntityType]


class EntityReviewService:
    
    def __init__(self, kg_service=None, bert_service=None):

        self.kg_service = kg_service
        self.bert_service = bert_service
        self._entity_type_cache: Dict[str, List[str]] = {}
        self._config_cache = None
        self._config_loaded_at = None
    
    async def _get_config(self) -> Dict:

        import asyncio
        from datetime import datetime, timedelta
        
        if self._config_cache and self._config_loaded_at:
            if datetime.utcnow() - self._config_loaded_at < timedelta(seconds=60):
                return self._config_cache
        
        try:
            from database.connection import async_session_maker
            from database.models import EntityConfigDB
            from sqlalchemy import select
            
            async with async_session_maker() as db:
                result = await db.execute(
                    select(EntityConfigDB).where(EntityConfigDB.is_active == True)
                )
                config = result.scalar_one_or_none()
                
                if config:
                    self._config_cache = {
                        "weights": {
                            "congruence": config.weight_congruence,
                            "coverage": config.weight_coverage,
                            "constraint": config.weight_constraint,
                            "completeness": config.weight_completeness,
                            "consistency": config.weight_consistency,
                        },
                        "thresholds": {
                            "pass_threshold": config.threshold_pass,
                            "review_threshold": config.threshold_review,
                        },
                        "settings": {
                            "congruence_min_similarity": config.congruence_min_similarity,
                            "congruence_exact_match_bonus": config.congruence_exact_match_bonus,
                            "coverage_novelty_threshold": config.coverage_novelty_threshold,
                            "coverage_novelty_bonus": config.coverage_novelty_bonus,
                            "constraint_min_length": config.constraint_min_length,
                            "constraint_max_length": config.constraint_max_length,
                            "constraint_violation_penalty": config.constraint_violation_penalty,
                            "completeness_required_fields": config.completeness_required_fields,
                            "completeness_optional_weight": config.completeness_optional_weight,
                            "consistency_agreement_bonus": config.consistency_agreement_bonus,
                            "consistency_base_score": config.consistency_base_score,
                        }
                    }
                    self._config_loaded_at = datetime.utcnow()
                    return self._config_cache
        except Exception as e:
            logger.warning(f"Could not load config from DB, using defaults: {e}")
        
        return {
            "weights": DEFAULT_WEIGHTS,
            "thresholds": DEFAULT_THRESHOLDS,
            "settings": DEFAULT_SETTINGS,
        }
    
    async def evaluate_entity(
        self,
        entity_text: str,
        entity_type: str,
        normalized_form: Optional[str] = None,
        context: Optional[str] = None,
        confidence: Optional[float] = None,
        source: Optional[str] = None,
        run_bert_validation: bool = True,
        run_llm_validation: bool = False
    ) -> Dict:

        logger.info(f"Evaluating entity: '{entity_text}' (type: {entity_type})")
        
        config = await self._get_config()
        settings = config["settings"]
        
        kg_matches = None
        if self.kg_service:
            try:
                kg_matches = await self.kg_service.find_matches_async(entity_text, max_matches=10, min_score=0.5)
            except Exception as e:
                logger.warning(f"Error fetching KG matches: {e}")
                kg_matches = []
        
        congruence = await self._evaluate_congruence(entity_text, entity_type, settings, kg_matches=kg_matches)
        
        coverage = await self._evaluate_coverage(entity_text, entity_type, settings, kg_matches=kg_matches)
        
        constraint = await self._evaluate_constraint(entity_text, entity_type, settings)
        
        completeness = self._evaluate_completeness(
            entity_text, entity_type, normalized_form, context, confidence, settings
        )
        
        consistency = await self._evaluate_consistency(
            entity_text, entity_type, source, confidence,
            run_bert_validation, run_llm_validation, settings
        )
        
        overall_score = self._calculate_overall_score(
            congruence.score,
            coverage.score,
            constraint.score,
            completeness.score,
            consistency.score,
            config["weights"]
        )
        
        review_status, recommendation = self._determine_status_and_recommendation(
            overall_score, congruence, constraint, consistency, config["thresholds"]
        )
        
        return {
            "entity_text": entity_text,
            "entity_type": entity_type,
            "congruence": congruence,
            "coverage": coverage,
            "constraint": constraint,
            "completeness": completeness,
            "consistency": consistency,
            "overall_score": overall_score,
            "review_status": review_status,
            "recommendation": recommendation,
            "evaluated_at": datetime.utcnow().isoformat()
        }
    
    async def _evaluate_congruence(
        self,
        entity_text: str,
        entity_type: str,
        settings: Dict,
        kg_matches: list = None
    ) -> CongruenceMetrics:

        nearest_entity = None
        nearest_uri = None
        embedding_distance = None
        score = 0.5
        method = "sequence_matcher"
        
        min_similarity = settings.get("congruence_min_similarity", 0.7)
        exact_match_bonus = settings.get("congruence_exact_match_bonus", 0.2)
        
        if kg_matches is not None:
            matches = [m for m in kg_matches if m.similarity_score >= min_similarity][:3]
            if matches:
                best_match = matches[0]
                nearest_entity = best_match.kg_label
                nearest_uri = best_match.kg_uri
                score = best_match.similarity_score
                embedding_distance = 1.0 - score
                if score >= 0.95:
                    score = min(1.0, score + exact_match_bonus)
            else:
                score = 0.5
        elif self.kg_service:
            try:
                matches = await self.kg_service.find_matches_async(entity_text, max_matches=3, min_score=min_similarity)
                
                if matches:
                    best_match = matches[0]
                    nearest_entity = best_match.kg_label
                    nearest_uri = best_match.kg_uri
                    
                    score = best_match.similarity_score
                    embedding_distance = 1.0 - score
                    
                    if score >= 0.95:
                        score = min(1.0, score + exact_match_bonus)
                else:
                    score = 0.5
                    
            except Exception as e:
                logger.warning(f"Error evaluating congruence: {e}")
        
        return CongruenceMetrics(
            score=score,
            nearest_entity=nearest_entity,
            nearest_uri=nearest_uri,
            embedding_distance=embedding_distance,
            method=method
        )
    
    async def _evaluate_coverage(
        self,
        entity_text: str,
        entity_type: str,
        settings: Dict,
        kg_matches: list = None
    ) -> CoverageMetrics:

        similar_count = 0
        is_novel = True
        fills_gap = False
        coverage_increase = 0.0
        score = 0.8
        
        novelty_threshold = settings.get("coverage_novelty_threshold", 3)
        novelty_bonus = settings.get("coverage_novelty_bonus", 0.3)
        
        matches = kg_matches
        if matches is None and self.kg_service:
            try:
                matches = await self.kg_service.find_matches_async(entity_text, max_matches=10, min_score=0.5)
            except Exception as e:
                logger.warning(f"Error evaluating coverage: {e}")
                matches = None
        
        if matches is not None:
            similar_count = len(matches)
                
            if similar_count == 0:
                is_novel = True
                fills_gap = True
                score = 0.5 + novelty_bonus
                coverage_increase = 5.0
            elif similar_count < novelty_threshold:
                best_similarity = matches[0].similarity_score if matches else 0
                if best_similarity < 0.8:
                    is_novel = True
                    fills_gap = True
                    score = 0.5 + (novelty_bonus * 0.7)
                    coverage_increase = 3.0
                else:
                    is_novel = False
                    fills_gap = False
                    score = 0.5
            else:
                is_novel = False
                fills_gap = False
                best_similarity = matches[0].similarity_score
                score = max(0.3, 0.6 - (0.3 * best_similarity))
        
        return CoverageMetrics(
            score=score,
            similar_entities_count=similar_count,
            is_novel=is_novel,
            fills_gap=fills_gap,
            coverage_increase=coverage_increase
        )
    
    async def _evaluate_constraint(
        self,
        entity_text: str,
        entity_type: str,
        settings: Dict
    ) -> ConstraintMetrics:

        violations = []
        ontology_valid = None
        type_valid = True
        format_valid = True
        score = 1.0
        
        min_length = settings.get("constraint_min_length", 2)
        max_length = settings.get("constraint_max_length", 200)
        violation_penalty = settings.get("constraint_violation_penalty", 0.15)
        
        if entity_type.lower() not in VALID_ENTITY_TYPES:
            type_valid = False
            violations.append(f"Invalid entity type: {entity_type}")
            score -= 0.3
        
        if len(entity_text) < min_length:
            format_valid = False
            violations.append(f"Entity text too short (< {min_length} characters)")
            shortness = 1.0 - (len(entity_text.strip()) / min_length)
            score -= 0.2 + 0.6 * shortness
        
        if len(entity_text) > max_length:
            format_valid = False
            violations.append(f"Entity text too long (> {max_length} characters)")
            score -= 0.1
        
        stripped = entity_text.strip()
        if len(stripped) <= 2 and not stripped.isalpha():
            format_valid = False
            violations.append("Entity is a single symbol or punctuation, not a valid medical term")
            score -= 0.5
        
        if re.search(r'[<>{}|\[\]\\]', entity_text):
            format_valid = False
            violations.append("Entity contains invalid characters")
            score -= violation_penalty
        
        if entity_type.lower() != "quantitative_measure" and re.match(r'^[\d\s.,-]+$', entity_text):
            violations.append("Non-quantitative entity contains only numbers")
            score -= 0.2
        
        type_violations = self._check_type_constraints(entity_text, entity_type)
        violations.extend(type_violations)
        score -= violation_penalty * len(type_violations)
        
        score = max(0.0, min(1.0, score))
        
        return ConstraintMetrics(
            score=score,
            violations=violations,
            ontology_valid=ontology_valid,
            type_valid=type_valid,
            format_valid=format_valid,
            clinical_plausibility=score
        )
    
    def _check_type_constraints(self, entity_text: str, entity_type: str) -> List[str]:
        violations = []
        text_lower = entity_text.lower()
        words = text_lower.split()
        word_count = len(words)
        
        population_indicators = ['patients', 'subjects', 'individuals', 'participants', 
                                 'cohort', 'group', 'male', 'female', 'adults', 'children',
                                 'men', 'women', 'cases', 'controls']
        population_word_count = sum(1 for w in words if w in population_indicators)
        
        if population_word_count >= 2 or (word_count > 5 and population_word_count >= 1):
            if entity_type.lower() in ['finding', 'disease', 'substance']:
                violations.append(f"Text appears to describe a patient population, not a {entity_type}")
        
        if word_count > 8 and entity_type.lower() in ['finding', 'disease', 'substance']:
            violations.append(f"Entity text too complex for type '{entity_type}' ({word_count} words)")
        
        if entity_type.lower() == "quantitative_measure":
            if not re.search(r'\d', entity_text):
                violations.append("Quantitative measure missing numeric value")
        
        elif entity_type.lower() == "finding":
            finding_indicators = ['pain', 'ache', 'fever', 'cough', 'symptom', 'sign',
                                  'elevated', 'decreased', 'abnormal', 'normal', 'positive',
                                  'negative', 'swelling', 'rash', 'bleeding', 'dysfunction']
            has_finding_indicator = any(ind in text_lower for ind in finding_indicators)
            
            if word_count > 4 and not has_finding_indicator and 'with' in text_lower:
                violations.append("Finding classification uncertain - text pattern suggests population description")
        
        elif entity_type.lower() == "procedure":
            procedure_indicators = ['scan', 'test', 'surgery', 'biopsy', 'exam', 
                                    'therapy', 'imaging', 'procedure', 'analysis',
                                    '-scopy', '-ectomy', '-plasty', '-gram']
            if not any(ind in text_lower for ind in procedure_indicators):
                pass
        
        elif entity_type.lower() == "substance":
            generic_terms = ['the', 'a', 'an', 'this', 'that']
            if text_lower in generic_terms:
                violations.append("Substance appears too generic")
        
        elif entity_type.lower() == "disease":
            if 'patient' in text_lower or 'subject' in text_lower:
                violations.append("Disease classification contains patient reference")
        
        return violations
    
    def _evaluate_completeness(
        self,
        entity_text: str,
        entity_type: str,
        normalized_form: Optional[str],
        context: Optional[str],
        confidence: Optional[float],
        settings: Dict
    ) -> CompletenessMetrics:

        missing_fields = []
        
        required_fields_str = settings.get("completeness_required_fields", "type,text")
        optional_weight = settings.get("completeness_optional_weight", 0.3)
        
        if isinstance(required_fields_str, str):
            required_fields = [f.strip().lower() for f in required_fields_str.split(",") if f.strip()]
        else:
            required_fields = ["type", "text"]
        
        required_fields_weight = 1.0 - optional_weight
        
        min_length = settings.get("constraint_min_length", 2)
        
        has_type = bool(entity_type)
        has_definition = bool(context)
        has_normalized_form = bool(normalized_form)
        has_context = bool(context)
        has_confidence = confidence is not None
        
        text_len = len(entity_text.strip()) if entity_text else 0
        if text_len == 0:
            has_text = False
        elif text_len < min_length:
            has_text = False
            missing_fields.append("text (too short to be a valid entity)")
        else:
            has_text = True
        
        field_status = {
            "type": has_type,
            "text": has_text,
            "context": has_context,
            "definition": has_definition,
            "normalized_form": has_normalized_form,
            "confidence": has_confidence,
        }
        
        required_present = 0
        total_required = len(required_fields)
        for field in required_fields:
            if field in field_status:
                if field_status[field]:
                    required_present += 1
                else:
                    missing_fields.append(field)
            else:
                missing_fields.append(field)
        
        required_score = required_present / total_required if total_required > 0 else 1.0
        
        optional_fields = [f for f in field_status.keys() if f not in required_fields]
        optional_present = sum(1 for f in optional_fields if field_status.get(f, False))
        optional_score = optional_present / len(optional_fields) if optional_fields else 1.0
        
        score = (required_score * required_fields_weight) + (optional_score * optional_weight)
        
        return CompletenessMetrics(
            score=score,
            has_type=has_type,
            has_definition=has_definition,
            has_normalized_form=has_normalized_form,
            has_context=has_context,
            has_confidence=has_confidence,
            missing_fields=missing_fields
        )
    
    async def _evaluate_consistency(
        self,
        entity_text: str,
        entity_type: str,
        source: Optional[str],
        confidence: Optional[float],
        run_bert_validation: bool,
        run_llm_validation: bool,
        settings: Dict
    ) -> ConsistencyMetrics:

        agreement_bonus = settings.get("consistency_agreement_bonus", 0.2)
        base_score = settings.get("consistency_base_score", 0.5)
        
        type_confidence = confidence if confidence is not None else base_score
        alternate_types = []
        bert_agreement = None
        llm_agreement = None
        cross_validation_score = None
        
        validation_available = run_bert_validation and self.bert_service
        
        if validation_available:
            score = type_confidence
        else:
            score = min(0.8, type_confidence)
        
        if run_bert_validation and self.bert_service:
            try:
                bert_result = await self._run_bert_classification(entity_text)
                if bert_result:
                    bert_type = bert_result.get("entity_type", "").lower()
                    bert_confidence = bert_result.get("confidence", 0.0)
                    bert_agreement = bert_type == entity_type.lower()
                    
                    if bert_agreement:
                        bonus = agreement_bonus * bert_confidence
                        score = min(1.0, score + bonus)
                    else:
                        penalty = agreement_bonus * 0.75 * bert_confidence
                        score = max(0.0, score - penalty)
                        alternate_types.append({
                            "type": bert_type,
                            "source": "bert",
                            "confidence": bert_confidence
                        })
                else:
                    score = max(0.0, score - 0.1)
            except Exception as e:
                logger.warning(f"BERT cross-validation failed: {e}")
                score = max(0.0, score - 0.05)
        
        agreements = [a for a in [bert_agreement, llm_agreement] if a is not None]
        if agreements:
            cross_validation_score = sum(1 for a in agreements if a) / len(agreements)
        
        return ConsistencyMetrics(
            score=min(1.0, max(0.0, score)),
            type_confidence=type_confidence,
            alternate_types=alternate_types,
            bert_agreement=bert_agreement,
            llm_agreement=llm_agreement,
            cross_validation_score=cross_validation_score
        )
    
    async def _run_bert_classification(self, text: str) -> Optional[Dict]:
        if not self.bert_service:
            return None
        
        try:
            result = self.bert_service.classify_text(text)
            if result and result.entities:
                best_entity = max(result.entities, key=lambda e: e.confidence)
                return {
                    "entity_type": best_entity.entity_type.value if hasattr(best_entity.entity_type, 'value') else str(best_entity.entity_type),
                    "confidence": best_entity.confidence
                }
        except Exception as e:
            logger.warning(f"BERT classification error: {e}")
        
        return None
    
    def _calculate_overall_score(
        self,
        congruence: float,
        coverage: float,
        constraint: float,
        completeness: float,
        consistency: float,
        weights: Dict
    ) -> float:
        overall = (
            congruence * weights.get("congruence", SCORE_WEIGHTS["congruence"]) +
            coverage * weights.get("coverage", SCORE_WEIGHTS["coverage"]) +
            constraint * weights.get("constraint", SCORE_WEIGHTS["constraint"]) +
            completeness * weights.get("completeness", SCORE_WEIGHTS["completeness"]) +
            consistency * weights.get("consistency", SCORE_WEIGHTS["consistency"])
        )
        return round(overall, 3)
    
    def _determine_status_and_recommendation(
        self,
        overall_score: float,
        congruence: CongruenceMetrics,
        constraint: ConstraintMetrics,
        consistency: ConsistencyMetrics,
        thresholds: Dict
    ) -> Tuple[str, str]:
        
        pass_threshold = thresholds.get("pass_threshold", PASS_THRESHOLD)
        review_threshold = thresholds.get("review_threshold", NEEDS_REVIEW_THRESHOLD)
        fail_threshold = review_threshold / 2
        
        if constraint.score < 0.4:
            return ReviewStatus.FAILED.value, "reject"
        
        if not constraint.type_valid:
            return ReviewStatus.FAILED.value, "reject"
        
        if overall_score >= pass_threshold:
            return ReviewStatus.PASSED.value, "approve"
        
        if overall_score >= review_threshold:
            if consistency.bert_agreement is False:
                return ReviewStatus.NEEDS_REVIEW.value, "verify_type"
            if congruence.score > 0.95:
                return ReviewStatus.NEEDS_REVIEW.value, "check_duplicate"
            return ReviewStatus.NEEDS_REVIEW.value, "manual_review"
        
        if overall_score < fail_threshold:
            return ReviewStatus.FAILED.value, "reject"
        
        return ReviewStatus.NEEDS_REVIEW.value, "manual_review"
    
    def get_score_explanation(self, evaluation: Dict) -> str:
        explanations = []
        
        overall = evaluation.get("overall_score", 0)
        status = evaluation.get("review_status", "unknown")
        rec = evaluation.get("recommendation", "unknown")
        explanations.append(f"Overall Score: {overall:.2%} ({status})")
        explanations.append(f"Recommendation: {rec}")
        explanations.append("")
        
        cong = evaluation.get("congruence", {})
        if isinstance(cong, CongruenceMetrics):
            cong = cong.model_dump()
        explanations.append(f"1. CONGRUENCE (Semantic Alignment): {cong.get('score', 0):.2%}")
        if cong.get("nearest_entity"):
            explanations.append(f"   - Nearest KG entity: {cong['nearest_entity']}")
        
        cov = evaluation.get("coverage", {})
        if isinstance(cov, CoverageMetrics):
            cov = cov.model_dump()
        explanations.append(f"2. COVERAGE (Representativeness): {cov.get('score', 0):.2%}")
        explanations.append(f"   - Is novel: {cov.get('is_novel', False)}")
        explanations.append(f"   - Similar entities in KG: {cov.get('similar_entities_count', 0)}")
        
        cons = evaluation.get("constraint", {})
        if isinstance(cons, ConstraintMetrics):
            cons = cons.model_dump()
        explanations.append(f"3. CONSTRAINT (Clinical Validity): {cons.get('score', 0):.2%}")
        violations = cons.get("violations", [])
        if violations:
            for v in violations:
                explanations.append(f"   ⚠ {v}")
        
        comp = evaluation.get("completeness", {})
        if isinstance(comp, CompletenessMetrics):
            comp = comp.model_dump()
        explanations.append(f"4. COMPLETENESS (Required Attributes): {comp.get('score', 0):.2%}")
        missing = comp.get("missing_fields", [])
        if missing:
            explanations.append(f"   - Missing: {', '.join(missing)}")
        
        cons2 = evaluation.get("consistency", {})
        if isinstance(cons2, ConsistencyMetrics):
            cons2 = cons2.model_dump()
        explanations.append(f"5. CONSISTENCY (Type Coherence): {cons2.get('score', 0):.2%}")
        if cons2.get("bert_agreement") is not None:
            explanations.append(f"   - BERT agreement: {'✓' if cons2['bert_agreement'] else '✗'}")
        
        return "\n".join(explanations)
