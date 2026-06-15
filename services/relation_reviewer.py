
import asyncio
import logging
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

VALID_RELATION_COMBOS = {
    "has_symptom": [
        ("disease", "symptom"),
        ("disease", "finding"),
    ],
    "has_finding": [
        ("disease", "finding"),
        ("disease", "symptom"),
        ("imaging_procedure", "finding"),
        ("examination_procedure", "finding"),
    ],
    "suggests": [
        ("finding", "disease"),
        ("symptom", "disease"),
        ("imaging_result", "disease"),
        ("examination_measure", "disease"),
        ("parameter", "disease"),
    ],
    "located_in": [
        ("disease", "organ"),
        ("finding", "organ"),
        ("symptom", "organ"),
        ("imaging_result", "organ"),
        ("therapeutic_procedure", "organ"),
    ],
    "indicated_for": [
        ("imaging_procedure", "disease"),
        ("examination_procedure", "disease"),
        ("therapeutic_procedure", "disease"),
        ("substance", "disease"),
    ],
    "produces_result": [
        ("imaging_procedure", "imaging_result"),
        ("examination_procedure", "examination_measure"),
        ("examination_procedure", "finding"),
        ("imaging_procedure", "finding"),
    ],
    "treats": [
        ("substance", "disease"),
        ("therapy", "disease"),
        ("therapeutic_procedure", "disease"),
        ("substance", "symptom"),
    ],
    "first_line_for": [
        ("substance", "disease"),
        ("therapy", "disease"),
    ],
    "contraindicated_in": [
        ("substance", "disease"),
        ("therapeutic_procedure", "disease"),
        ("substance", "adverse_event"),
    ],
    "rules_out": [
        ("finding", "disease"),
        ("imaging_result", "disease"),
        ("examination_measure", "disease"),
        ("parameter", "disease"),
    ],
    "assesses": [
        ("score", "disease"),
        ("parameter", "disease"),
        ("examination_procedure", "disease"),
        ("examination_measure", "disease"),
    ],
    "affects": [
        ("disease", "organ"),
        ("substance", "organ"),
        ("adverse_event", "organ"),
        ("disease", "parameter"),
    ],
}

DEFAULT_WEIGHTS = {
    "congruence": 0.25,
    "coverage": 0.15,
    "constraint": 0.25,
    "completeness": 0.15,
    "consistency": 0.20,
}

DEFAULT_THRESHOLDS = {
    "pass_threshold": 0.75,
    "review_threshold": 0.50,
}

DEFAULT_SETTINGS = {
    "congruence_type_penalty": 0.3,
    "congruence_kg_bonus": 0.2,
    "coverage_novelty_threshold": 3,
    "coverage_novelty_bonus": 0.3,
    "constraint_violation_penalty": 0.25,
    "constraint_min_confidence": 0.3,
    "constraint_require_context": False,
    "completeness_required_fields": "source_entity,target_entity,relation_type,source_type,target_type",
    "completeness_optional_weight": 0.15,
    "consistency_agreement_bonus": 0.3,
    "consistency_base_score": 0.5,
}

VALID_RELATION_TYPES = [
    "has_symptom", "has_finding", "suggests", "located_in",
    "indicated_for", "produces_result", "treats", "first_line_for",
    "contraindicated_in", "rules_out", "assesses", "affects",
]

VALID_ENTITY_TYPES = [
    "disease", "symptom", "finding", "organ",
    "imaging_procedure", "examination_procedure", "therapeutic_procedure",
    "imaging_result", "examination_measure", "parameter",
    "score", "therapy", "substance", "adverse_event",
]


class RelationReviewService:
    
    def __init__(self, kg_service=None):
        self.kg_service = kg_service
        self._config_cache = None
        self._config_loaded_at = None
    
    async def _get_config(self) -> Dict:

        if self._config_cache and self._config_loaded_at:
            if datetime.utcnow() - self._config_loaded_at < timedelta(seconds=60):
                return self._config_cache
        
        try:
            from database.connection import async_session_maker
            from database.models import RelationConfigDB
            from sqlalchemy import select
            
            async with async_session_maker() as db:
                result = await db.execute(
                    select(RelationConfigDB).where(RelationConfigDB.is_active == True)
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
                        }
                    }
                    self._config_loaded_at = datetime.utcnow()
                    return self._config_cache
        except Exception as e:
            logger.warning(f"Could not load relation config from DB, using defaults: {e}")
        
        return {
            "weights": DEFAULT_WEIGHTS,
            "thresholds": DEFAULT_THRESHOLDS,
            "settings": DEFAULT_SETTINGS,
        }
    
    async def evaluate_relation(
        self,
        source_entity: str,
        source_type: str,
        relation_type: str,
        target_entity: str,
        target_type: str,
        confidence: Optional[float] = None,
        context: Optional[str] = None,
        source: Optional[str] = None,
    ) -> Dict:

        logger.info(f"Evaluating relation: {source_entity} --{relation_type}--> {target_entity}")
        
        config = await self._get_config()
        settings = config["settings"]
        
        congruence_score = self._evaluate_congruence(
            source_type, relation_type, target_type, settings
        )
        
        coverage_score = await self._evaluate_coverage(
            source_entity, relation_type, target_entity, settings
        )
        
        constraint_score = self._evaluate_constraint(
            source_entity, source_type, relation_type,
            target_entity, target_type, confidence, context, settings
        )
        
        completeness_score = self._evaluate_completeness(
            source_entity, source_type, relation_type,
            target_entity, target_type, confidence, context, settings
        )
        
        consistency_score = self._evaluate_consistency(
            source, confidence, settings
        )
        
        weights = config["weights"]
        overall_score = (
            congruence_score * weights.get("congruence", 0.25) +
            coverage_score * weights.get("coverage", 0.15) +
            constraint_score * weights.get("constraint", 0.25) +
            completeness_score * weights.get("completeness", 0.15) +
            consistency_score * weights.get("consistency", 0.20)
        )
        overall_score = round(overall_score, 3)
        
        thresholds = config["thresholds"]
        review_status, recommendation = self._determine_status(
            overall_score, congruence_score, constraint_score, thresholds
        )
        
        return {
            "congruence_score": round(congruence_score, 3),
            "coverage_score": round(coverage_score, 3),
            "constraint_score": round(constraint_score, 3),
            "completeness_score": round(completeness_score, 3),
            "consistency_score": round(consistency_score, 3),
            "overall_score": overall_score,
            "review_status": review_status,
            "recommendation": recommendation,
        }
    
    def _evaluate_congruence(
        self,
        source_type: str,
        relation_type: str,
        target_type: str,
        settings: Dict,
    ) -> float:

        type_penalty = settings.get("congruence_type_penalty", 0.3)
        kg_bonus = settings.get("congruence_kg_bonus", 0.2)
        
        score = 0.5
        
        src = source_type.lower().strip()
        tgt = target_type.lower().strip()
        rel = relation_type.lower().strip()
        
        if rel not in VALID_RELATION_COMBOS:
            return max(0.0, score - type_penalty)
        
        valid_combos = VALID_RELATION_COMBOS[rel]
        
        if (src, tgt) in valid_combos:
            score = 0.9
            if self.kg_service:
                try:
                    score = min(1.0, score + kg_bonus * 0.5)
                except Exception:
                    pass
            return score
        
        valid_sources = {combo[0] for combo in valid_combos}
        valid_targets = {combo[1] for combo in valid_combos}
        
        if src in valid_sources and tgt in valid_targets:
            score = 0.6
        elif src in valid_sources or tgt in valid_targets:
            score = 0.4
        else:
            score = max(0.0, 0.3 - type_penalty)
        
        return max(0.0, min(1.0, score))
    
    async def _evaluate_coverage(
        self,
        source_entity: str,
        relation_type: str,
        target_entity: str,
        settings: Dict,
    ) -> float:

        novelty_threshold = settings.get("coverage_novelty_threshold", 3)
        novelty_bonus = settings.get("coverage_novelty_bonus", 0.3)
        
        score = 0.7
        
        if self.kg_service:
            try:
                similar_count = 0
                src_matches, tgt_matches = await asyncio.gather(
                    self.kg_service.find_matches_async(source_entity, max_matches=5, min_score=0.7),
                    self.kg_service.find_matches_async(target_entity, max_matches=5, min_score=0.7)
                )
                
                if src_matches and tgt_matches:
                    similar_count = min(len(src_matches), len(tgt_matches))
                
                if similar_count == 0:
                    score = 0.5 + novelty_bonus
                elif similar_count < novelty_threshold:
                    score = 0.5 + novelty_bonus * 0.5
                else:
                    score = max(0.3, 0.6 - 0.1 * similar_count)
                    
            except Exception as e:
                logger.warning(f"Error evaluating coverage: {e}")
                score = 0.6
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_constraint(
        self,
        source_entity: str,
        source_type: str,
        relation_type: str,
        target_entity: str,
        target_type: str,
        confidence: Optional[float],
        context: Optional[str],
        settings: Dict,
    ) -> float:

        violation_penalty = settings.get("constraint_violation_penalty", 0.25)
        min_confidence = settings.get("constraint_min_confidence", 0.3)
        require_context = settings.get("constraint_require_context", False)
        
        score = 1.0
        violations = []
        
        if relation_type.lower() not in VALID_RELATION_TYPES:
            violations.append(f"Invalid relation type: {relation_type}")
            score -= 0.3
        
        if source_type.lower() not in VALID_ENTITY_TYPES:
            violations.append(f"Invalid source type: {source_type}")
            score -= violation_penalty
        
        if target_type.lower() not in VALID_ENTITY_TYPES:
            violations.append(f"Invalid target type: {target_type}")
            score -= violation_penalty
        
        if source_entity.lower().strip() == target_entity.lower().strip():
            violations.append("Self-referencing relation (source == target)")
            score -= 0.3
        
        if len(source_entity.strip()) < 2:
            violations.append("Source entity too short")
            score -= violation_penalty
        
        if len(target_entity.strip()) < 2:
            violations.append("Target entity too short")
            score -= violation_penalty
        
        if confidence is not None and confidence < min_confidence:
            violations.append(f"Confidence {confidence:.2f} below minimum {min_confidence}")
            score -= violation_penalty
        
        if require_context and not context:
            violations.append("Context is required but missing")
            score -= violation_penalty * 0.5
        
        directional_relations = {
            "treats": ("substance|therapy|therapeutic_procedure", "disease|symptom"),
            "first_line_for": ("substance|therapy", "disease"),
            "produces_result": ("imaging_procedure|examination_procedure", "imaging_result|examination_measure|finding"),
            "assesses": ("score|parameter|examination_procedure|examination_measure", "disease"),
        }
        
        if relation_type.lower() in directional_relations:
            valid_src_pattern, valid_tgt_pattern = directional_relations[relation_type.lower()]
            src_valid = source_type.lower() in valid_src_pattern.split("|")
            tgt_valid = target_type.lower() in valid_tgt_pattern.split("|")
            
            if not src_valid or not tgt_valid:
                src_rev = source_type.lower() in valid_tgt_pattern.split("|")
                tgt_rev = target_type.lower() in valid_src_pattern.split("|")
                if src_rev and tgt_rev:
                    violations.append(f"Directionality reversed for '{relation_type}'")
                    score -= violation_penalty
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_completeness(
        self,
        source_entity: str,
        source_type: str,
        relation_type: str,
        target_entity: str,
        target_type: str,
        confidence: Optional[float],
        context: Optional[str],
        settings: Dict,
    ) -> float:
 
        required_fields_str = settings.get(
            "completeness_required_fields",
            "source_entity,target_entity,relation_type,source_type,target_type"
        )
        optional_weight = settings.get("completeness_optional_weight", 0.15)
        
        if isinstance(required_fields_str, str):
            required_fields = [f.strip().lower() for f in required_fields_str.split(",") if f.strip()]
        else:
            required_fields = ["source_entity", "target_entity", "relation_type", "source_type", "target_type"]
        
        field_values = {
            "source_entity": source_entity,
            "source_type": source_type,
            "relation_type": relation_type,
            "target_entity": target_entity,
            "target_type": target_type,
            "confidence": confidence,
            "context": context,
        }
        
        required_present = 0
        total_required = len(required_fields)
        for field in required_fields:
            val = field_values.get(field)
            if val is not None and (not isinstance(val, str) or val.strip()):
                required_present += 1
        
        required_score = required_present / total_required if total_required > 0 else 1.0
        
        optional_fields = [f for f in field_values if f not in required_fields]
        optional_present = sum(
            1 for f in optional_fields
            if field_values.get(f) is not None and (
                not isinstance(field_values[f], str) or field_values[f].strip()
            )
        )
        optional_score = optional_present / len(optional_fields) if optional_fields else 1.0
        
        required_weight = 1.0 - optional_weight
        score = (required_score * required_weight) + (optional_score * optional_weight)
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_consistency(
        self,
        source: Optional[str],
        confidence: Optional[float],
        settings: Dict,
    ) -> float:

        agreement_bonus = settings.get("consistency_agreement_bonus", 0.3)
        base_score = settings.get("consistency_base_score", 0.5)
        
        score = confidence if confidence is not None else base_score
        
        if source and source.lower() == "both":
            score = min(1.0, score + agreement_bonus)
        elif source and source.lower() in ("llm", "bert"):
            score = min(0.85, score)
        
        return max(0.0, min(1.0, score))
    
    def _determine_status(
        self,
        overall_score: float,
        congruence_score: float,
        constraint_score: float,
        thresholds: Dict,
    ) -> Tuple[str, str]:

        pass_threshold = thresholds.get("pass_threshold", 0.75)
        review_threshold = thresholds.get("review_threshold", 0.50)
        
        if constraint_score < 0.3:
            return "failed", "reject"
        
        if congruence_score < 0.2:
            return "failed", "reject"
        
        if overall_score >= pass_threshold:
            return "passed", "approve"
        
        if overall_score >= review_threshold:
            if congruence_score < 0.5:
                return "needs_review", "verify_type"
            return "needs_review", "manual_review"
        
        return "failed", "reject"
