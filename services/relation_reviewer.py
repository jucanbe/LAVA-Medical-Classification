
import logging
from typing import Optional, Dict, Tuple
from datetime import datetime, timedelta

from models.entities import normalize_entity_type, normalize_relation_type
from models.review_defaults import (
    RELATION_WEIGHTS,
    RELATION_SETTINGS,
    THRESHOLDS,
    SCORING_VERSION,
    UNVERIFIED_CONFIDENCE_CAP,
)

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

DEFAULT_WEIGHTS = RELATION_WEIGHTS
DEFAULT_THRESHOLDS = THRESHOLDS
DEFAULT_SETTINGS = RELATION_SETTINGS

VALID_RELATION_TYPES = list(VALID_RELATION_COMBOS)

# Relations whose direction is fixed: (allowed source types, allowed target types).
DIRECTIONAL_RELATIONS = {
    "treats": ({"substance", "therapy", "therapeutic_procedure"}, {"disease", "symptom"}),
    "first_line_for": ({"substance", "therapy"}, {"disease"}),
    "produces_result": ({"imaging_procedure", "examination_procedure"}, {"imaging_result", "examination_measure", "finding"}),
    "assesses": ({"score", "parameter", "examination_procedure", "examination_measure"}, {"disease"}),
}

_config_generation = 0


def invalidate_config_cache() -> None:
    """Make every RelationReviewService reload its configuration on next use."""
    global _config_generation
    _config_generation += 1


class RelationReviewService:

    def __init__(self, kg_service=None):
        self.kg_service = kg_service
        self._config_cache = None
        self._config_loaded_at = None
        self._config_generation = None

    async def _get_config(self) -> Dict:

        if (
            self._config_cache
            and self._config_loaded_at
            and self._config_generation == _config_generation
            and datetime.utcnow() - self._config_loaded_at < timedelta(seconds=60)
        ):
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
                    self._config_cache = self._build_config(config)
                    self._config_loaded_at = datetime.utcnow()
                    self._config_generation = _config_generation
                    return self._config_cache
        except Exception as e:
            logger.warning(f"Could not load relation review config from DB, using defaults: {e}")

        return {
            "weights": dict(DEFAULT_WEIGHTS),
            "thresholds": dict(DEFAULT_THRESHOLDS),
            "settings": dict(DEFAULT_SETTINGS),
        }

    @staticmethod
    def _build_config(config) -> Dict:
        """Merge a config row over the defaults; NULL columns fall back to defaults."""
        weights = {k: getattr(config, f"weight_{k}") for k in DEFAULT_WEIGHTS}
        thresholds = {
            "pass_threshold": config.threshold_pass,
            "review_threshold": config.threshold_review,
        }
        settings = {k: getattr(config, k) for k in DEFAULT_SETTINGS}
        return {
            "weights": {k: v if v is not None else DEFAULT_WEIGHTS[k] for k, v in weights.items()},
            "thresholds": {k: v if v is not None else DEFAULT_THRESHOLDS[k] for k, v in thresholds.items()},
            "settings": {k: v if v is not None else DEFAULT_SETTINGS[k] for k, v in settings.items()},
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

        logger.info(
            f"Evaluating relation from {source or 'unknown source'}: "
            f"{source_entity} --{relation_type}--> {target_entity}"
        )

        config = await self._get_config()
        settings = config["settings"]

        kg_evidence = self._lookup_triple(
            source_entity, source_type, relation_type, target_entity, target_type
        )

        congruence_score = self._evaluate_congruence(
            source_type, relation_type, target_type, settings, kg_evidence
        )

        coverage_score = self._evaluate_coverage(kg_evidence, settings)

        constraint_score = self._evaluate_constraint(
            source_entity, source_type, relation_type,
            target_entity, target_type, confidence, context, settings
        )

        completeness_score = self._evaluate_completeness(
            source_entity, source_type, relation_type,
            target_entity, target_type, confidence, context, settings
        )

        consistency_score = self._evaluate_consistency(confidence, settings)

        scores = {
            "congruence": congruence_score,
            "coverage": coverage_score,
            "constraint": constraint_score,
            "completeness": completeness_score,
            "consistency": consistency_score,
        }
        weights = config["weights"]
        assessed = {k: v for k, v in scores.items() if v is not None}
        total_weight = sum(weights[k] for k in assessed)
        overall_score = sum(weights[k] * v for k, v in assessed.items()) / total_weight if total_weight > 0 else 0.0
        overall_score = round(max(0.0, min(1.0, overall_score)), 3)

        review_status, recommendation = self._determine_status(
            overall_score, congruence_score, constraint_score, config["thresholds"]
        )

        def rounded(value):
            return None if value is None else round(value, 3)

        return {
            "congruence_score": rounded(congruence_score),
            "coverage_score": rounded(coverage_score),
            "constraint_score": rounded(constraint_score),
            "completeness_score": rounded(completeness_score),
            "consistency_score": rounded(consistency_score),
            "overall_score": overall_score,
            "review_status": review_status,
            "recommendation": recommendation,
            "kg_evidence": kg_evidence,
            "scoring_version": SCORING_VERSION,
        }

    def _lookup_triple(
        self,
        source_entity: str,
        source_type: str,
        relation_type: str,
        target_entity: str,
        target_type: str,
    ) -> Optional[Dict]:
        """KG evidence for the triple, or None when the KG cannot be consulted."""
        if self.kg_service is None:
            logger.warning("Relation review without a Knowledge Graph: Coverage not assessed")
            return None
        try:
            return self.kg_service.find_relation_triples(
                source_entity,
                normalize_relation_type(relation_type),
                target_entity,
                source_type=normalize_entity_type(source_type),
                target_type=normalize_entity_type(target_type),
            )
        except Exception as e:
            logger.error(
                f"KG triple lookup failed for {source_entity} --{relation_type}--> {target_entity}; "
                f"Coverage not assessed: {e}",
                exc_info=True,
            )
            return None

    def _evaluate_congruence(
        self,
        source_type: str,
        relation_type: str,
        target_type: str,
        settings: Dict,
        kg_evidence: Optional[Dict] = None,
    ) -> float:
        """Schema fit of (source type, relation, target type).

        A KG bonus is only added when the KG actually contains this triple.
        """

        type_penalty = settings["congruence_type_penalty"]
        kg_bonus = settings["congruence_kg_bonus"]

        src = normalize_entity_type(source_type)
        tgt = normalize_entity_type(target_type)
        rel = normalize_relation_type(relation_type)

        if rel not in VALID_RELATION_COMBOS:
            return max(0.0, 0.5 - type_penalty)

        valid_combos = VALID_RELATION_COMBOS[rel]

        if (src, tgt) in valid_combos:
            score = 0.9
            if kg_evidence and kg_evidence.get("triple_exists"):
                score = min(1.0, score + kg_bonus * 0.5)
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

    def _evaluate_coverage(self, kg_evidence: Optional[Dict], settings: Dict) -> Optional[float]:
        """Novelty of the triple itself.

        - triple already in the KG:                            0.3
        - new triple, source already has >= novelty_threshold
          relations of this type:                              0.5 + bonus / 2
        - new triple (including between known entities):       0.5 + bonus
        Returns None when the KG is unavailable.
        """
        if kg_evidence is None:
            return None

        novelty_threshold = settings["coverage_novelty_threshold"]
        novelty_bonus = settings["coverage_novelty_bonus"]

        if kg_evidence["triple_exists"]:
            score = 0.3
        elif kg_evidence["source_relation_count"] >= novelty_threshold:
            score = 0.5 + novelty_bonus * 0.5
        else:
            score = 0.5 + novelty_bonus

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

        violation_penalty = settings["constraint_violation_penalty"]
        min_confidence = settings["constraint_min_confidence"]
        require_context = settings["constraint_require_context"]

        src = normalize_entity_type(source_type)
        tgt = normalize_entity_type(target_type)
        rel = normalize_relation_type(relation_type)

        score = 1.0

        if rel not in VALID_RELATION_COMBOS:
            score -= 0.3

        if src is None:
            score -= violation_penalty

        if tgt is None:
            score -= violation_penalty

        if source_entity.lower().strip() == target_entity.lower().strip():
            score -= 0.3

        if len(source_entity.strip()) < 2:
            score -= violation_penalty

        if len(target_entity.strip()) < 2:
            score -= violation_penalty

        if confidence is not None and confidence < min_confidence:
            score -= violation_penalty

        if require_context and not context:
            score -= violation_penalty * 0.5

        if rel in DIRECTIONAL_RELATIONS:
            valid_sources, valid_targets = DIRECTIONAL_RELATIONS[rel]
            if not (src in valid_sources and tgt in valid_targets):
                if src in valid_targets and tgt in valid_sources:
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

        required_fields = [
            f.strip().lower()
            for f in str(settings["completeness_required_fields"]).split(",")
            if f.strip()
        ]
        optional_weight = settings["completeness_optional_weight"]

        field_values = {
            "source_entity": source_entity,
            "source_type": source_type,
            "relation_type": relation_type,
            "target_entity": target_entity,
            "target_type": target_type,
            "confidence": confidence,
            "context": context,
        }

        def present(value) -> bool:
            return value is not None and (not isinstance(value, str) or bool(value.strip()))

        for field in required_fields:
            if field not in field_values:
                logger.warning(f"Unknown required completeness field '{field}' is always counted as missing")

        required_present = sum(1 for f in required_fields if present(field_values.get(f)))
        required_score = required_present / len(required_fields) if required_fields else 1.0

        optional_fields = [f for f in field_values if f not in required_fields]
        optional_present = sum(1 for f in optional_fields if present(field_values[f]))
        optional_score = optional_present / len(optional_fields) if optional_fields else 1.0

        score = required_score * (1.0 - optional_weight) + optional_score * optional_weight

        return max(0.0, min(1.0, score))

    def _evaluate_consistency(
        self,
        confidence: Optional[float],
        settings: Dict,
    ) -> float:
        """Extractor confidence, treated the same for every extractor source.

        Relations have no independent validator: each triple is stored once,
        from the first extractor that produced it, so there is no second
        opinion to agree with. The confidence (or consistency_base_score when
        missing) is therefore always uncorroborated and capped at
        UNVERIFIED_CONFIDENCE_CAP. consistency_agreement_bonus is not used.
        """

        score = confidence if confidence is not None else settings["consistency_base_score"]
        return max(0.0, min(UNVERIFIED_CONFIDENCE_CAP, score))

    def _determine_status(
        self,
        overall_score: float,
        congruence_score: float,
        constraint_score: float,
        thresholds: Dict,
    ) -> Tuple[str, str]:

        pass_threshold = thresholds["pass_threshold"]
        review_threshold = thresholds["review_threshold"]

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
