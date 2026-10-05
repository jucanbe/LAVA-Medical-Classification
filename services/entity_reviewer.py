import asyncio
import logging
import re
from typing import Callable, Optional, List, Dict, Tuple
from datetime import datetime, timedelta

from models.entities import (
    NUMERIC_ENTITY_TYPES,
    normalize_entity_type,
    CongruenceMetrics,
    CoverageMetrics,
    ConstraintMetrics,
    CompletenessMetrics,
    ConsistencyMetrics,
    ReviewStatus,
)
from models.review_defaults import (
    ENTITY_WEIGHTS,
    ENTITY_SETTINGS,
    THRESHOLDS,
    SCORING_VERSION,
    UNVERIFIED_CONFIDENCE_CAP,
)

from services.bert_ner import BERTNERService

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = ENTITY_WEIGHTS
DEFAULT_THRESHOLDS = THRESHOLDS
DEFAULT_SETTINGS = ENTITY_SETTINGS

CRITERIA = ("congruence", "coverage", "constraint", "completeness", "consistency")

# KG matches below this similarity are not considered related at all.
KG_CANDIDATE_MIN_SIMILARITY = 0.5
# Similarity at or above which a KG label counts as the same concept.
KG_EXACT_SIMILARITY = 0.95
# Coverage: with few similar KG entities, a best match below this is still novel.
COVERAGE_NOVELTY_SIMILARITY = 0.8

_config_generation = 0


def invalidate_config_cache() -> None:
    """Make every EntityReviewService reload its configuration on next use."""
    global _config_generation
    _config_generation += 1


class EntityReviewService:

    def __init__(self, kg_service=None, bert_service=None, bert_model: Optional[str] = None,
                 type_normalizer: Optional[Callable[[str], Optional[str]]] = None,
                 numeric_types: Optional[frozenset] = None,
                 constraint_type_map: Optional[Dict[str, str]] = None,
                 config: Optional[Dict] = None):
        """All optional arguments default to the application's behaviour.

        bert_model           pin the Consistency cross-check to one entity model
        type_normalizer      canonical type for a label (None = not a valid type);
                             default: the project's entity types
        numeric_types        canonical types that may be purely numeric
        constraint_type_map  canonical type -> project type whose type-specific
                             Constraint heuristics apply (finding / disease / substance)
        config               fixed {"weights", "thresholds", "settings"} used instead
                             of the active database configuration
        """

        self.kg_service = kg_service
        self.bert_service = bert_service
        self.bert_model = bert_model
        self._normalize_type = type_normalizer or normalize_entity_type
        self._numeric_types = NUMERIC_ENTITY_TYPES if numeric_types is None else numeric_types
        self._constraint_type_map = constraint_type_map
        self._fixed_config = config
        self._config_cache = None
        self._config_loaded_at = None
        self._config_generation = None
        self._logged_bert_issues = set()

    async def _get_config(self) -> Dict:

        if self._fixed_config is not None:
            return self._fixed_config

        if (
            self._config_cache
            and self._config_loaded_at
            and self._config_generation == _config_generation
            and datetime.utcnow() - self._config_loaded_at < timedelta(seconds=60)
        ):
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
                    self._config_cache = self._build_config(config)
                    self._config_loaded_at = datetime.utcnow()
                    self._config_generation = _config_generation
                    return self._config_cache
        except Exception as e:
            logger.warning(f"Could not load entity review config from DB, using defaults: {e}")

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
        canonical_type = self._normalize_type(entity_type)

        kg_matches = await self._fetch_kg_matches(entity_text, canonical_type)

        congruence = self._evaluate_congruence(entity_text, entity_type, settings, kg_matches=kg_matches)

        coverage = self._evaluate_coverage(entity_text, entity_type, settings, kg_matches=kg_matches)

        constraint = self._evaluate_constraint(entity_text, entity_type, settings)

        completeness = self._evaluate_completeness(
            entity_text, entity_type, normalized_form, context, confidence, settings
        )

        consistency = await self._evaluate_consistency(
            entity_text, entity_type, source, confidence,
            run_bert_validation, run_llm_validation, settings
        )

        overall_score = self._calculate_overall_score(
            {
                "congruence": congruence.score if congruence else None,
                "coverage": coverage.score if coverage else None,
                "constraint": constraint.score,
                "completeness": completeness.score,
                "consistency": consistency.score,
            },
            config["weights"]
        )

        review_status, recommendation = self._determine_status_and_recommendation(
            overall_score, congruence, constraint, consistency, config["thresholds"]
        )

        return {
            "entity_text": entity_text,
            "entity_type": entity_type,
            "entity_type_normalized": canonical_type,
            "congruence": congruence,
            "coverage": coverage,
            "constraint": constraint,
            "completeness": completeness,
            "consistency": consistency,
            "overall_score": overall_score,
            "review_status": review_status,
            "recommendation": recommendation,
            "scoring_version": SCORING_VERSION,
            "evaluated_at": datetime.utcnow().isoformat()
        }

    async def _fetch_kg_matches(self, entity_text: str, canonical_type: Optional[str]) -> Optional[list]:
        """KG matches for the entity, restricted to compatible KG types.

        Returns None when the KG cannot be consulted, so that Congruence and
        Coverage are reported as not assessed instead of as "no match".
        """
        if self.kg_service is None:
            logger.warning("Entity review without a Knowledge Graph: Congruence and Coverage not assessed")
            return None
        try:
            return await self.kg_service.find_matches_async(
                entity_text,
                max_matches=10,
                min_score=KG_CANDIDATE_MIN_SIMILARITY,
                entity_type=canonical_type,
            )
        except Exception as e:
            logger.error(f"KG lookup failed for '{entity_text}'; Congruence and Coverage not assessed: {e}", exc_info=True)
            return None

    def _evaluate_congruence(
        self,
        entity_text: str,
        entity_type: str,
        settings: Dict,
        kg_matches: Optional[list],
    ) -> Optional[CongruenceMetrics]:

        if kg_matches is None:
            return None

        min_similarity = settings["congruence_min_similarity"]
        exact_match_bonus = settings["congruence_exact_match_bonus"]

        matches = [m for m in kg_matches if m.similarity_score >= min_similarity]
        if not matches:
            return CongruenceMetrics(score=0.5)

        best_match = matches[0]
        score = best_match.similarity_score
        if score >= KG_EXACT_SIMILARITY:
            score = min(1.0, score + exact_match_bonus)

        return CongruenceMetrics(
            score=score,
            nearest_entity=best_match.kg_label,
            nearest_uri=best_match.kg_uri,
            embedding_distance=1.0 - best_match.similarity_score,
        )

    def _evaluate_coverage(
        self,
        entity_text: str,
        entity_type: str,
        settings: Dict,
        kg_matches: Optional[list],
    ) -> Optional[CoverageMetrics]:

        if kg_matches is None:
            return None

        novelty_threshold = settings["coverage_novelty_threshold"]
        novelty_bonus = settings["coverage_novelty_bonus"]

        similar_count = len(kg_matches)
        best_similarity = kg_matches[0].similarity_score if kg_matches else 0.0

        if similar_count == 0:
            is_novel = True
            score = 0.5 + novelty_bonus
            coverage_increase = 5.0
        elif similar_count < novelty_threshold and best_similarity < COVERAGE_NOVELTY_SIMILARITY:
            is_novel = True
            score = 0.5 + (novelty_bonus * 0.7)
            coverage_increase = 3.0
        elif similar_count < novelty_threshold:
            is_novel = False
            score = 0.5
            coverage_increase = 0.0
        else:
            is_novel = False
            score = max(0.3, 0.6 - (0.3 * best_similarity))
            coverage_increase = 0.0

        return CoverageMetrics(
            score=score,
            similar_entities_count=similar_count,
            is_novel=is_novel,
            fills_gap=is_novel,
            coverage_increase=coverage_increase
        )

    def _evaluate_constraint(
        self,
        entity_text: str,
        entity_type: str,
        settings: Dict
    ) -> ConstraintMetrics:

        violations = []
        type_valid = True
        format_valid = True
        score = 1.0

        min_length = settings["constraint_min_length"]
        max_length = settings["constraint_max_length"]
        violation_penalty = settings["constraint_violation_penalty"]

        canonical_type = self._normalize_type(entity_type)
        stripped = entity_text.strip()

        if canonical_type is None:
            type_valid = False
            violations.append(f"Invalid entity type: {entity_type}")
            score -= 0.3

        if len(stripped) < min_length:
            format_valid = False
            violations.append(f"Entity text too short (< {min_length} characters)")
            shortness = 1.0 - (len(stripped) / min_length)
            score -= 0.2 + 0.6 * shortness

        if len(stripped) > max_length:
            format_valid = False
            violations.append(f"Entity text too long (> {max_length} characters)")
            score -= 0.1

        if len(stripped) <= 2 and not stripped.isalnum():
            format_valid = False
            violations.append("Entity is a single symbol or punctuation, not a valid medical term")
            score -= 0.5

        if re.search(r'[<>{}|\[\]\\]', entity_text):
            format_valid = False
            violations.append("Entity contains invalid characters")
            score -= violation_penalty

        if canonical_type not in self._numeric_types and re.match(r'^[\d\s.,-]+$', stripped):
            violations.append("Only numeric, but the entity type is not a measurement, parameter or score")
            score -= 0.2

        heuristic_type = canonical_type
        if self._constraint_type_map is not None:
            heuristic_type = self._constraint_type_map.get(canonical_type)
        type_violations = self._check_type_constraints(stripped, heuristic_type)
        violations.extend(type_violations)
        score -= violation_penalty * len(type_violations)

        score = max(0.0, min(1.0, score))

        return ConstraintMetrics(
            score=score,
            violations=violations,
            ontology_valid=None,
            type_valid=type_valid,
            format_valid=format_valid,
            clinical_plausibility=score
        )

    def _check_type_constraints(self, entity_text: str, canonical_type: Optional[str]) -> List[str]:
        violations = []
        if canonical_type is None:
            return violations

        text_lower = entity_text.lower()
        words = text_lower.split()
        word_count = len(words)

        population_indicators = ['patients', 'subjects', 'individuals', 'participants',
                                 'cohort', 'group', 'male', 'female', 'adults', 'children',
                                 'men', 'women', 'cases', 'controls']
        population_word_count = sum(1 for w in words if w in population_indicators)

        if population_word_count >= 2 or (word_count > 5 and population_word_count >= 1):
            if canonical_type in ['finding', 'disease', 'substance']:
                violations.append(f"Text appears to describe a patient population, not a {canonical_type}")

        if word_count > 8 and canonical_type in ['finding', 'disease', 'substance']:
            violations.append(f"Entity text too complex for type '{canonical_type}' ({word_count} words)")

        if canonical_type == "finding":
            finding_indicators = ['pain', 'ache', 'fever', 'cough', 'symptom', 'sign',
                                  'elevated', 'decreased', 'abnormal', 'normal', 'positive',
                                  'negative', 'swelling', 'rash', 'bleeding', 'dysfunction']
            has_finding_indicator = any(ind in text_lower for ind in finding_indicators)

            if word_count > 4 and not has_finding_indicator and 'with' in text_lower:
                violations.append("Finding classification uncertain - text pattern suggests population description")

        elif canonical_type == "substance":
            if text_lower in ['the', 'a', 'an', 'this', 'that']:
                violations.append("Substance appears too generic")

        elif canonical_type == "disease":
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

        required_fields = [
            f.strip().lower()
            for f in str(settings["completeness_required_fields"]).split(",")
            if f.strip()
        ]
        optional_weight = settings["completeness_optional_weight"]
        min_length = settings["constraint_min_length"]

        text_len = len(entity_text.strip()) if entity_text else 0
        has_text = text_len >= min_length
        if 0 < text_len < min_length:
            missing_fields.append("text (too short to be a valid entity)")

        # Each field is a distinct piece of information. The extractors do not
        # produce definitions, so "definition" is not scored.
        field_status = {
            "type": bool(entity_type),
            "text": has_text,
            "context": bool(context),
            "normalized_form": bool(normalized_form),
            "confidence": confidence is not None,
        }

        required_present = 0
        for field in required_fields:
            if field_status.get(field):
                required_present += 1
            else:
                if field not in field_status:
                    logger.warning(f"Unknown required completeness field '{field}' is always counted as missing")
                missing_fields.append(field)

        required_score = required_present / len(required_fields) if required_fields else 1.0

        optional_fields = [f for f in field_status if f not in required_fields]
        optional_present = sum(1 for f in optional_fields if field_status[f])
        optional_score = optional_present / len(optional_fields) if optional_fields else 1.0

        score = required_score * (1.0 - optional_weight) + optional_score * optional_weight

        return CompletenessMetrics(
            score=score,
            has_type=field_status["type"],
            has_definition=False,
            has_normalized_form=field_status["normalized_form"],
            has_context=field_status["context"],
            has_confidence=field_status["confidence"],
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
        """Extractor confidence, corroborated or contradicted by BERT.

        Uncorroborated confidence is capped at UNVERIFIED_CONFIDENCE_CAP. BERT
        agreement lifts the cap and adds agreement_bonus * p_bert; disagreement
        subtracts 0.75 * agreement_bonus * p_bert from the capped value. When
        BERT is unavailable, abstains or was not requested, the capped value is
        used unchanged (no penalty).
        """
        agreement_bonus = settings["consistency_agreement_bonus"]
        base_score = settings["consistency_base_score"]

        type_confidence = confidence if confidence is not None else base_score
        unverified = min(UNVERIFIED_CONFIDENCE_CAP, type_confidence)
        alternate_types = []
        bert_agreement = None

        if run_bert_validation:
            outcome = await self._run_bert_classification(entity_text, self._normalize_type(entity_type))
        else:
            outcome = {"status": "not_requested"}

        score = unverified
        if outcome["status"] == "agreed":
            bert_agreement = True
            score = min(1.0, type_confidence + agreement_bonus * outcome["confidence"])
        elif outcome["status"] == "disagreed":
            bert_agreement = False
            score = max(0.0, unverified - agreement_bonus * 0.75 * outcome["confidence"])
            alternate_types.append({
                "type": outcome["predicted_type"],
                "source": "bert",
                "confidence": outcome["confidence"]
            })

        return ConsistencyMetrics(
            score=min(1.0, max(0.0, score)),
            type_confidence=type_confidence,
            alternate_types=alternate_types,
            bert_agreement=bert_agreement,
            llm_agreement=None,
            cross_validation_score=None if bert_agreement is None else float(bert_agreement),
            bert_status=outcome["status"],
        )

    def _log_bert_issue_once(self, message: str) -> None:
        if message not in self._logged_bert_issues:
            self._logged_bert_issues.add(message)
            logger.warning(f"BERT cross-validation unavailable: {message}")

    def _select_bert_model(self, canonical_type: str) -> Tuple[Optional[str], Optional[str]]:
        """Pick an entity model whose labels are the project entity types.

        Returns (model_name, None) or (None, reason).
        """
        compatible = []
        incompatible = []
        for info in self.bert_service.get_available_models("entity"):
            if self.bert_model is not None and info["name"] != self.bert_model:
                continue
            model_types = BERTNERService.entity_types_from_labels(info.get("labels") or [])
            normalized = {self._normalize_type(t) for t in model_types}
            if model_types and None not in normalized:
                compatible.append((info["name"], normalized))
            else:
                incompatible.append(info["name"])

        if not compatible:
            if self.bert_model is not None:
                state = "is incompatible with the entity types" if incompatible else "was not found"
                return None, f"BERT model '{self.bert_model}' {state}"
            found = ", ".join(incompatible) or "none"
            return None, f"no BERT entity model uses the project entity types (found: {found})"
        for name, types in compatible:
            if canonical_type in types:
                return name, None
        return None, f"no compatible BERT model predicts type '{canonical_type}'"

    async def _run_bert_classification(self, text: str, canonical_type: Optional[str]) -> Dict:
        """Classify the entity text with BERT and compare with the proposed type.

        Returns {"status": ...} where status is agreed / disagreed / abstained
        or "unavailable: <reason>"; agreed/disagreed also carry predicted_type
        and confidence.
        """
        if self.bert_service is None:
            self._log_bert_issue_once("BERT service not initialized")
            return {"status": "unavailable: BERT service not initialized"}
        if canonical_type is None:
            return {"status": "unavailable: entity type is not a project type"}

        try:
            model_name, reason = self._select_bert_model(canonical_type)
        except Exception as e:
            logger.error(f"Could not inspect BERT models: {e}", exc_info=True)
            return {"status": f"unavailable: could not inspect BERT models ({e})"}
        if model_name is None:
            self._log_bert_issue_once(reason)
            return {"status": f"unavailable: {reason}"}

        try:
            entities, _ = await asyncio.to_thread(self.bert_service.classify, text, model_name)
        except Exception as e:
            logger.error(f"BERT inference failed for '{text}' with model '{model_name}': {e}", exc_info=True)
            return {"status": f"unavailable: BERT inference failed ({e})"}

        if not entities:
            return {"status": "abstained"}

        best = max(entities, key=lambda e: e["confidence"])
        predicted_type = self._normalize_type(best["type"])
        return {
            "status": "agreed" if predicted_type == canonical_type else "disagreed",
            "predicted_type": predicted_type,
            "confidence": float(best["confidence"]),
        }

    def _calculate_overall_score(
        self,
        scores: Dict[str, Optional[float]],
        weights: Dict
    ) -> float:
        """Weighted mean of the assessed criteria.

        Criteria that could not be assessed (None) are left out and the
        remaining weights are renormalized, so a missing optional service
        neither rewards nor penalizes the entity.
        """
        assessed = {k: v for k, v in scores.items() if v is not None}
        total_weight = sum(weights[k] for k in assessed)
        if total_weight <= 0:
            return 0.0
        overall = sum(weights[k] * v for k, v in assessed.items()) / total_weight
        return round(max(0.0, min(1.0, overall)), 3)

    def _determine_status_and_recommendation(
        self,
        overall_score: float,
        congruence: Optional[CongruenceMetrics],
        constraint: ConstraintMetrics,
        consistency: ConsistencyMetrics,
        thresholds: Dict
    ) -> Tuple[str, str]:

        pass_threshold = thresholds["pass_threshold"]
        review_threshold = thresholds["review_threshold"]
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
            if congruence is not None and congruence.score > 0.95:
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

        def as_dict(metric):
            return metric.model_dump() if hasattr(metric, "model_dump") else metric

        def score_line(index, title, metric):
            if metric is None:
                return f"{index}. {title}: not assessed (Knowledge Graph unavailable)"
            return f"{index}. {title}: {metric.get('score', 0):.2%}"

        cong = as_dict(evaluation.get("congruence"))
        explanations.append(score_line(1, "CONGRUENCE (Semantic Alignment)", cong))
        if cong and cong.get("nearest_entity"):
            explanations.append(f"   - Nearest KG entity: {cong['nearest_entity']}")

        cov = as_dict(evaluation.get("coverage"))
        explanations.append(score_line(2, "COVERAGE (Representativeness)", cov))
        if cov:
            explanations.append(f"   - Is novel: {cov.get('is_novel', False)}")
            explanations.append(f"   - Similar entities in KG: {cov.get('similar_entities_count', 0)}")

        cons = as_dict(evaluation.get("constraint")) or {}
        explanations.append(f"3. CONSTRAINT (Clinical Validity): {cons.get('score', 0):.2%}")
        for v in cons.get("violations", []):
            explanations.append(f"   ⚠ {v}")

        comp = as_dict(evaluation.get("completeness")) or {}
        explanations.append(f"4. COMPLETENESS (Required Attributes): {comp.get('score', 0):.2%}")
        missing = comp.get("missing_fields", [])
        if missing:
            explanations.append(f"   - Missing: {', '.join(missing)}")

        cons2 = as_dict(evaluation.get("consistency")) or {}
        explanations.append(f"5. CONSISTENCY (Type Coherence): {cons2.get('score', 0):.2%}")
        if cons2.get("bert_status"):
            explanations.append(f"   - BERT cross-check: {cons2['bert_status']}")

        return "\n".join(explanations)
