"""
Single source of truth for the review scorecard defaults.

These values are used by the review services (as fallbacks), by the
configuration routers (defaults / reset) and by the SQLAlchemy column
defaults, so they cannot drift apart.
"""

# Increment when the scoring logic changes so stored reviews can be told apart.
# Reviews without a version were produced by the pre-2.0 logic.
SCORING_VERSION = "2.1"

# Self-reported extractor confidence that is not corroborated by an independent
# validator (BERT agreement, or a second extractor for relations) is capped at
# this value. The same cap applies to every extractor source.
UNVERIFIED_CONFIDENCE_CAP = 0.8

# Entity weights calibrated on development data in the JBMS extension study (W3):
# the non-negative weights on a 0.05 grid that maximised AUROC against gold
# correctness. Coverage gets no weight because novelty is inversely related to
# correctness for single predictions. (Pre-2.1 defaults: 0.25/0.15/0.25/0.15/0.20.)
ENTITY_WEIGHTS = {
    "congruence": 0.20,
    "coverage": 0.00,
    "constraint": 0.05,
    "completeness": 0.30,
    "consistency": 0.45,
}

# Relation weights were not part of that calibration and keep the previous defaults.
RELATION_WEIGHTS = {
    "congruence": 0.25,
    "coverage": 0.15,
    "constraint": 0.25,
    "completeness": 0.15,
    "consistency": 0.20,
}

THRESHOLDS = {
    "pass_threshold": 0.75,
    "review_threshold": 0.50,
}

ENTITY_SETTINGS = {
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

RELATION_SETTINGS = {
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


def _flat_config(weights: dict, settings: dict) -> dict:
    config = {"config_name": "default"}
    config.update({f"weight_{k}": v for k, v in weights.items()})
    config["threshold_pass"] = THRESHOLDS["pass_threshold"]
    config["threshold_review"] = THRESHOLDS["review_threshold"]
    config.update(settings)
    return config


# Flat column-name -> value mappings used by the config tables and routers.
ENTITY_CONFIG_DEFAULTS = _flat_config(ENTITY_WEIGHTS, ENTITY_SETTINGS)
RELATION_CONFIG_DEFAULTS = _flat_config(RELATION_WEIGHTS, RELATION_SETTINGS)

# Settings that must lie in [0, 1] for every score to stay in [0, 1].
UNIT_INTERVAL_SETTINGS = {
    "congruence_min_similarity",
    "congruence_exact_match_bonus",
    "coverage_novelty_bonus",
    "constraint_violation_penalty",
    "completeness_optional_weight",
    "consistency_agreement_bonus",
    "consistency_base_score",
    "congruence_type_penalty",
    "congruence_kg_bonus",
    "constraint_min_confidence",
}


def validate_review_config(values: dict) -> None:
    """Raise ValueError if a flat config dict could produce invalid scores."""
    weights = [values[k] for k in values if k.startswith("weight_")]
    if any(w is None or w < 0 or w > 1 for w in weights):
        raise ValueError("Each weight must be between 0 and 1")
    total = sum(weights)
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Weights must sum to 1.0 (currently {total:.2f})")
    for key in ("threshold_pass", "threshold_review"):
        if not 0 <= values[key] <= 1:
            raise ValueError(f"{key} must be between 0 and 1")
    if values["threshold_pass"] <= values["threshold_review"]:
        raise ValueError("Pass threshold must be greater than review threshold")
    for key in UNIT_INTERVAL_SETTINGS & values.keys():
        if not 0 <= values[key] <= 1:
            raise ValueError(f"{key} must be between 0 and 1")
    if values.get("coverage_novelty_bonus", 0) > 0.5:
        raise ValueError("coverage_novelty_bonus must be at most 0.5 (Coverage = 0.5 + bonus)")
    if "coverage_novelty_threshold" in values and values["coverage_novelty_threshold"] < 1:
        raise ValueError("coverage_novelty_threshold must be at least 1")
    if "constraint_min_length" in values:
        if values["constraint_min_length"] < 1:
            raise ValueError("constraint_min_length must be at least 1")
        if values["constraint_min_length"] >= values["constraint_max_length"]:
            raise ValueError("Min length must be less than max length")
