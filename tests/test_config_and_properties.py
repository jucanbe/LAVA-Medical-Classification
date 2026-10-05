"""Configuration defaults, validation and score properties."""
import random
import unittest
from unittest.mock import AsyncMock

from tests.support import make_kg, StubBert, default_config

from database.models import EntityConfigDB, RelationConfigDB
from models.review_defaults import (
    ENTITY_CONFIG_DEFAULTS, RELATION_CONFIG_DEFAULTS, ENTITY_SETTINGS, RELATION_SETTINGS,
    ENTITY_WEIGHTS, THRESHOLDS, validate_review_config,
)
from routers import entity_config, relation_config
from services import entity_reviewer, relation_reviewer
from services.entity_reviewer import EntityReviewService
from services.relation_reviewer import RelationReviewService


class TestSingleSourceOfDefaults(unittest.TestCase):
    def column_defaults(self, model):
        return {c.name: c.default.arg for c in model.__table__.columns if c.default is not None and not callable(c.default.arg)}

    def test_entity_defaults_agree_everywhere(self):
        columns = self.column_defaults(EntityConfigDB)
        for key, value in ENTITY_CONFIG_DEFAULTS.items():
            with self.subTest(key=key):
                self.assertEqual(columns[key], value)
        self.assertIs(entity_config.DEFAULT_CONFIG, ENTITY_CONFIG_DEFAULTS)
        self.assertIs(entity_reviewer.DEFAULT_SETTINGS, ENTITY_SETTINGS)

    def test_relation_defaults_agree_everywhere(self):
        columns = self.column_defaults(RelationConfigDB)
        for key, value in RELATION_CONFIG_DEFAULTS.items():
            with self.subTest(key=key):
                self.assertEqual(columns[key], value)
        self.assertIs(relation_config.DEFAULT_CONFIG, RELATION_CONFIG_DEFAULTS)
        self.assertIs(relation_reviewer.DEFAULT_SETTINGS, RELATION_SETTINGS)

    def test_null_columns_fall_back_to_the_shared_defaults(self):
        row = EntityConfigDB(**ENTITY_CONFIG_DEFAULTS)
        row.coverage_novelty_bonus = None
        row.constraint_max_length = 100
        config = EntityReviewService._build_config(row)
        self.assertEqual(config["settings"]["coverage_novelty_bonus"], ENTITY_SETTINGS["coverage_novelty_bonus"])
        self.assertEqual(config["settings"]["constraint_max_length"], 100)

    def test_defaults_are_valid(self):
        validate_review_config({k: v for k, v in ENTITY_CONFIG_DEFAULTS.items() if k != "config_name"})
        validate_review_config({k: v for k, v in RELATION_CONFIG_DEFAULTS.items() if k != "config_name"})


class TestConfigValidation(unittest.TestCase):
    def cfg(self, **changes):
        values = {k: v for k, v in ENTITY_CONFIG_DEFAULTS.items() if k != "config_name"}
        values.update(changes)
        return values

    def assertRejected(self, **changes):
        with self.assertRaises(ValueError):
            validate_review_config(self.cfg(**changes))

    def test_default_entity_weights_are_the_calibrated_ones(self):
        self.assertEqual(ENTITY_WEIGHTS, {"congruence": 0.20, "coverage": 0.00, "constraint": 0.05,
                                          "completeness": 0.30, "consistency": 0.45})
        self.assertAlmostEqual(sum(ENTITY_WEIGHTS.values()), 1.0, places=9)

    def test_weights(self):
        self.assertRejected(weight_congruence=0.5)                            # sum above 1
        self.assertRejected(weight_congruence=-0.25, weight_coverage=0.65)    # sums to 1 but negative
        self.assertRejected(weight_congruence=1.2, weight_coverage=-0.05, weight_constraint=-0.15)
        validate_review_config(self.cfg(weight_consistency=ENTITY_WEIGHTS["consistency"] + 0.009))  # within tolerance

    def test_thresholds_and_settings(self):
        self.assertRejected(threshold_pass=0.5, threshold_review=0.6)
        self.assertRejected(threshold_pass=1.5)
        self.assertRejected(coverage_novelty_bonus=0.7)   # 0.5 + bonus would exceed 1
        self.assertRejected(consistency_base_score=-0.1)
        self.assertRejected(constraint_min_length=0)
        self.assertRejected(constraint_min_length=50, constraint_max_length=40)


def random_valid_config(rng: random.Random) -> dict:
    raw = [rng.random() for _ in ENTITY_WEIGHTS]
    weights = {k: v / sum(raw) for k, v in zip(ENTITY_WEIGHTS, raw)}
    review = rng.uniform(0.05, 0.9)
    settings = dict(ENTITY_SETTINGS)
    for key in ["congruence_min_similarity", "congruence_exact_match_bonus", "coverage_novelty_bonus",
                "constraint_violation_penalty", "completeness_optional_weight", "consistency_agreement_bonus",
                "consistency_base_score"]:
        settings[key] = rng.random() * (0.5 if key == "coverage_novelty_bonus" else 1.0)
    settings["coverage_novelty_threshold"] = rng.randint(1, 6)
    settings["constraint_min_length"] = rng.randint(1, 4)
    settings["constraint_max_length"] = rng.randint(5, 300)
    flat = {f"weight_{k}": v for k, v in weights.items()}
    flat.update(threshold_pass=rng.uniform(review + 0.01, 1.0), threshold_review=review, **settings)
    validate_review_config(flat)
    return {"weights": weights, "thresholds": {"pass_threshold": flat["threshold_pass"], "review_threshold": review}, "settings": settings}


TEXTS = ["Pneumonia", "Pneumonias", "7.5", "a", "%", "<x>", "male and female patients with fever and cough today", "x" * 250, "", "Heparin"]
TYPES = ["disease", "symptom", "parameter", "Substance", "organ_system", "ImagingProcedure", ""]


class TestScoreProperties(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.kg = make_kg()

    async def test_scores_stay_in_unit_interval_for_valid_configs(self):
        rng = random.Random(1234)
        for i in range(150):
            config = random_valid_config(rng)
            bert = rng.choice([None, StubBert(predicted_type=rng.choice(["disease", "symptom", None]), confidence=rng.random())])
            svc = EntityReviewService(kg_service=rng.choice([self.kg, None]), bert_service=bert)
            svc._get_config = AsyncMock(return_value=config)
            ev = await svc.evaluate_entity(
                rng.choice(TEXTS), rng.choice(TYPES),
                normalized_form=rng.choice([None, "n"]), context=rng.choice([None, "c"]),
                confidence=rng.choice([None, rng.random()]), run_bert_validation=rng.random() < 0.7,
            )
            with self.subTest(i=i):
                for name in ["congruence", "coverage", "constraint", "completeness", "consistency"]:
                    metric = ev[name]
                    if metric is not None:
                        self.assertGreaterEqual(metric.score, 0.0)
                        self.assertLessEqual(metric.score, 1.0)
                self.assertGreaterEqual(ev["overall_score"], 0.0)
                self.assertLessEqual(ev["overall_score"], 1.0)

    async def test_relation_scores_stay_in_unit_interval(self):
        rng = random.Random(99)
        entities = ["Pneumonia", "Fever", "Cough", "Heparin", "x", "Appendicitis"]
        types = ["disease", "symptom", "finding", "substance", "organ", "bogus"]
        relations = ["has_symptom", "treats", "located_in", "cures", "Treats"]
        for i in range(150):
            settings = dict(RELATION_SETTINGS)
            for key in ["congruence_type_penalty", "congruence_kg_bonus", "constraint_violation_penalty",
                        "constraint_min_confidence", "completeness_optional_weight", "consistency_base_score"]:
                settings[key] = rng.random()
            settings["coverage_novelty_bonus"] = rng.random() * 0.5
            svc = RelationReviewService(kg_service=rng.choice([self.kg, None]))
            svc._get_config = AsyncMock(return_value={"weights": dict(ENTITY_WEIGHTS), "thresholds": dict(THRESHOLDS), "settings": settings})
            ev = await svc.evaluate_relation(
                rng.choice(entities), rng.choice(types), rng.choice(relations), rng.choice(entities), rng.choice(types),
                confidence=rng.choice([None, rng.random()]), context=rng.choice([None, "c"]), source=rng.choice(["llm", "langextract"]),
            )
            with self.subTest(i=i):
                for key in ["congruence_score", "coverage_score", "constraint_score", "completeness_score", "consistency_score", "overall_score"]:
                    if ev[key] is not None:
                        self.assertTrue(0.0 <= ev[key] <= 1.0, (key, ev[key]))

    async def test_adding_completeness_information_never_lowers_completeness(self):
        svc = EntityReviewService()
        fields = ["context", "normalized_form", "confidence"]
        values = {"context": "ctx", "normalized_form": "norm", "confidence": 0.7}
        for required in ["type,text", "type,text,context", "type,text,confidence,normalized_form"]:
            settings = dict(ENTITY_SETTINGS, completeness_required_fields=required)
            for mask in range(8):
                present = {f: values[f] for bit, f in enumerate(fields) if mask & (1 << bit)}
                base = svc._evaluate_completeness("Pneumonia", "disease", present.get("normalized_form"), present.get("context"), present.get("confidence"), settings).score
                for extra in fields:
                    if extra in present:
                        continue
                    more = dict(present, **{extra: values[extra]})
                    richer = svc._evaluate_completeness("Pneumonia", "disease", more.get("normalized_form"), more.get("context"), more.get("confidence"), settings).score
                    with self.subTest(required=required, present=sorted(present), extra=extra):
                        self.assertGreaterEqual(richer, base)

    async def test_unavailable_optional_services_do_not_penalize(self):
        """No BERT vs. BERT unavailable vs. BERT abstaining: identical scores."""
        base = None
        for bert, run in [(None, True), (None, False), (StubBert(labels=["O", "B-Chemical"]), True),
                          (StubBert(predicted_type=None), True), (StubBert(error=RuntimeError("boom")), True)]:
            svc = EntityReviewService(kg_service=self.kg, bert_service=bert)
            svc._get_config = AsyncMock(return_value=default_config())
            ev = await svc.evaluate_entity("Pneumonia", "disease", context="c", confidence=0.9, run_bert_validation=run)
            if base is None:
                base = ev
            self.assertEqual(ev["consistency"].score, base["consistency"].score)
            self.assertEqual(ev["overall_score"], base["overall_score"])

    async def test_agreement_is_never_below_unverified_and_disagreement_never_above(self):
        for confidence in [None, 0.0, 0.3, 0.8, 0.95, 1.0]:
            for p in [0.1, 0.5, 0.99]:
                scores = {}
                for label, bert in [("agree", StubBert(predicted_type="disease", confidence=p)),
                                    ("none", None),
                                    ("disagree", StubBert(predicted_type="symptom", confidence=p))]:
                    svc = EntityReviewService(bert_service=bert)
                    result = await svc._evaluate_consistency("Pneumonia", "disease", "llm", confidence, True, False, ENTITY_SETTINGS)
                    scores[label] = result.score
                with self.subTest(confidence=confidence, p=p):
                    self.assertGreaterEqual(scores["agree"], scores["none"])
                    self.assertLessEqual(scores["disagree"], scores["none"])


if __name__ == "__main__":
    unittest.main()
