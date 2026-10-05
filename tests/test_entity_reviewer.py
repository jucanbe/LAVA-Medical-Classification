"""Entity review scorecard.

Expected values are worked out by hand from the documented formulas (see
the comments next to each literal), never by calling the implementation.
Settings: formula weights 0.25/0.15/0.25/0.15/0.20 (tests.support.FORMULA_WEIGHTS), exact-match bonus 0.2,
novelty bonus 0.3, violation penalty 0.25, optional weight 0.15,
agreement bonus 0.3, base score 0.5, unverified confidence cap 0.8.
"""
import unittest
from unittest.mock import AsyncMock

from tests.support import (
    make_kg, StubBert, FailingKG, MEDMENTIONS_LABELS, default_config, entity_settings,
)

from models.entities import (
    KGMatch, CongruenceMetrics, ConstraintMetrics, ConsistencyMetrics,
)
from services.entity_reviewer import EntityReviewService

TOL = 1e-9
ROUNDED = 6e-4  # overall scores are rounded to 3 decimals


def service(kg=None, bert=None, config=None) -> EntityReviewService:
    svc = EntityReviewService(kg_service=kg, bert_service=bert)
    svc._get_config = AsyncMock(return_value=config or default_config())
    return svc


def match(score: float, label: str = "X") -> KGMatch:
    return KGMatch(kg_uri=f"http://example.org/medical/{label}", kg_label=label, similarity_score=score)


class TestOverallScore(unittest.TestCase):
    def setUp(self):
        self.svc = service()
        self.w = default_config()["weights"]

    def test_weighted_mean_of_all_criteria(self):
        scores = {"congruence": 0.8, "coverage": 0.6, "constraint": 1.0, "completeness": 0.9, "consistency": 0.7}
        # .25*.8 + .15*.6 + .25*1 + .15*.9 + .20*.7 = .2 + .09 + .25 + .135 + .14 = 0.815
        self.assertAlmostEqual(self.svc._calculate_overall_score(scores, self.w), 0.815, delta=ROUNDED)

    def test_unassessed_criteria_are_renormalized_out(self):
        scores = {"congruence": 0.8, "coverage": None, "constraint": 1.0, "completeness": 0.9, "consistency": 0.7}
        # (.2 + .25 + .135 + .14) / (1 - .15) = .725 / .85 = 0.852941
        self.assertAlmostEqual(self.svc._calculate_overall_score(scores, self.w), 0.852941, delta=ROUNDED)

    def test_weights_not_summing_exactly_to_one_stay_in_range(self):
        weights = dict(self.w, consistency=0.21)  # sum 1.01, accepted by validation
        scores = dict.fromkeys(self.w, 1.0)
        self.assertLessEqual(self.svc._calculate_overall_score(scores, weights), 1.0)
        self.assertAlmostEqual(self.svc._calculate_overall_score(scores, weights), 1.0, delta=TOL)


class TestCongruence(unittest.TestCase):
    def setUp(self):
        self.svc = service()
        self.s = entity_settings()

    def score(self, matches):
        metric = self.svc._evaluate_congruence("x", "disease", self.s, kg_matches=matches)
        return metric.score

    def test_no_match_is_neutral(self):
        self.assertAlmostEqual(self.score([]), 0.5, delta=TOL)

    def test_similarity_threshold_boundary(self):
        self.assertAlmostEqual(self.score([match(0.70)]), 0.70, delta=TOL)  # >= 0.7 counts
        self.assertAlmostEqual(self.score([match(0.6999)]), 0.5, delta=TOL)  # below: treated as no match

    def test_exact_match_boundary(self):
        self.assertAlmostEqual(self.score([match(0.95)]), 1.0, delta=TOL)     # 0.95 + 0.2, capped at 1
        self.assertAlmostEqual(self.score([match(0.9499)]), 0.9499, delta=TOL)

    def test_kg_unavailable_is_not_assessed(self):
        self.assertIsNone(self.svc._evaluate_congruence("x", "disease", self.s, kg_matches=None))


class TestCoverage(unittest.TestCase):
    def setUp(self):
        self.svc = service()
        self.s = entity_settings()

    def coverage(self, scores):
        return self.svc._evaluate_coverage("x", "disease", self.s, kg_matches=[match(v) for v in scores])

    def test_novelty_bands(self):
        self.assertAlmostEqual(self.coverage([]).score, 0.8, delta=TOL)              # 0.5 + 0.3
        self.assertAlmostEqual(self.coverage([0.7999, 0.6]).score, 0.71, delta=TOL)  # 0.5 + 0.3*0.7
        self.assertAlmostEqual(self.coverage([0.8]).score, 0.5, delta=TOL)           # close duplicate
        self.assertAlmostEqual(self.coverage([0.6, 0.6, 0.6]).score, 0.42, delta=TOL)  # 0.6 - 0.3*0.6
        self.assertAlmostEqual(self.coverage([1.0, 1.0, 1.0]).score, 0.3, delta=TOL)   # floor
        self.assertTrue(self.coverage([]).is_novel)
        self.assertFalse(self.coverage([0.8]).is_novel)

    def test_kg_unavailable_is_not_assessed(self):
        self.assertIsNone(self.svc._evaluate_coverage("x", "disease", self.s, kg_matches=None))


class TestConstraint(unittest.TestCase):
    def setUp(self):
        self.svc = service()
        self.s = entity_settings()

    def c(self, text, etype):
        return self.svc._evaluate_constraint(text, etype, self.s)

    def test_clean_entity(self):
        result = self.c("Pneumonia", "disease")
        self.assertAlmostEqual(result.score, 1.0, delta=TOL)
        self.assertEqual(result.violations, [])

    def test_numeric_values_for_measurement_types_are_not_penalized(self):
        for text, etype in [("7.5", "parameter"), ("120 mmHg", "parameter"), ("12", "score"), ("4.2", "examination_measure")]:
            with self.subTest(text=text, etype=etype):
                self.assertAlmostEqual(self.c(text, etype).score, 1.0, delta=TOL)

    def test_numeric_only_text_for_other_types_is_penalized(self):
        result = self.c("7.5", "disease")
        self.assertAlmostEqual(result.score, 0.8, delta=TOL)  # 1 - 0.2
        self.assertEqual(len(result.violations), 1)

    def test_type_naming_variants_are_normalized(self):
        for etype in ["Disease", "DISEASE", " disease ", "ImagingProcedure", "imaging procedure", "imaging-procedure", "Drug"]:
            with self.subTest(etype=etype):
                result = self.c("Heparin" if etype == "Drug" else "CT scan", etype)
                self.assertTrue(result.type_valid)

    def test_invalid_type_is_rejected(self):
        result = self.c("adolescents", "organ_system")
        self.assertFalse(result.type_valid)
        self.assertAlmostEqual(result.score, 0.7, delta=TOL)  # 1 - 0.3
        status, rec = self.svc._determine_status_and_recommendation(0.9, None, result, ConsistencyMetrics(score=1.0), default_config()["thresholds"])
        self.assertEqual((status, rec), ("failed", "reject"))

    def test_length_rules(self):
        # 1 char with min_length 2: shortness 0.5 -> -(0.2 + 0.6*0.5) = -0.5
        self.assertAlmostEqual(self.c("a", "disease").score, 0.5, delta=TOL)
        # 1 punctuation char: -0.5 (short) - 0.5 (symbol)
        self.assertAlmostEqual(self.c("%", "finding").score, 0.0, delta=TOL)
        # max_length 200: 201 chars -> -0.1
        self.assertAlmostEqual(self.c("x" * 201, "disease").score, 0.9, delta=TOL)

    def test_invalid_characters(self):
        self.assertAlmostEqual(self.c("<fever>", "finding").score, 0.75, delta=TOL)  # -0.25

    def test_type_heuristics(self):
        # population description (-0.25) and patient reference for a disease (-0.25)
        self.assertAlmostEqual(self.c("male and female patients", "disease").score, 0.5, delta=TOL)


class TestCompleteness(unittest.TestCase):
    """Required: type, text. Optional (weight 0.15): context, normalized_form, confidence."""

    def setUp(self):
        self.svc = service()
        self.s = entity_settings()

    def comp(self, **kw):
        args = dict(normalized_form=None, context=None, confidence=None)
        args.update(kw)
        return self.svc._evaluate_completeness("Pneumonia", "disease", args["normalized_form"], args["context"], args["confidence"], self.s)

    def test_each_optional_field_counts_once(self):
        self.assertAlmostEqual(self.comp().score, 0.85, delta=TOL)
        self.assertAlmostEqual(self.comp(context="c").score, 0.85 + 0.15 / 3, delta=TOL)
        self.assertAlmostEqual(self.comp(context="c", normalized_form="n").score, 0.85 + 0.15 * 2 / 3, delta=TOL)
        self.assertAlmostEqual(self.comp(context="c", normalized_form="n", confidence=0.4).score, 1.0, delta=TOL)

    def test_context_is_not_also_a_definition(self):
        result = self.comp(context="c")
        self.assertTrue(result.has_context)
        self.assertFalse(result.has_definition)

    def test_required_context_missing(self):
        s = entity_settings(completeness_required_fields="type,text,context")
        result = self.svc._evaluate_completeness("Pneumonia", "disease", None, None, None, s)
        # required 2/3 * 0.85 + optional 0/2 * 0.15
        self.assertAlmostEqual(result.score, 0.85 * 2 / 3, delta=TOL)
        self.assertIn("context", result.missing_fields)

    def test_unknown_required_field_is_missing_and_logged(self):
        s = entity_settings(completeness_required_fields="type,text,definition")
        with self.assertLogs("services.entity_reviewer", level="WARNING"):
            result = self.svc._evaluate_completeness("Pneumonia", "disease", None, "c", None, s)
        self.assertIn("definition", result.missing_fields)


class TestConsistency(unittest.IsolatedAsyncioTestCase):
    async def consistency(self, bert, etype="disease", confidence=0.9, run_bert=True):
        svc = service(bert=bert)
        return await svc._evaluate_consistency("Pneumonia", etype, "llm", confidence, run_bert, False, entity_settings())

    async def test_bert_agreement(self):
        result = await self.consistency(StubBert(predicted_type="disease", confidence=0.9))
        self.assertEqual(result.bert_status, "agreed")
        self.assertIs(result.bert_agreement, True)
        self.assertAlmostEqual(result.score, 1.0, delta=TOL)  # min(1, 0.9 + 0.3*0.9)
        result = await self.consistency(StubBert(predicted_type="disease", confidence=0.8), confidence=0.5)
        self.assertAlmostEqual(result.score, 0.74, delta=TOL)  # 0.5 + 0.3*0.8

    async def test_bert_disagreement(self):
        result = await self.consistency(StubBert(predicted_type="symptom", confidence=0.8))
        self.assertEqual(result.bert_status, "disagreed")
        self.assertIs(result.bert_agreement, False)
        self.assertAlmostEqual(result.score, 0.62, delta=TOL)  # min(.8,.9) - 0.75*0.3*0.8
        self.assertEqual(result.alternate_types[0]["type"], "symptom")

    async def test_bert_abstains_without_penalty(self):
        result = await self.consistency(StubBert(predicted_type=None))
        self.assertEqual(result.bert_status, "abstained")
        self.assertIsNone(result.bert_agreement)
        self.assertAlmostEqual(result.score, 0.8, delta=TOL)  # min(0.8, 0.9)

    async def test_bert_unavailable_without_penalty(self):
        result = await self.consistency(None)
        self.assertTrue(result.bert_status.startswith("unavailable"))
        self.assertAlmostEqual(result.score, 0.8, delta=TOL)
        result = await self.consistency(None, confidence=0.6)
        self.assertAlmostEqual(result.score, 0.6, delta=TOL)

    async def test_incompatible_bert_labels_are_reported_not_used(self):
        stub = StubBert(labels=MEDMENTIONS_LABELS, predicted_type="Finding")
        result = await self.consistency(stub)
        self.assertIn("no BERT entity model uses the project entity types", result.bert_status)
        self.assertEqual(stub.calls, [])
        self.assertIsNone(result.bert_agreement)
        self.assertAlmostEqual(result.score, 0.8, delta=TOL)

    async def test_model_without_the_candidate_type(self):
        stub = StubBert(labels=["O", "B-disease", "I-disease"], predicted_type="disease")
        result = await self.consistency(stub, etype="symptom")
        self.assertIn("predicts type 'symptom'", result.bert_status)
        self.assertEqual(stub.calls, [])

    async def test_bert_failure_is_logged_and_not_penalized(self):
        stub = StubBert(error=RuntimeError("CUDA out of memory"))
        with self.assertLogs("services.entity_reviewer", level="ERROR") as logs:
            result = await self.consistency(stub)
        self.assertIn("CUDA out of memory", "".join(logs.output))
        self.assertIn("inference failed", result.bert_status)
        self.assertAlmostEqual(result.score, 0.8, delta=TOL)

    async def test_not_requested(self):
        result = await self.consistency(StubBert(predicted_type="disease"), run_bert=False)
        self.assertEqual(result.bert_status, "not_requested")
        self.assertAlmostEqual(result.score, 0.8, delta=TOL)

    async def test_missing_confidence_uses_base_score(self):
        result = await self.consistency(None, confidence=None)
        self.assertAlmostEqual(result.score, 0.5, delta=TOL)


class TestEvaluateEntityWithKG(unittest.IsolatedAsyncioTestCase):
    """End-to-end scoring against a real KnowledgeGraphService on a small TTL."""

    @classmethod
    def setUpClass(cls):
        cls.kg = make_kg()

    async def evaluate(self, text, etype="disease", kg="default", **kw):
        svc = service(kg=self.kg if kg == "default" else kg)
        args = dict(context="Patient with pneumonia.", confidence=0.9, run_bert_validation=False)
        args.update(kw)
        return await svc.evaluate_entity(text, etype, **args)

    async def test_kg_exact_match(self):
        ev = await self.evaluate("Pneumonia")
        self.assertAlmostEqual(ev["congruence"].score, 1.0, delta=TOL)
        self.assertEqual(ev["congruence"].nearest_entity, "Pneumonia")
        self.assertAlmostEqual(ev["coverage"].score, 0.5, delta=TOL)
        self.assertAlmostEqual(ev["completeness"].score, 0.95, delta=TOL)  # .85 + .15*2/3
        self.assertAlmostEqual(ev["consistency"].score, 0.8, delta=TOL)
        # .25 + .15*.5 + .25 + .15*.95 + .2*.8 = 0.8775
        self.assertAlmostEqual(ev["overall_score"], 0.8775, delta=ROUNDED)
        self.assertEqual(ev["review_status"], "passed")

    async def test_kg_similar_match(self):
        ev = await self.evaluate("Pneumonias")
        # difflib ratio = 2*M/T = 2*9/19
        self.assertAlmostEqual(ev["congruence"].score, 18 / 19, delta=TOL)
        self.assertAlmostEqual(ev["overall_score"], 0.25 * 18 / 19 + 0.6275, delta=ROUNDED)

    async def test_kg_no_match(self):
        ev = await self.evaluate("Appendicitis")
        self.assertAlmostEqual(ev["congruence"].score, 0.5, delta=TOL)
        self.assertAlmostEqual(ev["coverage"].score, 0.8, delta=TOL)
        self.assertTrue(ev["coverage"].is_novel)
        # .125 + .12 + .25 + .1425 + .16
        self.assertAlmostEqual(ev["overall_score"], 0.7975, delta=ROUNDED)

    async def test_matching_is_type_aware(self):
        # "Heparin" exists as a Substance; proposed as a disease it has no same-type match.
        ev = await self.evaluate("Heparin", "disease")
        self.assertAlmostEqual(ev["congruence"].score, 0.5, delta=TOL)
        ev = await self.evaluate("Heparin", "substance")
        self.assertAlmostEqual(ev["congruence"].score, 1.0, delta=TOL)

    async def test_kg_unavailable(self):
        with self.assertLogs("services.entity_reviewer", level="WARNING"):
            ev = await self.evaluate("Pneumonia", kg=None)
        self.assertIsNone(ev["congruence"])
        self.assertIsNone(ev["coverage"])
        # (.25*1 + .15*.95 + .2*.8) / .6
        self.assertAlmostEqual(ev["overall_score"], 0.5525 / 0.6, delta=ROUNDED)

    async def test_kg_failure_is_logged_not_scored_as_no_match(self):
        with self.assertLogs("services.entity_reviewer", level="ERROR"):
            ev = await self.evaluate("Pneumonia", kg=FailingKG())
        self.assertIsNone(ev["congruence"])
        self.assertIsNone(ev["coverage"])


class TestStatusBoundaries(unittest.TestCase):
    def setUp(self):
        self.svc = service()
        self.t = default_config()["thresholds"]
        self.ok = ConstraintMetrics(score=1.0)

    def status(self, overall, congruence=None, bert_agreement=None, constraint=None):
        return self.svc._determine_status_and_recommendation(
            overall, congruence, constraint or self.ok, ConsistencyMetrics(score=0.5, bert_agreement=bert_agreement), self.t
        )

    def test_threshold_edges(self):
        self.assertEqual(self.status(0.75), ("passed", "approve"))
        self.assertEqual(self.status(0.749), ("needs_review", "manual_review"))
        self.assertEqual(self.status(0.50), ("needs_review", "manual_review"))
        self.assertEqual(self.status(0.4999), ("needs_review", "manual_review"))
        self.assertEqual(self.status(0.25), ("needs_review", "manual_review"))
        self.assertEqual(self.status(0.2499), ("failed", "reject"))

    def test_review_band_recommendations(self):
        self.assertEqual(self.status(0.6, bert_agreement=False), ("needs_review", "verify_type"))
        self.assertEqual(self.status(0.6, congruence=CongruenceMetrics(score=0.96)), ("needs_review", "check_duplicate"))

    def test_constraint_gate(self):
        self.assertEqual(self.status(0.99, constraint=ConstraintMetrics(score=0.39)), ("failed", "reject"))
        self.assertEqual(self.status(0.99, constraint=ConstraintMetrics(score=0.4)), ("passed", "approve"))


if __name__ == "__main__":
    unittest.main()
