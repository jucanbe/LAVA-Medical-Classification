"""One regression test per bug fixed in the 2.0 scoring update.

Each test fails if the old behavior comes back.
"""
import unittest
from unittest.mock import AsyncMock, patch

from tests.support import make_kg, StubBert, default_config, entity_settings, relation_settings, TEST_KG_TTL

from database.models import EntityReviewDB
from services.entity_reviewer import EntityReviewService
from services.relation_reviewer import RelationReviewService

TOL = 1e-9


class RealShapedBert(StubBert):
    """Exactly the public surface of BERTNERService used by the reviewer: no classify_text()."""


class TestBertRegression(unittest.IsolatedAsyncioTestCase):
    async def test_bert_validation_runs_and_records_agreement(self):
        # Before: classify_text() did not exist -> AttributeError swallowed ->
        # bert_agreement NULL and Consistency - 0.1 on every review.
        bert = RealShapedBert(predicted_type="disease", confidence=0.9)
        self.assertFalse(hasattr(bert, "classify_text"))
        svc = EntityReviewService(bert_service=bert)
        result = await svc._evaluate_consistency("Pneumonia", "disease", "llm", 0.9, True, False, entity_settings())
        self.assertEqual(len(bert.calls), 1)
        self.assertIsNotNone(result.bert_agreement)
        self.assertAlmostEqual(result.score, 1.0, delta=TOL)
        self.assertNotAlmostEqual(result.score, 0.8 - 0.1, delta=1e-6)

    async def test_failed_bert_never_subtracts(self):
        for bert in [StubBert(error=AttributeError("classify_text")), StubBert(predicted_type=None), None]:
            svc = EntityReviewService(bert_service=bert)
            result = await svc._evaluate_consistency("Pneumonia", "disease", "llm", 0.9, True, False, entity_settings())
            self.assertAlmostEqual(result.score, 0.8, delta=TOL)


class TestRelationKGRegression(unittest.IsolatedAsyncioTestCase):
    async def test_router_service_has_the_shared_kg(self):
        # Before: `from services.kg_service import ...` failed silently -> no KG -> Coverage 0.7 always.
        from services import knowledge_graph
        from routers.relation_review import get_review_service
        kg = make_kg()
        with patch.object(knowledge_graph, "_kg_service", kg):
            svc = get_review_service()
            self.assertIs(svc.kg_service, kg)
            svc._get_config = AsyncMock(return_value=default_config(settings=relation_settings()))
            known = await svc.evaluate_relation("Pneumonia", "disease", "has_symptom", "Fever", "symptom", confidence=0.9)
            novel = await svc.evaluate_relation("Pneumonia", "disease", "has_symptom", "Cough", "symptom", confidence=0.9)
        self.assertAlmostEqual(known["coverage_score"], 0.3, delta=TOL)
        self.assertAlmostEqual(novel["coverage_score"], 0.8, delta=TOL)
        self.assertNotEqual(known["coverage_score"], 0.7)

    async def test_kg_initialization_failure_is_logged(self):
        from routers import relation_review
        with patch("services.knowledge_graph.get_kg_service", side_effect=RuntimeError("rdflib missing")):
            with self.assertLogs("routers.relation_review", level="ERROR"):
                svc = relation_review.get_review_service()
        self.assertIsNone(svc.kg_service)


class TestFakeKGBonusRegression(unittest.IsolatedAsyncioTestCase):
    async def test_no_bonus_just_because_a_kg_exists(self):
        # Before: +0.1 whenever kg_service was not None.
        empty_kg = make_kg(TEST_KG_TTL.split("med:pneumonia rel:has_symptom")[0])
        svc = RelationReviewService(kg_service=empty_kg)
        svc._get_config = AsyncMock(return_value=default_config(settings=relation_settings()))
        ev = await svc.evaluate_relation("Pneumonia", "disease", "has_symptom", "Fever", "symptom", confidence=0.9)
        self.assertFalse(ev["kg_evidence"]["triple_exists"])
        self.assertAlmostEqual(ev["congruence_score"], 0.9, delta=TOL)


class TestCompletenessRegression(unittest.TestCase):
    def test_context_counts_once(self):
        # Before: context also set has_definition -> 0.925 instead of 0.9.
        result = EntityReviewService()._evaluate_completeness("Pneumonia", "disease", None, "ctx", None, entity_settings())
        self.assertAlmostEqual(result.score, 0.85 + 0.15 / 3, delta=TOL)
        self.assertNotAlmostEqual(result.score, 0.925, delta=1e-6)
        self.assertFalse(result.has_definition)


class TestTypeValidRegression(unittest.TestCase):
    def test_false_is_preserved(self):
        # Before: `constraint_type_valid or True` always returned True.
        from routers.entity_review import _convert_to_response
        row = EntityReviewDB(id=1, entity_text="adolescents", entity_type="organ_system",
                             constraint_score=0.7, constraint_type_valid=False, overall_score=0.6, review_status="failed")
        self.assertIs(_convert_to_response(row).constraint.type_valid, False)
        row.constraint_type_valid = None
        self.assertIs(_convert_to_response(row).constraint.type_valid, True)


class TestTypeConstraintRegression(unittest.TestCase):
    def test_numeric_measurements_and_type_variants(self):
        # Before: numeric parameters were penalized via a check for a non-existent
        # "quantitative_measure" type, and CamelCase types were rejected.
        svc = EntityReviewService()
        self.assertAlmostEqual(svc._evaluate_constraint("7.5", "parameter", entity_settings()).score, 1.0, delta=TOL)
        self.assertTrue(svc._evaluate_constraint("CT angiography", "ImagingProcedure", entity_settings()).type_valid)
        self.assertFalse(svc._evaluate_constraint("adolescents", "organ_system", entity_settings()).type_valid)


class TestRelationConsistencyRegression(unittest.TestCase):
    def test_langextract_is_capped_like_llm(self):
        # Before: llm/bert capped at 0.85, langextract uncapped (0.95).
        svc = RelationReviewService()
        self.assertEqual(svc._evaluate_consistency(0.95, relation_settings()), 0.8)


class TestKGFailureRegression(unittest.IsolatedAsyncioTestCase):
    async def test_kg_error_is_not_scored_as_no_match(self):
        # Before: a failing KG lookup became [] -> Congruence 0.5 / Coverage 0.8 ("novel").
        from tests.support import FailingKG
        svc = EntityReviewService(kg_service=FailingKG())
        svc._get_config = AsyncMock(return_value=default_config())
        with self.assertLogs("services.entity_reviewer", level="ERROR"):
            ev = await svc.evaluate_entity("Pneumonia", "disease", run_bert_validation=False)
        self.assertIsNone(ev["coverage"])


if __name__ == "__main__":
    unittest.main()
