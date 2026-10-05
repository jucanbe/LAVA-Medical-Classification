"""Relation review scorecard, against a real KnowledgeGraphService.

The test KG contains the triple Pneumonia --has_symptom--> Fever and the
entities Cough, Heparin, Chest X-ray and Heart rate. Expected values are
worked out by hand from the documented formulas.
"""
import unittest
from unittest.mock import AsyncMock

from tests.support import make_kg, FailingKG, relation_settings, TEST_KG_TTL

from models.review_defaults import RELATION_WEIGHTS, THRESHOLDS, RELATION_SETTINGS
from services.relation_reviewer import RelationReviewService

TOL = 1e-9
ROUNDED = 6e-4


def service(kg=None, settings=None) -> RelationReviewService:
    svc = RelationReviewService(kg_service=kg)
    svc._get_config = AsyncMock(return_value={
        "weights": dict(RELATION_WEIGHTS),
        "thresholds": dict(THRESHOLDS),
        "settings": dict(settings or RELATION_SETTINGS),
    })
    return svc


class TestRelationCongruence(unittest.TestCase):
    def setUp(self):
        self.svc = service()
        self.s = relation_settings()

    def c(self, src, rel, tgt, evidence=None):
        return self.svc._evaluate_congruence(src, rel, tgt, self.s, evidence)

    def test_valid_schema(self):
        self.assertAlmostEqual(self.c("disease", "has_symptom", "symptom"), 0.9, delta=TOL)
        self.assertAlmostEqual(self.c("Disease", "hasSymptom", "Symptom"), 0.9, delta=TOL)  # naming variants

    def test_invalid_schema(self):
        # treats: sources {substance, therapy, therapeutic_procedure}, targets {disease, symptom}
        self.assertAlmostEqual(self.c("symptom", "treats", "disease"), 0.4, delta=TOL)  # target side allowed only
        self.assertAlmostEqual(self.c("organ", "treats", "organ"), 0.0, delta=TOL)      # 0.3 - 0.3
        self.assertAlmostEqual(self.c("disease", "cures", "disease"), 0.2, delta=TOL)  # unknown relation 0.5 - 0.3
        # both sides allowed but not as this pair: substance -> symptom is valid, therapy -> symptom is not
        self.assertAlmostEqual(self.c("therapy", "treats", "symptom"), 0.6, delta=TOL)

    def test_kg_bonus_requires_the_triple(self):
        no_triple = {"triple_exists": False, "source_known": True, "target_known": True, "source_relation_count": 1}
        with_triple = dict(no_triple, triple_exists=True)
        self.assertAlmostEqual(self.c("disease", "has_symptom", "symptom", no_triple), 0.9, delta=TOL)
        self.assertAlmostEqual(self.c("disease", "has_symptom", "symptom", with_triple), 1.0, delta=TOL)  # 0.9 + 0.2*0.5


class TestRelationWithKG(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.kg = make_kg()

    async def evaluate(self, src, stype, rel, tgt, ttype, kg="default", **kw):
        svc = service(self.kg if kg == "default" else kg)
        args = dict(confidence=0.9, context="Pneumonia with fever.", source="llm")
        args.update(kw)
        return await svc.evaluate_relation(src, stype, rel, tgt, ttype, **args)

    async def test_known_triple(self):
        ev = await self.evaluate("Pneumonia", "disease", "has_symptom", "Fever", "symptom")
        self.assertTrue(ev["kg_evidence"]["triple_exists"])
        self.assertAlmostEqual(ev["congruence_score"], 1.0, delta=TOL)
        self.assertAlmostEqual(ev["coverage_score"], 0.3, delta=TOL)
        # .25*1 + .15*.3 + .25*1 + .15*1 + .2*.8
        self.assertAlmostEqual(ev["overall_score"], 0.855, delta=ROUNDED)
        self.assertEqual(ev["review_status"], "passed")

    async def test_novel_triple_between_known_entities(self):
        ev = await self.evaluate("Pneumonia", "disease", "has_symptom", "Cough", "symptom")
        self.assertTrue(ev["kg_evidence"]["source_known"])
        self.assertTrue(ev["kg_evidence"]["target_known"])
        self.assertFalse(ev["kg_evidence"]["triple_exists"])
        self.assertAlmostEqual(ev["coverage_score"], 0.8, delta=TOL)  # 0.5 + 0.3
        self.assertAlmostEqual(ev["congruence_score"], 0.9, delta=TOL)

    async def test_completely_novel_entities_and_relation(self):
        ev = await self.evaluate("Appendicitis", "disease", "has_symptom", "Abdominal pain", "symptom")
        self.assertFalse(ev["kg_evidence"]["source_known"])
        self.assertFalse(ev["kg_evidence"]["target_known"])
        self.assertAlmostEqual(ev["coverage_score"], 0.8, delta=TOL)

    async def test_known_and_novel_triples_differ(self):
        known = await self.evaluate("Pneumonia", "disease", "has_symptom", "Fever", "symptom")
        novel = await self.evaluate("Pneumonia", "disease", "has_symptom", "Cough", "symptom")
        self.assertGreaterEqual(novel["coverage_score"] - known["coverage_score"], 0.5 - TOL)

    async def test_saturated_source_gets_half_bonus(self):
        ttl = TEST_KG_TTL + """
med:dyspnea a medtype:Finding ; rdfs:label "Dyspnea" .
med:chills a medtype:Finding ; rdfs:label "Chills" .
med:pneumonia rel:has_symptom med:cough , med:dyspnea .
"""
        ev = await self.evaluate("Pneumonia", "disease", "has_symptom", "Chills", "symptom", kg=make_kg(ttl))
        self.assertEqual(ev["kg_evidence"]["source_relation_count"], 3)
        self.assertAlmostEqual(ev["coverage_score"], 0.65, delta=TOL)  # 0.5 + 0.3/2

    async def test_kg_unavailable(self):
        with self.assertLogs("services.relation_reviewer", level="WARNING"):
            ev = await self.evaluate("Pneumonia", "disease", "has_symptom", "Fever", "symptom", kg=None)
        self.assertIsNone(ev["coverage_score"])
        self.assertAlmostEqual(ev["congruence_score"], 0.9, delta=TOL)
        # (.25*.9 + .25*1 + .15*1 + .2*.8) / .85
        self.assertAlmostEqual(ev["overall_score"], 0.785 / 0.85, delta=ROUNDED)

    async def test_kg_failure_is_logged(self):
        with self.assertLogs("services.relation_reviewer", level="ERROR"):
            ev = await self.evaluate("Pneumonia", "disease", "has_symptom", "Fever", "symptom", kg=FailingKG())
        self.assertIsNone(ev["coverage_score"])


class TestRelationConstraint(unittest.TestCase):
    def setUp(self):
        self.svc = service()

    def c(self, src="Heparin", stype="substance", rel="treats", tgt="Pneumonia", ttype="disease",
          confidence=0.9, context="ctx", **settings):
        return self.svc._evaluate_constraint(src, stype, rel, tgt, ttype, confidence, context, relation_settings(**settings))

    def test_valid(self):
        self.assertAlmostEqual(self.c(), 1.0, delta=TOL)

    def test_violations(self):
        self.assertAlmostEqual(self.c(rel="cures"), 0.7, delta=TOL)                    # unknown relation -0.3
        self.assertAlmostEqual(self.c(stype="gadget"), 0.75, delta=TOL)                # invalid source type -0.25
        self.assertAlmostEqual(self.c(src="pneumonia"), 0.7, delta=TOL)                # self reference -0.3
        self.assertAlmostEqual(self.c(src="Pneumonia", stype="disease", tgt="Heparin", ttype="substance"), 0.75, delta=TOL)  # reversed
        self.assertAlmostEqual(self.c(confidence=0.2), 0.75, delta=TOL)                # below min confidence 0.3
        self.assertAlmostEqual(self.c(context=None, constraint_require_context=True), 0.875, delta=TOL)
        self.assertAlmostEqual(self.c(src="H"), 0.75, delta=TOL)                       # too short


class TestRelationCompleteness(unittest.TestCase):
    def test_optional_fields(self):
        svc = service()
        s = relation_settings()
        args = ("Heparin", "substance", "treats", "Pneumonia", "disease")
        self.assertAlmostEqual(svc._evaluate_completeness(*args, None, None, s), 0.85, delta=TOL)
        self.assertAlmostEqual(svc._evaluate_completeness(*args, 0.8, None, s), 0.925, delta=TOL)
        self.assertAlmostEqual(svc._evaluate_completeness(*args, 0.8, "ctx", s), 1.0, delta=TOL)
        # A missing required field: 4/5 * 0.85 + 0
        self.assertAlmostEqual(svc._evaluate_completeness("Heparin", "", "treats", "Pneumonia", "disease", None, None, s), 0.68, delta=TOL)


class TestRelationConsistency(unittest.IsolatedAsyncioTestCase):
    async def test_every_extractor_is_treated_the_same(self):
        svc = service()
        for confidence, expected in [(0.95, 0.8), (0.6, 0.6), (None, 0.5)]:
            scores = set()
            for source in ["llm", "langextract", "bert", None]:
                ev = await svc.evaluate_relation("Heparin", "substance", "treats", "Pneumonia", "disease",
                                                 confidence=confidence, context="c", source=source)
                scores.add(ev["consistency_score"])
            with self.subTest(confidence=confidence):
                self.assertEqual(len(scores), 1)
                self.assertAlmostEqual(scores.pop(), expected, delta=TOL)


class TestRelationStatus(unittest.TestCase):
    def test_boundaries(self):
        svc = service()
        t = dict(THRESHOLDS)
        self.assertEqual(svc._determine_status(0.75, 0.9, 1.0, t), ("passed", "approve"))
        self.assertEqual(svc._determine_status(0.7499, 0.9, 1.0, t), ("needs_review", "manual_review"))
        self.assertEqual(svc._determine_status(0.6, 0.4, 1.0, t), ("needs_review", "verify_type"))
        self.assertEqual(svc._determine_status(0.4999, 0.9, 1.0, t), ("failed", "reject"))
        self.assertEqual(svc._determine_status(0.9, 0.9, 0.2999, t), ("failed", "reject"))
        self.assertEqual(svc._determine_status(0.9, 0.1999, 1.0, t), ("failed", "reject"))


if __name__ == "__main__":
    unittest.main()
