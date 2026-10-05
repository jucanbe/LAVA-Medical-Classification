"""Numerical validation: the audit cases re-run against the fixed code.

Each case pairs an independently hand-computed expectation with the value
the program produces. CASES is also used to print the validation table.
"""
import asyncio
import unittest
from difflib import SequenceMatcher
from unittest.mock import AsyncMock

from tests.support import make_kg, StubBert, default_config, entity_settings, relation_settings

from models.entities import KGMatch
from models.review_defaults import THRESHOLDS, RELATION_SETTINGS
from tests.support import FORMULA_WEIGHTS
from services.entity_reviewer import EntityReviewService
from services.relation_reviewer import RelationReviewService

TOLERANCE = 1e-9


def _run(coro):
    return asyncio.run(coro)


def _entity(bert=None, kg=None):
    svc = EntityReviewService(kg_service=kg, bert_service=bert)
    svc._get_config = AsyncMock(return_value=default_config())
    return svc


def _relation(kg=None):
    svc = RelationReviewService(kg_service=kg)
    svc._get_config = AsyncMock(return_value={"weights": dict(FORMULA_WEIGHTS), "thresholds": dict(THRESHOLDS), "settings": dict(RELATION_SETTINGS)})
    return svc


def _m(score):
    return KGMatch(kg_uri="u", kg_label="x", similarity_score=score)


def t1():
    s = {"congruence": .8, "coverage": .6, "constraint": 1.0, "completeness": .9, "consistency": .7}
    return _entity()._calculate_overall_score(s, FORMULA_WEIGHTS)


def t2():
    return _entity()._evaluate_completeness("pneumonia", "disease", None, "ctx", None, entity_settings()).score


def t3(score):
    return _entity()._evaluate_congruence("x", "disease", entity_settings(), [_m(score)] if score is not None else []).score


def t4(scores):
    return _entity()._evaluate_coverage("x", "disease", entity_settings(), [_m(v) for v in scores]).score


def t5(text, etype):
    return _entity()._evaluate_constraint(text, etype, entity_settings()).score


def t6(bert, confidence):
    return _run(_entity(bert=bert)._evaluate_consistency("pneumonia", "disease", "llm", confidence, True, False, entity_settings())).score


def r1(kg):
    return _run(_relation(kg).evaluate_relation("Pneumonia", "disease", "has_symptom", "Cough", "symptom", confidence=.9))["congruence_score"]


def r2(kg, target):
    return _run(_relation(kg).evaluate_relation("Pneumonia", "disease", "has_symptom", target, "symptom", confidence=.9))["coverage_score"]


def r3(source, confidence):
    return _relation()._evaluate_consistency(confidence, relation_settings()) if source else None


def r4():
    return _relation()._evaluate_completeness("a", "disease", "treats", "b", "disease", .8, None, relation_settings())


def k1():
    return SequenceMatcher(None, "pneumonia", "pneumonias").ratio()


_KG = None


def kg():
    global _KG
    if _KG is None:
        _KG = make_kg()
    return _KG


# (id, description, expected, callable). Expected values are hand-derived.
CASES = [
    ("T1", "overall = sum(w_i*s_i)/sum(w_i), s=(.8,.6,1,.9,.7)", 0.815, t1),
    ("T2", "completeness type+text+context: .85 + .15*1/3", 0.9, t2),
    ("T3a", "congruence best match 0.80", 0.8, lambda: t3(0.8)),
    ("T3b", "congruence best 0.96: min(1, .96+.2)", 1.0, lambda: t3(0.96)),
    ("T3c", "congruence no match", 0.5, lambda: t3(None)),
    ("T4a", "coverage no matches: .5+.3", 0.8, lambda: t4([])),
    ("T4b", "coverage one match 0.6: .5+.3*.7", 0.71, lambda: t4([.6])),
    ("T4c", "coverage one match 0.85", 0.5, lambda: t4([.85])),
    ("T4d", "coverage 3 matches best .6: .6-.3*.6", 0.42, lambda: t4([.6, .6, .6])),
    ("T5a", "constraint '7.5' parameter (numeric allowed)", 1.0, lambda: t5("7.5", "parameter")),
    ("T5b", "constraint '7.5' disease: 1-.2", 0.8, lambda: t5("7.5", "disease")),
    ("T5c", "constraint 'Pneumonia' 'Disease'", 1.0, lambda: t5("Pneumonia", "Disease")),
    ("T5d", "constraint 'pneumonia' 'ImagingProcedure' (valid type)", 1.0, lambda: t5("pneumonia", "ImagingProcedure")),
    ("T5e", "constraint invalid type organ_system: 1-.3", 0.7, lambda: t5("adolescents", "organ_system")),
    ("T6a", "consistency BERT agrees c=.9 p=.9: min(1,.9+.27)", 1.0, lambda: t6(StubBert(predicted_type="disease", confidence=.9), .9)),
    ("T6b", "consistency BERT disagrees c=.9 p=.8: .8-.225*.8", 0.62, lambda: t6(StubBert(predicted_type="symptom", confidence=.8), .9)),
    ("T6c", "consistency BERT unavailable c=.95: min(.8,.95)", 0.8, lambda: t6(None, .95)),
    ("T6d", "consistency BERT abstains c=.6", 0.6, lambda: t6(StubBert(predicted_type=None), .6)),
    ("R1a", "rel. congruence valid combo, KG present, triple absent", 0.9, lambda: r1(kg())),
    ("R1b", "rel. congruence valid combo, no KG", 0.9, lambda: r1(None)),
    ("R2a", "rel. coverage known triple", 0.3, lambda: r2(kg(), "Fever")),
    ("R2b", "rel. coverage new triple between known entities: .5+.3", 0.8, lambda: r2(kg(), "Cough")),
    ("R3a", "rel. consistency llm c=.95: min(.8,.95)", 0.8, lambda: r3("llm", .95)),
    ("R3b", "rel. consistency langextract c=.95", 0.8, lambda: r3("langextract", .95)),
    ("R4", "rel. completeness no context, c=.8: .85+.15*.5", 0.925, r4),
    ("K1", "difflib ratio 2M/T = 18/19", 18 / 19, k1),
]


class TestNumericalValidation(unittest.TestCase):
    def test_cases(self):
        for case_id, description, expected, fn in CASES:
            with self.subTest(case=case_id, description=description):
                self.assertAlmostEqual(fn(), expected, delta=TOLERANCE)


if __name__ == "__main__":
    unittest.main()
