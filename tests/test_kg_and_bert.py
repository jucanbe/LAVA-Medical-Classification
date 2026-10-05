"""Knowledge Graph matching and real BERT integration."""
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from tests import PROJECT_ROOT, TEST_TMP
from tests.support import make_kg, build_tiny_bert, TINY_FAVORED_LOGIT, default_config, entity_settings

from models.entities import MedicalEntity, MedicalEntityType
from services.bert_ner import BERTNERService, ENTITY_LABELS
from services.entity_reviewer import EntityReviewService

MEDMENTIONS_DIR = PROJECT_ROOT / "BERT_models" / "Entities" / "MedMentions"


class TestKGMatching(unittest.TestCase):
    def setUp(self):
        self.kg = make_kg()

    def labels(self, text, entity_type=None, min_score=0.5):
        return [m.kg_label for m in self.kg.find_matches(text, max_matches=10, min_score=min_score, entity_type=entity_type)]

    def test_schema_classes_are_not_candidate_entities(self):
        uris = set(self.kg._entities_cache)
        self.assertNotIn("http://example.org/medical/types/Finding", uris)
        self.assertEqual(self.labels("Finding"), [])
        self.assertEqual(self.labels("Fever finding class"), [])  # altLabel of a class
        self.assertEqual(self.labels("fever"), ["Fever"])          # the real entity still matches

    def test_legitimate_matching_is_preserved(self):
        self.assertEqual(self.labels("Lung infection"), ["Lung infection"])  # altLabel match, untyped query
        self.assertEqual(self.labels("pneumonia", "disease"), ["Pneumonia"])

    def test_type_aware_filtering(self):
        self.assertEqual(self.labels("Heparin", "disease"), [])
        self.assertEqual(self.labels("Heparin", "substance"), ["Heparin"])
        self.assertEqual(self.labels("Fever", "symptom"), ["Fever"])  # symptom accepts Finding

    def test_fine_grained_types_written_by_the_app_are_matched(self):
        self.kg.add_entity_to_kg(MedicalEntity(text="Night sweats", entity_type=MedicalEntityType.SYMPTOM, confidence=1.0))
        self.kg.add_entity_to_kg(MedicalEntity(text="Left lung", entity_type=MedicalEntityType.ORGAN, confidence=1.0))
        self.assertEqual(self.labels("Night sweats", "symptom"), ["Night sweats"])
        self.assertEqual(self.labels("Left lung", "organ"), ["Left lung"])

    def test_triple_lookup(self):
        known = self.kg.find_relation_triples("Pneumonia", "has_symptom", "Fever", "disease", "symptom")
        self.assertEqual(known, {"source_known": True, "target_known": True, "triple_exists": True, "source_relation_count": 1})
        novel = self.kg.find_relation_triples("Pneumonia", "has_symptom", "Cough", "disease", "symptom")
        self.assertFalse(novel["triple_exists"])
        self.assertFalse(self.kg.find_relation_triples("Fever", "has_symptom", "Pneumonia", "symptom", "disease")["triple_exists"])

    def test_added_relations_are_found_and_normalized(self):
        self.kg.add_relation_to_kg("Heparin", "substance", "Treats", "Pneumonia", "disease")
        self.assertTrue(self.kg.find_relation_triples("Heparin", "treats", "Pneumonia", "substance", "disease")["triple_exists"])

    def test_save_kg_writes_to_the_service_directory(self):
        path = Path(self.kg.save_kg())
        self.assertTrue(str(path).startswith(str(TEST_TMP)))


class TestTinyBertConfidence(unittest.TestCase):
    """A real BertForTokenClassification whose output probabilities are known."""

    @classmethod
    def setUpClass(cls):
        cls.models_dir = Path(tempfile.mkdtemp(prefix="bert-", dir=TEST_TMP))
        build_tiny_bert(cls.models_dir, "tiny_project", favored_label="B-disease")
        cls.bert = BERTNERService(str(cls.models_dir))

    def test_confidence_is_mean_softmax_probability(self):
        entities, _ = self.bert.classify("pneumonia", "tiny_project")
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]["type"], "disease")
        # Every token has logits [0, ..., 10, ..., 0] over 29 labels.
        n = len(ENTITY_LABELS)
        p = math.exp(TINY_FAVORED_LOGIT) / (math.exp(TINY_FAVORED_LOGIT) + (n - 1))
        self.assertAlmostEqual(entities[0]["confidence"], p, delta=1e-6)
        self.assertNotAlmostEqual(entities[0]["confidence"], 0.9, delta=1e-3)

    def test_available_models_expose_project_labels(self):
        models = self.bert.get_available_models("entity")
        self.assertEqual([m["name"] for m in models], ["tiny_project"])
        self.assertEqual(BERTNERService.entity_types_from_labels(models[0]["labels"]),
                         BERTNERService.entity_types_from_labels(ENTITY_LABELS))


class TestReviewerWithRealBert(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.models_dir = Path(tempfile.mkdtemp(prefix="bert-", dir=TEST_TMP))
        build_tiny_bert(cls.models_dir, "tiny_project", favored_label="B-disease")
        n = len(ENTITY_LABELS)
        cls.p = math.exp(TINY_FAVORED_LOGIT) / (math.exp(TINY_FAVORED_LOGIT) + (n - 1))

    def reviewer(self, bert):
        svc = EntityReviewService(bert_service=bert)
        svc._get_config = AsyncMock(return_value=default_config())
        return svc

    async def test_agreement_and_disagreement(self):
        svc = self.reviewer(BERTNERService(str(self.models_dir)))
        agreed = await svc._evaluate_consistency("Pneumonia", "disease", "llm", 0.5, True, False, entity_settings())
        self.assertEqual(agreed.bert_status, "agreed")
        self.assertAlmostEqual(agreed.score, 0.5 + 0.3 * self.p, delta=1e-6)
        disagreed = await svc._evaluate_consistency("Pneumonia", "symptom", "llm", 0.9, True, False, entity_settings())
        self.assertEqual(disagreed.bert_status, "disagreed")
        self.assertAlmostEqual(disagreed.score, 0.8 - 0.225 * self.p, delta=1e-6)
        self.assertAlmostEqual(disagreed.alternate_types[0]["confidence"], self.p, delta=1e-6)

    @unittest.skipUnless(MEDMENTIONS_DIR.exists(), "shipped MedMentions model not present")
    async def test_shipped_medmentions_model_is_detected_as_incompatible(self):
        svc = self.reviewer(BERTNERService(str(PROJECT_ROOT / "BERT_models")))
        result = await svc._evaluate_consistency("Pneumonia", "disease", "llm", 0.9, True, False, entity_settings())
        self.assertIn("no BERT entity model uses the project entity types (found: MedMentions)", result.bert_status)
        self.assertIsNone(result.bert_agreement)
        self.assertAlmostEqual(result.score, 0.8, delta=1e-9)


@unittest.skipUnless(MEDMENTIONS_DIR.exists(), "shipped MedMentions model not present")
class TestShippedModelProbabilities(unittest.TestCase):
    def test_real_model_reports_probabilities(self):
        bert = BERTNERService(str(PROJECT_ROOT / "BERT_models"))
        entities, _ = bert.classify(
            "The patient presented with fever, cough and elevated troponin after heparin infusion.", "MedMentions"
        )
        self.assertTrue(entities)
        for entity in entities:
            self.assertGreater(entity["confidence"], 0.0)
            self.assertLessEqual(entity["confidence"], 1.0)
        self.assertFalse(all(abs(e["confidence"] - 0.9) < 1e-9 for e in entities))


if __name__ == "__main__":
    unittest.main()
