"""Shared fixtures: a small on-disk KG, BERT stubs and a tiny real BERT model."""
import json
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from tests import PROJECT_ROOT, TEST_TMP

from models.review_defaults import ENTITY_SETTINGS, RELATION_SETTINGS, ENTITY_WEIGHTS, THRESHOLDS  # noqa: F401

# Fixed weights used by the hand-computed expectations in the tests (the pre-2.1 defaults).
# Kept separate from ENTITY_WEIGHTS so the formula tests do not depend on the shipped defaults.
FORMULA_WEIGHTS = {"congruence": 0.25, "coverage": 0.15, "constraint": 0.25, "completeness": 0.15, "consistency": 0.20}
from services.bert_ner import ENTITY_LABELS

TEST_KG_TTL = """
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix med: <http://example.org/medical/> .
@prefix medtype: <http://example.org/medical/types/> .
@prefix rel: <http://example.org/medical/relations/> .

medtype:Disease a owl:Class ; rdfs:label "Disease" .
medtype:Finding a owl:Class ; rdfs:label "Finding" ; skos:altLabel "Fever finding class" .
medtype:Substance a owl:Class ; rdfs:label "Substance" .
medtype:Procedure a owl:Class ; rdfs:label "Procedure" .
medtype:QuantitativeMeasure a owl:Class ; rdfs:label "QuantitativeMeasure" .

med:pneumonia a medtype:Disease ; rdfs:label "Pneumonia" ; skos:altLabel "Lung infection" .
med:fever a medtype:Finding ; rdfs:label "Fever" .
med:cough a medtype:Finding ; rdfs:label "Cough" .
med:heparin a medtype:Substance ; rdfs:label "Heparin" .
med:chest_xray a medtype:Procedure ; rdfs:label "Chest X-ray" .
med:heart_rate a medtype:QuantitativeMeasure ; rdfs:label "Heart rate" .

med:pneumonia rel:has_symptom med:fever .
"""

ALL_PROJECT_LABELS = list(ENTITY_LABELS)

# Labels of the shipped MedMentions model (UMLS semantic types), abbreviated.
MEDMENTIONS_LABELS = [
    "O", "B-AnatomicalStructure", "I-AnatomicalStructure", "B-Chemical", "I-Chemical",
    "B-Finding", "I-Finding", "B-InjuryOrPoisoning", "I-InjuryOrPoisoning",
]


def entity_settings(**overrides) -> dict:
    settings = dict(ENTITY_SETTINGS)
    settings.update(overrides)
    return settings


def relation_settings(**overrides) -> dict:
    settings = dict(RELATION_SETTINGS)
    settings.update(overrides)
    return settings


def default_config(weights=None, settings=None) -> dict:
    return {
        "weights": dict(weights or FORMULA_WEIGHTS),
        "thresholds": dict(THRESHOLDS),
        "settings": dict(settings or ENTITY_SETTINGS),
    }


def make_kg_dir(ttl: str = TEST_KG_TTL) -> Path:
    directory = Path(tempfile.mkdtemp(prefix="kg-", dir=TEST_TMP))
    (directory / "test_kg.ttl").write_text(ttl, encoding="utf-8")
    return directory


def make_kg(ttl: str = TEST_KG_TTL):
    from services.knowledge_graph import KnowledgeGraphService
    kg = KnowledgeGraphService(str(make_kg_dir(ttl)))
    kg.load_ttl_files()
    return kg


class StubBert:
    """Stands in for BERTNERService: same method names and return shapes."""

    def __init__(self, labels: Optional[List[str]] = None, predicted_type: Optional[str] = None,
                 confidence: float = 0.9, error: Optional[Exception] = None, model_name: str = "stub"):
        self.labels = ALL_PROJECT_LABELS if labels is None else labels
        self.predicted_type = predicted_type
        self.confidence = confidence
        self.error = error
        self.model_name = model_name
        self.calls = []

    def get_available_models(self, model_type=None):
        return [{"name": self.model_name, "labels": self.labels, "model_type": "entity"}]

    def classify(self, text, model_name=None):
        self.calls.append((text, model_name))
        if self.error:
            raise self.error
        if self.predicted_type is None:
            return [], 1.0
        return [{
            "text": text, "type": self.predicted_type,
            "start_pos": 0, "end_pos": len(text), "confidence": self.confidence,
        }], 1.0


class FailingKG:
    """A KG service whose lookups raise, e.g. an unreachable triple store."""

    async def find_matches_async(self, *args, **kwargs):
        raise RuntimeError("triple store unreachable")

    def find_relation_triples(self, *args, **kwargs):
        raise RuntimeError("triple store unreachable")


TINY_FAVORED_LOGIT = 10.0


# WordPiece tokenizer files (BiomedBERT vocabulary, 30,522 entries) kept with the tests so that
# they do not depend on any trained model being present.
TOKENIZER_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "tokenizer"


def build_tiny_bert(models_dir: Path, name: str = "tiny_project", favored_label: str = "B-disease") -> Path:
    """A real (tiny) BertForTokenClassification with the project label set.

    The classifier weights are zero and only the favored label has a bias, so
    every token's logits are exactly [0, ..., TINY_FAVORED_LOGIT, ..., 0] and
    its softmax probability is known in closed form.
    """
    import torch
    from transformers import BertConfig, BertForTokenClassification

    source_tokenizer = TOKENIZER_FIXTURE
    target = Path(models_dir) / "Entities" / name
    target.mkdir(parents=True, exist_ok=True)
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(source_tokenizer / filename, target / filename)

    id2label = {i: label for i, label in enumerate(ENTITY_LABELS)}
    config = BertConfig(
        vocab_size=30522, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, num_labels=len(ENTITY_LABELS),
        id2label=id2label, label2id={v: k for k, v in id2label.items()},
    )
    torch.manual_seed(0)
    model = BertForTokenClassification(config)
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.zero_()
        model.classifier.bias[ENTITY_LABELS.index(favored_label)] = TINY_FAVORED_LOGIT
    model.save_pretrained(str(target))
    (target / "model_metadata.json").write_text(json.dumps({"base_model": "tiny-test"}))
    return target
