"""End-to-end workflows through the HTTP API.

Only the LLM server (the OpenAI client's HTTP calls) is replaced. Extraction
parsing, KG validation, pending storage, review scoring, the real (tiny) BERT
model and persistence all run as in production.
"""
import unittest
from contextlib import ExitStack

from sqlalchemy import select

from tests.support_api import ApiTestCase, FakeLLM

from database.connection import async_session_maker
from database.models import PendingEntityDB, PendingRelationDB, EntityReviewDB, RelationReviewDB

TEXT = (
    "Patient with pneumonia presented with fever and a productive cough. "
    "Chest X-ray showed bilateral infiltrates. Heart rate 110 bpm. Started levofloxacin."
)

ENTITIES = [
    {"text": "pneumonia", "entity_type": "disease", "confidence": 0.95, "normalized_form": "Pneumonia"},
    {"text": "fever", "entity_type": "symptom", "confidence": 0.9, "normalized_form": None},
    {"text": "productive cough", "entity_type": "symptom", "confidence": 0.85, "normalized_form": None},
    {"text": "Chest X-ray", "entity_type": "imaging_procedure", "confidence": 0.9, "normalized_form": None},
    {"text": "bilateral infiltrates", "entity_type": "imaging_result", "confidence": 0.8, "normalized_form": None},
    {"text": "110 bpm", "entity_type": "parameter", "confidence": 0.9, "normalized_form": None},
    {"text": "levofloxacin", "entity_type": "substance", "confidence": 0.9, "normalized_form": "Levofloxacin"},
]

RELATIONS = [
    {"source_entity": "Pneumonia", "source_type": "disease", "relation_type": "has_symptom",
     "target_entity": "Fever", "target_type": "symptom", "confidence": 0.9},
    {"source_entity": "Pneumonia", "source_type": "disease", "relation_type": "has_symptom",
     "target_entity": "Cough", "target_type": "symptom", "confidence": 0.85},
    {"source_entity": "Levofloxacin", "source_type": "substance", "relation_type": "treats",
     "target_entity": "Pneumonia", "target_type": "disease", "confidence": 0.8},
]


class TestWorkflows(ApiTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.llm = FakeLLM(entities=ENTITIES, relations=RELATIONS)
        self.stack = ExitStack()
        for p in self.llm.patches():
            self.stack.enter_context(p)
        self.ok(await self.client.post("/llm-config/?verify_connection=true", json={
            "name": "fake", "base_url": "http://127.0.0.1:9/v1", "model_name": "fake-model", "is_default": True,
        }), status=201)

    async def asyncTearDown(self):
        self.stack.close()
        await super().asyncTearDown()

    async def test_entity_workflow(self):
        # Clinical text -> extraction -> KG validation -> pending entities
        body = self.ok(await self.client.post("/classify/", json={"text": TEXT}))
        self.assertTrue(body["success"], body.get("error"))
        statuses = {v["entity"]["text"]: v["validation_status"] for v in body["kg_validation"]["validated_entities"]}
        self.assertEqual(statuses["pneumonia"], "exact_match")
        self.assertEqual(statuses["fever"], "exact_match")          # symptom matches the Finding "Fever"
        self.assertEqual(statuses["Chest X-ray"], "exact_match")
        self.assertEqual(statuses["levofloxacin"], "not_found")

        async with async_session_maker() as db:
            pending = {p.text: p for p in (await db.execute(select(PendingEntityDB))).scalars()}
        self.assertEqual(set(pending), {"productive cough", "bilateral infiltrates", "110 bpm", "levofloxacin"})

        # Pending -> review -> stored review
        result = self.ok(await self.client.post("/entity-reviews/bulk-review", json={
            "entity_ids": [p.id for p in pending.values()], "run_bert_validation": True,
        }))
        self.assertEqual((result["reviewed"], result["errors"]), (4, 0))

        reviews = {r["entity_text"]: r for r in self.ok(await self.client.get("/entity-reviews/?page_size=100"))["reviews"]}
        self.assertEqual(set(reviews), set(pending))
        for text, review in reviews.items():
            with self.subTest(text=text):
                self.assertIsNotNone(review["congruence"])
                self.assertIsNotNone(review["coverage"])
                # The tiny model predicts "disease" for everything: all four disagree.
                self.assertEqual(review["consistency"]["bert_status"], "disagreed")
                self.assertIs(review["consistency"]["bert_agreement"], False)
                self.assertTrue(review["constraint"]["type_valid"])
        self.assertAlmostEqual(reviews["110 bpm"]["constraint"]["score"], 1.0, places=9)

        # Stored results equal a fresh evaluation of the same inputs.
        preview = self.ok(await self.client.post("/entity-reviews/re-evaluate-all?dry_run=true"))
        for row in preview["results"]:
            self.assertEqual(row["old_overall_score"], row["overall_score"])
            self.assertEqual(row["old_status"], row["new_status"])

        async with async_session_maker() as db:
            stored = (await db.execute(select(EntityReviewDB))).scalars().all()
        self.assertEqual(len(stored), 4)
        self.assertTrue(all(r.pending_entity_id for r in stored))

    async def test_relation_workflow(self):
        # Clinical text -> relation extraction -> pending relations
        body = self.ok(await self.client.post("/relations/classify", json={"text": TEXT}))
        self.assertTrue(body["success"], body.get("error"))
        self.assertEqual(body["relations_saved"], 3)

        # Pending -> review (with KG triple lookup) -> stored review
        result = self.ok(await self.client.post("/relation-reviews/bulk-review"))
        self.assertEqual((result["reviewed"], result["errors"]), (3, 0))

        reviews = {(r["source_entity"], r["target_entity"]): r
                   for r in self.ok(await self.client.get("/relation-reviews/"))["reviews"]}
        known = reviews[("Pneumonia", "Fever")]
        novel_known_entities = reviews[("Pneumonia", "Cough")]
        novel_entities = reviews[("Levofloxacin", "Pneumonia")]
        self.assertAlmostEqual(known["coverage_score"], 0.3, places=9)
        self.assertAlmostEqual(known["congruence_score"], 1.0, places=9)
        self.assertAlmostEqual(novel_known_entities["coverage_score"], 0.8, places=9)
        self.assertAlmostEqual(novel_known_entities["congruence_score"], 0.9, places=9)
        self.assertAlmostEqual(novel_entities["coverage_score"], 0.8, places=9)
        for review in reviews.values():
            self.assertAlmostEqual(review["consistency_score"], min(0.8, review_confidence(review)), places=9)

        async with async_session_maker() as db:
            stored = (await db.execute(select(RelationReviewDB))).scalars().all()
            pending = (await db.execute(select(PendingRelationDB))).scalars().all()
        self.assertEqual(len(stored), 3)
        self.assertEqual({p.source for p in pending}, {"llm"})

    async def test_approved_relation_becomes_a_known_triple(self):
        self.ok(await self.client.post("/relations/classify", json={"text": TEXT}))
        async with async_session_maker() as db:
            rel = (await db.execute(select(PendingRelationDB).where(PendingRelationDB.target_entity == "Cough"))).scalar_one()
        self.ok(await self.client.post(f"/pending-relations/{rel.id}/add-to-kg"))
        review = self.ok(await self.client.post("/relation-reviews/", json={
            "source_entity": "Pneumonia", "source_type": "disease", "relation_type": "has_symptom",
            "target_entity": "Cough", "target_type": "symptom",
        }))
        self.assertAlmostEqual(review["coverage_score"], 0.3, places=9)


class TestReasoningModelOutput(ApiTestCase):
    """LM Studio + reasoning model: structured output arrives in reasoning_content."""

    async def test_extraction_reads_structured_output_from_reasoning_content(self):
        llm = FakeLLM(entities=ENTITIES, relations=RELATIONS, reasoning_model=True)
        with ExitStack() as stack:
            for p in llm.patches():
                stack.enter_context(p)
            self.ok(await self.client.post("/llm-config/?verify_connection=true", json={
                "name": "reasoning", "base_url": "http://127.0.0.1:9/v1", "model_name": "fake-model", "is_default": True,
            }), status=201)
            body = self.ok(await self.client.post("/classify/", json={"text": TEXT}))
            relations = self.ok(await self.client.post("/relations/classify", json={"text": TEXT}))
        self.assertTrue(body["success"], body.get("error"))
        self.assertEqual(len(body["result"]["entities"]), len(ENTITIES))
        self.assertTrue(relations["success"], relations.get("error"))
        self.assertEqual(relations["relations_saved"], 3)


def review_confidence(review):
    return {("Pneumonia", "Fever"): 0.9, ("Pneumonia", "Cough"): 0.85, ("Levofloxacin", "Pneumonia"): 0.8}[
        (review["source_entity"], review["target_entity"])
    ]


if __name__ == "__main__":
    unittest.main()
