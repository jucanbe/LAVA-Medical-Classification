"""Persistence and API serialization of reviews and configuration."""
import unittest

from sqlalchemy import select, update

from tests.support_api import ApiTestCase

from database.connection import async_session_maker
from database.models import EntityReviewDB, PendingEntityDB
from models.review_defaults import ENTITY_CONFIG_DEFAULTS, RELATION_CONFIG_DEFAULTS, SCORING_VERSION
from routers.entity_review import get_review_service


async def add_pending_entity(**fields) -> int:
    values = dict(text="Pneumonia", entity_type="disease", confidence=0.9, context="Patient with pneumonia.", source="llm")
    values.update(fields)
    async with async_session_maker() as db:
        row = PendingEntityDB(**values)
        db.add(row)
        await db.commit()
        return row.id


class TestEntityReviewPersistence(ApiTestCase):
    async def test_type_valid_false_survives_persistence_and_api(self):
        created = self.ok(await self.client.post("/entity-reviews/", json={
            "entity_text": "adolescents", "entity_type": "organ_system", "run_bert_validation": False,
        }))
        self.assertIs(created["constraint"]["type_valid"], False)
        self.assertEqual(created["review_status"], "failed")

        async with async_session_maker() as db:
            stored = (await db.execute(select(EntityReviewDB).where(EntityReviewDB.id == created["id"]))).scalar_one()
            self.assertIs(stored.constraint_type_valid, False)

        fetched = self.ok(await self.client.get(f"/entity-reviews/{created['id']}"))
        self.assertIs(fetched["constraint"]["type_valid"], False)
        listed = self.ok(await self.client.get("/entity-reviews/"))
        self.assertIs(listed["reviews"][0]["constraint"]["type_valid"], False)

    async def test_legacy_rows_without_type_valid_default_to_true(self):
        async with async_session_maker() as db:
            db.add(EntityReviewDB(entity_text="fever", entity_type="finding", constraint_score=1.0,
                                  constraint_type_valid=None, overall_score=0.8, review_status="passed"))
            await db.commit()
        listed = self.ok(await self.client.get("/entity-reviews/"))
        self.assertIs(listed["reviews"][0]["constraint"]["type_valid"], True)
        self.assertIsNone(listed["reviews"][0]["scoring_version"])

    async def test_scores_round_trip(self):
        pending_id = await add_pending_entity(text="Pneumonias", normalized_text="Pneumonia")
        created = self.ok(await self.client.post("/entity-reviews/", json={
            "pending_entity_id": pending_id, "entity_text": "Pneumonias", "entity_type": "disease",
        }))
        expected = await get_review_service().evaluate_entity(
            "Pneumonias", "disease", normalized_form="Pneumonia", context="Patient with pneumonia.",
            confidence=0.9, source="llm", run_bert_validation=True,
        )
        fetched = self.ok(await self.client.get(f"/entity-reviews/{created['id']}"))
        for body in (created, fetched):
            for name in ["congruence", "coverage", "constraint", "completeness", "consistency"]:
                self.assertAlmostEqual(body[name]["score"], expected[name].score, places=12, msg=name)
            self.assertAlmostEqual(body["overall_score"], expected["overall_score"], places=12)
            self.assertEqual(body["review_status"], expected["review_status"])
            self.assertEqual(body["congruence"]["nearest_entity"], "Pneumonia")
            self.assertAlmostEqual(body["congruence"]["embedding_distance"], 1 - 18 / 19, places=12)
            self.assertIs(body["completeness"]["has_confidence"], True)
            self.assertIs(body["completeness"]["has_normalized_form"], True)
            self.assertIs(body["completeness"]["has_definition"], False)
            self.assertEqual(body["consistency"]["bert_status"], "agreed")   # tiny model predicts disease
            self.assertIs(body["consistency"]["bert_agreement"], True)
            self.assertEqual(body["scoring_version"], SCORING_VERSION)

    async def test_duplicate_reviews_do_not_crash_lookups(self):
        async with async_session_maker() as db:
            for _ in range(2):
                db.add(EntityReviewDB(entity_text="fever", entity_type="finding", overall_score=0.5, review_status="needs_review"))
            await db.commit()
        body = self.ok(await self.client.post("/entity-reviews/", json={"entity_text": "fever", "entity_type": "finding"}))
        self.assertEqual(body["entity_text"], "fever")


class TestReEvaluation(ApiTestCase):
    async def test_re_evaluation_keeps_history_and_dry_run_writes_nothing(self):
        created = self.ok(await self.client.post("/entity-reviews/", json={"entity_text": "Appendicitis", "entity_type": "disease"}))
        async with async_session_maker() as db:
            await db.execute(update(EntityReviewDB).values(scoring_version=None, overall_score=0.123, review_status="needs_review"))
            await db.commit()

        preview = self.ok(await self.client.post("/entity-reviews/re-evaluate-all?dry_run=true"))
        self.assertTrue(preview["dry_run"])
        self.assertEqual(preview["updated"], 0)
        self.assertEqual(preview["results"][0]["old_overall_score"], 0.123)
        unchanged = self.ok(await self.client.get(f"/entity-reviews/{created['id']}"))
        self.assertEqual(unchanged["overall_score"], 0.123)
        self.assertEqual(unchanged["score_history"], [])

        result = self.ok(await self.client.post("/entity-reviews/re-evaluate-all?only_legacy=true"))
        self.assertEqual(result["updated"], 1)
        after = self.ok(await self.client.get(f"/entity-reviews/{created['id']}"))
        self.assertEqual(after["scoring_version"], SCORING_VERSION)
        self.assertEqual(len(after["score_history"]), 1)
        self.assertEqual(after["score_history"][0]["overall_score"], 0.123)
        self.assertIsNone(after["score_history"][0]["scoring_version"])

        again = self.ok(await self.client.post("/entity-reviews/re-evaluate-all?only_legacy=true"))
        self.assertEqual(again["updated"], 0)

    async def test_single_relation_re_evaluation_keeps_history(self):
        created = self.ok(await self.client.post("/relation-reviews/", json={
            "source_entity": "Pneumonia", "source_type": "disease", "relation_type": "has_symptom",
            "target_entity": "Fever", "target_type": "symptom",
        }))
        result = self.ok(await self.client.post(f"/relation-reviews/{created['id']}/re-evaluate"))
        self.assertEqual(len(result["review"]["score_history"]), 1)
        self.assertEqual(result["review"]["score_history"][0]["overall_score"], created["overall_score"])


class TestConfiguration(ApiTestCase):
    async def test_stored_defaults_match_the_service(self):
        entity_cfg = self.ok(await self.client.get("/entity-config/"))["config"]
        relation_cfg = self.ok(await self.client.get("/relation-config/"))["config"]
        for key, value in ENTITY_CONFIG_DEFAULTS.items():
            self.assertEqual(entity_cfg[key], value, key)
        for key, value in RELATION_CONFIG_DEFAULTS.items():
            self.assertEqual(relation_cfg[key], value, key)
        service_cfg = await get_review_service()._get_config()
        self.assertEqual(service_cfg["weights"]["congruence"], entity_cfg["weight_congruence"])
        self.assertEqual(service_cfg["settings"]["constraint_max_length"], entity_cfg["constraint_max_length"])

    async def test_config_change_applies_to_the_next_review(self):
        self.ok(await self.client.get("/entity-config/"))
        params = {"entity_text": "Community acquired pneumonia", "entity_type": "disease", "run_bert": "false"}
        before = self.ok(await self.client.post("/entity-reviews/evaluate-text", params=params))
        self.assertEqual(before["evaluation"]["constraint"]["violations"], [])

        self.ok(await self.client.put("/entity-config/", params={"constraint_max_length": 10}))
        after = self.ok(await self.client.post("/entity-reviews/evaluate-text", params=params))
        self.assertIn("Entity text too long (> 10 characters)", after["evaluation"]["constraint"]["violations"])
        self.assertAlmostEqual(after["evaluation"]["constraint"]["score"], 0.9, places=9)

    async def test_relation_config_change_applies(self):
        self.ok(await self.client.get("/relation-config/"))
        self.ok(await self.client.put("/relation-config/", params={"congruence_kg_bonus": 0}))
        review = self.ok(await self.client.post("/relation-reviews/", json={
            "source_entity": "Pneumonia", "source_type": "disease", "relation_type": "has_symptom",
            "target_entity": "Fever", "target_type": "symptom",
        }))
        self.assertAlmostEqual(review["congruence_score"], 0.9, places=9)

    async def test_invalid_configuration_is_rejected_and_not_stored(self):
        self.ok(await self.client.get("/entity-config/"))
        for params in [{"weight_congruence": 0.5}, {"weight_congruence": -0.05, "weight_coverage": 0.45},
                       {"coverage_novelty_bonus": 0.8}, {"threshold_review": 0.9}]:
            with self.subTest(params=params):
                response = await self.client.put("/entity-config/", params=params)
                self.assertEqual(response.status_code, 400, response.text)
        cfg = self.ok(await self.client.get("/entity-config/"))["config"]
        self.assertEqual(cfg["weight_congruence"], ENTITY_CONFIG_DEFAULTS["weight_congruence"])
        self.assertEqual(cfg["coverage_novelty_bonus"], 0.3)


class TestAddToKG(ApiTestCase):
    async def test_review_of_a_numeric_parameter_can_be_added_to_the_kg(self):
        review = self.ok(await self.client.post("/entity-reviews/", json={"entity_text": "110 bpm", "entity_type": "parameter"}))
        self.assertAlmostEqual(review["constraint"]["score"], 1.0, places=9)
        body = self.ok(await self.client.post(f"/entity-reviews/{review['id']}/add-to-kg"))
        self.assertTrue(body["success"])
        matches = self.kg.find_matches("110 bpm", min_score=0.95, entity_type="parameter")
        self.assertEqual([m.kg_type for m in matches], ["Parameter"])
        saved = (self.kg.kg_directory / "medical_entities.ttl").read_text()
        self.assertIn("110 bpm", saved)
        self.assertEqual((await self.client.get(f"/entity-reviews/{review['id']}")).status_code, 404)

    async def test_pending_entity_can_be_added_to_the_kg(self):
        pending_id = await add_pending_entity(text="Levofloxacin", entity_type="Substance")
        body = self.ok(await self.client.post(f"/pending-entities/{pending_id}/add-to-kg"))
        self.assertTrue(body["success"])
        self.assertEqual([m.kg_label for m in self.kg.find_matches("Levofloxacin", min_score=0.95, entity_type="substance")], ["Levofloxacin"])

    async def test_invalid_type_is_refused(self):
        review = self.ok(await self.client.post("/entity-reviews/", json={"entity_text": "adolescents", "entity_type": "organ_system"}))
        response = await self.client.post(f"/entity-reviews/{review['id']}/add-to-kg")
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
