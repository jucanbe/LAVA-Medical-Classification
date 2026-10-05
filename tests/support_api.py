"""Base class for tests that drive the real FastAPI app over HTTP (in-process)."""
import json
import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from tests import TEST_TMP
from tests.support import make_kg, build_tiny_bert

import main
from database.connection import db_manager
from database.models import Base
from services import bert_ner, knowledge_graph
from services.bert_ner import BERTNERService
from services.entity_reviewer import invalidate_config_cache as invalidate_entity_config
from services.relation_reviewer import invalidate_config_cache as invalidate_relation_config

# main.py configures INFO logging for the app; keep test output readable.
logging.getLogger().setLevel(logging.WARNING)

_TINY_MODELS_DIR = None


def tiny_models_dir() -> Path:
    global _TINY_MODELS_DIR
    if _TINY_MODELS_DIR is None:
        _TINY_MODELS_DIR = Path(tempfile.mkdtemp(prefix="bert-api-", dir=TEST_TMP))
        build_tiny_bert(_TINY_MODELS_DIR, "tiny_project", favored_label="B-disease")
    return _TINY_MODELS_DIR


class FakeLLM:
    """Replaces the OpenAI HTTP boundary. Everything behind it is real code."""

    def __init__(self, entities=None, relations=None, reasoning_model=False):
        self.entities = entities or []
        self.relations = relations or []
        # Mimic LM Studio serving a reasoning model: constrained (response_format)
        # output arrives in reasoning_content with an empty content.
        self.reasoning_model = reasoning_model
        self.requests = []

    def _response(self, content, constrained=False):
        if self.reasoning_model and constrained:
            message = SimpleNamespace(content="", reasoning_content=content, model_extra={})
        else:
            message = SimpleNamespace(content=content, reasoning_content=None, model_extra={})
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])

    async def create(self, *args, **kwargs):
        messages = kwargs.get("messages", [])
        self.requests.append(messages)
        constrained = "response_format" in kwargs
        system = " ".join(m["content"] for m in messages if m["role"] == "system")
        if "ExtractedRelations" in system:
            return self._response(json.dumps({"relations": self.relations}), constrained)
        if "ExtractedEntities" in system:
            return self._response(json.dumps({"entities": self.entities}), constrained)
        return self._response("OK", constrained)

    async def list_models(self, *args, **kwargs):
        return SimpleNamespace(data=[SimpleNamespace(id="fake-model")])

    def patches(self):
        fake = self

        async def create(_self, *args, **kwargs):
            return await fake.create(*args, **kwargs)

        async def list_models(_self, *args, **kwargs):
            return await fake.list_models(*args, **kwargs)

        return [
            patch("openai.resources.chat.completions.AsyncCompletions.create", create),
            patch("openai.resources.models.AsyncModels.list", list_models),
        ]


class ApiTestCase(unittest.IsolatedAsyncioTestCase):
    """Fresh database, a temp KG and the tiny BERT model for every test."""

    async def asyncSetUp(self):
        async with db_manager.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await db_manager.init_db()
        self.kg = make_kg()
        knowledge_graph._kg_service = self.kg
        bert_ner._bert_ner_service = BERTNERService(str(tiny_models_dir()))
        invalidate_entity_config()
        invalidate_relation_config()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test", timeout=120)

    async def asyncTearDown(self):
        await self.client.aclose()
        await db_manager.engine.dispose()

    def ok(self, response, status=200):
        self.assertEqual(response.status_code, status, response.text)
        return response.json()
