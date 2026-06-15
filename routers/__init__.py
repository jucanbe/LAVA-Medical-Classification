from .llm_config import router as llm_config_router
from .classification import router as classification_router
from .bert_classification import router as bert_classification_router
from .frontend import router as frontend_router
from .pending_entities import router as pending_entities_router
from .entity_review import router as entity_review_router
from .entity_config import router as entity_config_router
from .relation_classification import router as relation_classification_router
from .pending_relations import router as pending_relations_router
from .relation_review import router as relation_review_router
from .relation_config import router as relation_config_router
from .sparql import router as sparql_router
from .triple_store import router as triple_store_router
from .classify_all import router as classify_all_router

__all__ = [
    "llm_config_router", 
    "classification_router", 
    "bert_classification_router",
    "frontend_router",
    "pending_entities_router",
    "entity_review_router",
    "entity_config_router",
    "relation_classification_router",
    "pending_relations_router",
    "relation_review_router",
    "relation_config_router",
    "sparql_router",
    "triple_store_router",
    "classify_all_router",
]
