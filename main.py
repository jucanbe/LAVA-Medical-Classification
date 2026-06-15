import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from database import db_manager
from routers import (
    llm_config_router, 
    classification_router, 
    bert_classification_router, 
    frontend_router,
    pending_entities_router,
    entity_review_router,
    entity_config_router,
    relation_classification_router,
    pending_relations_router,
    relation_review_router,
    relation_config_router,
    sparql_router,
    triple_store_router,
    classify_all_router
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the application lifecycle."""
    logger.info("Starting application...")
    await db_manager.init_db()
    logger.info("Database initialized")

    try:
        from database.connection import get_db as _get_db
        from sqlalchemy import select as _sel
        from database.models import TripleStoreConfigDB as _TSCDB
        from services.triple_store import (
            create_backend_from_config as _create_be,
            add_enabled_backend as _add_be,
        )

        async for session in _get_db():
            result = await session.execute(
                _sel(_TSCDB).where(_TSCDB.is_active == True)
            )
            active_rows = list(result.scalars().all())
            for row in active_rows:
                try:
                    be = await _create_be(row)
                    _add_be(row.id, be)
                    logger.info(
                        f"Triple store enabled on startup: {row.name} "
                        f"({row.store_type})"
                    )
                except Exception as exc:
                    logger.warning(
                        f"Failed to enable triple store '{row.name}' on startup: {exc}"
                    )
            if active_rows:
                try:
                    from services.knowledge_graph import reload_kg_service
                    reload_kg_service()
                except Exception:
                    pass
            break
    except Exception as e:
        logger.warning(f"Triple store auto-activation failed on startup: {e}")

    yield
    
    logger.info("Shutting down application...")
    await db_manager.close()
    logger.info("Database connection closed")


app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="""
## Medical Entity Classification API

This API provides two classification methods: LLM-based and BERT-based.

    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(frontend_router)
app.include_router(llm_config_router)
app.include_router(classification_router)
app.include_router(bert_classification_router)
app.include_router(pending_entities_router)
app.include_router(entity_review_router)
app.include_router(entity_config_router)
app.include_router(relation_classification_router)
app.include_router(pending_relations_router)
app.include_router(relation_review_router)
app.include_router(relation_config_router)
app.include_router(sparql_router)
app.include_router(triple_store_router)
app.include_router(classify_all_router)


@app.get("/api", tags=["Health"])
async def root():
    """Root endpoint to verify the API is running."""
    return {
        "message": "Medical Entity Classifier API",
        "version": settings.API_VERSION,
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True
    )
