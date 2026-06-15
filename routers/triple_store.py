import logging
from typing import List, Optional
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from datetime import datetime

from database.connection import get_db
from database.models import TripleStoreConfigDB
from models.triple_store import (
    TripleStoreConfigCreate,
    TripleStoreConfigUpdate,
    TripleStoreConfigResponse,
    TripleStoreStatus,
)
from services.triple_store import (
    get_active_backend,
    get_enabled_backends,
    add_enabled_backend,
    remove_enabled_backend,
    create_backend_from_config,
    InternalRDFLibBackend,
    ExternalSPARQLBackend,
    KG_DEFAULT_DIR,
    get_disabled_ttl_files,
    set_disabled_ttl_files,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/triple-store",
    tags=["Triple Store"],
    responses={404: {"description": "Not found"}},
)



@router.get(
    "/",
    response_model=List[TripleStoreConfigResponse],
    summary="List all triple store configurations",
)
async def list_configs(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(TripleStoreConfigDB).order_by(TripleStoreConfigDB.created_at.desc())
    )
    return [TripleStoreConfigResponse.model_validate(r) for r in result.scalars().all()]


@router.post(
    "/",
    response_model=TripleStoreConfigResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create triple store configuration",
)
async def create_config(
    config: TripleStoreConfigCreate,
    db: AsyncSession = Depends(get_db),
):
    existing = await db.execute(
        select(TripleStoreConfigDB).where(TripleStoreConfigDB.name == config.name)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A configuration named '{config.name}' already exists",
        )

    row = TripleStoreConfigDB(**config.model_dump())
    db.add(row)
    await db.flush()
    await db.refresh(row)

    if row.is_active:
        try:
            backend = await create_backend_from_config(row)
            add_enabled_backend(row.id, backend)
            logger.info(f"Auto-enabled backend on create: {row.name}")
            _reload_kg()
        except Exception as e:
            logger.warning(f"Could not auto-enable backend {row.name}: {e}")

    return TripleStoreConfigResponse.model_validate(row)


@router.get(
    "/{config_id}",
    response_model=TripleStoreConfigResponse,
    summary="Get triple store configuration",
)
async def get_config(config_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(TripleStoreConfigDB).where(TripleStoreConfigDB.id == config_id)
    )
    row = result.scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Configuration not found")
    return TripleStoreConfigResponse.model_validate(row)


@router.put(
    "/{config_id}",
    response_model=TripleStoreConfigResponse,
    summary="Update triple store configuration",
)
async def update_config(
    config_id: int,
    config_update: TripleStoreConfigUpdate,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(TripleStoreConfigDB).where(TripleStoreConfigDB.id == config_id)
    )
    row = result.scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Configuration not found")

    update_data = config_update.model_dump(exclude_unset=True)

    for key, value in update_data.items():
        setattr(row, key, value)
    row.updated_at = datetime.utcnow()
    await db.flush()
    await db.refresh(row)

    if row.is_active:
        try:
            backend = await create_backend_from_config(row)
            add_enabled_backend(row.id, backend)
            _reload_kg()
        except Exception as e:
            logger.warning(f"Could not refresh backend {row.name}: {e}")
    else:
        remove_enabled_backend(row.id)
        _reload_kg()

    return TripleStoreConfigResponse.model_validate(row)


@router.delete(
    "/{config_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete triple store configuration",
)
async def delete_config(config_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(TripleStoreConfigDB).where(TripleStoreConfigDB.id == config_id)
    )
    row = result.scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Configuration not found")
    was_active = row.is_active
    await db.delete(row)

    remove_enabled_backend(row.id)
    if was_active:
        _reload_kg()



def _reload_kg():
    """Silently reload the KG service singleton."""
    try:
        from services.knowledge_graph import reload_kg_service
        reload_kg_service()
    except Exception:
        pass



@router.post(
    "/{config_id}/toggle",
    response_model=TripleStoreConfigResponse,
    summary="Enable or disable a triple store configuration",
)
async def toggle_config(config_id: int, db: AsyncSession = Depends(get_db)):
    """Toggle is_active (enabled/disabled) for a store. Multiple stores can be enabled."""
    result = await db.execute(
        select(TripleStoreConfigDB).where(TripleStoreConfigDB.id == config_id)
    )
    row = result.scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Configuration not found")

    row.is_active = not row.is_active
    row.updated_at = datetime.utcnow()
    await db.flush()
    await db.refresh(row)

    if row.is_active:
        try:
            backend = await create_backend_from_config(row)
            add_enabled_backend(row.id, backend)
            logger.info(f"Triple store enabled: {row.name} ({row.store_type})")
        except Exception as e:
            logger.error(f"Failed to enable backend: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to enable backend: {e}",
            )
    else:
        remove_enabled_backend(row.id)
        logger.info(f"Triple store disabled: {row.name}")

    _reload_kg()
    return TripleStoreConfigResponse.model_validate(row)


@router.post(
    "/{config_id}/activate",
    response_model=TripleStoreConfigResponse,
    summary="Enable a triple store configuration (compat)",
    include_in_schema=False,
)
async def activate_config(config_id: int, db: AsyncSession = Depends(get_db)):
    return await toggle_config(config_id, db)


@router.post(
    "/{config_id}/test",
    summary="Test connection for a triple store configuration",
)
async def test_config(config_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(TripleStoreConfigDB).where(TripleStoreConfigDB.id == config_id)
    )
    row = result.scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Configuration not found")

    try:
        backend = await create_backend_from_config(row)
        ok, msg = await backend.test_connection()
        return {"success": ok, "message": msg}
    except Exception as e:
        return {"success": False, "message": str(e)}


@router.post(
    "/{config_id}/load-ttl",
    summary="Upload a TTL file into a triple store backend",
)
async def load_ttl_into_store(
    config_id: int,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    if not file.filename or not file.filename.endswith(".ttl"):
        raise HTTPException(status_code=400, detail="Only .ttl files are accepted")

    result = await db.execute(
        select(TripleStoreConfigDB).where(TripleStoreConfigDB.id == config_id)
    )
    row = result.scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Configuration not found")

    content = (await file.read()).decode("utf-8")

    if row.store_type == "internal":
        ttl_dir = Path(row.ttl_directory) if row.ttl_directory else KG_DEFAULT_DIR
        ttl_dir.mkdir(parents=True, exist_ok=True)
        dest = ttl_dir / file.filename
        with open(dest, "w", encoding="utf-8") as fp:
            fp.write(content)
        logger.info(f"TTL file saved to {dest}")

    try:
        backend = await create_backend_from_config(row)
        new_triples = await backend.load_ttl_data(content)

        if row.is_active:
            add_enabled_backend(row.id, backend)
            _reload_kg()

        return {
            "success": True,
            "filename": file.filename,
            "new_triples": new_triples,
            "total_triples": await backend.get_triple_count(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading TTL: {e}")


@router.get(
    "/status/active",
    summary="Get the status of all enabled triple stores",
)
async def get_active_status(db: AsyncSession = Depends(get_db)):
    """Return status of all enabled stores. Auto-initialises backends if needed."""
    result = await db.execute(
        select(TripleStoreConfigDB).where(TripleStoreConfigDB.is_active == True)
    )
    enabled_rows = list(result.scalars().all())
    backends = get_enabled_backends()

    if not enabled_rows:
        return TripleStoreStatus(
            backend_type="none",
            triple_count=0,
            is_connected=False,
            message="No triple store enabled.",
        )

    configs_status = []
    total_triples = 0
    any_connected = False

    for row in enabled_rows:
        cfg = TripleStoreConfigResponse.model_validate(row)
        be = backends.get(row.id)

        if be is None:
            try:
                be = await create_backend_from_config(row)
                add_enabled_backend(row.id, be)
                logger.info(f"Auto-initialized backend: {row.name}")
            except Exception as e:
                configs_status.append({
                    "config": cfg.model_dump(),
                    "is_connected": False,
                    "triple_count": 0,
                    "message": f"Init failed: {e}",
                })
                continue

        ok, msg = await be.test_connection()
        count = await be.get_triple_count()
        cache_st = None
        if isinstance(be, InternalRDFLibBackend):
            cache_st = be.cache_status

        configs_status.append({
            "config": cfg.model_dump(),
            "is_connected": ok,
            "triple_count": count,
            "cache_status": cache_st,
            "message": msg,
        })
        total_triples += count
        if ok:
            any_connected = True

    if len(backends) != len(get_enabled_backends()):
        _reload_kg()

    return TripleStoreStatus(
        configs=configs_status,
        backend_type="multiple" if len(enabled_rows) > 1 else enabled_rows[0].store_type,
        triple_count=total_triples,
        is_connected=any_connected,
        message=f"{len(enabled_rows)} store(s) enabled — {total_triples:,} triples total",
    )


@router.get(
    "/files/list",
    summary="List TTL files in the default KnowledgeGraph directory",
)
async def list_ttl_files(
    directory: Optional[str] = Query(None, description="Custom directory path"),
):
    ttl_dir = Path(directory) if directory else KG_DEFAULT_DIR
    if not ttl_dir.exists():
        return {"files": []}

    disabled = set(get_disabled_ttl_files(ttl_dir))
    files = []
    for f in sorted(ttl_dir.glob("*.ttl")):
        st = f.stat()
        files.append({
            "name": f.name,
            "size": st.st_size,
            "modified": st.st_mtime,
            "enabled": f.name not in disabled,
        })
    return {"files": files}


@router.post(
    "/files/toggle",
    summary="Enable or disable a specific TTL file",
)
async def toggle_ttl_file(
    filename: str = Query(..., description="TTL filename to toggle"),
    directory: Optional[str] = Query(None, description="Custom directory path"),
):
    """Toggle whether a TTL file is loaded by the internal backend."""
    ttl_dir = Path(directory) if directory else KG_DEFAULT_DIR
    file_path = ttl_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    disabled = get_disabled_ttl_files(ttl_dir)
    if filename in disabled:
        disabled.remove(filename)
        now_enabled = True
    else:
        disabled.append(filename)
        now_enabled = False

    set_disabled_ttl_files(disabled, ttl_dir)

    backends = get_enabled_backends()
    for be in backends.values():
        if isinstance(be, InternalRDFLibBackend) and be.ttl_directory == ttl_dir:
            await be.load_all_ttl_files()
    _reload_kg()

    return {"success": True, "filename": filename, "enabled": now_enabled}


@router.get(
    "/{config_id}/entity-stats",
    summary="Get entity statistics from a triple store via SPARQL",
    description="Runs preset SPARQL queries against a configured store to get entity counts by type, "
                "total triples, sample entities, etc.",
)
async def get_store_entity_stats(config_id: int, db: AsyncSession = Depends(get_db)):
    """Get entity/type statistics from a triple store backend via SPARQL queries."""
    result = await db.execute(
        select(TripleStoreConfigDB).where(TripleStoreConfigDB.id == config_id)
    )
    row = result.scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Configuration not found")

    backends = get_enabled_backends()
    be = backends.get(config_id)
    if be is not None:
        backend = be
    else:
        try:
            backend = await create_backend_from_config(row)
        except Exception as e:
            return {"success": False, "error": str(e)}

    try:
        total_triples = await backend.get_triple_count()
    except Exception:
        total_triples = 0

    entities_by_type = {}
    total_entities = 0
    try:
        q_types = (
            "SELECT ?type (COUNT(DISTINCT ?entity) AS ?count) "
            "WHERE { ?entity a ?type . } "
            "GROUP BY ?type ORDER BY DESC(?count)"
        )
        raw = await backend.sparql_query(q_types)
        for b in raw.get("results", {}).get("bindings", []):
            type_uri = b.get("type", {}).get("value", "")
            count = int(b.get("count", {}).get("value", "0"))
            type_name = type_uri.split("/")[-1].split("#")[-1]
            entities_by_type[type_name] = count
            total_entities += count
    except Exception as e:
        logger.warning(f"Error getting type stats for store {config_id}: {e}")

    distinct_entities = 0
    try:
        q_ent = "SELECT (COUNT(DISTINCT ?s) AS ?c) WHERE { ?s a ?type . }"
        raw = await backend.sparql_query(q_ent)
        bindings = raw.get("results", {}).get("bindings", [])
        if bindings:
            distinct_entities = int(bindings[0].get("c", {}).get("value", "0"))
    except Exception:
        pass

    total_labels = 0
    try:
        q_labels = (
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> "
            "SELECT (COUNT(?label) AS ?c) WHERE { ?s rdfs:label ?label . }"
        )
        raw = await backend.sparql_query(q_labels)
        bindings = raw.get("results", {}).get("bindings", [])
        if bindings:
            total_labels = int(bindings[0].get("c", {}).get("value", "0"))
    except Exception:
        pass

    sample_entities = []
    try:
        q_samples = (
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> "
            "SELECT ?entity ?label ?type WHERE { "
            "  ?entity rdfs:label ?label . "
            "  ?entity a ?type . "
            "} LIMIT 30"
        )
        raw = await backend.sparql_query(q_samples)
        seen = set()
        for b in raw.get("results", {}).get("bindings", []):
            label = b.get("label", {}).get("value", "")
            type_uri = b.get("type", {}).get("value", "")
            type_name = type_uri.split("/")[-1].split("#")[-1]
            if label and label not in seen and len(sample_entities) < 20:
                sample_entities.append({"label": label, "type": type_name})
                seen.add(label)
    except Exception:
        pass

    total_predicates = 0
    try:
        q_preds = "SELECT (COUNT(DISTINCT ?p) AS ?c) WHERE { ?s ?p ?o . }"
        raw = await backend.sparql_query(q_preds)
        bindings = raw.get("results", {}).get("bindings", [])
        if bindings:
            total_predicates = int(bindings[0].get("c", {}).get("value", "0"))
    except Exception:
        pass

    return {
        "success": True,
        "store_name": row.name,
        "store_type": row.store_type,
        "total_triples": total_triples,
        "total_entities": distinct_entities,
        "total_labels": total_labels,
        "total_predicates": total_predicates,
        "entities_by_type": entities_by_type,
        "sample_entities": sample_entities,
    }
