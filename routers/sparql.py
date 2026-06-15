import logging
import time
from typing import Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from database.connection import get_db
from database.models import TripleStoreConfigDB
from services.knowledge_graph import get_kg_service, KG_DEFAULT_DIR
from services.triple_store import (
    get_enabled_backends,
    create_backend_from_config,
    ExternalSPARQLBackend,
    InternalRDFLibBackend,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/sparql",
    tags=["SPARQL"],
    responses={404: {"description": "Not found"}}
)


class SPARQLQueryRequest(BaseModel):
    """Request body for SPARQL queries."""
    query: str = Field(..., description="SPARQL query string")
    ttl_file: Optional[str] = Field(
        None,
        description="Specific TTL file to query. If not provided, queries the full loaded graph."
    )
    store_id: Optional[int] = Field(
        None,
        description="Triple store config ID to route query to. Overrides ttl_file."
    )
    query_all: bool = Field(
        False,
        description="When True, query ALL sources (local KG + every configured store) and merge results."
    )


class SPARQLQueryResponse(BaseModel):
    """Response for SPARQL queries."""
    success: bool
    query: str
    ttl_file: Optional[str] = None
    result_type: str = Field(
        ..., description="Type of result: 'bindings' (SELECT), 'boolean' (ASK), or 'graph' (CONSTRUCT/DESCRIBE)"
    )
    columns: list = Field(default_factory=list, description="Column names for SELECT queries")
    results: list = Field(default_factory=list, description="Query result rows")
    total_results: int = 0
    execution_time_ms: float = 0.0
    error: Optional[str] = None


@router.post(
    "/query",
    response_model=SPARQLQueryResponse,
    summary="Execute a SPARQL query"
)
async def execute_sparql_query(
    request: SPARQLQueryRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Execute a SPARQL query against the Knowledge Graph.
    
    Supports SELECT, ASK, CONSTRUCT and DESCRIBE queries.
    - If store_id is provided, routes query to that triple store backend.
    - If a specific TTL file is provided, loads only that file into a temporary graph.
    - Otherwise queries the full loaded Knowledge Graph.
    """
    start_time = time.time()

    try:
        sparql_query = request.query.strip()
        if not sparql_query:
            raise HTTPException(status_code=400, detail="Empty SPARQL query")

        if request.query_all:
            return await _query_all_sources(sparql_query, start_time, db)

        if request.store_id is not None:
            return await _query_store_backend(request.store_id, sparql_query, start_time, db)

        from rdflib import Graph

        if request.ttl_file:
            kg_dir = KG_DEFAULT_DIR
            file_path = kg_dir / request.ttl_file

            if not file_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"TTL file not found: {request.ttl_file}"
                )

            graph = Graph()
            graph.parse(str(file_path), format="turtle")
        else:
            kg_service = get_kg_service()
            graph = kg_service.graph

        query_result = graph.query(sparql_query)
        execution_time = (time.time() - start_time) * 1000

        import re as _re
        query_body = _re.sub(
            r'(?i)^\s*(PREFIX\s+\S+\s*<[^>]*>\s*|BASE\s*<[^>]*>\s*)+',
            '', sparql_query
        ).strip().upper()

        if query_body.startswith("ASK"):
            return SPARQLQueryResponse(
                success=True,
                query=sparql_query,
                ttl_file=request.ttl_file,
                result_type="boolean",
                columns=["result"],
                results=[{"result": bool(query_result)}],
                total_results=1,
                execution_time_ms=round(execution_time, 2),
            )

        elif query_body.startswith("SELECT"):
            columns = [str(v) for v in query_result.vars] if query_result.vars else []
            rows = []
            for row in query_result:
                row_dict = {}
                for i, var in enumerate(columns):
                    value = row[i]
                    row_dict[var] = str(value) if value is not None else None
                rows.append(row_dict)

            return SPARQLQueryResponse(
                success=True,
                query=sparql_query,
                ttl_file=request.ttl_file,
                result_type="bindings",
                columns=columns,
                results=rows,
                total_results=len(rows),
                execution_time_ms=round(execution_time, 2),
            )

        elif query_body.startswith("CONSTRUCT") or query_body.startswith("DESCRIBE"):
            triples = []
            result_graph = query_result.graph if hasattr(query_result, 'graph') else Graph()
            for s, p, o in result_graph:
                triples.append({
                    "subject": str(s),
                    "predicate": str(p),
                    "object": str(o),
                })

            return SPARQLQueryResponse(
                success=True,
                query=sparql_query,
                ttl_file=request.ttl_file,
                result_type="graph",
                columns=["subject", "predicate", "object"],
                results=triples,
                total_results=len(triples),
                execution_time_ms=round(execution_time, 2),
            )

        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported SPARQL query type. Use SELECT, ASK, CONSTRUCT, or DESCRIBE."
            )

    except HTTPException:
        raise
    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        logger.error(f"SPARQL query error: {e}")
        return SPARQLQueryResponse(
            success=False,
            query=request.query,
            ttl_file=request.ttl_file,
            result_type="error",
            error=str(e),
            execution_time_ms=round(execution_time, 2),
        )


@router.get(
    "/files",
    summary="List available TTL files for SPARQL queries"
)
async def list_ttl_files():
    """List all TTL files available in the KnowledgeGraph directory."""
    try:
        kg_dir = KG_DEFAULT_DIR
        if not kg_dir.exists():
            return {"success": True, "files": []}

        files = []
        for f in sorted(kg_dir.glob("*.ttl")):
            stat = f.stat()
            files.append({
                "name": f.name,
                "size": stat.st_size,
                "modified": stat.st_mtime,
            })

        return {"success": True, "files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/prefixes",
    summary="Get common RDF prefixes"
)
async def get_prefixes():
    """Return common RDF prefixes to help users build SPARQL queries."""
    kg_service = get_kg_service()
    
    graph_prefixes = {}
    for prefix, ns in kg_service.graph.namespaces():
        if prefix:
            graph_prefixes[prefix] = str(ns)

    common = {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "owl": "http://www.w3.org/2002/07/owl#",
        "skos": "http://www.w3.org/2004/02/skos/core#",
        "xsd": "http://www.w3.org/2001/XMLSchema#",
        "med": "http://example.org/medical/",
        "medtype": "http://example.org/medical/types/",
        "medrel": "http://example.org/medical/relations/",
    }
    common.update(graph_prefixes)

    return {"success": True, "prefixes": common}


@router.get(
    "/examples",
    summary="Get example SPARQL queries"
)
async def get_example_queries():
    """Return a list of example SPARQL queries for the medical KG."""
    examples = [
        {
            "name": "List all entities with labels",
            "description": "Get all entities that have an rdfs:label",
            "query": "SELECT ?entity ?label\nWHERE {\n  ?entity rdfs:label ?label .\n}\nORDER BY ?label\nLIMIT 50"
        },
        {
            "name": "Count entities by type",
            "description": "Count how many entities of each type exist",
            "query": "SELECT ?type (COUNT(?entity) AS ?count)\nWHERE {\n  ?entity rdf:type ?type .\n}\nGROUP BY ?type\nORDER BY DESC(?count)"
        },
        {
            "name": "Find all diseases",
            "description": "List all entities typed as Disease",
            "query": "PREFIX medtype: <http://example.org/medical/types/>\n\nSELECT ?entity ?label\nWHERE {\n  ?entity rdf:type medtype:Disease .\n  ?entity rdfs:label ?label .\n}\nORDER BY ?label"
        },
        {
            "name": "Find relations between entities",
            "description": "Show all relation triples (subject → relation → object)",
            "query": "PREFIX medrel: <http://example.org/medical/relations/>\n\nSELECT ?source ?sourceLabel ?relation ?target ?targetLabel\nWHERE {\n  ?source ?relation ?target .\n  FILTER(STRSTARTS(STR(?relation), STR(medrel:)))\n  OPTIONAL { ?source rdfs:label ?sourceLabel }\n  OPTIONAL { ?target rdfs:label ?targetLabel }\n}\nLIMIT 50"
        },
        {
            "name": "Search entity by name",
            "description": "Find entities containing a specific text (replace 'diabetes')",
            "query": "SELECT ?entity ?label ?type\nWHERE {\n  ?entity rdfs:label ?label .\n  OPTIONAL { ?entity rdf:type ?type }\n  FILTER(CONTAINS(LCASE(STR(?label)), \"diabetes\"))\n}\nLIMIT 20"
        },
        {
            "name": "Disease-Symptom relations",
            "description": "Find all has_symptom relationships",
            "query": "PREFIX medrel: <http://example.org/medical/relations/>\n\nSELECT ?disease ?diseaseLabel ?symptom ?symptomLabel\nWHERE {\n  ?disease medrel:has_symptom ?symptom .\n  ?disease rdfs:label ?diseaseLabel .\n  ?symptom rdfs:label ?symptomLabel .\n}\nORDER BY ?diseaseLabel"
        },
        {
            "name": "Total triple count",
            "description": "Count total number of triples in the graph",
            "query": "SELECT (COUNT(*) AS ?total)\nWHERE {\n  ?s ?p ?o .\n}"
        },
        {
            "name": "All predicates used",
            "description": "List all distinct predicates (properties) in the graph",
            "query": "SELECT DISTINCT ?predicate (COUNT(?s) AS ?usage)\nWHERE {\n  ?s ?predicate ?o .\n}\nGROUP BY ?predicate\nORDER BY DESC(?usage)"
        },
    ]
    return {"success": True, "examples": examples}



async def _query_all_sources(
    sparql_query: str,
    start_time: float,
    db: AsyncSession,
) -> SPARQLQueryResponse:
    """Run query against local KG + every configured triple store, merge results."""
    import asyncio
    from rdflib import Graph as RDFGraph

    merged_columns: list = []
    merged_rows: list = []
    seen_row_keys: set = set()
    errors: list = []

    try:
        kg_service = get_kg_service()
        graph = kg_service.graph
        qr = graph.query(sparql_query)

        if hasattr(qr, 'vars') and qr.vars:
            cols = [str(v) for v in qr.vars]
            if not merged_columns:
                merged_columns = cols
            for row in qr:
                row_dict = {}
                for i, var in enumerate(cols):
                    value = row[i]
                    row_dict[var] = str(value) if value is not None else None
                rk = tuple(sorted(row_dict.items()))
                if rk not in seen_row_keys:
                    seen_row_keys.add(rk)
                    merged_rows.append(row_dict)
        elif hasattr(qr, 'askAnswer'):
            merged_columns = ['result']
            merged_rows.append({'result': bool(qr)})
    except Exception as e:
        errors.append(f"Local KG: {e}")

    try:
        result = await db.execute(
            select(TripleStoreConfigDB).order_by(TripleStoreConfigDB.name)
        )
        store_rows = list(result.scalars().all())
    except Exception:
        store_rows = []

    enabled = get_enabled_backends()

    for cfg_row in store_rows:
        try:
            backend = enabled.get(cfg_row.id)
            if backend is None:
                backend = await create_backend_from_config(cfg_row)

            raw = await backend.sparql_query(sparql_query)

            if 'boolean' in raw:
                if not merged_columns:
                    merged_columns = ['result']
                merged_rows.append({'result': raw['boolean']})
                continue

            head = raw.get('head', {})
            cols = head.get('vars', [])
            if cols and not merged_columns:
                merged_columns = cols

            for b in raw.get('results', {}).get('bindings', []):
                row_dict = {}
                for var in cols:
                    cell = b.get(var)
                    row_dict[var] = cell['value'] if cell else None
                rk = tuple(sorted(row_dict.items()))
                if rk not in seen_row_keys:
                    seen_row_keys.add(rk)
                    merged_rows.append(row_dict)
        except Exception as e:
            errors.append(f"{cfg_row.name}: {e}")

    execution_time = (time.time() - start_time) * 1000

    error_msg = None
    if errors and not merged_rows:
        error_msg = "; ".join(errors)

    return SPARQLQueryResponse(
        success=len(merged_rows) > 0 or not errors,
        query=sparql_query,
        result_type='bindings' if merged_columns != ['result'] else 'boolean',
        columns=merged_columns,
        results=merged_rows,
        total_results=len(merged_rows),
        execution_time_ms=round(execution_time, 2),
        error=error_msg,
    )



async def _query_store_backend(
    store_id: int,
    sparql_query: str,
    start_time: float,
    db: AsyncSession,
) -> SPARQLQueryResponse:
    """Execute a SPARQL query against a configured triple store backend."""
    result = await db.execute(
        select(TripleStoreConfigDB).where(TripleStoreConfigDB.id == store_id)
    )
    row = result.scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail=f"Triple store config {store_id} not found")

    enabled = get_enabled_backends()
    backend = enabled.get(store_id)
    if backend is None:
        backend = await create_backend_from_config(row)

    try:
        raw = await backend.sparql_query(sparql_query)
    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        return SPARQLQueryResponse(
            success=False,
            query=sparql_query,
            result_type="error",
            error=str(e),
            execution_time_ms=round(execution_time, 2),
        )

    execution_time = (time.time() - start_time) * 1000

    if "boolean" in raw:
        return SPARQLQueryResponse(
            success=True,
            query=sparql_query,
            result_type="boolean",
            columns=["result"],
            results=[{"result": raw["boolean"]}],
            total_results=1,
            execution_time_ms=round(execution_time, 2),
        )

    head = raw.get("head", {})
    columns = head.get("vars", [])
    bindings = raw.get("results", {}).get("bindings", [])

    rows = []
    for b in bindings:
        row_dict = {}
        for var in columns:
            cell = b.get(var)
            row_dict[var] = cell["value"] if cell else None
        rows.append(row_dict)

    return SPARQLQueryResponse(
        success=True,
        query=sparql_query,
        result_type="bindings",
        columns=columns,
        results=rows,
        total_results=len(rows),
        execution_time_ms=round(execution_time, 2),
    )


@router.get(
    "/sources",
    summary="List all available SPARQL query sources",
    description="Returns TTL files and configured triple store backends for the source selector.",
)
async def list_query_sources(db: AsyncSession = Depends(get_db)):
    """List all available targets for SPARQL queries."""
    sources = []

    sources.append({
        "id": None,
        "name": "All (Local KG + External Stores)",
        "type": "all",
    })

    try:
        kg_service = get_kg_service()
        local_triples = len(kg_service.graph) if kg_service.graph else 0
    except Exception:
        local_triples = 0

    sources.append({
        "id": None,
        "name": "Local KG (all TTL files)",
        "type": "local",
        "triple_count": local_triples,
    })

    kg_dir = KG_DEFAULT_DIR
    if kg_dir.exists():
        for f in sorted(kg_dir.glob("*.ttl")):
            sources.append({
                "id": None,
                "name": f.name,
                "type": "ttl_file",
                "ttl_file": f.name,
                "size": f.stat().st_size,
            })

    try:
        result = await db.execute(
            select(TripleStoreConfigDB).order_by(TripleStoreConfigDB.name)
        )
        for cfg in result.scalars().all():
            label = cfg.name
            if cfg.is_active:
                label += " ★"
            sources.append({
                "id": cfg.id,
                "name": label,
                "type": "store",
                "store_type": cfg.store_type,
                "is_active": cfg.is_active,
                "endpoint": cfg.sparql_query_endpoint or cfg.ttl_directory or "",
            })
    except Exception as e:
        logger.warning(f"Error listing triple store configs: {e}")

    return {"success": True, "sources": sources}
