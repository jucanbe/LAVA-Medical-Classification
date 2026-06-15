from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field


class TripleStoreConfigBase(BaseModel):
    """Base model for Triple Store configuration."""

    name: str = Field(..., min_length=1, max_length=100, description="Configuration name")
    store_type: str = Field(
        ...,
        description="Backend type: 'internal' (rdflib in-memory + cache) or 'sparql_endpoint' (external SPARQL server)"
    )

    ttl_directory: Optional[str] = Field(
        None,
        description="Directory containing TTL files. Defaults to KnowledgeGraph/"
    )
    use_cache: bool = Field(
        True,
        description="Cache parsed graph as N-Triples for faster subsequent loads (internal only)"
    )

    sparql_query_endpoint: Optional[str] = Field(
        None,
        description="SPARQL query endpoint URL (e.g. http://localhost:3030/ds/query)"
    )
    sparql_update_endpoint: Optional[str] = Field(
        None,
        description="SPARQL update endpoint URL (e.g. http://localhost:3030/ds/update)"
    )
    sparql_gsp_endpoint: Optional[str] = Field(
        None,
        description="Graph Store Protocol endpoint URL for data upload (e.g. http://localhost:3030/ds/data)"
    )
    auth_username: Optional[str] = Field(None, description="Username for SPARQL endpoint authentication")
    auth_password: Optional[str] = Field(None, description="Password for SPARQL endpoint authentication")
    named_graph: Optional[str] = Field(None, description="Named graph URI (optional)")

    is_active: bool = Field(True, description="Whether this triple store is enabled (queried in searches)")


class TripleStoreConfigCreate(TripleStoreConfigBase):
    """Model for creating a new Triple Store configuration."""
    pass


class TripleStoreConfigUpdate(BaseModel):
    """Model for updating a Triple Store configuration."""

    name: Optional[str] = Field(None, min_length=1, max_length=100)
    store_type: Optional[str] = None
    ttl_directory: Optional[str] = None
    use_cache: Optional[bool] = None
    sparql_query_endpoint: Optional[str] = None
    sparql_update_endpoint: Optional[str] = None
    sparql_gsp_endpoint: Optional[str] = None
    auth_username: Optional[str] = None
    auth_password: Optional[str] = None
    named_graph: Optional[str] = None
    is_active: Optional[bool] = None


class TripleStoreConfigResponse(TripleStoreConfigBase):
    """Response model for Triple Store configuration."""

    model_config = {"from_attributes": True}

    id: int = Field(..., description="Unique ID")
    created_at: datetime
    updated_at: datetime


class TripleStoreStatus(BaseModel):
    """Status of the triple store backends."""

    active_config: Optional[TripleStoreConfigResponse] = None
    configs: Optional[list] = None
    backend_type: str = "none"
    triple_count: int = 0
    is_connected: bool = False
    cache_status: Optional[str] = None
    message: str = ""
