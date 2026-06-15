import os
import json
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import httpx

try:
    from rdflib import Graph
    HAS_RDFLIB = True
except ImportError:
    HAS_RDFLIB = False

logger = logging.getLogger(__name__)

CACHE_DIR_NAME = ".triple_store_cache"
DISABLED_TTL_FILE = ".disabled_ttl_files.json"
KG_DEFAULT_DIR = Path(__file__).parent.parent / "KnowledgeGraph"



def get_disabled_ttl_files(ttl_directory: Optional[Path] = None) -> List[str]:
    d = ttl_directory or KG_DEFAULT_DIR
    p = d / DISABLED_TTL_FILE
    if not p.exists():
        return []
    try:
        with open(p, "r") as fp:
            return json.load(fp)
    except Exception:
        return []


def set_disabled_ttl_files(disabled: List[str], ttl_directory: Optional[Path] = None):
    d = ttl_directory or KG_DEFAULT_DIR
    d.mkdir(parents=True, exist_ok=True)
    p = d / DISABLED_TTL_FILE
    with open(p, "w") as fp:
        json.dump(disabled, fp)



class TripleStoreBackend(ABC):
    """Abstract interface for triple store backends."""

    @abstractmethod
    async def load_ttl_file(self, file_path: str) -> int:
        """Load a TTL file. Returns number of new triples."""

    @abstractmethod
    async def load_ttl_data(self, ttl_content: str) -> int:
        """Load TTL data from a string. Returns number of new triples."""

    @abstractmethod
    async def sparql_query(self, sparql: str) -> Dict[str, Any]:
        """Execute a SPARQL SELECT/ASK query. Returns SPARQL-JSON results dict."""

    @abstractmethod
    async def get_triple_count(self) -> int:
        """Return total number of triples."""

    @abstractmethod
    async def test_connection(self) -> Tuple[bool, str]:
        """Test the connection. Returns (ok, message)."""

    @abstractmethod
    async def clear(self) -> bool:
        """Clear all data."""

    @abstractmethod
    def get_rdflib_graph(self) -> Optional["Graph"]:
        """Return the underlying rdflib Graph (internal only). None for external."""



class InternalRDFLibBackend(TripleStoreBackend):
    """rdflib in-memory graph with optional N-Triples disk cache."""

    def __init__(self, ttl_directory: str, use_cache: bool = True):
        if not HAS_RDFLIB:
            raise ImportError("rdflib is required for the internal store.")
        self.ttl_directory = Path(ttl_directory)
        self.use_cache = use_cache
        self.graph: Graph = Graph()
        self._cache_dir = self.ttl_directory / CACHE_DIR_NAME
        self._loaded_files: List[str] = []
        self.cache_status: Optional[str] = None


    async def load_all_ttl_files(self) -> int:
        """Load every *.ttl from the configured directory (with cache), skipping disabled files."""
        if not self.ttl_directory.exists():
            os.makedirs(self.ttl_directory, exist_ok=True)
            return 0

        disabled = set(get_disabled_ttl_files(self.ttl_directory))
        ttl_files = sorted(
            f for f in self.ttl_directory.glob("*.ttl")
            if f.name not in disabled
        )
        if not ttl_files:
            return 0

        if self.use_cache and self._is_cache_valid(ttl_files):
            count = self._load_from_cache()
            if count > 0:
                self._loaded_files = [str(f) for f in ttl_files]
                self.cache_status = "hit"
                return count

        start = time.time()
        self.graph = Graph()

        for ttl_file in ttl_files:
            try:
                self.graph.parse(str(ttl_file), format="turtle")
                self._loaded_files.append(str(ttl_file))
                logger.info(f"Parsed: {ttl_file.name}")
            except Exception as e:
                logger.error(f"Error parsing {ttl_file}: {e}")

        elapsed = time.time() - start
        count = len(self.graph)
        logger.info(f"Loaded {count} triples from {len(ttl_files)} files in {elapsed:.2f}s")

        if self.use_cache and count > 0:
            self._save_cache(ttl_files)
            self.cache_status = "miss"
        else:
            self.cache_status = "disabled" if not self.use_cache else "miss"

        return count


    async def load_ttl_file(self, file_path: str) -> int:
        before = len(self.graph)
        self.graph.parse(file_path, format="turtle")
        after = len(self.graph)
        if self.use_cache:
            ttl_files = sorted(self.ttl_directory.glob("*.ttl"))
            self._save_cache(ttl_files)
        return after - before

    async def load_ttl_data(self, ttl_content: str) -> int:
        before = len(self.graph)
        self.graph.parse(data=ttl_content, format="turtle")
        after = len(self.graph)
        if self.use_cache:
            ttl_files = sorted(self.ttl_directory.glob("*.ttl"))
            self._save_cache(ttl_files)
        return after - before

    async def sparql_query(self, sparql: str) -> Dict[str, Any]:
        """Execute SPARQL on local rdflib graph and return SPARQL-JSON-like dict."""
        result = self.graph.query(sparql)

        if hasattr(result, "vars") and result.vars:
            columns = [str(v) for v in result.vars]
            bindings = []
            for row in result:
                binding = {}
                for i, var in enumerate(columns):
                    value = row[i]
                    if value is not None:
                        binding[var] = {"type": "literal", "value": str(value)}
                bindings.append(binding)
            return {
                "head": {"vars": columns},
                "results": {"bindings": bindings}
            }
        elif hasattr(result, "askAnswer"):
            return {
                "head": {},
                "boolean": bool(result.askAnswer)
            }
        else:
            return {"head": {}, "results": {"bindings": []}}

    async def get_triple_count(self) -> int:
        return len(self.graph)

    async def test_connection(self) -> Tuple[bool, str]:
        try:
            count = len(self.graph)
            return True, f"Internal store ready — {count} triples loaded"
        except Exception as e:
            return False, str(e)

    async def clear(self) -> bool:
        self.graph = Graph()
        self._loaded_files = []
        return True

    def get_rdflib_graph(self) -> Optional[Graph]:
        return self.graph


    def _fingerprint(self, ttl_files: List[Path]) -> str:
        parts = []
        for f in sorted(ttl_files):
            st = f.stat()
            parts.append(f"{f.name}:{st.st_size}:{st.st_mtime}")
        return hashlib.md5("|".join(parts).encode()).hexdigest()

    def _is_cache_valid(self, ttl_files: List[Path]) -> bool:
        meta_file = self._cache_dir / "meta.json"
        cache_file = self._cache_dir / "graph.nt"
        if not meta_file.exists() or not cache_file.exists():
            return False
        try:
            with open(meta_file, "r") as fp:
                meta = json.load(fp)
            return meta.get("fingerprint") == self._fingerprint(ttl_files)
        except Exception:
            return False

    def _save_cache(self, ttl_files: List[Path]):
        try:
            os.makedirs(self._cache_dir, exist_ok=True)
            cache_file = self._cache_dir / "graph.nt"
            self.graph.serialize(str(cache_file), format="nt")
            meta_file = self._cache_dir / "meta.json"
            with open(meta_file, "w") as fp:
                json.dump({
                    "fingerprint": self._fingerprint(ttl_files),
                    "triple_count": len(self.graph),
                    "cached_at": time.time(),
                    "files": [f.name for f in ttl_files],
                }, fp)
            logger.info(f"Cache saved: {len(self.graph)} triples → {cache_file}")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def _load_from_cache(self) -> int:
        try:
            start = time.time()
            cache_file = self._cache_dir / "graph.nt"
            self.graph = Graph()
            self.graph.parse(str(cache_file), format="nt")
            elapsed = time.time() - start
            count = len(self.graph)
            logger.info(f"Loaded {count} triples from cache in {elapsed:.2f}s")
            return count
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return 0



class ExternalSPARQLBackend(TripleStoreBackend):
    """External triple store accessed via a SPARQL endpoint."""

    def __init__(
        self,
        query_endpoint: str,
        update_endpoint: Optional[str] = None,
        gsp_endpoint: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        named_graph: Optional[str] = None,
    ):
        self.query_endpoint = query_endpoint
        self.update_endpoint = update_endpoint
        self.gsp_endpoint = gsp_endpoint
        self.auth = (username, password) if username else None
        self.named_graph = named_graph
        self._client = httpx.AsyncClient(timeout=60.0)


    async def load_ttl_file(self, file_path: str) -> int:
        with open(file_path, "r", encoding="utf-8") as fp:
            content = fp.read()
        return await self.load_ttl_data(content)

    async def load_ttl_data(self, ttl_content: str) -> int:
        """Upload TTL using the Graph Store Protocol endpoint."""
        endpoint = self.gsp_endpoint
        if not endpoint:
            raise ValueError("No GSP endpoint configured for data upload")

        params = {"graph": self.named_graph} if self.named_graph else {"default": ""}
        headers = {"Content-Type": "text/turtle; charset=utf-8"}

        before = await self.get_triple_count()
        resp = await self._client.post(
            endpoint,
            content=ttl_content.encode("utf-8"),
            headers=headers,
            params=params,
            auth=self.auth,
        )
        resp.raise_for_status()
        after = await self.get_triple_count()
        return after - before

    _COMMON_PREFIXES: Dict[str, str] = {
        "rdf":     "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs":    "http://www.w3.org/2000/01/rdf-schema#",
        "owl":     "http://www.w3.org/2002/07/owl#",
        "xsd":     "http://www.w3.org/2001/XMLSchema#",
        "skos":    "http://www.w3.org/2004/02/skos/core#",
        "dc":      "http://purl.org/dc/elements/1.1/",
        "dcterms": "http://purl.org/dc/terms/",
        "foaf":    "http://xmlns.com/foaf/0.1/",
        "schema":  "http://schema.org/",
        "med":     "http://example.org/medical/",
        "medtype": "http://example.org/medical/types/",
        "medrel":  "http://example.org/medical/relations/",
    }

    def _ensure_prefixes(self, sparql: str) -> str:
        """Auto-prepend PREFIX declarations for common prefixes used but not declared."""
        import re
        declared = {
            m.group(1).lower()
            for m in re.finditer(r'(?i)PREFIX\s+(\w+)\s*:', sparql)
        }
        used = {
            m.group(1).lower()
            for m in re.finditer(r'(?<!\w)(\w+):\w+', sparql)
            if not m.group(0).startswith('http')
        }
        missing_lines = []
        for prefix, uri in self._COMMON_PREFIXES.items():
            if prefix in used and prefix not in declared:
                missing_lines.append(f"PREFIX {prefix}: <{uri}>")
        if missing_lines:
            return "\n".join(missing_lines) + "\n" + sparql
        return sparql

    async def sparql_query(self, sparql: str) -> Dict[str, Any]:
        """Execute SPARQL query against the external endpoint.

        Auto-prepends common PREFIX declarations when they are used
        but not declared (external stores like Jena require explicit prefixes).
        """
        sparql = self._ensure_prefixes(sparql)
        headers = {
            "Accept": "application/sparql-results+json",
        }
        resp = await self._client.post(
            self.query_endpoint,
            data={"query": sparql},
            headers=headers,
            auth=self.auth,
        )
        resp.raise_for_status()
        return resp.json()

    async def get_triple_count(self) -> int:
        try:
            if self.named_graph:
                q = f"SELECT (COUNT(*) AS ?c) WHERE {{ GRAPH <{self.named_graph}> {{ ?s ?p ?o }} }}"
            else:
                q = "SELECT (COUNT(*) AS ?c) WHERE { ?s ?p ?o }"
            data = await self.sparql_query(q)
            bindings = data.get("results", {}).get("bindings", [])
            return int(bindings[0]["c"]["value"]) if bindings else 0
        except Exception:
            return 0

    async def test_connection(self) -> Tuple[bool, str]:
        try:
            count = await self.get_triple_count()
            return True, f"Connected to SPARQL endpoint — {count} triples"
        except httpx.ConnectError:
            return False, "Cannot connect to SPARQL endpoint"
        except Exception as e:
            return False, f"Error: {e}"

    async def clear(self) -> bool:
        if not self.update_endpoint:
            return False
        try:
            sparql = f"CLEAR GRAPH <{self.named_graph}>" if self.named_graph else "CLEAR DEFAULT"
            resp = await self._client.post(
                self.update_endpoint,
                data={"update": sparql},
                auth=self.auth,
            )
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Error clearing external store: {e}")
            return False

    def get_rdflib_graph(self) -> Optional["Graph"]:
        return None



_enabled_backends: Dict[int, TripleStoreBackend] = {}


def get_enabled_backends() -> Dict[int, TripleStoreBackend]:
    """Return dict of all enabled backends keyed by config id."""
    return _enabled_backends


def get_active_backend() -> Optional[TripleStoreBackend]:
    """Backwards-compat: return the first internal backend, or the first backend overall."""
    for be in _enabled_backends.values():
        if be.get_rdflib_graph() is not None:
            return be
    if _enabled_backends:
        return next(iter(_enabled_backends.values()))
    return None


def set_active_backend(backend: Optional[TripleStoreBackend]):
    """Backwards-compat shim: add/replace the backend with id 0."""
    global _enabled_backends
    if backend is None:
        _enabled_backends.clear()
    else:
        _enabled_backends[0] = backend


def add_enabled_backend(config_id: int, backend: TripleStoreBackend):
    """Register a backend under its config id."""
    _enabled_backends[config_id] = backend


def remove_enabled_backend(config_id: int):
    """Remove a backend by config id."""
    _enabled_backends.pop(config_id, None)


def get_enabled_external_backends() -> List[TripleStoreBackend]:
    """Return all enabled external (non-rdflib) backends."""
    return [be for be in _enabled_backends.values() if be.get_rdflib_graph() is None]


async def create_backend_from_config(config) -> TripleStoreBackend:
    """
    Instantiate the appropriate backend from a TripleStoreConfigDB row.

    For internal backends the TTL files are loaded automatically.
    """
    if config.store_type == "internal":
        directory = config.ttl_directory or str(KG_DEFAULT_DIR)
        backend = InternalRDFLibBackend(
            ttl_directory=directory,
            use_cache=config.use_cache if config.use_cache is not None else True,
        )
        await backend.load_all_ttl_files()
        return backend
    else:
        return ExternalSPARQLBackend(
            query_endpoint=config.sparql_query_endpoint,
            update_endpoint=config.sparql_update_endpoint,
            gsp_endpoint=config.sparql_gsp_endpoint,
            username=config.auth_username,
            password=config.auth_password,
            named_graph=config.named_graph,
        )
