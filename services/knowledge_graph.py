import os
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from difflib import SequenceMatcher
import re
import asyncio

try:
    from rdflib import Graph, Namespace, RDF, RDFS, OWL
    from rdflib.namespace import SKOS
    HAS_RDFLIB = True
except ImportError:
    HAS_RDFLIB = False

from models.entities import (
    MedicalEntity,
    MedicalEntityType,
    ValidatedEntity,
    KGMatch,
    KGMatchStatus,
    KGValidationResult,
    EntityClassificationResult
)

logger = logging.getLogger(__name__)

KG_DEFAULT_DIR = Path(__file__).parent.parent / "KnowledgeGraph"

_ENTITY_TYPE_TO_KG_TYPES: Dict[str, List[str]] = {
    "disease": ["Disease"],
    "symptom": ["Finding"],
    "finding": ["Finding"],
    "organ": ["Finding"],
    "imaging_procedure": ["Procedure"],
    "examination_procedure": ["Procedure"],
    "therapeutic_procedure": ["Procedure"],
    "imaging_result": ["Finding"],
    "examination_measure": ["QuantitativeMeasure"],
    "parameter": ["QuantitativeMeasure"],
    "score": ["QuantitativeMeasure"],
    "therapy": ["Substance", "Procedure"],
    "substance": ["Substance"],
    "adverse_event": ["Finding", "Disease"],
}


def _try_get_active_backend():
    try:
        from services.triple_store import get_active_backend
        return get_active_backend()
    except Exception:
        return None


def _try_get_external_backends():
    try:
        from services.triple_store import get_enabled_external_backends
        return get_enabled_external_backends()
    except Exception:
        return []


class KnowledgeGraphService:
    
    EXACT_MATCH_THRESHOLD = 0.95
    SIMILAR_MATCH_THRESHOLD = 0.70
    LOW_MATCH_THRESHOLD = 0.60
    
    def __init__(self, kg_directory: Optional[str] = None):

        if not HAS_RDFLIB:
            raise ImportError(
                "rdflib is required for Knowledge Graph operations. "
                "Install it with: pip install rdflib"
            )
        
        self.kg_directory = Path(kg_directory) if kg_directory else KG_DEFAULT_DIR
        self.graph = Graph()
        self._entities_cache: Dict[str, Dict] = {}
        self._loaded_files: List[str] = []
        
        self.namespaces = {
            'rdf': RDF,
            'rdfs': RDFS,
            'owl': OWL,
            'skos': SKOS
        }

        backend = _try_get_active_backend()
        if backend is not None:
            rdf_graph = backend.get_rdflib_graph()
            if rdf_graph is not None:
                self.graph = rdf_graph
                logger.info("KG service: using rdflib graph from active triple store backend")
    
    def load_ttl_files(self, file_patterns: Optional[List[str]] = None) -> int:

        backend = _try_get_active_backend()
        if backend is not None:
            rdf_graph = backend.get_rdflib_graph()
            if rdf_graph is not None and len(rdf_graph) > 0:
                self.graph = rdf_graph
                self._build_entity_cache()
                logger.info(f"KG loaded from triple store backend: {len(self.graph)} triples")
                return len(self.graph)

        if not self.kg_directory.exists():
            logger.warning(f"KG directory does not exist: {self.kg_directory}")
            os.makedirs(self.kg_directory, exist_ok=True)
            return 0
        
        patterns = file_patterns or ["*.ttl"]
        initial_count = len(self.graph)
        
        for pattern in patterns:
            for ttl_file in self.kg_directory.glob(pattern):
                try:
                    self.graph.parse(ttl_file, format="turtle")
                    self._loaded_files.append(str(ttl_file))
                    logger.info(f"Loaded KG file: {ttl_file}")
                except Exception as e:
                    logger.error(f"Error loading {ttl_file}: {e}")
        
        new_triples = len(self.graph) - initial_count
        logger.info(f"Loaded {new_triples} triples from {len(self._loaded_files)} files")
        
        self._build_entity_cache()
        
        return new_triples
    
    def _build_entity_cache(self):
        self._entities_cache.clear()
        
        label_predicates = [
            RDFS.label,
            SKOS.prefLabel,
            SKOS.altLabel,
        ]
        
        for subj in self.graph.subjects():
            uri = str(subj)
            if uri not in self._entities_cache:
                self._entities_cache[uri] = {
                    'uri': uri,
                    'labels': [],
                    'types': [],
                    'normalized_labels': []
                }
            
            for pred in label_predicates:
                for obj in self.graph.objects(subj, pred):
                    label = str(obj)
                    if label not in self._entities_cache[uri]['labels']:
                        self._entities_cache[uri]['labels'].append(label)
                        self._entities_cache[uri]['normalized_labels'].append(
                            self._normalize_text(label)
                        )
            
            for obj in self.graph.objects(subj, RDF.type):
                type_uri = str(obj)
                if type_uri not in self._entities_cache[uri]['types']:
                    self._entities_cache[uri]['types'].append(type_uri)
        
        logger.info(f"Built entity cache with {len(self._entities_cache)} entities")
    
    def _normalize_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using SequenceMatcher."""
        return SequenceMatcher(None, text1, text2).ratio()
    
    @staticmethod
    def _kg_type_local(type_uri: str) -> str:
        """Extract local name from a KG type URI."""
        return type_uri.split('#')[-1].split('/')[-1]

    def _matches_entity_type(self, entity_data: Dict, entity_type: Optional[str]) -> bool:
        """Check if a cached entity matches the requested entity_type filter."""
        if entity_type is None:
            return True
        allowed_kg_types = _ENTITY_TYPE_TO_KG_TYPES.get(entity_type)
        if not allowed_kg_types:
            return True
        for t in entity_data['types']:
            if self._kg_type_local(t) in allowed_kg_types:
                return True
        return False

    def find_matches(
        self,
        entity_text: str,
        max_matches: int = 5,
        min_score: float = None,
        entity_type: Optional[str] = None,
    ) -> List[KGMatch]:

        if min_score is None:
            min_score = self.SIMILAR_MATCH_THRESHOLD
            
        normalized_query = self._normalize_text(entity_text)
        matches = []
        
        for uri, entity_data in self._entities_cache.items():
            if not self._matches_entity_type(entity_data, entity_type):
                continue

            best_score = 0.0
            best_label = ""
            
            for i, norm_label in enumerate(entity_data['normalized_labels']):
                if normalized_query == norm_label:
                    best_score = 1.0
                    best_label = entity_data['labels'][i]
                    break
                
                score = self._calculate_similarity(normalized_query, norm_label)
                if score > best_score:
                    best_score = score
                    best_label = entity_data['labels'][i]
                
                if normalized_query in norm_label or norm_label in normalized_query:
                    containment_score = min(len(normalized_query), len(norm_label)) / max(len(normalized_query), len(norm_label))
                    if containment_score > best_score:
                        best_score = containment_score
                        best_label = entity_data['labels'][i]
            
            if best_score >= min_score:
                kg_type = entity_data['types'][0] if entity_data['types'] else None
                if kg_type:
                    kg_type = self._kg_type_local(kg_type)
                
                matches.append(KGMatch(
                    kg_uri=uri,
                    kg_label=best_label,
                    kg_type=kg_type,
                    similarity_score=best_score
                ))
        
        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        return matches[:max_matches]


    async def _find_matches_external(
        self,
        entity_text: str,
        max_matches: int = 5,
        min_score: float = None,
        entity_type: Optional[str] = None,
    ) -> List[KGMatch]:

        if min_score is None:
            min_score = self.SIMILAR_MATCH_THRESHOLD

        backends = _try_get_external_backends()
        if not backends:
            return []

        normalized_query = self._normalize_text(entity_text)
        if not normalized_query:
            return []

        search_token = entity_text.replace("\\", "\\\\").replace("'", "\\'")

        type_filter = ""
        if entity_type:
            allowed = _ENTITY_TYPE_TO_KG_TYPES.get(entity_type)
            if allowed:
                type_uris = ", ".join(
                    f"<http://example.org/medical/types/{t}>" for t in allowed
                )
                type_filter = f"  ?entity a ?type .\n  FILTER(?type IN ({type_uris}))\n"

        if type_filter:
            sparql = (
                "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
                "PREFIX skos: <http://www.w3.org/2004/02/skos/core#>\n"
                "SELECT DISTINCT ?entity ?label ?type WHERE {\n"
                "  ?entity ?labelPred ?label .\n"
                "  FILTER(?labelPred IN (rdfs:label, skos:prefLabel, skos:altLabel))\n"
                f"  FILTER(CONTAINS(LCASE(STR(?label)), LCASE('{search_token}')))\n"
                + type_filter
                + f"}} LIMIT {max_matches * 10}\n"
            )
        else:
            sparql = (
                "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
                "PREFIX skos: <http://www.w3.org/2004/02/skos/core#>\n"
                "SELECT DISTINCT ?entity ?label ?type WHERE {\n"
                "  ?entity ?labelPred ?label .\n"
                "  FILTER(?labelPred IN (rdfs:label, skos:prefLabel, skos:altLabel))\n"
                f"  FILTER(CONTAINS(LCASE(STR(?label)), LCASE('{search_token}')))\n"
                "  OPTIONAL {{ ?entity a ?type }}\n"
                f"}} LIMIT {max_matches * 10}\n"
            )

        all_matches: List[KGMatch] = []

        for backend in backends:
            try:
                raw = await backend.sparql_query(sparql)
            except Exception as e:
                logger.warning(f"External SPARQL search failed for '{entity_text}': {e}")
                continue

            for b in raw.get("results", {}).get("bindings", []):
                uri = b.get("entity", {}).get("value", "")
                label = b.get("label", {}).get("value", "")
                type_uri = b.get("type", {}).get("value", "")

                if not uri or not label:
                    continue

                norm_label = self._normalize_text(label)
                if normalized_query == norm_label:
                    score = 1.0
                else:
                    score = self._calculate_similarity(normalized_query, norm_label)
                    if normalized_query in norm_label or norm_label in normalized_query:
                        containment = min(len(normalized_query), len(norm_label)) / max(len(normalized_query), len(norm_label))
                        score = max(score, containment)

                if score < min_score:
                    continue

                kg_type = None
                if type_uri:
                    kg_type = type_uri.split("#")[-1].split("/")[-1]

                all_matches.append(KGMatch(
                    kg_uri=uri,
                    kg_label=label,
                    kg_type=kg_type,
                    similarity_score=score,
                ))

        by_uri: Dict[str, KGMatch] = {}
        for m in all_matches:
            existing = by_uri.get(m.kg_uri)
            if existing is None or m.similarity_score > existing.similarity_score:
                by_uri[m.kg_uri] = m
        result = sorted(by_uri.values(), key=lambda x: x.similarity_score, reverse=True)
        return result[:max_matches]

    def _merge_matches(
        self,
        local: List[KGMatch],
        external: List[KGMatch],
        max_matches: int = 5,
    ) -> List[KGMatch]:
        """Merge local and external match lists, deduplicated by URI."""
        by_uri: Dict[str, KGMatch] = {}
        for m in local + external:
            existing = by_uri.get(m.kg_uri)
            if existing is None or m.similarity_score > existing.similarity_score:
                by_uri[m.kg_uri] = m
        merged = sorted(by_uri.values(), key=lambda x: x.similarity_score, reverse=True)
        return merged[:max_matches]


    async def find_matches_async(
        self,
        entity_text: str,
        max_matches: int = 5,
        min_score: float = None,
        entity_type: Optional[str] = None,
        kg_backend: str = "all",
    ) -> List[KGMatch]:

        local: List[KGMatch] = []
        external: List[KGMatch] = []

        if kg_backend in ("all", "internal"):
            local = self.find_matches(
                entity_text, max_matches=max_matches,
                min_score=min_score, entity_type=entity_type,
            )
        if kg_backend in ("all", "external"):
            external = await self._find_matches_external(
                entity_text, max_matches=max_matches,
                min_score=min_score, entity_type=entity_type,
            )
        return self._merge_matches(local, external, max_matches=max_matches)

    async def validate_entity_async(
        self, entity: MedicalEntity, kg_backend: str = "all"
    ) -> ValidatedEntity:

        etype = entity.entity_type.value if hasattr(entity.entity_type, 'value') else str(entity.entity_type)
        matches = await self.find_matches_async(
            entity.text,
            min_score=self.LOW_MATCH_THRESHOLD,
            entity_type=etype,
            kg_backend=kg_backend,
        )
        if not matches:
            matches = await self.find_matches_async(
                entity.text,
                min_score=self.LOW_MATCH_THRESHOLD,
                entity_type=None,
                kg_backend=kg_backend,
            )
        return self._build_validated_entity(entity, matches)

    async def validate_entities_async(
        self,
        classification_result: EntityClassificationResult,
        kg_backend: str = "all",
    ) -> KGValidationResult:

        import time
        start_time = time.time()

        tasks = [
            self.validate_entity_async(entity, kg_backend=kg_backend)
            for entity in classification_result.entities
        ]
        validated_entities = await asyncio.gather(*tasks)

        exact_matches = 0
        similar_matches = 0
        low_matches = 0
        not_found = 0

        for validated in validated_entities:
            if validated.validation_status == KGMatchStatus.EXACT_MATCH:
                exact_matches += 1
            elif validated.validation_status == KGMatchStatus.SIMILAR_MATCH:
                similar_matches += 1
            elif validated.validation_status == KGMatchStatus.LOW_MATCH:
                low_matches += 1
            else:
                not_found += 1

        processing_time = (time.time() - start_time) * 1000

        return KGValidationResult(
            validated_entities=list(validated_entities),
            total_entities=len(validated_entities),
            exact_matches=exact_matches,
            similar_matches=similar_matches,
            low_matches=low_matches,
            not_found=not_found,
            processing_time_ms=processing_time,
        )


    def _build_validated_entity(
        self, entity: MedicalEntity, matches: List[KGMatch]
    ) -> ValidatedEntity:
        if matches:
            best_match = matches[0]

            if best_match.similarity_score >= self.EXACT_MATCH_THRESHOLD:
                status = KGMatchStatus.EXACT_MATCH
                is_validated = True
                notes = f"Exact match: '{best_match.kg_label}' ({best_match.similarity_score:.0%})"
            elif best_match.similarity_score >= self.SIMILAR_MATCH_THRESHOLD:
                status = KGMatchStatus.SIMILAR_MATCH
                is_validated = True
                notes = f"Similar match: '{best_match.kg_label}' ({best_match.similarity_score:.0%})"
            else:
                status = KGMatchStatus.LOW_MATCH
                is_validated = False
                notes = f"Low match: '{best_match.kg_label}' ({best_match.similarity_score:.0%})"
        else:
            status = KGMatchStatus.NOT_FOUND
            is_validated = False
            notes = "Not found in Knowledge Graph"

        return ValidatedEntity(
            entity=entity,
            validation_status=status,
            kg_matches=matches,
            is_validated=is_validated,
            validation_notes=notes,
        )

    def validate_entity(self, entity: MedicalEntity) -> ValidatedEntity:
        matches = self.find_matches(entity.text, min_score=self.LOW_MATCH_THRESHOLD)
        return self._build_validated_entity(entity, matches)

    def validate_entities(
        self,
        classification_result: EntityClassificationResult
    ) -> KGValidationResult:

        import time
        start_time = time.time()
        
        validated_entities = []
        exact_matches = 0
        similar_matches = 0
        low_matches = 0
        not_found = 0
        
        for entity in classification_result.entities:
            validated = self.validate_entity(entity)
            validated_entities.append(validated)
            
            if validated.validation_status == KGMatchStatus.EXACT_MATCH:
                exact_matches += 1
            elif validated.validation_status == KGMatchStatus.SIMILAR_MATCH:
                similar_matches += 1
            elif validated.validation_status == KGMatchStatus.LOW_MATCH:
                low_matches += 1
            else:
                not_found += 1
        
        processing_time = (time.time() - start_time) * 1000
        
        return KGValidationResult(
            validated_entities=validated_entities,
            total_entities=len(validated_entities),
            exact_matches=exact_matches,
            similar_matches=similar_matches,
            low_matches=low_matches,
            not_found=not_found,
            processing_time_ms=processing_time
        )
    
    def get_kg_stats(self) -> Dict:
        total_labels = 0
        entities_by_type = {}
        sample_entities = []
        
        for uri, entity_data in self._entities_cache.items():
            total_labels += len(entity_data.get('labels', []))
            
            types = entity_data.get('types', [])
            for type_uri in types:
                type_name = type_uri.split('/')[-1].lower()
                if type_name in ['disease', 'symptom', 'finding', 'organ', 
                                 'imagingprocedure', 'imaging_procedure',
                                 'examinationprocedure', 'examination_procedure',
                                 'therapeuticprocedure', 'therapeutic_procedure',
                                 'imagingresult', 'imaging_result',
                                 'examinationmeasure', 'examination_measure',
                                 'parameter', 'score', 'therapy',
                                 'substance', 'adverseevent', 'adverse_event',
                                 'quantitativemeasure', 'quantitative_measure',
                                 'procedure']:
                    kg_type_normalize = {
                        'quantitativemeasure': 'parameter',
                        'quantitative_measure': 'parameter',
                        'imagingprocedure': 'imaging_procedure',
                        'examinationprocedure': 'examination_procedure',
                        'therapeuticprocedure': 'therapeutic_procedure',
                        'imagingresult': 'imaging_result',
                        'examinationmeasure': 'examination_measure',
                        'adverseevent': 'adverse_event',
                        'procedure': 'therapeutic_procedure',
                    }
                    type_name = kg_type_normalize.get(type_name, type_name)
                    
                    if type_name not in entities_by_type:
                        entities_by_type[type_name] = 0
                    entities_by_type[type_name] += 1
                    
                    if len(sample_entities) < 20:
                        labels = entity_data.get('labels', [])
                        if labels:
                            sample_entities.append({
                                'label': labels[0],
                                'type': type_name,
                                'uri': uri
                            })
        
        return {
            "loaded_files": self._loaded_files,
            "total_triples": len(self.graph),
            "total_entities": len(self._entities_cache),
            "total_labels": total_labels,
            "entities_by_type": entities_by_type,
            "sample_entities": sample_entities,
            "kg_directory": str(self.kg_directory)
        }
    
    def search_entities(self, query: str, max_results: int = 10) -> List[Dict]:

        if not query or not query.strip():
            return []
        
        normalized_query = self._normalize_text(query)
        results = []
        
        for uri, entity_data in self._entities_cache.items():
            labels = entity_data.get('labels', [])
            normalized_labels = entity_data.get('normalized_labels', [])
            
            for i, norm_label in enumerate(normalized_labels):
                if normalized_query in norm_label or norm_label in normalized_query:
                    similarity = 1.0 if normalized_query == norm_label else self._calculate_similarity(normalized_query, norm_label)
                    
                    types = entity_data.get('types', [])
                    entity_type = 'unknown'
                    for type_uri in types:
                        type_name = type_uri.split('/')[-1].lower()
                        if type_name in ['disease', 'symptom', 'finding', 'organ',
                                         'imagingprocedure', 'imaging_procedure',
                                         'examinationprocedure', 'examination_procedure',
                                         'therapeuticprocedure', 'therapeutic_procedure',
                                         'imagingresult', 'imaging_result',
                                         'examinationmeasure', 'examination_measure',
                                         'parameter', 'score', 'therapy',
                                         'substance', 'adverseevent', 'adverse_event',
                                         'quantitativemeasure', 'quantitative_measure',
                                         'procedure']:
                            entity_type = type_name
                            kg_type_normalize = {
                                'quantitativemeasure': 'parameter',
                                'quantitative_measure': 'parameter',
                                'imagingprocedure': 'imaging_procedure',
                                'examinationprocedure': 'examination_procedure',
                                'therapeuticprocedure': 'therapeutic_procedure',
                                'imagingresult': 'imaging_result',
                                'examinationmeasure': 'examination_measure',
                                'adverseevent': 'adverse_event',
                                'procedure': 'therapeutic_procedure',
                            }
                            entity_type = kg_type_normalize.get(entity_type, entity_type)
                            break
                    
                    results.append({
                        'label': labels[i] if i < len(labels) else norm_label,
                        'type': entity_type,
                        'uri': uri,
                        'similarity': similarity
                    })
                    break
            
            if len(results) >= max_results * 2:
                break
        
        if len(results) < max_results:
            for uri, entity_data in self._entities_cache.items():
                if any(r['uri'] == uri for r in results):
                    continue
                
                labels = entity_data.get('labels', [])
                normalized_labels = entity_data.get('normalized_labels', [])
                
                for i, norm_label in enumerate(normalized_labels):
                    similarity = self._calculate_similarity(normalized_query, norm_label)
                    if similarity >= 0.5:
                        types = entity_data.get('types', [])
                        entity_type = 'unknown'
                        for type_uri in types:
                            type_name = type_uri.split('/')[-1].lower()
                            if type_name in ['disease', 'symptom', 'finding', 'organ',
                                             'imagingprocedure', 'imaging_procedure',
                                             'examinationprocedure', 'examination_procedure',
                                             'therapeuticprocedure', 'therapeutic_procedure',
                                             'imagingresult', 'imaging_result',
                                             'examinationmeasure', 'examination_measure',
                                             'parameter', 'score', 'therapy',
                                             'substance', 'adverseevent', 'adverse_event',
                                             'quantitativemeasure', 'quantitative_measure',
                                             'procedure']:
                                entity_type = type_name
                                kg_type_normalize = {
                                    'quantitativemeasure': 'parameter',
                                    'quantitative_measure': 'parameter',
                                    'imagingprocedure': 'imaging_procedure',
                                    'examinationprocedure': 'examination_procedure',
                                    'therapeuticprocedure': 'therapeutic_procedure',
                                    'imagingresult': 'imaging_result',
                                    'examinationmeasure': 'examination_measure',
                                    'adverseevent': 'adverse_event',
                                    'procedure': 'therapeutic_procedure',
                                }
                                entity_type = kg_type_normalize.get(entity_type, entity_type)
                                break
                        
                        results.append({
                            'label': labels[i] if i < len(labels) else norm_label,
                            'type': entity_type,
                            'uri': uri,
                            'similarity': similarity
                        })
                        break
                
                if len(results) >= max_results * 2:
                    break
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:max_results]
    
    def add_entity_to_kg(
        self,
        entity: MedicalEntity,
        entity_uri: Optional[str] = None,
        link_to_uri: Optional[str] = None,
        link_type: Optional[str] = None
    ) -> str:

        from rdflib import URIRef, Literal
        
        if not entity_uri:
            safe_name = re.sub(r'[^a-zA-Z0-9]', '_', entity.text)
            import time
            timestamp = int(time.time() * 1000)
            entity_uri = f"http://example.org/medical/{safe_name}_{timestamp}"
        
        uri_ref = URIRef(entity_uri)
        
        self.graph.add((uri_ref, RDFS.label, Literal(entity.text)))
        
        type_mapping = {
            MedicalEntityType.DISEASE: "Disease",
            MedicalEntityType.SYMPTOM: "Symptom",
            MedicalEntityType.FINDING: "Finding",
            MedicalEntityType.ORGAN: "Organ",
            MedicalEntityType.IMAGING_PROCEDURE: "ImagingProcedure",
            MedicalEntityType.EXAMINATION_PROCEDURE: "ExaminationProcedure",
            MedicalEntityType.THERAPEUTIC_PROCEDURE: "TherapeuticProcedure",
            MedicalEntityType.IMAGING_RESULT: "ImagingResult",
            MedicalEntityType.EXAMINATION_MEASURE: "ExaminationMeasure",
            MedicalEntityType.PARAMETER: "Parameter",
            MedicalEntityType.SCORE: "Score",
            MedicalEntityType.THERAPY: "Therapy",
            MedicalEntityType.SUBSTANCE: "Substance",
            MedicalEntityType.ADVERSE_EVENT: "AdverseEvent"
        }
        
        type_name = type_mapping.get(entity.entity_type, "Finding")
        type_uri = URIRef(f"http://example.org/medical/types/{type_name}")
        self.graph.add((uri_ref, RDF.type, type_uri))
        
        subclass_mapping = {
            "ImagingProcedure": "Procedure",
            "ExaminationProcedure": "Procedure",
            "TherapeuticProcedure": "Procedure",
            "ExaminationMeasure": "QuantitativeMeasure",
            "Parameter": "QuantitativeMeasure",
            "Score": "QuantitativeMeasure",
        }
        if type_name in subclass_mapping:
            parent_type = subclass_mapping[type_name]
            parent_uri = URIRef(f"http://example.org/medical/types/{parent_type}")
            self.graph.add((type_uri, RDFS.subClassOf, parent_uri))
        
        if entity.normalized_form:
            self.graph.add((uri_ref, SKOS.prefLabel, Literal(entity.normalized_form)))
        
        if link_to_uri and link_type:
            linked_uri = URIRef(link_to_uri)
            link_predicates = {
                'sameAs': OWL.sameAs,
                'relatedTo': SKOS.related,
                'narrowerThan': SKOS.narrower,
                'broaderThan': SKOS.broader,
                'closeMatch': SKOS.closeMatch,
                'exactMatch': SKOS.exactMatch,
            }
            predicate = link_predicates.get(link_type, SKOS.related)
            self.graph.add((uri_ref, predicate, linked_uri))
            logger.info(f"Linked {entity.text} to {link_to_uri} via {link_type}")
        
        self._build_entity_cache()
        
        logger.info(f"Added entity to KG: {entity.text} ({entity_uri})")
        return entity_uri

    def add_relation_to_kg(
        self,
        source_entity: str,
        source_type: str,
        relation_type: str,
        target_entity: str,
        target_type: str,
    ) -> Dict[str, str]:

        from rdflib import URIRef, Literal
        import time

        def _get_or_create_entity_uri(text: str, etype: str) -> URIRef:
            for s, _p, o in self.graph.triples((None, RDFS.label, Literal(text))):
                return s
            safe = re.sub(r'[^a-zA-Z0-9]', '_', text)
            ts = int(time.time() * 1000)
            uri = URIRef(f"http://example.org/medical/{safe}_{ts}")
            self.graph.add((uri, RDFS.label, Literal(text)))

            str_to_type = {
                'disease': 'Disease', 'symptom': 'Symptom', 'finding': 'Finding',
                'organ': 'Organ', 'imaging_procedure': 'ImagingProcedure',
                'examination_procedure': 'ExaminationProcedure',
                'therapeutic_procedure': 'TherapeuticProcedure',
                'imaging_result': 'ImagingResult',
                'examination_measure': 'ExaminationMeasure',
                'parameter': 'Parameter', 'score': 'Score',
                'therapy': 'Therapy', 'substance': 'Substance',
                'adverse_event': 'AdverseEvent',
            }
            type_name = str_to_type.get(etype.lower(), 'Finding')
            type_uri = URIRef(f"http://example.org/medical/types/{type_name}")
            self.graph.add((uri, RDF.type, type_uri))

            subclass_map = {
                'ImagingProcedure': 'Procedure',
                'ExaminationProcedure': 'Procedure',
                'TherapeuticProcedure': 'Procedure',
                'ExaminationMeasure': 'QuantitativeMeasure',
                'Parameter': 'QuantitativeMeasure',
                'Score': 'QuantitativeMeasure',
            }
            if type_name in subclass_map:
                parent = subclass_map[type_name]
                self.graph.add((
                    type_uri,
                    RDFS.subClassOf,
                    URIRef(f"http://example.org/medical/types/{parent}")
                ))

            return uri

        source_uri = _get_or_create_entity_uri(source_entity, source_type)
        target_uri = _get_or_create_entity_uri(target_entity, target_type)

        relation_uri = URIRef(f"http://example.org/medical/relations/{relation_type}")
        self.graph.add((source_uri, relation_uri, target_uri))

        self._build_entity_cache()

        logger.info(
            f"Added relation to KG: {source_entity} --[{relation_type}]--> {target_entity}"
        )
        return {
            "source_uri": str(source_uri),
            "relation_uri": str(relation_uri),
            "target_uri": str(target_uri),
        }

    def save_kg(self, filename: str = "medical_entities.ttl") -> str:

        filepath = self.kg_directory / filename
        self.graph.serialize(destination=str(filepath), format="turtle")
        logger.info(f"Saved KG to: {filepath}")
        return str(filepath)


_kg_service: Optional[KnowledgeGraphService] = None


def _resolve_kg_directory() -> Optional[str]:

    try:
        from services.triple_store import get_active_backend, InternalRDFLibBackend
        backend = get_active_backend()
        if isinstance(backend, InternalRDFLibBackend):
            return str(backend.ttl_directory)
    except Exception:
        pass
    return None


def get_kg_service(kg_directory: Optional[str] = None) -> KnowledgeGraphService:

    global _kg_service
    
    if _kg_service is None:
        resolved_dir = kg_directory or _resolve_kg_directory()
        _kg_service = KnowledgeGraphService(resolved_dir)
        _kg_service.load_ttl_files()
    
    return _kg_service


def reload_kg_service(kg_directory: Optional[str] = None) -> KnowledgeGraphService:

    global _kg_service
    resolved_dir = kg_directory or _resolve_kg_directory()
    _kg_service = KnowledgeGraphService(resolved_dir)
    _kg_service.load_ttl_files()
    return _kg_service
