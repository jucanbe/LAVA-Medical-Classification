import json
import os
import random
import re
import logging
import tempfile
import uuid
from typing import List, Dict, Optional

from models.llm_config import DEFAULT_CLASSIFY_ALL_PROMPT

logger = logging.getLogger(__name__)

try:
    import langextract as lx
    HAS_LANGEXTRACT = True
except ImportError:
    HAS_LANGEXTRACT = False
    lx = None

_visualizations: Dict[str, str] = {}
_MAX_VISUALIZATIONS = 50

ENTITY_TYPES = [
    "disease", "symptom", "finding", "organ",
    "imaging_procedure", "examination_procedure", "therapeutic_procedure",
    "imaging_result", "examination_measure", "parameter", "score",
    "therapy", "substance", "adverse_event",
]

RELATION_TYPES = [
    "has_symptom", "has_finding", "suggests", "located_in",
    "indicated_for", "produces_result", "treats", "first_line_for",
    "contraindicated_in", "rules_out", "assesses", "affects",
]

_KG_TYPE_NORMALIZE = {
    "quantitativemeasure": "parameter",
    "quantitative_measure": "parameter",
    "imagingprocedure": "imaging_procedure",
    "examinationprocedure": "examination_procedure",
    "therapeuticprocedure": "therapeutic_procedure",
    "imagingresult": "imaging_result",
    "examinationmeasure": "examination_measure",
    "adverseevent": "adverse_event",
    "procedure": "therapeutic_procedure",
}

CLASSIFY_ALL_SYSTEM_PROMPT = DEFAULT_CLASSIFY_ALL_PROMPT


USER_PROMPT_TEMPLATE = """Analyze the following medical text. Extract ALL medical entities and relationships.

{kg_examples}TEXT TO ANALYZE:
\"\"\"
{text}
\"\"\"

Return ONLY valid JSON with "entities" and "relations" arrays."""


class ClassifyAllService:
    """Unified entity + relation extraction using the project's own LLMClient
    and langextract only for visualization."""

    def __init__(self):
        if not HAS_LANGEXTRACT:
            raise ImportError(
                "langextract is required for Classify All visualization. "
                "Install with: pip install langextract"
            )

    def build_kg_examples_text(
        self,
        kg_service,
        num_entities: int = 8,
        custom_samples: Optional[dict] = None,
        use_random_kg: bool = True,
        use_known_samples: bool = True,
    ) -> str:
        all_entities: List[Dict] = []
        all_relations: List[Dict] = []
        text_parts: List[str] = []

        if use_known_samples and custom_samples:
            for e in custom_samples.get("entities", []):
                all_entities.append({
                    "text": e["text"],
                    "entity_type": e["entity_type"],
                    "attributes": e.get("attributes", {}),
                })
                text_parts.append(e["text"])

            for r in custom_samples.get("relations", []):
                all_relations.append({
                    "source_entity": r["source_entity"],
                    "source_type": r.get("source_type", "unknown"),
                    "relation_type": r["relation_type"],
                    "target_entity": r["target_entity"],
                    "target_type": r.get("target_type", "unknown"),
                })

        if use_random_kg:
            sampled = self._sample_kg_entities(kg_service, num_entities)
            for s in sampled:
                if not any(e["text"].lower() == s["label"].lower() for e in all_entities):
                    all_entities.append({
                        "text": s["label"],
                        "entity_type": s["type"],
                        "attributes": {"source": "knowledge_graph"},
                    })
                    text_parts.append(s["label"])

        if not all_entities and not all_relations:
            return ""

        example_json = json.dumps({"entities": all_entities, "relations": all_relations}, indent=2)

        if text_parts:
            example_text = "Clinical findings include: " + ", ".join(text_parts) + "."
            return f'EXAMPLE (for reference):\nText: "{example_text}"\nExpected output:\n{example_json}\n\n'
        else:
            return f'EXAMPLE (for reference):\nExpected output:\n{example_json}\n\n'

    def build_kg_examples(self, kg_service, num_entities: int = 8) -> list:
        sampled = self._sample_kg_entities(kg_service, num_entities)
        if not sampled:
            return []

        text = "Clinical findings include: " + ", ".join(e["label"] for e in sampled) + "."
        extractions = [
            lx.data.Extraction(
                extraction_class=e["type"],
                extraction_text=e["label"],
                attributes={"source": "knowledge_graph"},
            )
            for e in sampled
        ]
        return [lx.data.ExampleData(text=text, extractions=extractions)]

    @staticmethod
    def _sample_kg_entities(kg_service, num_entities: int = 8) -> List[Dict]:
        cache = getattr(kg_service, "_entities_cache", {})
        if not cache:
            return []

        eligible: List[Dict] = []
        for uri, data in cache.items():
            labels = data.get("labels", [])
            types = data.get("types", [])
            if not labels or not types:
                continue
            raw_type = types[0].split("/")[-1].split("#")[-1].lower()
            norm_type = _KG_TYPE_NORMALIZE.get(raw_type, raw_type)
            if norm_type in ENTITY_TYPES:
                eligible.append({"label": labels[0], "type": norm_type})

        if not eligible:
            return []

        random.shuffle(eligible)
        sampled: List[Dict] = []
        types_seen: set = set()
        for ent in eligible:
            if ent["type"] not in types_seen or len(sampled) < num_entities:
                sampled.append(ent)
                types_seen.add(ent["type"])
            if len(sampled) >= num_entities:
                break
        return sampled

    @staticmethod
    def _build_system_prompt(
        classify_all_prompt: Optional[str] = None,
        entity_prompt: Optional[str] = None,
        relation_prompt: Optional[str] = None,
    ) -> str:
        prompt = classify_all_prompt if classify_all_prompt else CLASSIFY_ALL_SYSTEM_PROMPT
        extras = []
        if entity_prompt:
            extras.append(f"\nAdditional entity extraction instructions:\n{entity_prompt[:800]}")
        if relation_prompt:
            extras.append(f"\nAdditional relation extraction instructions:\n{relation_prompt[:800]}")
        if extras:
            prompt += "\n" + "\n".join(extras)
        return prompt

    @staticmethod
    def _parse_llm_response(raw: str) -> dict:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", cleaned)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.warning("Could not parse LLM response as JSON")
                    return {"entities": [], "relations": []}
            else:
                return {"entities": [], "relations": []}

        valid_entities = []
        for ent in data.get("entities", []):
            if isinstance(ent, dict) and ent.get("text") and ent.get("entity_type"):
                valid_entities.append({
                    "text": str(ent["text"]),
                    "entity_type": str(ent["entity_type"]),
                    "attributes": ent.get("attributes", {}),
                })

        valid_relations = []
        for rel in data.get("relations", []):
            if isinstance(rel, dict) and rel.get("source_entity") and rel.get("target_entity") and rel.get("relation_type"):
                valid_relations.append({
                    "source_entity": str(rel["source_entity"]),
                    "source_type": str(rel.get("source_type", "unknown")),
                    "relation_type": str(rel["relation_type"]),
                    "target_entity": str(rel["target_entity"]),
                    "target_type": str(rel.get("target_type", "unknown")),
                    "context": str(rel.get("context", "")),
                })

        return {"entities": valid_entities, "relations": valid_relations}

    @staticmethod
    def _find_char_interval(text_lower: str, needle: str, used_positions: set):
        needle_lower = needle.lower()
        start = 0
        while True:
            pos = text_lower.find(needle_lower, start)
            if pos == -1:
                return None
            end = pos + len(needle)
            if pos not in used_positions:
                used_positions.add(pos)
                return lx.data.CharInterval(start_pos=pos, end_pos=end)
            start = pos + 1

    def _build_visualization(self, text: str, entities: List[dict], relations: List[dict]) -> str:
        text_lower = text.lower()
        used_positions: set = set()
        extractions = []

        for ent in entities:
            interval = self._find_char_interval(text_lower, ent["text"], used_positions)
            extractions.append(
                lx.data.Extraction(
                    extraction_class=ent["entity_type"],
                    extraction_text=ent["text"],
                    attributes=ent.get("attributes", {}),
                    char_interval=interval,
                )
            )

        for rel in relations:
            rel_text = rel.get("context") or rel["source_entity"]
            interval = self._find_char_interval(text_lower, rel_text, used_positions)
            extractions.append(
                lx.data.Extraction(
                    extraction_class=rel["relation_type"],
                    extraction_text=rel_text,
                    attributes={
                        "source_entity": rel["source_entity"],
                        "source_type": rel["source_type"],
                        "target_entity": rel["target_entity"],
                        "target_type": rel["target_type"],
                    },
                    char_interval=interval,
                )
            )

        doc = lx.data.AnnotatedDocument(
            document_id="classify_all",
            extractions=extractions,
            text=text,
        )

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                jsonl_name = "results.jsonl"
                lx.io.save_annotated_documents([doc], output_name=jsonl_name, output_dir=tmpdir)
                jsonl_path = os.path.join(tmpdir, jsonl_name)
                html_content = lx.visualize(jsonl_path)
                if hasattr(html_content, "data"):
                    return html_content.data
                return str(html_content)
        except Exception as e:
            logger.warning(f"Error generating langextract visualization: {e}")
            return (
                "<html><body style='font-family:sans-serif;padding:2rem'>"
                f"<p>Visualization generation error: {e}</p></body></html>"
            )

    async def extract(
        self,
        text: str,
        llm_client,
        classify_all_prompt: Optional[str] = None,
        entity_prompt: Optional[str] = None,
        relation_prompt: Optional[str] = None,
        kg_service=None,
        kg_samples: Optional[dict] = None,
        use_random_kg: bool = True,
        use_known_samples: bool = True,
    ) -> dict:
        system_prompt = self._build_system_prompt(classify_all_prompt, entity_prompt, relation_prompt)

        kg_example_text = ""
        if kg_service:
            kg_example_text = self.build_kg_examples_text(
                kg_service,
                custom_samples=kg_samples,
                use_random_kg=use_random_kg,
                use_known_samples=use_known_samples,
            )

        user_prompt = USER_PROMPT_TEMPLATE.format(
            kg_examples=kg_example_text,
            text=text,
        )

        logger.info(f"Classify All: sending {len(text)} chars to LLM")

        raw_response = await llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=min(4096, max(1024, 4096 - len(text) // 3)),
            frequency_penalty=0.3,
        )

        logger.info(f"Classify All: received {len(raw_response)} chars from LLM")

        parsed = self._parse_llm_response(raw_response)
        entities = parsed["entities"]
        relations = parsed["relations"]

        viz_html = self._build_visualization(text, entities, relations)

        viz_id = str(uuid.uuid4())
        _visualizations[viz_id] = viz_html
        while len(_visualizations) > _MAX_VISUALIZATIONS:
            oldest_key = next(iter(_visualizations))
            del _visualizations[oldest_key]

        return {
            "entities": entities,
            "relations": relations,
            "visualization_id": viz_id,
            "total_extractions": len(entities) + len(relations),
        }

    async def extract_chunked(
        self,
        chunks: list,
        full_text: str,
        llm_client,
        classify_all_prompt: Optional[str] = None,
        entity_prompt: Optional[str] = None,
        relation_prompt: Optional[str] = None,
        kg_service=None,
        kg_samples: Optional[dict] = None,
        use_random_kg: bool = True,
        use_known_samples: bool = True,
    ) -> dict:
        all_entities: List[dict] = []
        all_relations: List[dict] = []
        seen_ent_keys: set = set()
        seen_rel_keys: set = set()

        total_chunks = len(chunks)
        for idx, chunk in enumerate(chunks):
            chunk_text = chunk.text if hasattr(chunk, "text") else str(chunk)
            logger.info(f"Classify All: chunk {idx + 1}/{total_chunks} ({len(chunk_text)} chars)")

            try:
                result = await self.extract(
                    text=chunk_text,
                    llm_client=llm_client,
                    classify_all_prompt=classify_all_prompt,
                    entity_prompt=entity_prompt,
                    relation_prompt=relation_prompt,
                    kg_service=kg_service,
                    kg_samples=kg_samples,
                    use_random_kg=use_random_kg,
                    use_known_samples=use_known_samples,
                )

                for ent in result.get("entities", []):
                    key = (ent["text"].strip().lower(), ent["entity_type"])
                    if key not in seen_ent_keys:
                        seen_ent_keys.add(key)
                        all_entities.append(ent)

                for rel in result.get("relations", []):
                    key = (
                        rel["source_entity"].strip().lower(),
                        rel["relation_type"],
                        rel["target_entity"].strip().lower(),
                    )
                    if key not in seen_rel_keys:
                        seen_rel_keys.add(key)
                        all_relations.append(rel)

            except Exception as exc:
                logger.warning(f"Classify All: chunk {idx + 1} failed: {exc}")
                continue

        viz_html = self._build_visualization(full_text, all_entities, all_relations)
        viz_id = str(uuid.uuid4())
        _visualizations[viz_id] = viz_html
        while len(_visualizations) > _MAX_VISUALIZATIONS:
            oldest_key = next(iter(_visualizations))
            del _visualizations[oldest_key]

        logger.info(
            f"Classify All: document done — {len(all_entities)} entities, "
            f"{len(all_relations)} relations from {total_chunks} chunks"
        )

        return {
            "entities": all_entities,
            "relations": all_relations,
            "visualization_id": viz_id,
            "total_extractions": len(all_entities) + len(all_relations),
        }


def get_visualization_html(viz_id: str) -> Optional[str]:
    return _visualizations.get(viz_id)


_classify_all_service: Optional[ClassifyAllService] = None


def get_classify_all_service() -> ClassifyAllService:
    """Get (or create) the ClassifyAllService singleton."""
    global _classify_all_service
    if _classify_all_service is None:
        _classify_all_service = ClassifyAllService()
    return _classify_all_service
