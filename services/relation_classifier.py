import time
import logging
from typing import Optional, List
from pydantic import BaseModel, Field

from .llm_client import LLMClient
from models.relations import (
    MedicalRelation,
    MedicalRelationType,
    RelationClassificationResult,
)

logger = logging.getLogger(__name__)


class ExtractedRelations(BaseModel):
    
    relations: List[MedicalRelation] = Field(
        default_factory=list,
        description="List of medical relations found in the text"
    )


class MedicalRelationClassifier:
    
    SYSTEM_PROMPT = """You are a medical relation extraction expert.
Your task is to identify and classify medical RELATIONS between entities in the given text.

You MUST identify ONLY these 12 types of medical relations (use these exact strings):
- "has_symptom": A disease/condition has a symptom (e.g., Pulmonary embolism → has_symptom → Dyspnea)
- "has_finding": A disease/condition has a clinical finding (e.g., Pulmonary embolism → has_finding → Filling defect)
- "suggests": A finding suggests a disease/condition (e.g., Filling defect → suggests → Pulmonary embolism)
- "located_in": An entity is located in an anatomical location (e.g., Filling defect → located_in → Pulmonary artery)
- "indicated_for": A procedure is indicated for a condition (e.g., CT angiography → indicated_for → Suspected PE)
- "produces_result": A procedure produces a result (e.g., CT angiography → produces_result → CT shows embolus)
- "treats": A substance/procedure treats a condition (e.g., Anticoagulation → treats → Pulmonary embolism)
- "first_line_for": A substance is first-line treatment (e.g., Heparin → first_line_for → Pulmonary embolism)
- "contraindicated_in": Contraindicated in a condition (e.g., Thrombolysis → contraindicated_in → Active bleeding)
- "rules_out": A finding/procedure rules out a condition (e.g., Normal CT → rules_out → Pulmonary embolism)
- "assesses": A tool/score assesses a condition (e.g., Wells score → assesses → Pulmonary embolism)
- "affects": A condition affects an organ/system (e.g., Pulmonary embolism → affects → Lung)

CRITICAL: relation_type must be a STRING like "treats", NOT an object.

For each relation found, provide:
1. source_entity: The subject entity text as it appears in the text
2. source_type: Type of source entity (disease, symptom, finding, organ, imaging_procedure, examination_procedure, therapeutic_procedure, imaging_result, examination_measure, parameter, score, therapy, substance, adverse_event)
3. relation_type: One of the 12 strings above
4. target_entity: The object entity text as it appears in the text
5. target_type: Type of target entity
6. confidence: A score from 0.0 to 1.0

Be thorough but precise. Only identify genuine medical relations present in the text."""

    USER_PROMPT_TEMPLATE = """Analyze the following medical text and extract all medical RELATIONS between entities.

TEXT TO ANALYZE:
\"\"\"
{text}
\"\"\"

Extract all medical relations. For each relation:
- source_entity: exact subject entity text from the document
- source_type: type of source entity (e.g., "disease", "symptom", "finding", "organ", "imaging_procedure", "substance", "therapy")
- relation_type: MUST be one of: "has_symptom", "has_finding", "suggests", "located_in", "indicated_for", "produces_result", "treats", "first_line_for", "contraindicated_in", "rules_out", "assesses", "affects"
- target_entity: exact object entity text from the document
- target_type: type of target entity
- confidence: number between 0.0 and 1.0

Return ONLY valid JSON. Example format:
{{"relations": [{{"source_entity": "Pulmonary embolism", "source_type": "disease", "relation_type": "has_symptom", "target_entity": "Dyspnea", "target_type": "symptom", "confidence": 0.9}}]}}"""

    def __init__(self, llm_client: LLMClient, custom_system_prompt: str = None):
        """Initialize the medical relation classifier."""
        self.llm_client = llm_client
        self.custom_system_prompt = custom_system_prompt
    
    def get_system_prompt(self) -> str:
        """Get the system prompt to use for classification."""
        if self.custom_system_prompt:
            return self.custom_system_prompt
        return self.SYSTEM_PROMPT
    
    async def classify(
        self,
        text: str,
        min_confidence: float = 0.5
    ) -> RelationClassificationResult:

        start_time = time.time()
        
        try:
            user_prompt = self.USER_PROMPT_TEMPLATE.format(text=text)
            
            extracted = await self.llm_client.generate_structured(
                prompt=user_prompt,
                response_model=ExtractedRelations,
                system_prompt=self.get_system_prompt(),
                temperature=0.1
            )
            
            filtered_relations = [
                rel for rel in extracted.relations
                if rel.confidence >= min_confidence
            ]
            
            filtered_relations = self._enrich_relations(text, filtered_relations)
            
            processing_time = (time.time() - start_time) * 1000
            
            return RelationClassificationResult(
                relations=filtered_relations,
                total_relations=len(filtered_relations),
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error classifying relations: {e}")
            raise
    
    def _enrich_relations(
        self,
        text: str,
        relations: List[MedicalRelation]
    ) -> List[MedicalRelation]:

        enriched = []
        text_lower = text.lower()
        
        for rel in relations:
            source_pos = text_lower.find(rel.source_entity.lower())
            target_pos = text_lower.find(rel.target_entity.lower())
            
            if source_pos != -1 and target_pos != -1:
                start = min(source_pos, target_pos)
                end = max(
                    source_pos + len(rel.source_entity),
                    target_pos + len(rel.target_entity)
                )
                context_start = max(0, start - 30)
                context_end = min(len(text), end + 30)
                context = text[context_start:context_end]
                
                if context_start > 0:
                    context = "..." + context
                if context_end < len(text):
                    context = context + "..."
                
                enriched.append(MedicalRelation(
                    source_entity=rel.source_entity,
                    source_type=rel.source_type,
                    relation_type=rel.relation_type,
                    target_entity=rel.target_entity,
                    target_type=rel.target_type,
                    confidence=rel.confidence,
                    context=context
                ))
            else:
                enriched.append(rel)
        
        return enriched
    
    @staticmethod
    def get_supported_relation_types() -> List[dict]:

        from models.relations import RELATION_DESCRIPTIONS
        return [
            {"type": rt.value, "description": RELATION_DESCRIPTIONS.get(rt.value, "")}
            for rt in MedicalRelationType
        ]
