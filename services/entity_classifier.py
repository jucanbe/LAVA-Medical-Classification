import time
import logging
import re
from typing import Optional, List, Tuple
from pydantic import BaseModel, Field

from .llm_client import LLMClient, LLMGenerationError
from models.entities import (
    MedicalEntity,
    MedicalEntityType,
    EntityClassificationResult,
    KGValidationResult,
    ValidatedEntity
)

logger = logging.getLogger(__name__)

TOKEN_LIMIT_PATTERNS = [
    r"Expecting ',' delimiter",
    r"Expecting ':' delimiter",
    r"Unterminated string",
    r"Expecting value",
    r"Expecting property name",
    r"Invalid \\escape",
    r"char \d{4,}",
]


class ExtractedEntities(BaseModel):
    """Model for the LLM response with extracted entities."""
    
    entities: List[MedicalEntity] = Field(
        default_factory=list,
        description="List of medical entities found in the text"
    )


class MedicalEntityClassifier:    
    SYSTEM_PROMPT = """You are a medical named entity recognition (NER) expert. 
Your task is to identify and classify medical entities in the given text.

You MUST identify ONLY these 14 types of medical entities (use these exact strings):
- "disease": Pathological conditions (e.g., Pulmonary embolism, diabetes, hypertension, pneumonia)
- "symptom": Symptoms (e.g., Dyspnea, fever, coughing, back pain, headache, nausea)
- "finding": Clinical or imaging-based findings (e.g., Filling defect, elevated troponin)
- "organ": Anatomical structures (e.g., Pulmonary artery, lung, liver, heart)
- "imaging_procedure": Imaging procedures (e.g., CT pulmonary angiography, MRI, X-ray, ultrasound)
- "examination_procedure": Clinical exams or lab tests (e.g., blood test, physical examination)
- "therapeutic_procedure": Therapeutic procedures (e.g., Thrombectomy, surgery, biopsy)
- "imaging_result": Imaging results (e.g., CT shows embolus, MRI reveals lesion)
- "examination_measure": Examination measures (e.g., blood count, urinalysis result)
- "parameter": Quantitative parameters (e.g., D-dimer level, Heart rate, blood pressure 120/80)
- "score": Clinical scores (e.g., Wells score, APACHE score, Glasgow coma scale)
- "therapy": Therapies (e.g., Anticoagulation therapy, chemotherapy, radiotherapy)
- "substance": Drugs or substances (e.g., Heparin, Contrast agent, insulin, aspirin)
- "adverse_event": Adverse events (e.g., Bleeding, allergic reaction, drug interaction)

CRITICAL: entity_type must be a STRING like "disease", NOT an object like {}.

For each entity found, provide:
1. text: The exact text as it appears in the document
2. entity_type: One of the 14 strings above
3. confidence: A score from 0.0 to 1.0
4. normalized_form: The standard form if applicable, or null

Be thorough but precise. Only identify genuine medical entities."""

    USER_PROMPT_TEMPLATE = """Analyze the following medical text and extract all medical entities.

TEXT TO ANALYZE:
\"\"\"
{text}
\"\"\"

Extract all medical entities. For each entity:
- text: exact text from the document
- entity_type: MUST be one of these strings: "disease", "symptom", "finding", "organ", "imaging_procedure", "examination_procedure", "therapeutic_procedure", "imaging_result", "examination_measure", "parameter", "score", "therapy", "substance", "adverse_event"
- confidence: number between 0.0 and 1.0
- normalized_form: standard form or null

Return ONLY valid JSON. Example format:
{{"entities": [{{"text": "Pulmonary embolism", "entity_type": "disease", "confidence": 0.95, "normalized_form": null}}]}}"""

    MIN_CHUNK_SIZE = 300
    
    MAX_SPLIT_DEPTH = 3

    def __init__(self, llm_client: LLMClient, custom_system_prompt: str = None):
        self.llm_client = llm_client
        self.custom_system_prompt = custom_system_prompt
    
    def get_system_prompt(self) -> str:
        if self.custom_system_prompt:
            return self.custom_system_prompt
        return self.SYSTEM_PROMPT
    
    def _is_token_limit_error(self, error: Exception) -> bool:
        error_str = str(error)
        for pattern in TOKEN_LIMIT_PATTERNS:
            if re.search(pattern, error_str, re.IGNORECASE):
                char_match = re.search(r'char (\d+)', error_str)
                if char_match:
                    char_pos = int(char_match.group(1))
                    if char_pos > 2000:
                        logger.info(f"Detected likely token limit at char {char_pos}")
                        return True
                if 'Expecting' in error_str:
                    return True
        return False
    
    def _split_text(self, text: str) -> List[str]:
        mid = len(text) // 2
        
        best_split = mid
        search_range = min(500, len(text) // 4)
        
        for i in range(search_range):
            for pos in [mid - i, mid + i]:
                if 0 < pos < len(text) - 1:
                    if text[pos] in '.!?' and text[pos + 1] in ' \n\t':
                        best_split = pos + 1
                        break
            else:
                continue
            break
        
        if best_split == mid:
            for i in range(search_range):
                for pos in [mid - i, mid + i]:
                    if 0 < pos < len(text) - 1:
                        if text[pos] == '\n':
                            best_split = pos + 1
                            break
                else:
                    continue
                break
        
        chunk1 = text[:best_split].strip()
        chunk2 = text[best_split:].strip()
        
        return [chunk1, chunk2] if chunk1 and chunk2 else [text]
    
    async def classify_with_auto_split(
        self,
        text: str,
        include_context: bool = True,
        min_confidence: float = 0.5,
        depth: int = 0
    ) -> EntityClassificationResult:
        start_time = time.time()
        
        try:
            result = await self.classify(
                text=text,
                include_context=include_context,
                min_confidence=min_confidence
            )
            return result
            
        except LLMGenerationError as e:
            if self._is_token_limit_error(e) and depth < self.MAX_SPLIT_DEPTH:
                if len(text) > self.MIN_CHUNK_SIZE * 2:
                    logger.info(f"Token limit detected, splitting text (depth={depth}, len={len(text)})")
                    
                    sub_chunks = self._split_text(text)
                    
                    if len(sub_chunks) == 2:
                        logger.info(f"Split into chunks of {len(sub_chunks[0])} and {len(sub_chunks[1])} chars")
                        
                        all_entities = []
                        seen_entities = set()
                        
                        for i, sub_chunk in enumerate(sub_chunks):
                            try:
                                sub_result = await self.classify_with_auto_split(
                                    text=sub_chunk,
                                    include_context=include_context,
                                    min_confidence=min_confidence,
                                    depth=depth + 1
                                )
                                
                                for entity in sub_result.entities:
                                    entity_key = (entity.text.lower(), entity.entity_type.value)
                                    if entity_key not in seen_entities:
                                        seen_entities.add(entity_key)
                                        all_entities.append(entity)
                                        
                            except Exception as sub_e:
                                logger.warning(f"Sub-chunk {i} failed: {sub_e}")
                                continue
                        
                        processing_time = (time.time() - start_time) * 1000
                        
                        return EntityClassificationResult(
                            entities=all_entities,
                            total_entities=len(all_entities),
                            processing_time_ms=processing_time
                        )
            
            raise
    
    async def classify(
        self,
        text: str,
        include_context: bool = True,
        min_confidence: float = 0.5
    ) -> EntityClassificationResult:
        start_time = time.time()
        
        try:
            user_prompt = self.USER_PROMPT_TEMPLATE.format(text=text)
            
            extracted = await self.llm_client.generate_structured(
                prompt=user_prompt,
                response_model=ExtractedEntities,
                system_prompt=self.get_system_prompt(),
                temperature=0.1
            )
            
            filtered_entities = [
                entity for entity in extracted.entities
                if entity.confidence >= min_confidence
            ]
            
            if include_context:
                filtered_entities = self._enrich_entities(text, filtered_entities)
            
            processing_time = (time.time() - start_time) * 1000
            
            return EntityClassificationResult(
                entities=filtered_entities,
                total_entities=len(filtered_entities),
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error classifying entities: {e}")
            raise
    
    def _enrich_entities(
        self,
        text: str,
        entities: List[MedicalEntity]
    ) -> List[MedicalEntity]:
        enriched = []
        text_lower = text.lower()
        
        for entity in entities:
            entity_text_lower = entity.text.lower()
            
            start_pos = text_lower.find(entity_text_lower)
            
            if start_pos != -1:
                end_pos = start_pos + len(entity.text)
                
                context_start = max(0, start_pos - 50)
                context_end = min(len(text), end_pos + 50)
                context = text[context_start:context_end]
                
                if context_start > 0:
                    context = "..." + context
                if context_end < len(text):
                    context = context + "..."
                
                enriched_entity = MedicalEntity(
                    text=entity.text,
                    entity_type=entity.entity_type,
                    confidence=entity.confidence,
                    start_position=start_pos,
                    end_position=end_pos,
                    context=context,
                    normalized_form=entity.normalized_form
                )
                enriched.append(enriched_entity)
            else:
                enriched.append(entity)
        
        return enriched
    
    async def classify_batch(
        self,
        texts: List[str],
        include_context: bool = True,
        min_confidence: float = 0.5
    ) -> List[EntityClassificationResult]:

        results = []
        for text in texts:
            result = await self.classify(text, include_context, min_confidence)
            results.append(result)
        return results
    
    async def classify_and_validate(
        self,
        text: str,
        include_context: bool = True,
        min_confidence: float = 0.5,
        kg_directory: Optional[str] = None
    ) -> tuple[EntityClassificationResult, KGValidationResult]:
        classification_result = await self.classify(
            text=text,
            include_context=include_context,
            min_confidence=min_confidence
        )
        
        from .knowledge_graph import get_kg_service
        kg_service = get_kg_service(kg_directory)
        
        validation_result = await kg_service.validate_entities_async(classification_result)
        
        return classification_result, validation_result
    
    def get_supported_entity_types(self) -> List[str]:
        return [e.value for e in MedicalEntityType]
