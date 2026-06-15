from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


DEFAULT_ENTITY_PROMPT = """You are a medical named entity recognition (NER) expert. 
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

DEFAULT_RELATION_PROMPT = """You are a medical relation extraction expert.
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

DEFAULT_CLASSIFY_ALL_PROMPT = """You are a medical NER and relation extraction expert.
Your task is to identify medical entities AND relationships in the given text.

=== Entity types (use these exact strings) ===
- "disease": Pathological conditions (e.g., Pulmonary embolism, diabetes)
- "symptom": Symptoms (e.g., Dyspnea, fever, headache)
- "finding": Clinical or imaging-based findings (e.g., Filling defect, elevated troponin)
- "organ": Anatomical structures (e.g., Pulmonary artery, lung, heart)
- "imaging_procedure": Imaging procedures (e.g., CT angiography, MRI, X-ray)
- "examination_procedure": Clinical exams or lab tests (e.g., blood test)
- "therapeutic_procedure": Therapeutic procedures (e.g., Thrombectomy, surgery)
- "imaging_result": Imaging results (e.g., CT shows embolus)
- "examination_measure": Examination measures (e.g., blood count)
- "parameter": Quantitative parameters (e.g., D-dimer, Heart rate, BP 120/80)
- "score": Clinical scores (e.g., Wells score, APACHE score)
- "therapy": Therapies (e.g., Anticoagulation therapy, chemotherapy)
- "substance": Drugs / substances (e.g., Heparin, aspirin, insulin)
- "adverse_event": Adverse events (e.g., Bleeding, allergic reaction)

=== Relation types (use these exact strings) ===
- "has_symptom": A disease has a symptom
- "has_finding": A disease has a clinical finding
- "suggests": A finding suggests a disease
- "located_in": An entity is located in an anatomical location
- "indicated_for": A procedure is indicated for a condition
- "produces_result": A procedure produces a result
- "treats": A substance/procedure treats a condition
- "first_line_for": A substance is first-line treatment
- "contraindicated_in": Contraindicated in a condition
- "rules_out": A finding/procedure rules out a condition
- "assesses": A tool/score assesses a condition
- "affects": A condition affects an organ/system

Extract entities and relations in order of appearance.
Use exact text from the document for extractions. Do not paraphrase or overlap entities.
Provide meaningful attributes for each entity to add context.

Return ONLY valid JSON with this exact format:
{
  "entities": [
    {"text": "exact text", "entity_type": "disease", "attributes": {"detail": "optional info"}}
  ],
  "relations": [
    {"source_entity": "aspirin", "source_type": "substance", "relation_type": "treats", "target_entity": "headache", "target_type": "symptom", "context": "surrounding text snippet"}
  ]
}"""

DEFAULT_SYSTEM_PROMPT = DEFAULT_ENTITY_PROMPT


class LLMConfigBase(BaseModel):   
    model_config = {"protected_namespaces": ()}
    
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Identifying name for the configuration"
    )
    base_url: str = Field(
        ...,
        description="Base URL of the VLLM server (e.g., http://localhost:8000/v1)"
    )
    model_name: str = Field(
        ...,
        description="Name of the model deployed on VLLM"
    )
    api_key: Optional[str] = Field(
        default="EMPTY",
        description="API key for the VLLM server (default 'EMPTY')"
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for generation"
    )
    max_tokens: int = Field(
        default=2048,
        ge=1,
        le=8192,
        description="Maximum tokens to generate"
    )
    entity_prompt: Optional[str] = Field(
        default=None,
        description="Custom system prompt for entity classification. If empty, uses default prompt."
    )
    relation_prompt: Optional[str] = Field(
        default=None,
        description="Custom system prompt for relation classification. If empty, uses default prompt."
    )
    classify_all_prompt: Optional[str] = Field(
        default=None,
        description="Custom system prompt for Classify All (joint entity+relation extraction). If empty, uses default prompt."
    )
    kg_samples: Optional[str] = Field(
        default=None,
        description="JSON string with custom entity/relation samples for few-shot examples."
    )
    is_default: bool = Field(
        default=False,
        description="Indicates if this is the default configuration"
    )


class LLMConfigCreate(LLMConfigBase):
    pass


class LLMConfigUpdate(BaseModel):    
    model_config = {"protected_namespaces": ()}
    
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    base_url: Optional[str] = None
    model_name: Optional[str] = None
    api_key: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=8192)
    entity_prompt: Optional[str] = None
    relation_prompt: Optional[str] = None
    classify_all_prompt: Optional[str] = None
    kg_samples: Optional[str] = None
    is_default: Optional[bool] = None


class LLMConfigResponse(LLMConfigBase):    
    model_config = {"protected_namespaces": (), "from_attributes": True}
    
    id: int = Field(..., description="Unique ID of the configuration")
    created_at: datetime = Field(..., description="Creation date")
    updated_at: datetime = Field(..., description="Last update date")
