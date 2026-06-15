import httpx
from openai import AsyncOpenAI
from typing import Optional, Dict, Any, Type, TypeVar, Tuple
from pydantic import BaseModel, ValidationError
import json
import logging
import re
import asyncio

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMConnectionError(Exception):
    pass


class LLMGenerationError(Exception):
    pass


class LLMClient:
 
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 1.0
    
    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str = "EMPTY",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):

        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self._supports_json_schema = False
        self._supports_json_object = False
        self._connection_verified = False
        
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=120.0
        )
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:

        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                **kwargs
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating with LLM: {e}")
            raise
    
    async def verify_connection(self) -> Tuple[bool, str, list]:

        try:
            models = await self.client.models.list()
            model_list = [model.id for model in models.data]
            
            if not model_list:
                return False, "Server reachable but no models available", []
            
            if self.model_name not in model_list:
                return False, f"Model '{self.model_name}' not found. Available: {model_list}", model_list
            
            test_response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Respond with exactly: OK"}],
                temperature=0,
                max_tokens=10
            )
            
            if test_response.choices and test_response.choices[0].message.content:
                await self._detect_capabilities()
                self._connection_verified = True
                return True, f"Successfully connected. Model '{self.model_name}' is responding.", model_list
            else:
                return False, "Model responded but returned empty content", model_list
                
        except httpx.ConnectError as e:
            return False, f"Cannot connect to server at {self.base_url}: Connection refused", []
        except httpx.TimeoutException as e:
            return False, f"Connection timeout to {self.base_url}", []
        except Exception as e:
            return False, f"Connection error: {str(e)}", []
    
    async def _detect_capabilities(self):
        try:
            await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                temperature=0,
                max_tokens=5,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "test",
                        "strict": True,
                        "schema": {"type": "object", "properties": {"test": {"type": "string"}}}
                    }
                }
            )
            self._supports_json_schema = True
            logger.info("Server supports json_schema response format")
        except Exception as e:
            logger.debug(f"Server does not support json_schema: {e}")
            self._supports_json_schema = False
        
        if not self._supports_json_schema:
            try:
                await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": "Respond with JSON: {\"test\": \"ok\"}"}],
                    temperature=0,
                    max_tokens=20,
                    response_format={"type": "json_object"}
                )
                self._supports_json_object = True
                logger.info("Server supports json_object response format")
            except Exception as e:
                logger.debug(f"Server does not support json_object: {e}")
                self._supports_json_object = False
    
    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[int] = None
    ) -> T:

        retries = max_retries if max_retries is not None else self.max_retries
        last_error = None
        
        for attempt in range(retries):
            try:
                result = await self._generate_structured_attempt(
                    prompt=prompt,
                    response_model=response_model,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    attempt=attempt
                )
                return result
                
            except (json.JSONDecodeError, ValidationError) as e:
                last_error = e
                logger.warning(f"Structured generation attempt {attempt + 1}/{retries} failed: {e}")
                
                if attempt < retries - 1:
                    await asyncio.sleep(self.retry_delay)
                    continue
                    
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error in structured generation: {e}")
                raise
        
        raise LLMGenerationError(
            f"Failed to generate valid structured response after {retries} attempts. "
            f"Last error: {last_error}"
        )
    
    async def _generate_structured_attempt(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
        attempt: int
    ) -> T:
        """Single attempt at structured generation."""
        
        schema = response_model.model_json_schema()
        schema_str = json.dumps(schema, indent=2)
        
        json_instruction = f"""You MUST respond with a valid JSON object that strictly conforms to this schema:

{schema_str}

CRITICAL RULES:
1. Your response must be ONLY the JSON object - no explanations, no schema repetition, no preamble.
2. Start your response directly with {{ and end with }}
3. All required fields must be present.
4. Use the exact field names from the schema.
5. All enum values must be exact strings as defined in the schema.
6. confidence must be a number between 0.0 and 1.0
7. Do NOT output the schema itself. Output ONLY the data conforming to the schema."""

        full_system_prompt = system_prompt or ""
        if full_system_prompt:
            full_system_prompt += "\n\n"
        full_system_prompt += json_instruction
        
        if attempt > 0:
            full_system_prompt += f"\n\nPREVIOUS ATTEMPT FAILED. Please ensure your response is valid JSON only."
        
        messages = [
            {"role": "system", "content": full_system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        extra_params = {}
        
        if self._supports_json_schema:
            extra_params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": schema
                }
            }
        elif self._supports_json_object:
            extra_params["response_format"] = {"type": "json_object"}
        
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            frequency_penalty=0.3,
            **extra_params
        )
        
        content = response.choices[0].message.content
        
        if not content:
            raise ValueError("LLM returned empty response")
        
        logger.debug(f"Original content (first 500 chars): {content[:500]}")
        content = self._repair_json(content)
        logger.debug(f"Repaired content (first 500 chars): {content[:500]}")
        
        try:
            data = json.loads(content)
            logger.debug(f"Parsed data keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
        except json.JSONDecodeError:
            extracted = self._extract_json(content)
            if extracted:
                extracted = self._repair_json(extracted)
                data = json.loads(extracted)
            else:
                raise
        
        try:
            result = response_model.model_validate(data)
            logger.debug(f"Validation successful for {response_model.__name__}")
            return result
        except Exception as e:
            logger.error(f"Pydantic validation error: {e}")
            logger.error(f"Data that failed validation: {json.dumps(data, indent=2)[:1000]}")
            raise
    
    def _repair_json(self, text: str) -> str:

        import json
        
        text = re.sub(
            r'"entity_type"\s*:\s*\{\s*\}',
            '"entity_type": "finding"',
            text,
            flags=re.MULTILINE | re.DOTALL
        )
        
        text = re.sub(
            r'"entity_type"\s*:\s*\{[\s\n]*\}',
            '"entity_type": "finding"',
            text,
            flags=re.MULTILINE | re.DOTALL
        )
        
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        
        try:
            data = json.loads(text)
            if isinstance(data, dict) and 'entities' in data:
                fixed_entities = []
                for entity in data['entities']:
                    if isinstance(entity, dict):
                        entity_type = entity.get('entity_type')
                        if entity_type is None or entity_type == {} or not isinstance(entity_type, str) or entity_type == '':
                            entity_text = entity.get('text', '').lower()
                            entity['entity_type'] = self._infer_entity_type(entity_text)
                        if entity.get('text'):
                            fixed_entities.append(entity)
                data['entities'] = fixed_entities
            return json.dumps(data)
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode failed during repair, trying regex fixes: {e}")
            pass
        
        text = re.sub(
            r'"entity_type"\s*:\s*\{[^}]*\}',
            '"entity_type": "finding"',
            text,
            flags=re.MULTILINE
        )
        
        text = re.sub(r'}\s*{', '}, {', text)
        
        return text
    
    def _infer_entity_type(self, text: str) -> str:

        text_lower = text.lower()
        
        disease_indicators = ['itis', 'osis', 'emia', 'oma', 'pathy', 'disease', 'syndrome', 
                             'disorder', 'infection', 'cancer', 'tumor', 'endometriosis']
        if any(ind in text_lower for ind in disease_indicators):
            return "disease"
        
        imaging_proc_indicators = ['scan', 'mri', 'ct', 'x-ray', 'xray', 'ultrasound',
                                   'angiography', 'radiograph', 'echocardiography', 'pet']
        if any(ind in text_lower for ind in imaging_proc_indicators):
            return "imaging_procedure"
        
        therapeutic_proc_indicators = ['surgery', 'biopsy', 'laparoscopy', 'thrombectomy',
                                       'transplant', 'resection', 'excision', 'implant']
        if any(ind in text_lower for ind in therapeutic_proc_indicators):
            return "therapeutic_procedure"
        
        exam_proc_indicators = ['test', 'examination', 'screening', 'assessment', 'lab']
        if any(ind in text_lower for ind in exam_proc_indicators):
            return "examination_procedure"
        
        substance_indicators = ['mg', 'ml', 'dose', 'drug', 'medication', 'medicine',
                               'tablet', 'capsule', 'injection', 'heparin', 'insulin']
        if any(ind in text_lower for ind in substance_indicators):
            return "substance"
        
        score_indicators = ['score', 'scale', 'index', 'wells', 'apache', 'glasgow']
        if any(ind in text_lower for ind in score_indicators):
            return "score"
        
        parameter_indicators = ['%', 'level', 'count', 'rate', 'pressure', 'temperature',
                               'mmhg', 'mg/dl', 'd-dimer']
        if any(ind in text_lower for ind in parameter_indicators):
            return "parameter"
        
        organ_indicators = ['artery', 'vein', 'bowel', 'bladder', 'ureter', 'ovarian', 
                           'uterus', 'kidney', 'liver', 'heart', 'lung', 'brain', 'bone']
        if any(ind in text_lower for ind in organ_indicators):
            return "organ"
        
        symptom_indicators = ['pain', 'fever', 'cough', 'nausea', 'dyspnea', 'fatigue',
                             'headache', 'vomiting', 'dizziness']
        if any(ind in text_lower for ind in symptom_indicators):
            return "symptom"
        
        return "finding"

    def _extract_json(self, text: str) -> Optional[str]:
        text = text.strip()
        
        code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if code_block:
            return code_block.group(1).strip()
        
        json_objects = []
        brace_count = 0
        start_idx = None
        
        for i, char in enumerate(text):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx is not None:
                    json_objects.append(text[start_idx:i + 1])
                    start_idx = None
        
        if not json_objects:
            json_match = re.search(r"\{[\s\S]*\}", text)
            if json_match:
                return json_match.group(0)
            return None
        
        data_keys = {"entities", "relations"}
        for json_str in json_objects:
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict) and data_keys.intersection(parsed.keys()):
                    return json_str
            except json.JSONDecodeError:
                continue
        
        return json_objects[-1] if json_objects else None
    
    async def health_check(self) -> bool:

        try:
            models = await self.client.models.list()
            return len(models.data) > 0
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
    
    async def get_available_models(self) -> list:

        try:
            models = await self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            logger.error(f"Error getting models: {e}")
            return []
    
    @classmethod
    def from_config(cls, config) -> "LLMClient":

        return cls(
            base_url=config.base_url,
            model_name=config.model_name,
            api_key=config.api_key or "EMPTY",
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
    
    @classmethod
    async def from_config_verified(cls, config) -> "LLMClient":

        client = cls.from_config(config)
        success, message, models = await client.verify_connection()
        
        if not success:
            raise LLMConnectionError(message)
        
        return client
