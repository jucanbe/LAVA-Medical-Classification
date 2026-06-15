from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from database import get_db, db_manager
from models import LLMConfigCreate, LLMConfigResponse, LLMConfigUpdate
from models.llm_config import DEFAULT_ENTITY_PROMPT, DEFAULT_RELATION_PROMPT, DEFAULT_CLASSIFY_ALL_PROMPT
from services import LLMClient
from services.llm_client import LLMConnectionError

router = APIRouter(prefix="/llm-config", tags=["LLM Configuration"])


@router.post(
    "/",
    response_model=LLMConfigResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create LLM configuration",
    description="Create a new connection configuration for the VLLM/LM Studio server. Optionally verifies the connection before saving."
)
async def create_llm_config(
    config: LLMConfigCreate,
    db: AsyncSession = Depends(get_db),
    verify_connection: bool = Query(
        default=True,
        description="Verify connection to the LLM server before saving the configuration"
    )
):
    """Create a new LLM configuration and save it to the database."""
    
    existing = await db_manager.get_llm_config_by_name(db, config.name)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A configuration with the name '{config.name}' already exists"
        )
    
    if verify_connection:
        temp_client = LLMClient(
            base_url=config.base_url,
            model_name=config.model_name,
            api_key=config.api_key or "EMPTY",
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        success, message, models = await temp_client.verify_connection()
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": "Failed to verify LLM connection",
                    "message": message,
                    "available_models": models
                }
            )
    
    db_config = await db_manager.create_llm_config(
        session=db,
        name=config.name,
        base_url=config.base_url,
        model_name=config.model_name,
        api_key=config.api_key,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        is_default=config.is_default
    )
    
    return LLMConfigResponse.model_validate(db_config)


@router.get(
    "/",
    response_model=List[LLMConfigResponse],
    summary="List LLM configurations",
    description="Get all saved LLM configurations"
)
async def list_llm_configs(db: AsyncSession = Depends(get_db)):
    """List all LLM configurations."""
    configs = await db_manager.get_all_llm_configs(db)
    return [LLMConfigResponse.model_validate(c) for c in configs]


@router.get(
    "/default",
    response_model=LLMConfigResponse,
    summary="Get default configuration",
    description="Get the LLM configuration marked as default"
)
async def get_default_config(db: AsyncSession = Depends(get_db)):
    """Get the default LLM configuration."""
    config = await db_manager.get_default_llm_config(db)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No default configuration has been set"
        )
    return LLMConfigResponse.model_validate(config)


@router.get(
    "/default-prompt",
    summary="Get default system prompts",
    description="Get the default system prompts for entity and relation classification"
)
async def get_default_prompt():
    """Get the default system prompts used for classification."""
    return {
        "entity_prompt": DEFAULT_ENTITY_PROMPT,
        "relation_prompt": DEFAULT_RELATION_PROMPT,
        "classify_all_prompt": DEFAULT_CLASSIFY_ALL_PROMPT,
        "description": "Default system prompts for medical entity classification, relation classification, and Classify All"
    }


@router.get(
    "/{config_id}",
    response_model=LLMConfigResponse,
    summary="Get configuration by ID",
    description="Get a specific LLM configuration by its ID"
)
async def get_llm_config(
    config_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get an LLM configuration by its ID."""
    config = await db_manager.get_llm_config_by_id(db, config_id)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration with ID {config_id} not found"
        )
    return LLMConfigResponse.model_validate(config)


@router.put(
    "/{config_id}",
    response_model=LLMConfigResponse,
    summary="Update LLM configuration",
    description="Update an existing LLM configuration. Optionally verifies the connection before saving."
)
async def update_llm_config(
    config_id: int,
    config_update: LLMConfigUpdate,
    db: AsyncSession = Depends(get_db),
    verify_connection: bool = Query(
        default=True,
        description="Verify connection to the LLM server before saving the configuration"
    )
):
    """Update an LLM configuration."""
    
    existing = await db_manager.get_llm_config_by_id(db, config_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration with ID {config_id} not found"
        )
    
    if config_update.name and config_update.name != existing.name:
        name_exists = await db_manager.get_llm_config_by_name(db, config_update.name)
        if name_exists:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"A configuration with the name '{config_update.name}' already exists"
            )
    
    if verify_connection:
        test_config = {
            "base_url": config_update.base_url or existing.base_url,
            "model_name": config_update.model_name or existing.model_name,
            "api_key": config_update.api_key if config_update.api_key is not None else existing.api_key,
            "temperature": config_update.temperature if config_update.temperature is not None else existing.temperature,
            "max_tokens": config_update.max_tokens if config_update.max_tokens is not None else existing.max_tokens
        }
        
        temp_client = LLMClient(
            base_url=test_config["base_url"],
            model_name=test_config["model_name"],
            api_key=test_config["api_key"] or "EMPTY",
            temperature=test_config["temperature"],
            max_tokens=test_config["max_tokens"]
        )
        success, message, models = await temp_client.verify_connection()
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": "Failed to verify LLM connection",
                    "message": message,
                    "available_models": models
                }
            )
    
    updated = await db_manager.update_llm_config(
        session=db,
        config_id=config_id,
        **config_update.model_dump(exclude_unset=True)
    )
    
    return LLMConfigResponse.model_validate(updated)


@router.delete(
    "/{config_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete LLM configuration",
    description="Delete an LLM configuration"
)
async def delete_llm_config(
    config_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Delete an LLM configuration."""
    deleted = await db_manager.delete_llm_config(db, config_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration with ID {config_id} not found"
        )


@router.post(
    "/{config_id}/test",
    summary="Test LLM connection",
    description="Test the connection to the VLLM/LM Studio server using a configuration. Also detects server capabilities."
)
async def test_llm_connection(
    config_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Test the connection to the VLLM/LM Studio server."""
    config = await db_manager.get_llm_config_by_id(db, config_id)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration with ID {config_id} not found"
        )
    
    client = LLMClient.from_config(config)
    success, message, models = await client.verify_connection()
    
    if success:
        return {
            "status": "connected",
            "message": message,
            "available_models": models,
            "capabilities": {
                "supports_json_schema": client._supports_json_schema,
                "supports_json_object": client._supports_json_object
            }
        }
    else:
        return {
            "status": "disconnected",
            "message": message,
            "available_models": models,
            "capabilities": None
        }


@router.post(
    "/{config_id}/set-default",
    response_model=LLMConfigResponse,
    summary="Set configuration as default",
    description="Set an LLM configuration as the default one"
)
async def set_default_config(
    config_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Set an LLM configuration as the default."""
    config = await db_manager.get_llm_config_by_id(db, config_id)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration with ID {config_id} not found"
        )
    
    updated = await db_manager.update_llm_config(
        session=db,
        config_id=config_id,
        is_default=True
    )
    
    return LLMConfigResponse.model_validate(updated)
