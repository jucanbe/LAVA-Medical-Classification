from pydantic import BaseModel
from typing import Optional
import os


class BaseSettings(BaseModel):
    """Base class for configuration."""
    
    class Config:
        extra = "allow"


class Settings(BaseSettings):
    """General application configuration."""
    
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./entity_classifier.db")
    
    DEFAULT_VLLM_BASE_URL: str = os.getenv("DEFAULT_VLLM_BASE_URL", "http://localhost:8000/v1")
    DEFAULT_MODEL_NAME: str = os.getenv("DEFAULT_MODEL_NAME", "meta-llama/Llama-2-7b-chat-hf")
    
    API_TITLE: str = os.getenv("API_TITLE", "Medical Classifier API")
    API_VERSION: str = os.getenv("API_VERSION", "1.0.0")


settings = Settings()
