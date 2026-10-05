"""
SQLite database connection management.
"""
import logging
from pathlib import Path
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select, update, inspect, text
from typing import Optional, List, AsyncGenerator
from datetime import datetime

from .models import Base, LLMConfigDB
from config import settings

logger = logging.getLogger(__name__)


# Columns added after the first release. create_all() does not alter existing
# tables, so they are added here with ALTER TABLE ... ADD COLUMN. The migration
# only adds nullable columns: existing rows and data are left untouched.
_ADDED_COLUMNS = [
    ("entity_reviews", "completeness_has_confidence", "BOOLEAN"),
    ("entity_reviews", "consistency_bert_status", "VARCHAR(300)"),
    ("entity_reviews", "scoring_version", "VARCHAR(20)"),
    ("entity_reviews", "score_history", "TEXT"),
    ("relation_reviews", "scoring_version", "VARCHAR(20)"),
    ("relation_reviews", "score_history", "TEXT"),
]


def _add_missing_columns(sync_conn) -> None:
    inspector = inspect(sync_conn)
    for table, column, ddl_type in _ADDED_COLUMNS:
        existing = {c["name"] for c in inspector.get_columns(table)}
        if column not in existing:
            sync_conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}"))
            logger.info(f"Migrated database: added column {table}.{column}")


class DatabaseManager:
    """
    Class to manage database connection and operations.
    Implements the Singleton pattern for the connection.
    """
    
    _instance: Optional["DatabaseManager"] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.engine = create_async_engine(
                settings.DATABASE_URL,
                echo=False,
                future=True,
                connect_args={"timeout": 30},
            )
            self.async_session = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            DatabaseManager._initialized = True
    
    async def init_db(self):
        """Initialize the database by creating all tables.

        For SQLite backends the parent directory of the DB file is
        created automatically so that first-run inside Docker (or any
        fresh environment) works without manual setup.
        """
        db_url = str(self.engine.url)
        if db_url.startswith("sqlite"):
            # Extract the file path from the URL (skip 'sqlite+aiosqlite:///')
            parts = db_url.split("///", 1)
            if len(parts) == 2 and parts[1]:
                db_path = Path(parts[1])
                db_path.parent.mkdir(parents=True, exist_ok=True)

        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await conn.run_sync(_add_missing_columns)
    
    async def close(self):
        """Close the database connection."""
        await self.engine.dispose()
    
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session."""
        async with self.async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    
    async def create_llm_config(
        self,
        session: AsyncSession,
        name: str,
        base_url: str,
        model_name: str,
        api_key: str = "EMPTY",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        is_default: bool = False
    ) -> LLMConfigDB:
        """Create a new LLM configuration."""
        
        if is_default:
            await self._clear_default_config(session)
        
        config = LLMConfigDB(
            name=name,
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            is_default=is_default
        )
        session.add(config)
        await session.flush()
        await session.refresh(config)
        return config
    
    async def get_llm_config_by_id(
        self,
        session: AsyncSession,
        config_id: int
    ) -> Optional[LLMConfigDB]:
        """Get an LLM configuration by its ID."""
        result = await session.execute(
            select(LLMConfigDB).where(LLMConfigDB.id == config_id)
        )
        return result.scalar_one_or_none()
    
    async def get_llm_config_by_name(
        self,
        session: AsyncSession,
        name: str
    ) -> Optional[LLMConfigDB]:
        """Get an LLM configuration by its name."""
        result = await session.execute(
            select(LLMConfigDB).where(LLMConfigDB.name == name)
        )
        return result.scalar_one_or_none()
    
    async def get_default_llm_config(
        self,
        session: AsyncSession
    ) -> Optional[LLMConfigDB]:
        """Get the default LLM configuration."""
        result = await session.execute(
            select(LLMConfigDB).where(LLMConfigDB.is_default == True)
        )
        return result.scalar_one_or_none()
    
    async def get_all_llm_configs(
        self,
        session: AsyncSession
    ) -> List[LLMConfigDB]:
        """Get all LLM configurations."""
        result = await session.execute(
            select(LLMConfigDB).order_by(LLMConfigDB.created_at.desc())
        )
        return list(result.scalars().all())
    
    async def update_llm_config(
        self,
        session: AsyncSession,
        config_id: int,
        **kwargs
    ) -> Optional[LLMConfigDB]:
        """Update an LLM configuration."""
        
        if kwargs.get("is_default"):
            await self._clear_default_config(session)
        
        update_data = {k: v for k, v in kwargs.items() if v is not None}
        
        if update_data:
            update_data["updated_at"] = datetime.utcnow()
            await session.execute(
                update(LLMConfigDB)
                .where(LLMConfigDB.id == config_id)
                .values(**update_data)
            )
        
        return await self.get_llm_config_by_id(session, config_id)
    
    async def delete_llm_config(
        self,
        session: AsyncSession,
        config_id: int
    ) -> bool:
        """Delete an LLM configuration."""
        config = await self.get_llm_config_by_id(session, config_id)
        if config:
            await session.delete(config)
            return True
        return False
    
    async def _clear_default_config(self, session: AsyncSession):
        """Remove the default flag from all configurations."""
        await session.execute(
            update(LLMConfigDB)
            .where(LLMConfigDB.is_default == True)
            .values(is_default=False)
        )


db_manager = DatabaseManager()

async_session_maker = db_manager.async_session


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get a database session in FastAPI."""
    async for session in db_manager.get_session():
        yield session
