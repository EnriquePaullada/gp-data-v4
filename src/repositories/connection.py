"""
MongoDB Connection Management
Singleton Motor client with connection pooling and lifecycle management.
"""
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from typing import Optional
from ..config import settings
from ..utils.observability import logger


class DatabaseManager:
    """
    Singleton MongoDB client manager with async Motor.
    Handles connection lifecycle, pooling, and graceful shutdown.
    """

    _instance: Optional["DatabaseManager"] = None
    _client: Optional[AsyncIOMotorClient] = None
    _database: Optional[AsyncIOMotorDatabase] = None

    def __new__(cls) -> "DatabaseManager":
        """Enforce singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def connect(self) -> None:
        """
        Initialize MongoDB connection with configured pool settings.
        Idempotent - safe to call multiple times.
        """
        if self._client:
            try:
                await self._client.admin.command("ping")
                logger.debug("Reusing healthy MongoDB connection")
                return 
            except (RuntimeError, Exception):
                logger.warning("Event loop closed or connection lost. Rebuilding client...")
                self._client = None
                self._database = None

        logger.info(
            f"Connecting to MongoDB at {settings.mongodb_uri}",
            extra={
                "database": settings.mongodb_database,
                "max_pool_size": settings.mongodb_max_pool_size,
                "environment": settings.environment
            }
        )
        self._client = AsyncIOMotorClient(
            settings.mongodb_uri,
            maxPoolSize=settings.mongodb_max_pool_size,
            minPoolSize=settings.mongodb_min_pool_size,
            serverSelectionTimeoutMS=settings.mongodb_server_selection_timeout_ms,
        )

        self._database = self._client[settings.mongodb_database]

    async def disconnect(self) -> None:
        """
        Close MongoDB connection and cleanup resources.
        Idempotent - safe to call multiple times.
        """
        if self._client is None:
            logger.debug("MongoDB client already disconnected")
            return

        logger.info("Closing MongoDB connection")
        self._client.close()
        self._client = None
        self._database = None
        logger.info("MongoDB connection closed")

    @property
    def database(self) -> AsyncIOMotorDatabase:
        """
        Get the database instance.
        Raises RuntimeError if not connected.
        """
        if self._database is None:
            raise RuntimeError(
                "Database not connected. Call await db_manager.connect() first."
            )
        return self._database

    @property
    def client(self) -> AsyncIOMotorClient:
        """
        Get the Motor client instance.
        Raises RuntimeError if not connected.
        """
        if self._client is None:
            raise RuntimeError(
                "Database client not connected. Call await db_manager.connect() first."
            )
        return self._client

    async def create_indexes(self) -> None:
        """
        Create all required indexes for optimal query performance.
        Should be called during application startup.
        """
        db = self.database

        logger.info("Creating MongoDB indexes")

        # Leads collection indexes
        await db.leads.create_index("lead_id", unique=True, name="idx_lead_id_unique")
        await db.leads.create_index(
            [("current_stage", 1), ("last_interaction_at", -1)],
            name="idx_stage_last_interaction"
        )
        await db.leads.create_index(
            "next_followup_at",
            name="idx_next_followup",
            sparse=True  # Only index documents with this field
        )

        # Messages collection indexes
        await db.messages.create_index(
            [("lead_id", 1), ("timestamp", -1)],
            name="idx_lead_messages"
        )
        await db.messages.create_index(
            "timestamp",
            name="idx_timestamp"
        )

        # Optional: TTL index for message archival (if enabled)
        if settings.enable_message_archival and settings.message_retention_days > 0:
            await db.messages.create_index(
                "created_at",
                name="idx_message_ttl",
                expireAfterSeconds=settings.message_retention_days * 86400  # days to seconds
            )

        logger.info("MongoDB indexes created successfully")


# Singleton instance
db_manager = DatabaseManager()


async def get_database() -> AsyncIOMotorDatabase:
    """
    Dependency injection helper for repositories.
    Returns the connected database instance.
    """
    return db_manager.database
