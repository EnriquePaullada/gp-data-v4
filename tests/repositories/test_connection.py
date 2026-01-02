"""
Database Connection Tests
Tests for MongoDB client lifecycle and connection management.
"""
import pytest
from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorClient

from src.repositories.connection import DatabaseManager, db_manager, get_database
from src.config import settings


pytestmark = pytest.mark.asyncio


class TestDatabaseManager:
    """Test suite for DatabaseManager singleton."""

    async def test_singleton_pattern(self):
        """DatabaseManager should return same instance."""
        manager1 = DatabaseManager()
        manager2 = DatabaseManager()

        assert manager1 is manager2

    async def test_connect_initializes_client(self):
        """Connect should initialize Motor client and database."""
        manager = DatabaseManager()

        # Clean state
        await manager.disconnect()

        # Connect
        await manager.connect()

        assert manager._client is not None
        assert manager._database is not None
        assert isinstance(manager.client, AsyncIOMotorClient)
        assert isinstance(manager.database, AsyncIOMotorDatabase)

        # Cleanup
        await manager.disconnect()

    async def test_connect_is_idempotent(self):
        """Multiple connect calls should not create new clients."""
        manager = DatabaseManager()

        await manager.connect()
        client1 = manager._client

        await manager.connect()
        client2 = manager._client

        assert client1 is client2

        # Cleanup
        await manager.disconnect()

    async def test_disconnect_cleans_up(self):
        """Disconnect should close client and clear references."""
        manager = DatabaseManager()

        await manager.connect()
        await manager.disconnect()

        assert manager._client is None
        assert manager._database is None

    async def test_disconnect_is_idempotent(self):
        """Multiple disconnect calls should not raise errors."""
        manager = DatabaseManager()

        await manager.connect()
        await manager.disconnect()
        await manager.disconnect()  # Should not raise

        assert manager._client is None

    async def test_database_property_raises_when_not_connected(self):
        """Accessing database before connect should raise RuntimeError."""
        manager = DatabaseManager()
        await manager.disconnect()  # Ensure disconnected

        with pytest.raises(RuntimeError, match="Database not connected"):
            _ = manager.database

    async def test_client_property_raises_when_not_connected(self):
        """Accessing client before connect should raise RuntimeError."""
        manager = DatabaseManager()
        await manager.disconnect()  # Ensure disconnected

        with pytest.raises(RuntimeError, match="Database client not connected"):
            _ = manager.client

    async def test_connection_uses_settings(self):
        """Connection should use configuration from settings."""
        manager = DatabaseManager()
        await manager.connect()

        # Verify database name matches settings
        assert manager.database.name == settings.mongodb_database

        # Cleanup
        await manager.disconnect()

    async def test_ping_verifies_connection(self):
        """Connect should verify connection with ping."""
        manager = DatabaseManager()

        # This should not raise if MongoDB is running
        await manager.connect()

        # Manually verify we can ping
        result = await manager.client.admin.command("ping")
        assert result.get("ok") == 1.0

        # Cleanup
        await manager.disconnect()

    async def test_create_indexes_creates_all_indexes(self):
        """create_indexes should create all required indexes."""
        manager = DatabaseManager()
        await manager.connect()

        # Create indexes
        await manager.create_indexes()

        # Verify leads indexes
        leads_indexes = await manager.database.leads.index_information()
        assert "idx_lead_id_unique" in leads_indexes
        assert "idx_stage_last_interaction" in leads_indexes
        assert "idx_next_followup" in leads_indexes

        # Verify messages indexes
        messages_indexes = await manager.database.messages.index_information()
        assert "idx_lead_messages" in messages_indexes
        assert "idx_timestamp" in messages_indexes

        # Cleanup
        await manager.disconnect()

    async def test_get_database_helper(self):
        """get_database helper should return connected database."""
        await db_manager.connect()

        db = await get_database()

        assert isinstance(db, AsyncIOMotorDatabase)
        assert db.name == settings.mongodb_database

        # Cleanup
        await db_manager.disconnect()


@pytest.fixture(scope="function", autouse=True)
async def cleanup_db_manager():
    """Ensure db_manager is in clean state after each test."""
    yield
    # Cleanup after test
    try:
        await db_manager.disconnect()
    except:
        pass
