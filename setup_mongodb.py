"""
MongoDB Atlas Setup Script
Tests connection and initializes database with collections and indexes.
"""
import asyncio
from src.repositories import db_manager
from src.config import settings


async def setup_mongodb():
    """Initialize MongoDB Atlas database with collections and indexes."""
    print("🔄 Connecting to MongoDB Atlas...")
    print(f"   Database: {settings.mongodb_database}")
    print(f"   Cluster: gp-data-production.ghvw52k.mongodb.net")
    print()

    try:
        # Connect to MongoDB
        await db_manager.connect()
        print("✅ Connection successful!")
        print()

        # Get database instance
        db = db_manager.database

        # Show existing collections
        existing_collections = await db.list_collection_names()
        print(f"📦 Existing collections: {existing_collections or 'None'}")
        print()

        # Create indexes
        print("🔨 Creating indexes...")
        await db_manager.create_indexes()
        print("✅ Indexes created successfully!")
        print()

        # Verify indexes
        print("📊 Verifying indexes:")

        # Leads collection indexes
        leads_indexes = await db.leads.index_information()
        print(f"   Leads collection: {len(leads_indexes)} indexes")
        for idx_name in leads_indexes:
            print(f"      - {idx_name}")

        # Messages collection indexes
        messages_indexes = await db.messages.index_information()
        print(f"   Messages collection: {len(messages_indexes)} indexes")
        for idx_name in messages_indexes:
            print(f"      - {idx_name}")

        print()
        print("🎉 MongoDB Atlas setup complete!")
        print()
        print("📝 Summary:")
        print(f"   ✅ Database: {settings.mongodb_database}")
        print(f"   ✅ Collections: leads, messages")
        print(f"   ✅ Indexes: {len(leads_indexes) + len(messages_indexes)} total")
        print()

    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        print("💡 Troubleshooting:")
        print("   1. Verify your IP is whitelisted in Atlas Network Access")
        print("   2. Check that the username and password are correct")
        print("   3. Ensure the cluster is running (not paused)")
        raise

    finally:
        # Clean disconnect
        await db_manager.disconnect()
        print("👋 Disconnected from MongoDB")


if __name__ == "__main__":
    asyncio.run(setup_mongodb())
