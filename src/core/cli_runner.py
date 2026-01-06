"""
CLI Runner for Conversation Orchestrator
Interactive command-line tool to test the orchestrator with real conversations.
"""
import asyncio
from loguru import logger
from src.core.conversation_orchestrator import ConversationOrchestrator
from src.models.lead import Lead, SalesStage
from src.utils.observability import configure_logging


async def run_conversation():
    """
    Run an interactive conversation session with the orchestrator.
    Demonstrates the complete 3-agent pipeline in action with MongoDB persistence.
    """
    # Configure logging
    configure_logging()

    logger.info("=" * 70)
    logger.info("🤖 GP Data Conversation Orchestrator - Interactive Demo")
    logger.info("=" * 70)

    # Initialize orchestrator
    orchestrator = ConversationOrchestrator()

    try:
        # Initialize MongoDB connection and repositories
        logger.info("\n🔌 Connecting to MongoDB...")
        await orchestrator.initialize()
        logger.info("✅ Connected to MongoDB with persistence enabled")

        # Load or create lead from database
        phone_number = "+5215538899800"
        logger.info(f"\n📞 Loading lead: {phone_number}")

        lead = await orchestrator.lead_repo.get_or_create(
            phone_number=phone_number,
            full_name="Carlos Rodriguez"
        )

        if lead.message_count > 0:
            logger.info(f"✅ Loaded existing lead from database")
            logger.info(f"   Previous messages: {lead.message_count}")
            logger.info(f"   Signals collected: {len(lead.signals)}")
        else:
            logger.info(f"✅ Created new lead in database")

        logger.info(f"\n📋 Lead Profile:")
        logger.info(f"   Name: {lead.full_name}")
        logger.info(f"   ID: {lead.lead_id}")
        logger.info(f"   Stage: {lead.current_stage}")
        logger.info(f"   Message Count: {lead.message_count}")

        # Test conversation scenarios
        test_messages = [
            "Hola, me interesa su producto",
            "Somos una empresa de 25 personas",
            "¿Cuánto cuesta?",
            "Necesitamos integración con HubSpot"
        ]

        print("\n" + "=" * 70)
        print("💬 Starting Conversation Simulation")
        print("=" * 70 + "\n")

        for i, message in enumerate(test_messages, 1):
            print(f"\n{'─' * 70}")
            print(f"🗣️  LEAD ({i}/{len(test_messages)}): {message}")
            print(f"{'─' * 70}")

            try:
                # Process through orchestrator
                result = await orchestrator.process_message(message, lead)

                # Display results
                print(f"\n📊 Classification:")
                print(f"   Intent: {result.classification.intent}")
                print(f"   Sentiment: {result.classification.sentiment}")
                print(f"   Language: {result.classification.language}")
                print(f"   Confidence: {result.classification.intent_confidence:.2%}")

                print(f"\n🎯 Strategic Decision:")
                print(f"   Action: {result.strategy.action}")
                print(f"   Tone: {result.strategy.message_strategy.tone}")
                print(f"   Focus: {result.strategy.focus_dimension}")

                print(f"\n💬 Response Generated:")
                print(f"   {result.execution.message.content}")

                print(f"\n⚡ Performance:")
                print(f"   Duration: {result.total_duration_ms:.0f}ms")
                print(f"   Agreement: {result.execution.agreement_level:.1%}")

                # Update lead reference
                lead = result.lead_updated

                # Pause between messages for readability
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"❌ Failed to process message: {e}")
                break

        # Final lead state
        print(f"\n\n{'=' * 70}")
        print("📈 Final Lead State")
        print(f"{'=' * 70}")
        print(f"Total Messages: {lead.message_count}")
        print(f"Signals Collected: {len(lead.signals)}")
        print(f"BANT Summary:")
        for dimension, value in lead.bant_summary.items():
            print(f"   {dimension}: {value}")

        print(f"\n💾 All conversation data persisted to MongoDB")
        print(f"   Lead: {lead.lead_id}")
        print(f"   Messages stored: {lead.message_count}")
        print(f"\n✅ Orchestration Demo Complete!\n")

    finally:
        # Ensure graceful shutdown
        logger.info("🔌 Disconnecting from MongoDB...")
        await orchestrator.shutdown()
        logger.info("✅ Shutdown complete")


async def run_single_message():
    """
    Run a single message through the orchestrator.
    Quick test for debugging with MongoDB persistence.
    """
    configure_logging()

    orchestrator = ConversationOrchestrator()

    try:
        # Initialize with persistence
        await orchestrator.initialize()

        # Load or create lead
        lead = await orchestrator.lead_repo.get_or_create(
            phone_number="+5215538899800",
            full_name="Test User"
        )

        message = "I need pricing for 20 users with HubSpot integration"

        logger.info(f"Processing: {message}")

        result = await orchestrator.process_message(message, lead)

        print(f"\n✅ Response: {result.outbound_message}")
        print(f"⚡ Duration: {result.total_duration_ms:.0f}ms")
        print(f"💾 Persisted to MongoDB: {lead.lead_id}")

    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    import sys

    # Choose demo mode
    if len(sys.argv) > 1 and sys.argv[1] == "single":
        asyncio.run(run_single_message())
    else:
        asyncio.run(run_conversation())
