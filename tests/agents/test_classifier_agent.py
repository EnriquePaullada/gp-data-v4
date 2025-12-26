import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

from src.models.lead import Lead
from src.models.message import Message, MessageRole
from src.agents.classifier_agent import ClassifierAgent

async def test_classifier_agent():
    """Founding Engineer Test: Classification with Full Context."""
    
    # 1. Environment Setup
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(env_path)
    
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("No API Key found. Check your .env file.")
        return

    # 2. Initialize Infrastructure
    agent = ClassifierAgent()
    
    # 3. Create a Lead with "Memory"
    # We simulate a lead who has already greeted the bot
    lead = Lead(lead_id="+5215538899800", full_name="John Doe")
    
    # Message 1 (History)
    greet_msg = Message(
        lead_id=lead.lead_id, 
        role=MessageRole.LEAD, 
        content="Hi, I'm John from a medical startup."
    )
    lead.add_message(greet_msg)
    
    # 4. The CURRENT incoming message
    new_input = "We have 20 reps and need HubSpot integration. How much is it?"
    
    logger.info(f"Running classification for lead: {lead.lead_id}")
    
    # 5. Execute with Lead Context
    response = await agent.classify(new_input, lead)
    
    # 6. Structured Output Audit
    print("\n" + "═"*30 + " AGENT OUTPUT " + "═"*30)
    print(f"INTENT:     {response.intent}")
    print(f"TOPIC:      {response.topic}")
    print(f"REASONING:  {response.reasoning}")
    
    # Check if AI correctly linked to history
    print(f"SENTIMENT:  {response.sentiment}")
    print(f"NEW SIGNALS: {len(response.new_signals)} facts extracted")
    print("═" * 74 + "\n")

if __name__ == "__main__":
    try:
        asyncio.run(test_classifier_agent())
    except Exception as e:
        logger.exception("Test Suite Crashed")
        exit(1)