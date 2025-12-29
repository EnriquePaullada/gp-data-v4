from pydantic_ai import Agent
from src.models.executor_response import ExecutorResponse
from src.models.director_response import DirectorResponse
from src.models.lead import Lead
from loguru import logger

# Alena Gomez: The Communication Agent
alena_agent: Agent[None, ExecutorResponse] = Agent(
    'openai:gpt-4o-mini',
    output_type=ExecutorResponse,
    instructions=(
        "You are Alena Gomez, the warm and professional Voice of GP Data. "
        "Your mission is to turn the Strategic Director's command into WhatsApp-first human empathy & drive sales."
        
        "### TONE & STYLE ###"
        "1. CONCISE: 1-3 sentences. WhatsApp is a fast-paced channel."
        "2. PERSONA: You are a consultative expert. Be encouraging and inviting, not pushy."
        "3. MIRRORING: Match the lead's energy. If they are frustrated, lead with heavy empathy."
        "4. COPYWRITER: Your word-choice is of paramount importance. Think always in terms of human emotion and empathy first, business objectives second."
        "5. LOCALE: Use natural Mexican Spanish ('tú') if the strategy requires Spanish."
        
        "### COMPLIANCE GATEKEEPER ###"
        "NEVER promise exact ROI or guarantees. Use words like 'typically', 'could', or 'often'."
        "NEVER offer unauthorized discounts. Stick to $99/month."
        "NEVER forget to evaluate whether your messaging could represent a liability to the brand or negatively impact the company's ROI."
    )
)

class ExecutorService:
    """
    The Orchestration logic for Agent 3.
    Bridges the 'General' (Director) to the 'User' (WhatsApp).
    """

    async def craft_message(self, lead: Lead, strategy: DirectorResponse) -> ExecutorResponse:
        logger.info(f"🎙️ Alena Gomez generating response for Lead: {lead.lead_id}")

        # --- THE DYNAMIC INJECTION (Using our imports) ---
        # We format the history so Alena knows what was JUST said.
        history_transcript = "\n".join([
            f"{m.role.upper()}: {m.content}" 
            for m in lead.recent_history[-3:] # Last 3 turns for flow
        ])

        # We construct the "Battle Plan" for this specific message.
        prompt = f"""
        [THE STRATEGIC PLAN]
        Action: {strategy.action}
        Tone: {strategy.message_strategy.tone}
        Points to include: {strategy.message_strategy.key_points}
        Empathy focus: {strategy.message_strategy.empathy_points}
        Goal (CTA): {strategy.message_strategy.conversational_goal}

        [CONVERSATIONAL CONTEXT]
        Lead Name: {lead.full_name or 'Friend'}
        Language Requirement: {strategy.message_strategy.language}
        Recent History:
        {history_transcript}

        #TASK
        You will first consider the full richness of the context provided to you and determine what the best message for Outbound to the lead is.
        You will then build the ExecutorResponse object around that message you have determined, which will be sent directly to the lead's WhatsApp
        This is your chance to provide feedback to your director as to what you do not agree with, and what you believe is very accurate to highlight.
        Once you have built the full object, you will review it once more to ensure no compliance or liability risks are present.
        Finally, you will execute any upgrades or changes that you deem necessary in accordance with the previous steps and emit your final output.
        """

        try:
            result = await alena_agent.run(prompt)
            logger.success(f"🎙️ Alena message created. Agreement: {result.output.agreement_level}")
            return result.output
            
        except Exception as e:
            logger.error(f"🎙️ Alena failed to speak: {str(e)}")
            raise