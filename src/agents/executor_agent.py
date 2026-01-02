from pydantic_ai import Agent
from src.models.executor_response import ExecutorResponse
from src.models.director_response import DirectorResponse
from src.models.lead import Lead
from loguru import logger
from src.config import get_settings
from src.utils.llm_client import run_agent_with_retry
from src.utils.cost_tracker import get_cost_tracker
from src.utils.observability import log_agent_execution, log_llm_call
import time

# Initialize settings
settings = get_settings()

# Alena Gomez: The Communication Agent
alena_agent: Agent[None, ExecutorResponse] = Agent(
    settings.executor_model,
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

    def __init__(self):
        self.model_name = settings.executor_model

    async def craft_message(self, lead: Lead, strategy: DirectorResponse) -> ExecutorResponse:
        """
        Generate customer-facing message with retry logic and cost tracking.
        """
        start_time = time.time()
        cost_tracker = get_cost_tracker()

        logger.info(f"🎙️ Alena Gomez generating response for Lead: {lead.lead_id}")

        # --- THE DYNAMIC INJECTION (Using our imports) ---
        # Use DRY-compliant format_history method - last 3 turns for flow
        history_transcript = lead.format_history(limit=3)

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
            # Execute with retry logic
            result = await run_agent_with_retry(
                agent=alena_agent,
                prompt=prompt
            )

            # Track costs
            try:
                usage = result.usage() if hasattr(result, 'usage') else None
                if usage:
                    cost = cost_tracker.track_completion(
                        model=self.model_name,
                        input_tokens=usage.request_tokens if hasattr(usage, 'request_tokens') else 0,
                        output_tokens=usage.response_tokens if hasattr(usage, 'response_tokens') else 0,
                        agent_name="ExecutorAgent"
                    )

                    duration_ms = (time.time() - start_time) * 1000
                    log_llm_call(
                        agent_name="ExecutorAgent",
                        model=self.model_name,
                        input_tokens=usage.request_tokens if hasattr(usage, 'request_tokens') else 0,
                        output_tokens=usage.response_tokens if hasattr(usage, 'response_tokens') else 0,
                        cost_usd=cost,
                        duration_ms=duration_ms,
                        success=True
                    )
            except Exception as tracking_error:
                logger.warning(f"Cost tracking failed (non-critical): {tracking_error}")

            # Log agent execution
            duration_ms = (time.time() - start_time) * 1000
            log_agent_execution(
                agent_name="ExecutorAgent",
                lead_id=lead.lead_id,
                action="craft_message",
                duration_ms=duration_ms,
                agreement_level=result.agreement_level
            )

            logger.success(f"🎙️ Alena message created. Agreement: {result.agreement_level}")
            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            log_llm_call(
                agent_name="ExecutorAgent",
                model=self.model_name,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                duration_ms=duration_ms,
                success=False,
                error=str(e)
            )
            logger.error(f"🎙️ Alena failed to speak: {str(e)}")
            raise