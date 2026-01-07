from pydantic_ai import Agent
from loguru import logger
from src.models.classifier_response import ClassifierResponse, Intent, UrgencyLevel
from src.models.intelligence import Sentiment
from src.models.lead import Lead
from src.config import get_settings
from src.utils.llm_client import run_agent_with_fallback
from src.utils.cost_tracker import get_cost_tracker
from src.utils.observability import log_agent_execution, log_llm_call
import datetime as dt
import time

class ClassifierAgent:
    """
    Agent responsible for high-precision classification.
    Aligned with PydanticAI v1.x specifications (Late 2025).
    """
    
    def __init__(self, model_override: str | None = None):
        settings = get_settings()
        model_name = model_override or settings.classifier_model

        # 1. We type the Agent using Generics: Agent[Dependencies, OutputType]
        self.agent: Agent[None, ClassifierResponse] = Agent(
            model_name,
            output_type=ClassifierResponse,
            instructions=(
                "You are a high-precision sales intelligence engine. "
                "Classify messages based on the provided schema using evidence from history. "
                "THE STANDARD: Your primary goal is high-precision extraction. "
                "ESCAPE VALVE: Use 'unclear' labels ONLY for genuine ambiguity."
            )
        )
        self.model_name = model_name
        logger.info(f"ClassifierAgent initialized on 2025-spec with model: {model_name}")

    async def classify(self, content: str, lead: Lead) -> ClassifierResponse:
        """
        Executes classification with full awareness of the lead's history.
        Now with retry logic, cost tracking, and structured logging.
        """
        start_time = time.time()
        cost_tracker = get_cost_tracker()

        # 1. Format the 'Working Memory' for the LLM using DRY-compliant method
        history_str = lead.format_history()

        # 2. Construct the high-fidelity prompt
        # Note: We provide the current time so the AI understands 'recency'
        context_prompt = f"""
        CURRENT DATE:
        {dt.datetime.now(dt.UTC).isoformat()}

        CONVERSATION CONTEXT:
        {history_str}

        NEW MESSAGE TO CLASSIFY:
        {content}
        """

        logger.debug(f"Classifying for {lead.lead_id} with {len(lead.recent_history)} context messages.")

        # 3. Define fallback for critical failures
        def safe_fallback():
            """Return a safe default classification when LLM completely fails."""
            return ClassifierResponse(
                intent=Intent.UNCLEAR,
                intent_confidence=0.0,
                topic="system_error",
                topic_confidence=0.0,
                urgency=UrgencyLevel.UNCLEAR,
                urgency_confidence=0.0,
                language="english",
                sentiment=Sentiment.UNCLEAR,
                engagement_level="low",
                requires_human_escalation=True,
                reasoning="LLM failure - classification unavailable. Human review required.",
                new_signals=[]
            )

        try:
            # 4. Execute with retry logic
            result = await run_agent_with_fallback(
                agent=self.agent,
                prompt=context_prompt,
                fallback_factory=safe_fallback
            )

            # 5. Track costs (extract usage from result if available)
            try:
                usage = result.usage() if hasattr(result, 'usage') else None
                if usage:
                    cost = cost_tracker.track_completion(
                        model=self.model_name,
                        input_tokens=usage.request_tokens if hasattr(usage, 'request_tokens') else 0,
                        output_tokens=usage.response_tokens if hasattr(usage, 'response_tokens') else 0,
                        agent_name="ClassifierAgent"
                    )

                    # 6. Structured logging
                    duration_ms = (time.time() - start_time) * 1000
                    log_llm_call(
                        agent_name="ClassifierAgent",
                        model=self.model_name,
                        input_tokens=usage.request_tokens if hasattr(usage, 'request_tokens') else 0,
                        output_tokens=usage.response_tokens if hasattr(usage, 'response_tokens') else 0,
                        cost_usd=cost,
                        duration_ms=duration_ms,
                        success=True
                    )
            except Exception as tracking_error:
                logger.warning(f"Cost tracking failed (non-critical): {tracking_error}")

            # 7. Log agent execution
            duration_ms = (time.time() - start_time) * 1000
            log_agent_execution(
                agent_name="ClassifierAgent",
                lead_id=lead.lead_id,
                action="classify",
                duration_ms=duration_ms,
                intent=result.intent,
                confidence=result.intent_confidence
            )

            logger.success(f"Classification complete: {result.intent}")
            return result

        except Exception as e:
            # This should rarely happen due to fallback, but just in case
            duration_ms = (time.time() - start_time) * 1000
            log_llm_call(
                agent_name="ClassifierAgent",
                model=self.model_name,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                duration_ms=duration_ms,
                success=False,
                error=str(e)
            )
            logger.error(f"Agent logic failed catastrophically: {str(e)}")
            raise