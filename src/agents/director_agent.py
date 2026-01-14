from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from loguru import logger
from src.models.director_response import DirectorResponse,StrategicAction
from src.models.lead import Lead
from src.models.classifier_response import ClassifierResponse, Intent
from src.config import get_settings
from src.utils.llm_client import run_agent_with_circuit_breaker
from src.utils.fallback_responses import get_fallback_strategy
from src.utils.cost_tracker import get_cost_tracker
from src.utils.observability import log_agent_execution, log_llm_call
import time

@dataclass
class DirectorDeps:
    """External context for the Director (Playbook, current pricing, etc.)"""
    playbook_version: str
    enterprise_threshold: int

# Initialize director_agent with settings
settings = get_settings()

director_agent: Agent[DirectorDeps, DirectorResponse] = Agent(
    settings.director_model,
    deps_type=DirectorDeps,
    output_type=DirectorResponse,
    # We use 'instructions' for the static role
    instructions=(
        "You are the Strategic Sales Director for GP Data. Your purpose is to convert raw conversational data into decisive, sales-oriented, strategic commands."
        "You receive the full lead state and rich context, and analyze it thoroughly. You are an **empathetic consultant**, not an interrogator. "
        "Make the lead feel heard and understood, earning you the right to ask questions. Offer to be helpful first, ask later."
        "Your ultimate objective is to maximize the **qualified demo conversion rate** by building trust, not by rushing through a checklist."
        "##PRIME DIRECTIVE: Rapport First, Qualification Second. Your insight will be used by the Communication Executor to write the copy for the final message to the lead."
        "###Communication channel: WhatsApp, be sure to consider the message length limitations in your guidance"
    )
)

# 1. DYNAMIC CONTEXT
@director_agent.system_prompt
async def inject_lead_context(ctx: RunContext[DirectorDeps]) -> str:
    """
    We pass the 'Live Dashboard' of the Lead.
    """
    return (
        f"Context: GP Data v4. Playbook: {ctx.deps.playbook_version}. "
        f"Enterprise Qualification Threshold: {ctx.deps.enterprise_threshold} seats."
    )

class DirectorService:
    """The Logic Hub for Agent 2."""

    def __init__(self, playbook_version: str = "2025.12.v2"):
        settings = get_settings()
        self.deps = DirectorDeps(
            playbook_version=playbook_version,
            enterprise_threshold=settings.enterprise_threshold
        )
        self.model_name = settings.director_model

    async def decide_next_move(self, lead: Lead, classification: ClassifierResponse) -> DirectorResponse:
        """
        DETERMINISTIC GATING + DYNAMIC CONTEXT + LLM STRATEGY
        Now with retry logic, cost tracking, and structured logging.
        """
        start_time = time.time()
        cost_tracker = get_cost_tracker()

        logger.info(f"🧠 Director analyzing Lead: {lead.lead_id} | Stage: {lead.current_stage}")

        # --- LAYER 1: DETERMINISTIC HARD GATES ---
        # If the classifier says 'Ready to Buy', we don't even ask the LLM to 'think'.
        # We force a 'CLOSE' action.
        if classification.intent == Intent.READY_TO_BUY:
            logger.warning("🎯 High Intent Detected: Triggering Deterministic CLOSE action.")
            return DirectorResponse(
                action=StrategicAction.CLOSE,
                strategic_reasoning="Hard-gate: Classifier detected explicit buying intent. Bypassing discovery.",
                message_strategy={
                    "tone": "enthusiastic",
                    "language": classification.language if classification.language != "mixed" else "english",
                    "empathy_points": ["Acknowledge their readiness to transform their sales process"],
                    "key_points": ["Confirm the demo is the fastest path to ROI"],
                    "conversational_goal": "Finalize the calendar invite"
                },
                focus_dimension="timeline"
            )

        # --- LAYER 2: DYNAMIC CONTEXT PREPARATION ---
        # Use DRY-compliant format_history method
        transcript = lead.format_history()

        # --- LAYER 3: THE STRATEGIC PROMPT ---
        prompt = f"""
        #LEAD DOSSIER
            Name: {lead.full_name}
            Sales Stage: {lead.current_stage}
            BANT KNOWLEDGE GRAPH:: {lead.bant_summary}

        #RECENT CONVERSATION HISTORY:
        [{transcript}]

        #LATEST SIGNAL
        - Intent: {classification.intent}
        - Reasoning: {classification.reasoning}
        - New Signals Extracted: {classification.new_signals}

        TASK:
        Analyze the rich lead context and think of several ways of achieving your purpose.
        Finally, provide the Communication Executor the best strategic guidance.
        """

        try:
            # Execute with circuit breaker, retry logic, and fallback
            result = await run_agent_with_circuit_breaker(
                agent=director_agent,
                prompt=prompt,
                fallback_factory=get_fallback_strategy,
                deps=self.deps
            )

            # Track costs
            try:
                usage = result.usage() if hasattr(result, 'usage') else None
                if usage:
                    cost = cost_tracker.track_completion(
                        model=self.model_name,
                        input_tokens=usage.request_tokens if hasattr(usage, 'request_tokens') else 0,
                        output_tokens=usage.response_tokens if hasattr(usage, 'response_tokens') else 0,
                        agent_name="DirectorAgent"
                    )

                    duration_ms = (time.time() - start_time) * 1000
                    log_llm_call(
                        agent_name="DirectorAgent",
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
                agent_name="DirectorAgent",
                lead_id=lead.lead_id,
                action="decide_strategy",
                duration_ms=duration_ms,
                strategic_action=result.action,
                stage=lead.current_stage
            )

            logger.success(f"♟️ Strategy Decided: {result.action}")
            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            log_llm_call(
                agent_name="DirectorAgent",
                model=self.model_name,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                duration_ms=duration_ms,
                success=False,
                error=str(e)
            )
            logger.error(f"Director strategy failed: {str(e)}")
            raise