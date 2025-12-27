from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from loguru import logger
from src.models.director_response import DirectorResponse,StrategicAction
from src.models.lead import Lead

@dataclass
class DirectorDeps:
    """External context for the Director (Playbook, current pricing, etc.)"""
    playbook_version: str
    enterprise_threshold: int = 30

director_agent: Agent[DirectorDeps, DirectorResponse] = Agent(
    'openai:gpt-4o',
    deps_type=DirectorDeps,
    output_type=DirectorResponse,
    # We use 'instructions' for the static role
    instructions=(
        "You are the Strategic Sales Director for GP Data. Your purpose is to convert raw conversational data into decisive, strategic commands."
        "You are an **empathetic consultant**, not an interrogator. "
        "Make the lead feel heard and understood, earning you the right to ask questions. Offer to be helpful first, ask later."
        "Your ultimate objective is to maximize the **qualified demo conversion rate** by building trust, not by rushing through a checklist."
        "##PRIME DIRECTIVE: Rapport First, Qualification Second."
    )
)

# 1. DYNAMIC CONTEXT (Using 'Lead' and 'logger')
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
        self.deps = DirectorDeps(playbook_version=playbook_version)

    async def decide_next_move(self, lead: Lead, last_classification: str) -> DirectorResponse:
        """
        DETERMINISTIC GATING + LLM STRATEGY
        """
        logger.info(f"🧠 Director analyzing Lead: {lead.lead_id} | Stage: {lead.current_stage}")
        
        # --- LAYER 1: DETERMINISTIC HARD GATES ---
        # If the classifier says 'Ready to Buy', we don't even ask the LLM to 'think'.
        # We force a 'CLOSE' action.
        if last_classification == "ready_to_buy":
            logger.warning("🎯 High Intent Detected: Triggering Deterministic CLOSE action.")
            return DirectorResponse(
                action=StrategicAction.CLOSE,
                strategic_reasoning="Classifier detected explicit buying intent. Bypassing discovery.",
                message_strategy={
                    "tone": "enthusiastic",
                    "language": "english", # Or detect from lead
                    "empathy_points": ["Acknowledge their readiness to transform their sales process"],
                    "key_points": ["Confirm the demo is the fastest path to ROI"],
                    "conversational_goal": "Finalize the calendar invite"
                },
                focus_dimension="timeline"
            )

        # --- LAYER 2: LLM STRATEGIC REASONING ---
        # If it's not a hard-gate scenario, we let the LLM look at the BANT summary.
        prompt = f"""
        CURRENT BANT SUMMARY: {lead.bant_summary}
        LAST SALES STAGE: {lead.current_stage}
        
        Analyze the state and decide the conversational_goal for the next message.
        """
        
        result = await director_agent.run(prompt, deps=self.deps)
        
        logger.success(f"♟️ Strategy Decided: {result.output.action}")
        return result.output