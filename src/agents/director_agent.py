from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from loguru import logger
from src.models.director_response import DirectorResponse,StrategicAction
from src.models.lead import Lead
from src.models.classifier_response import ClassifierResponse, Intent

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
        self.deps = DirectorDeps(playbook_version=playbook_version)

    async def decide_next_move(self, lead: Lead, classification: ClassifierResponse) -> DirectorResponse:
        """
        DETERMINISTIC GATING + DYNAMIC CONTEXT + LLM LLM STRATEGY
        """
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
        # We format the recent conversation history so the LLM sees a clean transcript
        transcript = "\n".join([
            f"{msg.role.upper()}: {msg.content}" 
            for msg in lead.recent_history
        ])


        # --- LAYER 3: THE STRATEGIC PROMPT ---
        prompt = f"""
        BANT KNOWLEDGE GRAPH:
        {lead.bant_summary}
        
        CURRENT SALES STAGE: 
        {lead.current_stage}

        RECENT CONVERSATION HISTORY:
        {transcript}

        LATEST MESSAGE CLASSIFICATION:
        - Intent: {classification.intent}
        - Reasoning: {classification.reasoning}
        - New Signals Extracted: {classification.new_signals}

        TASK: 
        Analyze the rich lead context and think of several ways of achieving your purpose.
        Finally, provide the Communication Executor the best strategic guidance.
        """
        
        result = await director_agent.run(prompt, deps=self.deps)
        logger.success(f"♟️ Strategy Decided: {result.output.action}")
        return result.output