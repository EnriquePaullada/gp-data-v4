from pydantic_ai import Agent
from loguru import logger
from src.models.classifier_response import ClassifierResponse
from src.models.lead import Lead
import datetime as dt

class ClassifierAgent:
    """
    Agent responsible for high-precision classification.
    Aligned with PydanticAI v1.x specifications (Late 2025).
    """
    
    def __init__(self, model_override: str | None = None):
        model_name = model_override or 'openai:gpt-4o-mini'
        
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
        logger.info(f"ClassifierAgent initialized on 2025-spec with model: {model_name}")

    async def classify(self, content: str, lead: Lead) -> ClassifierResponse:
        """
        Executes classification with full awareness of the lead's history.
        """
        # 1. Format the 'Working Memory' for the LLM
        history_str = ""
        for msg in lead.recent_history:
            # We capitalize the role for clarity (LEAD: ..., ASSISTANT: ...)
            history_str += f"{msg.role.upper()}: {msg.content}\n"

        # 2. Construct the high-fidelity prompt
        # Note: We provide the current time so the AI understands 'recency'
        context_prompt = f"""
        CURRENT DATE:
        {dt.UTC}
        
        CONVERSATION CONTEXT:
        {history_str}

        NEW MESSAGE TO CLASSIFY:
        {content}
        """

        logger.debug(f"Classifying for {lead.lead_id} with {len(lead.recent_history)} context messages.")
        
        try:
            # Aligned with PydanticAI v1.x standards (output_type and .output)
            result = await self.agent.run(context_prompt)
            
            logger.success(f"Classification complete: {result.output.intent}")
            return result.output
            
        except Exception as e:
            logger.error(f"Agent logic failed: {str(e)}")
            raise