"""
Follow-up Scheduler for Lead Re-engagement

Proactively schedules follow-up prompts for inactive leads.
Sales-oriented approach to keep conversations warm and
maximize conversion opportunities.
"""

import datetime as dt
from enum import StrEnum
from dataclasses import dataclass
from typing import Optional
from src.config import get_settings
from src.models.lead import Lead, SalesStage
from src.utils.observability import logger


class FollowUpType(StrEnum):
    """Types of follow-up prompts."""
    CHECK_IN = "check_in"           # "Hey, just checking in..."
    VALUE_REMINDER = "value_reminder"  # Remind of product benefits
    URGENCY = "urgency"             # Create urgency (limited time, etc.)
    SOCIAL_PROOF = "social_proof"   # Share success stories
    FINAL_ATTEMPT = "final_attempt"  # Last chance before going cold


@dataclass
class FollowUpAction:
    """Scheduled follow-up action for a lead."""
    lead_id: str
    followup_type: FollowUpType
    scheduled_at: dt.datetime
    attempt_number: int
    reason: str


class FollowUpScheduler:
    """
    Schedules and manages follow-up prompts for lead re-engagement.

    Tracks inactivity and schedules appropriate follow-up types
    based on lead stage and conversation history.

    Usage:
        scheduler = FollowUpScheduler()

        # Check if lead needs follow-up
        action = scheduler.get_next_followup(lead)
        if action:
            # Schedule or execute follow-up
            lead.next_followup_at = action.scheduled_at

        # After sending follow-up
        scheduler.record_followup_sent(lead)
    """

    # Stages that should receive follow-ups
    FOLLOWUP_STAGES = {
        SalesStage.NEW,
        SalesStage.DISCOVERY,
        SalesStage.QUALIFIED,
    }

    # Stages that are terminal (no follow-ups)
    TERMINAL_STAGES = {
        SalesStage.DEMO_SCHEDULED,
        SalesStage.WON,
        SalesStage.LOST,
    }

    def __init__(self):
        settings = get_settings()
        self._initial_delay_hours = settings.followup_initial_delay_hours
        self._max_attempts = settings.followup_max_attempts
        self._escalation_hours = settings.followup_escalation_hours

    def get_next_followup(self, lead: Lead) -> Optional[FollowUpAction]:
        """
        Determine if and when lead needs a follow-up.

        Args:
            lead: Lead to evaluate

        Returns:
            FollowUpAction if follow-up needed, None otherwise
        """
        # Skip terminal stages
        if lead.current_stage in self.TERMINAL_STAGES:
            return None

        # Skip ON_HOLD (requires manual intervention)
        if lead.current_stage == SalesStage.ON_HOLD:
            return None

        # Calculate inactivity
        now = dt.datetime.now(dt.UTC)
        hours_inactive = (now - lead.last_interaction_at).total_seconds() / 3600

        # Determine attempt number from followup count
        attempt = self._get_attempt_number(lead)

        # Check if max attempts exceeded
        if attempt > self._max_attempts:
            logger.info(f"Lead {lead.lead_id} exceeded max follow-up attempts")
            return None

        # Get delay for this attempt
        delay_hours = self._get_delay_for_attempt(attempt)

        # Check if enough time has passed
        if hours_inactive < delay_hours:
            return None

        # Determine follow-up type based on attempt and stage
        followup_type = self._select_followup_type(lead, attempt)

        return FollowUpAction(
            lead_id=lead.lead_id,
            followup_type=followup_type,
            scheduled_at=now,
            attempt_number=attempt,
            reason=self._generate_reason(lead, attempt, hours_inactive),
        )

    def schedule_followup(self, lead: Lead, delay_hours: Optional[int] = None) -> dt.datetime:
        """
        Schedule next follow-up for a lead.

        Args:
            lead: Lead to schedule follow-up for
            delay_hours: Override delay (uses config default if None)

        Returns:
            Scheduled datetime
        """
        attempt = self._get_attempt_number(lead)
        hours = delay_hours or self._get_delay_for_attempt(attempt + 1)

        scheduled = dt.datetime.now(dt.UTC) + dt.timedelta(hours=hours)
        lead.next_followup_at = scheduled

        logger.info(
            f"Scheduled follow-up for {lead.lead_id}",
            extra={"scheduled_at": scheduled.isoformat(), "attempt": attempt + 1}
        )

        return scheduled

    def should_mark_cold(self, lead: Lead) -> bool:
        """
        Check if lead should be marked as cold (ON_HOLD).

        Args:
            lead: Lead to evaluate

        Returns:
            True if lead should go to ON_HOLD
        """
        attempt = self._get_attempt_number(lead)
        return attempt > self._max_attempts

    def mark_cold(self, lead: Lead) -> None:
        """
        Mark lead as cold after exhausting follow-up attempts.

        Args:
            lead: Lead to mark cold
        """
        lead.current_stage = SalesStage.ON_HOLD
        lead.next_followup_at = None

        logger.info(
            f"Marked lead {lead.lead_id} as cold (ON_HOLD)",
            extra={"previous_stage": lead.current_stage}
        )

    def _get_attempt_number(self, lead: Lead) -> int:
        """
        Infer follow-up attempt number from lead signals.

        Counts signals with 'followup' in confidence reasoning.
        """
        followup_count = sum(
            1 for s in lead.signals
            if "followup" in (s.confidence.reasoning or "").lower()
        )
        return followup_count + 1

    def _get_delay_for_attempt(self, attempt: int) -> int:
        """Get delay hours for given attempt number."""
        if attempt <= 0:
            return self._initial_delay_hours

        # Use escalation schedule, cap at last value
        index = min(attempt - 1, len(self._escalation_hours) - 1)
        return self._escalation_hours[index]

    def _select_followup_type(self, lead: Lead, attempt: int) -> FollowUpType:
        """Select appropriate follow-up type based on context."""
        if attempt >= self._max_attempts:
            return FollowUpType.FINAL_ATTEMPT

        if attempt == 1:
            return FollowUpType.CHECK_IN

        if lead.current_stage == SalesStage.QUALIFIED:
            return FollowUpType.URGENCY

        if attempt == 2:
            return FollowUpType.VALUE_REMINDER

        return FollowUpType.SOCIAL_PROOF

    def _generate_reason(self, lead: Lead, attempt: int, hours_inactive: float) -> str:
        """Generate human-readable reason for follow-up."""
        days = int(hours_inactive / 24)
        return (
            f"Lead inactive for {days} days. "
            f"Stage: {lead.current_stage.value}. "
            f"Follow-up attempt {attempt}/{self._max_attempts}."
        )

    def get_followup_prompt_guidance(self, followup_type: FollowUpType) -> dict:
        """
        Get messaging guidance for follow-up type.

        Returns dict suitable for Director agent message_strategy.
        """
        guidance = {
            FollowUpType.CHECK_IN: {
                "tone": "warm and casual",
                "empathy_points": ["Acknowledge they're busy", "No pressure"],
                "key_points": ["Just checking in", "Here if you need anything"],
                "conversational_goal": "Re-establish contact without being pushy",
            },
            FollowUpType.VALUE_REMINDER: {
                "tone": "helpful and informative",
                "empathy_points": ["Understand their challenges"],
                "key_points": ["Specific benefit reminder", "Quick win they could achieve"],
                "conversational_goal": "Remind them why they were interested",
            },
            FollowUpType.URGENCY: {
                "tone": "professional with gentle urgency",
                "empathy_points": ["Respect their time"],
                "key_points": ["Limited availability", "Others are moving forward"],
                "conversational_goal": "Create motivation to act",
            },
            FollowUpType.SOCIAL_PROOF: {
                "tone": "enthusiastic and credible",
                "empathy_points": ["Others had similar hesitation"],
                "key_points": ["Success story", "Concrete results"],
                "conversational_goal": "Build confidence through peer validation",
            },
            FollowUpType.FINAL_ATTEMPT: {
                "tone": "respectful and direct",
                "empathy_points": ["Respect if timing isn't right"],
                "key_points": ["Last check-in", "Door always open"],
                "conversational_goal": "Give graceful exit while leaving door open",
            },
        }
        return guidance.get(followup_type, guidance[FollowUpType.CHECK_IN])


# Singleton instance
_scheduler: Optional[FollowUpScheduler] = None


def get_followup_scheduler() -> FollowUpScheduler:
    """Get or create the follow-up scheduler singleton."""
    global _scheduler
    if _scheduler is None:
        _scheduler = FollowUpScheduler()
    return _scheduler
