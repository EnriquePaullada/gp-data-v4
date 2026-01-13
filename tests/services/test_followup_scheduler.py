"""
Tests for Follow-up Scheduler

Validates lead re-engagement scheduling logic.
"""

import pytest
import datetime as dt
from src.services.followup_scheduler import (
    FollowUpScheduler,
    FollowUpType,
    FollowUpAction,
    get_followup_scheduler,
)
from src.models.lead import Lead, SalesStage
from src.models.intelligence import IntelligenceSignal, BANTDimension, ConfidenceScore


def make_followup_signal() -> IntelligenceSignal:
    """Create a valid signal indicating a follow-up was sent."""
    return IntelligenceSignal(
        dimension=BANTDimension.NEED,
        extracted_value="pending",
        confidence=ConfidenceScore(value=0.5, reasoning="followup_sent: automated check-in"),
        source_message_id="followup_msg_001",
        raw_evidence="[System: Follow-up message sent]",
    )


class TestFollowUpScheduler:
    """Tests for FollowUpScheduler class."""

    @pytest.fixture
    def scheduler(self):
        return FollowUpScheduler()

    @pytest.fixture
    def active_lead(self):
        """Lead with recent activity."""
        return Lead(
            lead_id="+525512345678",
            full_name="Carlos Test",
            current_stage=SalesStage.DISCOVERY,
            last_interaction_at=dt.datetime.now(dt.UTC),
        )

    @pytest.fixture
    def inactive_lead(self):
        """Lead inactive for 25 hours."""
        return Lead(
            lead_id="+525512345678",
            full_name="Carlos Test",
            current_stage=SalesStage.DISCOVERY,
            last_interaction_at=dt.datetime.now(dt.UTC) - dt.timedelta(hours=25),
        )

    @pytest.fixture
    def very_inactive_lead(self):
        """Lead inactive for 5 days."""
        return Lead(
            lead_id="+525512345678",
            full_name="Carlos Test",
            current_stage=SalesStage.DISCOVERY,
            last_interaction_at=dt.datetime.now(dt.UTC) - dt.timedelta(days=5),
        )

    # ===========================================
    # Follow-up Detection
    # ===========================================

    def test_active_lead_no_followup(self, scheduler, active_lead):
        """Active leads should not need follow-up."""
        action = scheduler.get_next_followup(active_lead)
        assert action is None

    def test_inactive_lead_needs_followup(self, scheduler, inactive_lead):
        """Inactive leads should need follow-up."""
        action = scheduler.get_next_followup(inactive_lead)
        assert action is not None
        assert action.lead_id == inactive_lead.lead_id
        assert action.attempt_number == 1

    def test_first_followup_is_check_in(self, scheduler, inactive_lead):
        """First follow-up should be CHECK_IN type."""
        action = scheduler.get_next_followup(inactive_lead)
        assert action.followup_type == FollowUpType.CHECK_IN

    # ===========================================
    # Terminal Stages
    # ===========================================

    def test_demo_scheduled_no_followup(self, scheduler, inactive_lead):
        """DEMO_SCHEDULED leads should not get follow-ups."""
        inactive_lead.current_stage = SalesStage.DEMO_SCHEDULED
        action = scheduler.get_next_followup(inactive_lead)
        assert action is None

    def test_won_no_followup(self, scheduler, inactive_lead):
        """WON leads should not get follow-ups."""
        inactive_lead.current_stage = SalesStage.WON
        action = scheduler.get_next_followup(inactive_lead)
        assert action is None

    def test_lost_no_followup(self, scheduler, inactive_lead):
        """LOST leads should not get follow-ups."""
        inactive_lead.current_stage = SalesStage.LOST
        action = scheduler.get_next_followup(inactive_lead)
        assert action is None

    def test_on_hold_no_followup(self, scheduler, inactive_lead):
        """ON_HOLD leads should not get automated follow-ups."""
        inactive_lead.current_stage = SalesStage.ON_HOLD
        action = scheduler.get_next_followup(inactive_lead)
        assert action is None

    # ===========================================
    # Follow-up Escalation
    # ===========================================

    def test_second_attempt_after_signal(self, scheduler, very_inactive_lead):
        """Should detect second attempt from signals."""
        very_inactive_lead.signals.append(make_followup_signal())

        action = scheduler.get_next_followup(very_inactive_lead)
        assert action is not None
        assert action.attempt_number == 2

    def test_second_followup_type(self, scheduler, very_inactive_lead):
        """Second follow-up should be VALUE_REMINDER."""
        very_inactive_lead.signals.append(make_followup_signal())

        action = scheduler.get_next_followup(very_inactive_lead)
        assert action.followup_type == FollowUpType.VALUE_REMINDER

    def test_qualified_lead_gets_urgency(self, scheduler, very_inactive_lead):
        """Qualified leads should get URGENCY follow-up."""
        very_inactive_lead.current_stage = SalesStage.QUALIFIED
        very_inactive_lead.signals.append(make_followup_signal())

        action = scheduler.get_next_followup(very_inactive_lead)
        assert action.followup_type == FollowUpType.URGENCY

    def test_final_attempt_type(self, scheduler, very_inactive_lead):
        """Last attempt should be FINAL_ATTEMPT type."""
        for _ in range(2):
            very_inactive_lead.signals.append(make_followup_signal())

        action = scheduler.get_next_followup(very_inactive_lead)
        assert action.followup_type == FollowUpType.FINAL_ATTEMPT

    def test_max_attempts_exceeded(self, scheduler, very_inactive_lead):
        """Should return None after max attempts."""
        for _ in range(3):
            very_inactive_lead.signals.append(make_followup_signal())

        action = scheduler.get_next_followup(very_inactive_lead)
        assert action is None

    # ===========================================
    # Scheduling
    # ===========================================

    def test_schedule_followup(self, scheduler, active_lead):
        """Should schedule follow-up datetime."""
        scheduled = scheduler.schedule_followup(active_lead)

        assert active_lead.next_followup_at == scheduled
        assert scheduled > dt.datetime.now(dt.UTC)

    def test_schedule_with_custom_delay(self, scheduler, active_lead):
        """Should respect custom delay."""
        scheduled = scheduler.schedule_followup(active_lead, delay_hours=48)

        expected_min = dt.datetime.now(dt.UTC) + dt.timedelta(hours=47)
        expected_max = dt.datetime.now(dt.UTC) + dt.timedelta(hours=49)

        assert expected_min < scheduled < expected_max

    # ===========================================
    # Cold Leads
    # ===========================================

    def test_should_mark_cold(self, scheduler, very_inactive_lead):
        """Should detect when lead should go cold."""
        for _ in range(3):
            very_inactive_lead.signals.append(make_followup_signal())

        assert scheduler.should_mark_cold(very_inactive_lead) is True

    def test_should_not_mark_cold_early(self, scheduler, inactive_lead):
        """Should not mark cold before max attempts."""
        assert scheduler.should_mark_cold(inactive_lead) is False

    def test_mark_cold(self, scheduler, inactive_lead):
        """Should mark lead as ON_HOLD."""
        scheduler.mark_cold(inactive_lead)

        assert inactive_lead.current_stage == SalesStage.ON_HOLD
        assert inactive_lead.next_followup_at is None

    # ===========================================
    # Prompt Guidance
    # ===========================================

    def test_get_prompt_guidance_check_in(self, scheduler):
        """Should return guidance for CHECK_IN."""
        guidance = scheduler.get_followup_prompt_guidance(FollowUpType.CHECK_IN)

        assert "tone" in guidance
        assert "warm" in guidance["tone"].lower()
        assert "empathy_points" in guidance
        assert "key_points" in guidance

    def test_get_prompt_guidance_urgency(self, scheduler):
        """Should return guidance for URGENCY."""
        guidance = scheduler.get_followup_prompt_guidance(FollowUpType.URGENCY)

        assert "urgency" in guidance["tone"].lower()

    def test_get_prompt_guidance_all_types(self, scheduler):
        """Should have guidance for all follow-up types."""
        for followup_type in FollowUpType:
            guidance = scheduler.get_followup_prompt_guidance(followup_type)
            assert guidance is not None
            assert "tone" in guidance


class TestFollowUpAction:
    """Tests for FollowUpAction dataclass."""

    def test_action_fields(self):
        """Should hold all required fields."""
        action = FollowUpAction(
            lead_id="+525512345678",
            followup_type=FollowUpType.CHECK_IN,
            scheduled_at=dt.datetime.now(dt.UTC),
            attempt_number=1,
            reason="Test reason",
        )

        assert action.lead_id == "+525512345678"
        assert action.followup_type == FollowUpType.CHECK_IN
        assert action.attempt_number == 1


class TestSingleton:
    """Tests for singleton accessor."""

    def test_get_followup_scheduler_singleton(self):
        """Should return same instance."""
        s1 = get_followup_scheduler()
        s2 = get_followup_scheduler()
        assert s1 is s2
