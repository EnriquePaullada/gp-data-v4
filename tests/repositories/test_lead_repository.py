"""
Lead Repository Tests
Comprehensive tests for Lead persistence and business queries.
"""
import pytest
import datetime as dt
from typing import List

from src.repositories import db_manager, LeadRepository
from src.models.lead import Lead, SalesStage
from src.models.intelligence import IntelligenceSignal, BANTDimension, ConfidenceScore


pytestmark = pytest.mark.asyncio


@pytest.fixture(scope="function")
async def lead_repo():
    """Provide a clean LeadRepository instance with connected database."""
    await db_manager.connect()
    await db_manager.create_indexes()
    db = db_manager.database

    # Clean up test data before each test
    await db.leads.delete_many({"lead_id": {"$regex": "^\\+test"}})

    repo = LeadRepository(db)
    yield repo

    # Cleanup after test
    await db.leads.delete_many({"lead_id": {"$regex": "^\\+test"}})


@pytest.fixture
def sample_lead() -> Lead:
    """Provide a sample Lead instance for testing."""
    return Lead(
        lead_id="+test5215538899800",
        full_name="Carlos Test",
        current_stage=SalesStage.DISCOVERY
    )


@pytest.fixture
def sample_signal() -> IntelligenceSignal:
    """Provide a sample IntelligenceSignal for testing."""
    return IntelligenceSignal(
        dimension=BANTDimension.BUDGET,
        extracted_value="high",
        confidence=ConfidenceScore(value=0.9, reasoning="Explicit mention of budget"),
        source_message_id="msg_123",
        is_inferred=False,
        raw_evidence="We have $50k budget"
    )


class TestLeadRepositoryBasicCRUD:
    """Test basic CRUD operations."""

    async def test_create_lead(self, lead_repo: LeadRepository, sample_lead: Lead):
        """Should create a lead and return it with _id populated."""
        created_lead = await lead_repo.create(sample_lead)

        assert created_lead.id is not None
        assert created_lead.lead_id == sample_lead.lead_id
        assert created_lead.full_name == sample_lead.full_name
        assert created_lead.created_at is not None
        assert created_lead.updated_at is not None

    async def test_find_by_id(self, lead_repo: LeadRepository, sample_lead: Lead):
        """Should retrieve a lead by MongoDB ObjectId."""
        created_lead = await lead_repo.create(sample_lead)

        found_lead = await lead_repo.find_by_id(created_lead.id)

        assert found_lead is not None
        assert found_lead.id == created_lead.id
        assert found_lead.lead_id == created_lead.lead_id

    async def test_find_by_id_nonexistent(self, lead_repo: LeadRepository):
        """Should return None for non-existent ObjectId."""
        from bson import ObjectId

        fake_id = str(ObjectId())
        result = await lead_repo.find_by_id(fake_id)

        assert result is None

    async def test_update_lead(self, lead_repo: LeadRepository, sample_lead: Lead):
        """Should update an existing lead."""
        created_lead = await lead_repo.create(sample_lead)

        # Modify and update
        created_lead.full_name = "Updated Name"
        created_lead.current_stage = SalesStage.QUALIFIED

        updated_lead = await lead_repo.update(created_lead)

        assert updated_lead.full_name == "Updated Name"
        assert updated_lead.current_stage == SalesStage.QUALIFIED

        # Verify in database
        db_lead = await lead_repo.find_by_id(created_lead.id)
        assert db_lead.full_name == "Updated Name"

    async def test_delete_lead(self, lead_repo: LeadRepository, sample_lead: Lead):
        """Should delete a lead by ObjectId."""
        created_lead = await lead_repo.create(sample_lead)

        deleted = await lead_repo.delete(created_lead.id)
        assert deleted is True

        # Verify deletion
        found = await lead_repo.find_by_id(created_lead.id)
        assert found is None

    async def test_delete_nonexistent_lead(self, lead_repo: LeadRepository):
        """Should return False when deleting non-existent lead."""
        from bson import ObjectId

        fake_id = str(ObjectId())
        result = await lead_repo.delete(fake_id)

        assert result is False


class TestLeadRepositoryBusinessQueries:
    """Test Lead-specific business queries."""

    async def test_get_by_phone(self, lead_repo: LeadRepository, sample_lead: Lead):
        """Should retrieve lead by E.164 phone number."""
        await lead_repo.create(sample_lead)

        found_lead = await lead_repo.get_by_phone(sample_lead.lead_id)

        assert found_lead is not None
        assert found_lead.lead_id == sample_lead.lead_id

    async def test_get_by_phone_nonexistent(self, lead_repo: LeadRepository):
        """Should return None for non-existent phone number."""
        result = await lead_repo.get_by_phone("+test9999999999")

        assert result is None

    async def test_get_or_create_existing(self, lead_repo: LeadRepository, sample_lead: Lead):
        """Should return existing lead without creating duplicate."""
        created_lead = await lead_repo.create(sample_lead)

        result = await lead_repo.get_or_create(sample_lead.lead_id)

        assert result.id == created_lead.id
        assert result.lead_id == created_lead.lead_id

        # Verify no duplicate created
        count = await lead_repo.count({"lead_id": sample_lead.lead_id})
        assert count == 1

    async def test_get_or_create_new(self, lead_repo: LeadRepository):
        """Should create new lead if not found."""
        phone = "+test5219999999999"

        result = await lead_repo.get_or_create(phone, full_name="New Lead")

        assert result.id is not None
        assert result.lead_id == phone
        assert result.full_name == "New Lead"
        assert result.current_stage == SalesStage.NEW

    async def test_update_stage(self, lead_repo: LeadRepository, sample_lead: Lead):
        """Should update lead's sales stage."""
        await lead_repo.create(sample_lead)

        success = await lead_repo.update_stage(
            sample_lead.lead_id,
            SalesStage.DEMO_SCHEDULED
        )

        assert success is True

        # Verify update
        updated_lead = await lead_repo.get_by_phone(sample_lead.lead_id)
        assert updated_lead.current_stage == SalesStage.DEMO_SCHEDULED

    async def test_update_stage_nonexistent(self, lead_repo: LeadRepository):
        """Should return False when updating non-existent lead stage."""
        result = await lead_repo.update_stage(
            "+test9999999999",
            SalesStage.QUALIFIED
        )

        assert result is False


class TestLeadRepositoryAnalyticsQueries:
    """Test analytics and business intelligence queries."""

    async def test_get_leads_by_stage(self, lead_repo: LeadRepository):
        """Should retrieve all leads in a specific stage."""
        # Create leads in different stages
        leads: List[Lead] = [
            Lead(lead_id="+test1", current_stage=SalesStage.DISCOVERY),
            Lead(lead_id="+test2", current_stage=SalesStage.DISCOVERY),
            Lead(lead_id="+test3", current_stage=SalesStage.QUALIFIED),
        ]

        for lead in leads:
            await lead_repo.create(lead)

        # Query by stage
        discovery_leads = await lead_repo.get_leads_by_stage(SalesStage.DISCOVERY)

        assert len(discovery_leads) == 2
        assert all(l.current_stage == SalesStage.DISCOVERY for l in discovery_leads)

    async def test_get_leads_needing_followup(self, lead_repo: LeadRepository):
        """Should retrieve leads with scheduled follow-ups."""
        now = dt.datetime.now(dt.UTC)
        past = now - dt.timedelta(hours=1)
        future = now + dt.timedelta(hours=1)

        # Create leads with different follow-up times
        leads: List[Lead] = [
            Lead(lead_id="+test1", next_followup_at=past),  # Overdue
            Lead(lead_id="+test2", next_followup_at=future),  # Future
            Lead(lead_id="+test3", next_followup_at=None),  # No follow-up
        ]

        for lead in leads:
            await lead_repo.create(lead)

        # Query overdue follow-ups
        overdue_leads = await lead_repo.get_leads_needing_followup(before=now)

        assert len(overdue_leads) == 1
        assert overdue_leads[0].lead_id == "+test1"

    async def test_get_stale_leads(self, lead_repo: LeadRepository):
        """Should find leads with no recent activity."""
        now = dt.datetime.now(dt.UTC)
        old_date = now - dt.timedelta(days=60)
        recent_date = now - dt.timedelta(days=5)

        # Create leads with different activity dates
        leads: List[Lead] = [
            Lead(
                lead_id="+test1",
                last_interaction_at=old_date,
                current_stage=SalesStage.DISCOVERY
            ),
            Lead(
                lead_id="+test2",
                last_interaction_at=recent_date,
                current_stage=SalesStage.DISCOVERY
            ),
            Lead(
                lead_id="+test3",
                last_interaction_at=old_date,
                current_stage=SalesStage.WON  # Should be excluded
            ),
        ]

        for lead in leads:
            await lead_repo.create(lead)

        # Query stale leads (30+ days inactive)
        stale_leads = await lead_repo.get_stale_leads(days_inactive=30)

        assert len(stale_leads) == 1
        assert stale_leads[0].lead_id == "+test1"

    async def test_get_high_intent_leads(self, lead_repo: LeadRepository):
        """Should retrieve engaged leads with high message count."""
        leads: List[Lead] = [
            Lead(
                lead_id="+test1",
                message_count=10,
                current_stage=SalesStage.QUALIFIED
            ),
            Lead(
                lead_id="+test2",
                message_count=2,
                current_stage=SalesStage.QUALIFIED
            ),
            Lead(
                lead_id="+test3",
                message_count=15,
                current_stage=SalesStage.WON  # Wrong stage
            ),
        ]

        for lead in leads:
            await lead_repo.create(lead)

        # Query high-intent leads (5+ messages, specific stages)
        high_intent = await lead_repo.get_high_intent_leads(min_message_count=5)

        assert len(high_intent) == 1
        assert high_intent[0].lead_id == "+test1"

    async def test_count_by_stage(self, lead_repo: LeadRepository):
        """Should return lead count for each pipeline stage."""
        leads: List[Lead] = [
            Lead(lead_id="+test1", current_stage=SalesStage.NEW),
            Lead(lead_id="+test2", current_stage=SalesStage.NEW),
            Lead(lead_id="+test3", current_stage=SalesStage.DISCOVERY),
            Lead(lead_id="+test4", current_stage=SalesStage.WON),
        ]

        for lead in leads:
            await lead_repo.create(lead)

        stage_counts = await lead_repo.count_by_stage()

        assert stage_counts[SalesStage.NEW] == 2
        assert stage_counts[SalesStage.DISCOVERY] == 1
        assert stage_counts[SalesStage.WON] == 1
        assert stage_counts[SalesStage.QUALIFIED] == 0


class TestLeadRepositorySaveUpsert:
    """Test the save() upsert method (orchestrator integration)."""

    async def test_save_creates_new_lead(self, lead_repo: LeadRepository, sample_lead: Lead):
        """Save should create lead if it doesn't exist."""
        saved_lead = await lead_repo.save(sample_lead)

        assert saved_lead.id is not None
        assert saved_lead.lead_id == sample_lead.lead_id

        # Verify in database
        db_lead = await lead_repo.get_by_phone(sample_lead.lead_id)
        assert db_lead is not None

    async def test_save_updates_existing_lead_by_id(
        self,
        lead_repo: LeadRepository,
        sample_lead: Lead
    ):
        """Save should update lead if ID is populated."""
        created_lead = await lead_repo.create(sample_lead)

        # Modify and save
        created_lead.full_name = "Updated via Save"
        saved_lead = await lead_repo.save(created_lead)

        assert saved_lead.full_name == "Updated via Save"

        # Verify only one lead exists
        count = await lead_repo.count({"lead_id": sample_lead.lead_id})
        assert count == 1

    async def test_save_updates_existing_lead_by_phone(
        self,
        lead_repo: LeadRepository,
        sample_lead: Lead
    ):
        """Save should update lead if phone number matches (no ID)."""
        await lead_repo.create(sample_lead)

        # Create new Lead object with same phone, no ID
        updated_lead = Lead(
            lead_id=sample_lead.lead_id,
            full_name="Updated without ID"
        )

        saved_lead = await lead_repo.save(updated_lead)

        assert saved_lead.full_name == "Updated without ID"

        # Verify only one lead exists
        count = await lead_repo.count({"lead_id": sample_lead.lead_id})
        assert count == 1

    async def test_save_preserves_signals(
        self,
        lead_repo: LeadRepository,
        sample_lead: Lead,
        sample_signal: IntelligenceSignal
    ):
        """Save should preserve intelligence signals across updates."""
        sample_lead.add_signal(sample_signal)

        created_lead = await lead_repo.create(sample_lead)
        assert len(created_lead.signals) == 1

        # Update via save
        created_lead.full_name = "Updated"
        saved_lead = await lead_repo.save(created_lead)

        assert len(saved_lead.signals) == 1
        assert saved_lead.signals[0].dimension == BANTDimension.BUDGET


@pytest.fixture(scope="module", autouse=True)
async def cleanup_database():
    """Clean up test database after all tests."""
    yield
    await db_manager.connect()
    await db_manager.database.leads.delete_many({"lead_id": {"$regex": "^\\+test"}})
    await db_manager.disconnect()
