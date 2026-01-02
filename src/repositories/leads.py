"""
Lead Repository
Lead-specific persistence and query operations.
"""
from typing import Optional, List
from motor.motor_asyncio import AsyncIOMotorDatabase
import datetime as dt

from .base import BaseRepository
from ..models.lead import Lead, SalesStage
from ..utils.observability import logger


class LeadRepository(BaseRepository[Lead]):
    """
    Repository for Lead persistence and business queries.
    Extends BaseRepository with Lead-specific operations.
    """

    def __init__(self, database: AsyncIOMotorDatabase):
        """Initialize Lead repository with database connection."""
        super().__init__(database, "leads", Lead)

    async def get_by_phone(self, phone_number: str) -> Optional[Lead]:
        """
        Retrieve a Lead by their E.164 phone number.

        Args:
            phone_number: E.164 formatted phone number (e.g., "+5215538899800")

        Returns:
            Lead instance or None if not found
        """
        return await self.find_one({"lead_id": phone_number})

    async def get_or_create(self, phone_number: str, full_name: Optional[str] = None) -> Lead:
        """
        Get existing lead or create new one if not found.
        Idempotent operation for conversation initialization.

        Args:
            phone_number: E.164 formatted phone number
            full_name: Optional lead name for new leads

        Returns:
            Lead instance (existing or newly created)
        """
        existing_lead = await self.get_by_phone(phone_number)

        if existing_lead:
            logger.debug(f"Found existing lead: {phone_number}")
            return existing_lead

        # Create new lead
        new_lead = Lead(
            lead_id=phone_number,
            full_name=full_name,
            current_stage=SalesStage.NEW,
            last_interaction_at=dt.datetime.now(dt.UTC)
        )

        created_lead = await self.create(new_lead)
        logger.info(f"Created new lead: {phone_number}", extra={"lead_id": phone_number})

        return created_lead

    async def update_stage(self, lead_id: str, new_stage: SalesStage) -> bool:
        """
        Update a lead's sales pipeline stage.

        Args:
            lead_id: Lead's phone number
            new_stage: Target SalesStage

        Returns:
            True if updated, False if lead not found
        """
        result = await self.collection.update_one(
            {"lead_id": lead_id},
            {
                "$set": {
                    "current_stage": new_stage.value,
                    "updated_at": dt.datetime.now(dt.UTC)
                }
            }
        )

        if result.modified_count > 0:
            logger.info(
                f"Updated lead stage: {lead_id} -> {new_stage}",
                extra={"lead_id": lead_id, "new_stage": new_stage.value}
            )
            return True

        return False

    async def get_leads_by_stage(
        self,
        stage: SalesStage,
        limit: int = 100,
        skip: int = 0
    ) -> List[Lead]:
        """
        Retrieve all leads in a specific pipeline stage.

        Args:
            stage: Target SalesStage to filter by
            limit: Maximum number of leads to return
            skip: Number of leads to skip (pagination)

        Returns:
            List of Lead instances
        """
        return await self.find_many(
            filter_dict={"current_stage": stage.value},
            limit=limit,
            skip=skip,
            sort=[("last_interaction_at", -1)]  # Most recent first
        )

    async def get_leads_needing_followup(
        self,
        before: Optional[dt.datetime] = None,
        limit: int = 100
    ) -> List[Lead]:
        """
        Retrieve leads that need follow-up (scheduled or overdue).

        Args:
            before: Get leads with follow-up scheduled before this time (default: now)
            limit: Maximum number of leads to return

        Returns:
            List of Lead instances sorted by follow-up time
        """
        before = before or dt.datetime.now(dt.UTC)

        return await self.find_many(
            filter_dict={
                "next_followup_at": {"$lte": before, "$ne": None}
            },
            limit=limit,
            sort=[("next_followup_at", 1)]  # Earliest first
        )

    async def get_stale_leads(
        self,
        days_inactive: int = 30,
        exclude_stages: Optional[List[SalesStage]] = None,
        limit: int = 100
    ) -> List[Lead]:
        """
        Find leads with no recent activity (potential re-engagement targets).

        Args:
            days_inactive: Number of days without interaction
            exclude_stages: Stages to exclude (e.g., WON, LOST)
            limit: Maximum number of leads to return

        Returns:
            List of Lead instances sorted by last interaction
        """
        exclude_stages = exclude_stages or [SalesStage.WON, SalesStage.LOST]
        cutoff_date = dt.datetime.now(dt.UTC) - dt.timedelta(days=days_inactive)

        filter_dict = {
            "last_interaction_at": {"$lt": cutoff_date},
            "current_stage": {"$nin": [stage.value for stage in exclude_stages]}
        }

        return await self.find_many(
            filter_dict=filter_dict,
            limit=limit,
            sort=[("last_interaction_at", 1)]  # Oldest first
        )

    async def get_high_intent_leads(
        self,
        min_message_count: int = 5,
        stages: Optional[List[SalesStage]] = None,
        limit: int = 100
    ) -> List[Lead]:
        """
        Retrieve engaged leads with high message count (proxy for intent).

        Args:
            min_message_count: Minimum number of messages exchanged
            stages: Filter by specific stages (default: DISCOVERY, QUALIFIED, DEMO_SCHEDULED)
            limit: Maximum number of leads to return

        Returns:
            List of Lead instances sorted by message count
        """
        stages = stages or [
            SalesStage.DISCOVERY,
            SalesStage.QUALIFIED,
            SalesStage.DEMO_SCHEDULED
        ]

        filter_dict = {
            "message_count": {"$gte": min_message_count},
            "current_stage": {"$in": [stage.value for stage in stages]}
        }

        return await self.find_many(
            filter_dict=filter_dict,
            limit=limit,
            sort=[("message_count", -1)]  # Most engaged first
        )

    async def count_by_stage(self) -> dict[SalesStage, int]:
        """
        Get lead count for each pipeline stage (analytics).

        Returns:
            Dictionary mapping SalesStage to count
        """
        pipeline = [
            {
                "$group": {
                    "_id": "$current_stage",
                    "count": {"$sum": 1}
                }
            }
        ]

        results = await self.collection.aggregate(pipeline).to_list(length=None)

        # Initialize all stages with 0
        stage_counts = {stage: 0 for stage in SalesStage}

        # Populate with actual counts
        for result in results:
            stage_value = result["_id"]
            try:
                stage = SalesStage(stage_value)
                stage_counts[stage] = result["count"]
            except ValueError:
                logger.warning(f"Unknown stage value in database: {stage_value}")

        return stage_counts

    async def save(self, lead: Lead) -> Lead:
        """
        Upsert operation: Create if new, update if exists.
        Primary method for orchestrator integration.

        Args:
            lead: Lead instance to persist

        Returns:
            Persisted Lead instance with updated timestamps
        """
        if lead.id:
            # Existing lead - update
            return await self.update(lead)
        else:
            # Check if lead exists by phone number
            existing = await self.get_by_phone(lead.lead_id)
            if existing:
                # Update existing lead's ID and update
                lead.id = existing.id
                return await self.update(lead)
            else:
                # Create new lead
                return await self.create(lead)
