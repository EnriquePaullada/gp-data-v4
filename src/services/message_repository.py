"""
Message Repository
Time-series message storage and conversation history retrieval.
"""
from typing import List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
import datetime as dt

from ..repositories.base import BaseRepository
from ..models.message import Message, MessageRole
from ..utils.observability import logger


class MessageRepository(BaseRepository[Message]):
    """
    Repository for Message persistence and conversation queries.
    Optimized for time-series data and conversation history retrieval.
    """

    def __init__(self, database: AsyncIOMotorDatabase):
        """Initialize Message repository with database connection."""
        super().__init__(database, "messages", Message)

    async def get_conversation_history(
        self,
        lead_id: str,
        limit: int = 100,
        before: Optional[dt.datetime] = None
    ) -> List[Message]:
        """
        Retrieve full conversation history for a lead.

        Args:
            lead_id: Lead's phone number
            limit: Maximum number of messages to return
            before: Get messages before this timestamp (pagination)

        Returns:
            List of Message instances sorted by timestamp (oldest first)
        """
        filter_dict = {"lead_id": lead_id}

        if before:
            filter_dict["timestamp"] = {"$lt": before}

        messages = await self.find_many(
            filter_dict=filter_dict,
            limit=limit,
            sort=[("timestamp", 1)]  # Chronological order
        )

        logger.debug(
            f"Retrieved conversation history for {lead_id}",
            extra={"lead_id": lead_id, "message_count": len(messages)}
        )

        return messages

    async def get_recent_messages(
        self,
        lead_id: str,
        limit: int = 20
    ) -> List[Message]:
        """
        Get the most recent N messages for a lead.
        Used to populate Lead.recent_history working memory.

        Args:
            lead_id: Lead's phone number
            limit: Number of recent messages (default: 20 for working memory)

        Returns:
            List of Message instances sorted by timestamp (oldest first)
        """
        messages = await self.find_many(
            filter_dict={"lead_id": lead_id},
            limit=limit,
            sort=[("timestamp", -1)]  # Get latest first
        )

        # Reverse to get chronological order
        return list(reversed(messages))

    async def save_messages(self, messages: List[Message]) -> List[Message]:
        """
        Persist multiple messages in a single operation.
        Primary method for orchestrator integration.

        Args:
            messages: List of Message instances to persist

        Returns:
            List of persisted Message instances
        """
        if not messages:
            return []

        # Filter out messages that already have an ID (already persisted)
        new_messages = [msg for msg in messages if not msg.id]

        if not new_messages:
            logger.debug("All messages already persisted, skipping save")
            return messages

        persisted = await self.bulk_create(new_messages)

        logger.info(
            f"Persisted {len(persisted)} messages",
            extra={"message_count": len(persisted)}
        )

        return persisted

    async def get_messages_by_role(
        self,
        lead_id: str,
        role: MessageRole,
        limit: int = 50
    ) -> List[Message]:
        """
        Retrieve messages filtered by sender role.

        Args:
            lead_id: Lead's phone number
            role: MessageRole to filter by
            limit: Maximum number of messages to return

        Returns:
            List of Message instances
        """
        return await self.find_many(
            filter_dict={"lead_id": lead_id, "role": role.value},
            limit=limit,
            sort=[("timestamp", -1)]
        )

    async def get_messages_in_timerange(
        self,
        lead_id: str,
        start_time: dt.datetime,
        end_time: dt.datetime
    ) -> List[Message]:
        """
        Retrieve messages within a specific time window.

        Args:
            lead_id: Lead's phone number
            start_time: Start of time range (inclusive)
            end_time: End of time range (inclusive)

        Returns:
            List of Message instances sorted chronologically
        """
        return await self.find_many(
            filter_dict={
                "lead_id": lead_id,
                "timestamp": {"$gte": start_time, "$lte": end_time}
            },
            limit=1000,  # Higher limit for analytics queries
            sort=[("timestamp", 1)]
        )

    async def count_messages_for_lead(self, lead_id: str) -> int:
        """
        Get total message count for a lead (all time).

        Args:
            lead_id: Lead's phone number

        Returns:
            Total number of messages
        """
        return await self.count({"lead_id": lead_id})

    async def get_total_tokens_for_lead(self, lead_id: str) -> int:
        """
        Calculate total token usage for a lead's conversation.
        Used for cost attribution and analytics.

        Args:
            lead_id: Lead's phone number

        Returns:
            Total tokens consumed
        """
        pipeline = [
            {"$match": {"lead_id": lead_id}},
            {"$group": {"_id": None, "total_tokens": {"$sum": "$tokens"}}}
        ]

        result = await self.collection.aggregate(pipeline).to_list(length=1)

        if result:
            return result[0]["total_tokens"]

        return 0

    async def get_average_response_time(
        self,
        lead_id: str,
        hours: int = 24
    ) -> Optional[float]:
        """
        Calculate average time between lead message and assistant response.
        Business intelligence metric for responsiveness.

        Args:
            lead_id: Lead's phone number
            hours: Time window to analyze (recent hours)

        Returns:
            Average response time in seconds, or None if insufficient data
        """
        cutoff_time = dt.datetime.now(dt.UTC) - dt.timedelta(hours=hours)

        # Get recent messages sorted by timestamp
        messages = await self.find_many(
            filter_dict={
                "lead_id": lead_id,
                "timestamp": {"$gte": cutoff_time}
            },
            limit=200,
            sort=[("timestamp", 1)]
        )

        if len(messages) < 2:
            return None

        response_times = []
        last_lead_message_time: Optional[dt.datetime] = None

        for msg in messages:
            if msg.role == MessageRole.LEAD:
                last_lead_message_time = msg.timestamp
            elif msg.role == MessageRole.ASSISTANT and last_lead_message_time:
                time_diff = (msg.timestamp - last_lead_message_time).total_seconds()
                response_times.append(time_diff)
                last_lead_message_time = None

        if not response_times:
            return None

        return sum(response_times) / len(response_times)

    async def delete_messages_for_lead(self, lead_id: str) -> int:
        """
        Delete all messages for a lead.
        Use cautiously - intended for GDPR compliance or data cleanup.

        Args:
            lead_id: Lead's phone number

        Returns:
            Number of messages deleted
        """
        result = await self.collection.delete_many({"lead_id": lead_id})

        logger.warning(
            f"Deleted all messages for lead {lead_id}",
            extra={"lead_id": lead_id, "deleted_count": result.deleted_count}
        )

        return result.deleted_count
