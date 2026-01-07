"""
Tests for Rate Limiter.
"""

import pytest
import asyncio
from datetime import datetime, timedelta, timezone

from src.utils.rate_limiter import InMemoryRateLimiter, RateLimitResult


class TestInMemoryRateLimiter:
    """Test suite for InMemoryRateLimiter."""

    @pytest.fixture
    def rate_limiter(self):
        """Create rate limiter with test-friendly settings."""
        return InMemoryRateLimiter(
            max_requests=5,
            window_seconds=60,  # 1 minute
            spike_threshold=3,
            spike_window_seconds=10,
            ban_duration_seconds=30
        )

    @pytest.mark.asyncio
    async def test_first_request_allowed(self, rate_limiter):
        """Test that first request is always allowed."""
        result = await rate_limiter.check_rate_limit("+5215538899800")

        assert result.allowed is True
        assert result.remaining == 4  # 5 max - 1 used
        assert result.retry_after is None

    @pytest.mark.asyncio
    async def test_requests_within_limit(self, rate_limiter):
        """Test multiple requests within limit."""
        lead_id = "+5215538899800"

        # Send 5 requests (max limit)
        for i in range(5):
            result = await rate_limiter.check_rate_limit(lead_id)
            assert result.allowed is True
            assert result.remaining == 5 - (i + 1)

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, rate_limiter):
        """Test that 6th request is blocked."""
        lead_id = "+5215538899800"

        # Send 5 requests (max limit)
        for _ in range(5):
            await rate_limiter.check_rate_limit(lead_id)

        # 6th request should be blocked
        result = await rate_limiter.check_rate_limit(lead_id)

        assert result.allowed is False
        assert result.remaining == 0
        assert result.retry_after is not None
        assert result.retry_after > 0
        assert "Rate limit exceeded" in result.reason

    @pytest.mark.asyncio
    async def test_rate_limit_per_lead(self, rate_limiter):
        """Test that rate limits are per-lead."""
        lead1 = "+5215538899800"
        lead2 = "+5215538899801"

        # Exhaust limit for lead1
        for _ in range(5):
            await rate_limiter.check_rate_limit(lead1)

        # lead1 should be blocked
        result1 = await rate_limiter.check_rate_limit(lead1)
        assert result1.allowed is False

        # lead2 should still be allowed
        result2 = await rate_limiter.check_rate_limit(lead2)
        assert result2.allowed is True

    @pytest.mark.asyncio
    async def test_ban_lead(self, rate_limiter):
        """Test banning a lead."""
        lead_id = "+5215538899800"

        # Ban the lead
        await rate_limiter.ban_lead(lead_id, 60, "Spam detected")

        # Check if banned
        is_banned = await rate_limiter.is_banned(lead_id)
        assert is_banned is True

    @pytest.mark.asyncio
    async def test_banned_lead_gets_ban_info(self, rate_limiter):
        """Test getting ban information."""
        lead_id = "+5215538899800"

        await rate_limiter.ban_lead(lead_id, 60, "Abusive behavior")

        ban_info = await rate_limiter.get_ban_info(lead_id)
        assert ban_info is not None

        ban_until, reason = ban_info
        assert reason == "Abusive behavior"
        assert ban_until > datetime.now(timezone.utc)

    @pytest.mark.asyncio
    async def test_ban_expires(self, rate_limiter):
        """Test that bans expire after duration."""
        lead_id = "+5215538899800"

        # Ban for 1 second
        await rate_limiter.ban_lead(lead_id, 1, "Test ban")

        # Should be banned immediately
        assert await rate_limiter.is_banned(lead_id) is True

        # Wait for ban to expire
        await asyncio.sleep(1.2)

        # Should no longer be banned
        assert await rate_limiter.is_banned(lead_id) is False

    @pytest.mark.asyncio
    async def test_spike_detection(self, rate_limiter):
        """Test spike detection."""
        lead_id = "+5215538899800"

        # Send 3 requests quickly (triggers spike threshold)
        for _ in range(3):
            await rate_limiter.check_rate_limit(lead_id)

        # Should detect spike
        spike_detected = await rate_limiter.detect_spike(lead_id)
        assert spike_detected is True

    @pytest.mark.asyncio
    async def test_spike_not_detected_if_below_threshold(self, rate_limiter):
        """Test that spike is not detected below threshold."""
        lead_id = "+5215538899800"

        # Send only 2 requests (below threshold of 3)
        for _ in range(2):
            await rate_limiter.check_rate_limit(lead_id)

        # Should NOT detect spike
        spike_detected = await rate_limiter.detect_spike(lead_id)
        assert spike_detected is False

    @pytest.mark.asyncio
    async def test_spike_window(self, rate_limiter):
        """Test that spike detection uses time window."""
        lead_id = "+5215538899800"

        # Manually add old requests
        old_time = datetime.now(timezone.utc) - timedelta(seconds=20)
        rate_limiter._requests[lead_id] = [old_time, old_time]

        # Add 1 recent request
        await rate_limiter.check_rate_limit(lead_id)

        # Should NOT detect spike (only 1 in recent window)
        spike_detected = await rate_limiter.detect_spike(lead_id)
        assert spike_detected is False

    @pytest.mark.asyncio
    async def test_multiple_leads_concurrent(self, rate_limiter):
        """Test concurrent requests from multiple leads."""
        leads = [f"+521553889980{i}" for i in range(10)]

        # Process 3 requests per lead concurrently
        async def make_requests(lead_id):
            results = []
            for _ in range(3):
                result = await rate_limiter.check_rate_limit(lead_id)
                results.append(result)
            return results

        # Run concurrently
        results_per_lead = await asyncio.gather(*[make_requests(lead) for lead in leads])

        # All requests should be allowed (3 < 5 limit)
        for results in results_per_lead:
            assert all(r.allowed for r in results)

    @pytest.mark.asyncio
    async def test_rate_limit_headers(self, rate_limiter):
        """Test that rate limit result includes correct header values."""
        lead_id = "+5215538899800"

        # Make first request
        result = await rate_limiter.check_rate_limit(lead_id)

        assert result.allowed is True
        assert result.remaining == 4
        assert result.reset_at > datetime.now(timezone.utc)

    @pytest.mark.asyncio
    async def test_clean_up_old_requests(self, rate_limiter):
        """Test that old requests are cleaned up."""
        lead_id = "+5215538899800"

        # Add old request manually
        old_time = datetime.now(timezone.utc) - timedelta(seconds=120)
        rate_limiter._requests[lead_id] = [old_time]

        # Make new request
        result = await rate_limiter.check_rate_limit(lead_id)

        # Old request should be cleaned up
        # Only the new request should remain
        assert len(rate_limiter._requests[lead_id]) == 1
        assert result.remaining == 4  # Should count as first request

    @pytest.mark.asyncio
    async def test_ban_overrides_rate_limit(self, rate_limiter):
        """Test that banned status is independent of rate limit."""
        lead_id = "+5215538899800"

        # Ban the lead
        await rate_limiter.ban_lead(lead_id, 60, "Spam")

        # Even with rate limit available, lead should be banned
        assert await rate_limiter.is_banned(lead_id) is True

        # Rate limit check should still work (but ban check should happen first in production)
        result = await rate_limiter.check_rate_limit(lead_id)
        assert result.allowed is True  # Rate limit itself allows, but ban should prevent

    @pytest.mark.asyncio
    async def test_sliding_window_behavior(self, rate_limiter):
        """Test sliding window algorithm."""
        lead_id = "+5215538899800"

        # Make 5 requests (max limit)
        for _ in range(5):
            await rate_limiter.check_rate_limit(lead_id)

        # 6th request should fail
        result = await rate_limiter.check_rate_limit(lead_id)
        assert result.allowed is False

        # Wait for 1 second
        await asyncio.sleep(1)

        # Window hasn't moved enough yet (window is 60 seconds)
        result = await rate_limiter.check_rate_limit(lead_id)
        assert result.allowed is False

    @pytest.mark.asyncio
    async def test_different_leads_independent(self, rate_limiter):
        """Test that different leads have independent limits."""
        lead1 = "+5215538899800"
        lead2 = "+5215538899801"
        lead3 = "+5215538899802"

        # Exhaust limits for lead1 and lead2
        for _ in range(5):
            await rate_limiter.check_rate_limit(lead1)
            await rate_limiter.check_rate_limit(lead2)

        # Both should be rate limited
        assert (await rate_limiter.check_rate_limit(lead1)).allowed is False
        assert (await rate_limiter.check_rate_limit(lead2)).allowed is False

        # lead3 should still have full quota
        result = await rate_limiter.check_rate_limit(lead3)
        assert result.allowed is True
        assert result.remaining == 4
