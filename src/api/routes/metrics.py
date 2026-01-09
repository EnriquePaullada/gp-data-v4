"""
Metrics Endpoints

Prometheus-compatible metrics and queue statistics for observability.
"""
from fastapi import APIRouter, Request
from fastapi.responses import Response, JSONResponse
from loguru import logger

from src.message_queue import InMemoryQueue
from src.utils.metrics import metrics

router = APIRouter(tags=["Metrics"])


@router.get("/metrics")
async def prometheus_metrics(request: Request):
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus text exposition format for scraping.
    Includes:
    - Request counts and latencies
    - Queue depth and processing stats
    - Agent invocation counts and durations
    - Token usage (input/output) by agent
    - Cost tracking (USD) by agent
    - Rate limiting and security events

    Content-Type: text/plain; version=0.0.4; charset=utf-8
    """
    try:
        # Update queue gauges from current state
        queue: InMemoryQueue = request.app.state.queue
        queue_stats = await queue.get_metrics()

        metrics.queue_pending.set(queue_stats.pending)
        metrics.queue_processing.set(queue_stats.processing)
        metrics.queue_failed.set(queue_stats.failed)

        # Export all metrics in Prometheus format
        output = metrics.export()

        return Response(
            content=output,
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )

    except Exception as e:
        logger.error(f"Failed to export metrics: {e}", exc_info=True)
        return Response(
            content=f"# Error exporting metrics: {e}\n",
            media_type="text/plain",
            status_code=500
        )


@router.get("/metrics/queue")
async def queue_metrics(request: Request):
    """
    Get message queue metrics.

    Returns statistics about queue performance:
    - Pending messages
    - Messages being processed
    - Completed messages
    - Failed messages
    - Dead letter queue size
    - Average processing time
    - Error rate

    Returns:
        Queue metrics as JSON
    """
    try:
        queue: InMemoryQueue = request.app.state.queue
        queue_stats = await queue.get_metrics()

        return {
            "status": "ok",
            "metrics": queue_stats.model_dump()
        }

    except Exception as e:
        logger.error(f"Failed to get queue metrics: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e)
            }
        )
