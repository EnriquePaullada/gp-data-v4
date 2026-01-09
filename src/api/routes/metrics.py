"""
Metrics Endpoints

Queue statistics for observability.
Prometheus metrics endpoint will be added in a separate commit.
"""
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from loguru import logger

from src.message_queue import InMemoryQueue

router = APIRouter(tags=["Metrics"])


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
