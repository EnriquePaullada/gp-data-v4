"""
Health and Readiness Endpoints

Kubernetes-compatible health probes for load balancers and orchestration.
"""
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from loguru import logger

from src.repositories import db_manager

router = APIRouter(tags=["Health"])

# API version - single source of truth
API_VERSION = "1.3.0"


@router.get("/health")
async def health_check():
    """
    Basic health check endpoint.

    Returns 200 if service is running.
    Used by load balancers and monitoring systems.
    """
    return {
        "status": "healthy",
        "service": "gp-data-v4",
        "version": API_VERSION
    }


@router.get("/ready")
async def readiness_check(request: Request):
    """
    Readiness probe - checks if service can handle requests.

    Verifies:
    - MongoDB connection is active
    - Orchestrator is initialized

    Returns 200 if ready, 503 if not ready.
    """
    try:
        # Check orchestrator is initialized
        orchestrator = request.app.state.orchestrator
        if not orchestrator.lead_repo or not orchestrator.message_repo:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "reason": "Repositories not initialized"
                }
            )

        # Check MongoDB connection
        await db_manager.client.admin.command("ping")

        return {
            "status": "ready",
            "mongodb": "connected",
            "orchestrator": "initialized"
        }

    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "reason": str(e)
            }
        )


@router.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "GP Data v4 API",
        "version": API_VERSION,
        "endpoints": {
            "health": "/health",
            "ready": "/ready",
            "metrics": "/metrics",
            "twilio_webhook": "/webhooks/twilio (POST)",
            "queue_metrics": "/metrics/queue"
        }
    }
