"""
API Routes

Modular route definitions for the GP Data v4 API.
"""
from src.api.routes.health import router as health_router
from src.api.routes.webhooks import router as webhooks_router
from src.api.routes.metrics import router as metrics_router

__all__ = [
    "health_router",
    "webhooks_router",
    "metrics_router",
]
