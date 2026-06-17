"""Shared FastAPI runtime for IQUANA AI segmentation services."""
from iquana_service_core.app import create_service_app
from iquana_service_core.lifespan import build_lifespan
from iquana_service_core.routers.health import build_health_router
from iquana_service_core.routers.models import build_model_routers

__all__ = [
    "create_service_app",
    "build_lifespan",
    "build_health_router",
    "build_model_routers",
]
