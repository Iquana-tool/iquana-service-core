"""Shared health endpoint for IQUANA AI services.

The base health check has no heavy dependencies (no torch). Services that want
to report device/runtime details inject an ``extra`` callable so service-core
stays dependency-light.
"""
from logging import getLogger
from typing import Callable, Optional

from fastapi import APIRouter

logger = getLogger(__name__)

# A callable returning a JSON-serializable dict merged into the health payload.
HealthExtra = Callable[[], dict]


def build_health_router(extra: Optional[HealthExtra] = None) -> APIRouter:
    """Build the ``/health`` router.

    Args:
        extra: Optional callable returning extra fields to merge into the
            response (e.g. a torch device probe defined in the service).
    """
    router = APIRouter()

    @router.get("/health", tags=["health"])
    async def health_check():
        payload = {"status": "ok"}
        if extra is not None:
            try:
                payload.update(extra())
            except Exception as e:  # never let diagnostics break the health check
                logger.warning("Health extra probe failed: %s", e)
                payload["extra_error"] = str(e)
        return payload

    return router
