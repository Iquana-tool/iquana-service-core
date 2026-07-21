"""Shared model-registry routes for IQUANA AI services.

Every service exposes the identical surface:

  GET  /models/all                     -> all models for this service's task
  GET  /models/all/available           -> task models with status == "ready"
  GET  /models/{key}                   -> info for one model
  GET  /annotation_session/models/{key}/preload  -> warm a model into cache

The only thing that differs per service is the ``task`` tag, so this is a
factory parameterized by registry + task.
"""
from logging import getLogger
from typing import Tuple

from fastapi import APIRouter
from iquana_toolbox.mlflow import MLFlowModelRegistry

logger = getLogger(__name__)


def _model_infos_via_tags(registry: MLFlowModelRegistry, tags: dict):
    """Look up models by tags across toolbox versions.

    MIGRATION BRIDGE: the toolbox renamed ``get_models_via_tags`` ->
    ``get_model_infos_via_tags``. Services are mid-migration and pin different
    toolbox revisions. Once every repo pins the canonical (renamed) toolbox,
    delete this helper and call ``registry.get_model_infos_via_tags`` directly.
    """
    getter = getattr(registry, "get_model_infos_via_tags", None) or getattr(
        registry, "get_models_via_tags"
    )
    return getter(tags=tags)


def build_model_routers(registry: MLFlowModelRegistry, task: str) -> Tuple[APIRouter, APIRouter]:
    """Build the (public, session) model routers for a service.

    Args:
        registry: The shared MLflow-backed model registry.
        task: The ``task`` tag identifying this service's models
            (e.g. ``"instance-suggestion"``, ``"prompted-segmentation"``).

    Returns:
        ``(router, session_router)`` — mount both on the app.
    """
    router = APIRouter()
    session_router = APIRouter(prefix="/annotation_session", tags=["annotation_session"])

    @router.get("/models/all", tags=["models"])
    async def list_models():
        """List all models registered for this service's task."""
        models = _model_infos_via_tags(registry, {"task": task})
        return {
            "success": True,
            "message": f"Retrieved {len(models)} models.",
            "result": models,
        }

    @router.get("/models/all/available", tags=["models"])
    async def list_available_models():
        """List task models that are ready to serve."""
        models = _model_infos_via_tags(registry, {"task": task, "status": "ready"})
        return {
            "success": True,
            "message": f"Retrieved {len(models)} available models.",
            "result": models,
        }

    @router.get("/models/{model_registry_key}", tags=["models"])
    async def get_model(model_registry_key: str):
        """Return registry info for a single model."""
        return {
            "success": True,
            "message": "Retrieved model information.",
            "result": registry.get_model_info(model_registry_key),
        }

    @session_router.get("/models/{model_registry_key}/preload", tags=["models"])
    async def preload_model(model_registry_key: str, user_id: str):
        """Warm a model into the registry cache at the start of a session.

        Models load lazily on first use; this is a convenience to avoid a cold
        first inference. ``user_id`` is accepted for symmetry with the session
        contract even though the cache is process-global.
        """
        registry.get_model_by_alias(model_registry_key, "latest")
        return {
            "success": True,
            "message": f"Preloaded model '{model_registry_key}'.",
        }

    return router, session_router
