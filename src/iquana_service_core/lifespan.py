"""Shared startup/shutdown lifecycle for IQUANA AI services."""
import os
from contextlib import asynccontextmanager
from logging import getLogger
from typing import Callable, Optional

from fastapi import FastAPI
from iquana_toolbox.mlflow import MLFlowModelRegistry

from iquana_service_core.discovery import import_models_package, load_plugin_models
from iquana_service_core.registry import collected_models

logger = getLogger(__name__)

# Called once at startup with the registry; registers the service's models.
RegisterModels = Callable[[MLFlowModelRegistry], None]


def build_lifespan(
    registry: MLFlowModelRegistry,
    register_models: Optional[RegisterModels] = None,
    *,
    models_package: Optional[str] = None,
    hf_login: bool = False,
):
    """Build a FastAPI lifespan that optionally logs into HuggingFace and
    registers the service's models in MLflow on startup.

    Two registration paths, either or both:

    * ``models_package`` -- auto-discover and register every class decorated
      with :func:`iquana_service_core.registry.register_model` in that package
      (plus any ``iquana.models`` entry-point plugins). Preferred.
    * ``register_models`` -- a callable that registers the service's models
      itself. Kept for services not (yet) using the decorator.

    Args:
        registry: Shared model registry.
        register_models: Optional callable that registers this service's models.
        models_package: Optional dotted name of the package to auto-discover.
        hf_login: If True, log into HuggingFace using ``HF_ACCESS_TOKEN``
            (needed by services that pull gated weights).
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if hf_login:
            _hf_login()
        if models_package:
            import_models_package(models_package)
            load_plugin_models()
            classes = collected_models()
            logger.info("Registering %d auto-discovered model(s)...", len(classes))
            _register_into_mlflow(registry, classes)
        if register_models is not None:
            logger.info("Registering models via service-provided callable...")
            register_models(registry)
        yield
        if hf_login:
            _hf_logout()
        logger.info("Service shutting down.")

    return lifespan


def _register_into_mlflow(registry: MLFlowModelRegistry, model_classes: list) -> None:
    """Register collected model classes into the MLflow registry.

    MIGRATION BRIDGE: the toolbox registry method was renamed
    ``ensure_models_are_registered`` -> ``register_models`` and services pin
    different toolbox revisions, so resolve whichever exists. Phase 2 moves the
    registry into service-core and deletes this shim.
    """
    register = getattr(registry, "register_models", None) or getattr(
        registry, "ensure_models_are_registered", None
    )
    if register is None:
        raise AttributeError(
            "Registry exposes neither 'register_models' nor "
            "'ensure_models_are_registered'."
        )
    register(model_classes)


def _hf_login() -> None:
    token = os.getenv("HF_ACCESS_TOKEN")
    if not token:
        logger.warning("HF_ACCESS_TOKEN not set; skipping HuggingFace login.")
        return
    try:
        from huggingface_hub import login, whoami

        login(token=token)
        logger.info("Logged into HuggingFace as: %s", whoami().get("name"))
    except Exception as e:
        logger.warning("HuggingFace login failed: %s", e)


def _hf_logout() -> None:
    try:
        from huggingface_hub import logout

        logout()
    except Exception as e:
        logger.debug("HuggingFace logout failed: %s", e)
