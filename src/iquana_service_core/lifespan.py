"""Shared startup/shutdown lifecycle for IQUANA AI services."""
import os
from contextlib import asynccontextmanager
from logging import getLogger
from typing import Callable

from fastapi import FastAPI
from iquana_toolbox.mlflow import MLFlowModelRegistry

logger = getLogger(__name__)

# Called once at startup with the registry; registers the service's models.
RegisterModels = Callable[[MLFlowModelRegistry], None]


def build_lifespan(
    registry: MLFlowModelRegistry,
    register_models: RegisterModels,
    *,
    hf_login: bool = False,
):
    """Build a FastAPI lifespan that optionally logs into HuggingFace and
    registers the service's models in MLflow on startup.

    Args:
        registry: Shared model registry.
        register_models: Callable that registers this service's models.
        hf_login: If True, log into HuggingFace using ``HF_ACCESS_TOKEN``
            (needed by services that pull gated weights).
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if hf_login:
            _hf_login()
        logger.info("Registering models in the registry...")
        register_models(registry)
        yield
        if hf_login:
            _hf_logout()
        logger.info("Service shutting down.")

    return lifespan


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
