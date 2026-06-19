"""FastAPI app factory shared by all IQUANA AI segmentation services.

A service becomes, in full:

    from iquana_service_core import create_service_app
    from app.state import MODEL_REGISTRY
    from app.routes.inference import router, session_router

    app = create_service_app(
        title="Instance Discovery API",
        task="instance-discovery",
        registry=MODEL_REGISTRY,
        models_package="models",   # auto-discovers @register_model classes
        inference_routers=[router, session_router],
        hf_login=True,
    )

Everything else (health, /models/*, the preload route, CORS, lifespan) is
provided here so it lives in exactly one place.
"""
import os
from logging import getLogger
from typing import Iterable, Optional, Sequence

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from iquana_toolbox.mlflow import MLFlowModelRegistry

from iquana_service_core.lifespan import RegisterModels, build_lifespan
from iquana_service_core.routers.health import HealthExtra, build_health_router
from iquana_service_core.routers.models import build_model_routers

logger = getLogger(__name__)


def create_service_app(
    *,
    title: str,
    task: str,
    registry: MLFlowModelRegistry,
    inference_routers: Sequence[APIRouter],
    models_package: Optional[str] = None,
    register_models: Optional[RegisterModels] = None,
    description: str = "",
    version: str = "0.1.0",
    hf_login: bool = False,
    health_extra: Optional[HealthExtra] = None,
    allowed_origins: Optional[Iterable[str]] = None,
    root_path: str = "",
) -> FastAPI:
    """Build a fully-wired FastAPI app for an AI service.

    Model registration takes one of two forms (at least one is required):

    * ``models_package`` -- auto-discover every class decorated with
      :func:`iquana_service_core.registry.register_model` in that package
      (e.g. ``"models"``). Preferred: adding a model is just dropping a file.
    * ``register_models`` -- a callable the service supplies to register its
      models itself. Kept for services not yet using the decorator.

    Args:
        title: OpenAPI title.
        task: The service's ``task`` tag, used to filter the model registry.
        registry: Shared MLflow-backed model registry.
        inference_routers: Service-specific routers (the only thing a service
            must still author itself).
        models_package: Dotted package to auto-discover models from.
        register_models: Callable invoked at startup to register models.
        description: OpenAPI description.
        version: OpenAPI version.
        hf_login: Log into HuggingFace on startup (for gated weights).
        health_extra: Optional callable returning extra ``/health`` fields.
        allowed_origins: CORS origins; defaults to ``$ALLOWED_ORIGINS`` (comma
            separated) or ``http://localhost:8000``.
        root_path: ASGI root path when served behind a proxy prefix.
    """
    if models_package is None and register_models is None:
        raise ValueError(
            "create_service_app requires either 'models_package' (auto-discovery) "
            "or a 'register_models' callable."
        )

    if allowed_origins is None:
        allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000").split(",")
    allowed_origins = [o.strip() for o in allowed_origins if o.strip()]

    app = FastAPI(
        title=title,
        description=description,
        version=version,
        root_path=root_path,
        lifespan=build_lifespan(
            registry,
            register_models,
            models_package=models_package,
            hf_login=hf_login,
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Shared surface.
    app.include_router(build_health_router(health_extra))
    model_router, model_session_router = build_model_routers(registry, task)
    app.include_router(model_router)
    app.include_router(model_session_router)

    # Service-specific surface.
    for router in inference_routers:
        app.include_router(router)

    logger.debug("Created service app '%s' (task=%s)", title, task)
    return app
