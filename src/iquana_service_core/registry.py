"""Decorator-based model collection for IQUANA services.

Add a model by dropping a class in your service's ``models`` package and
decorating it -- no hand-maintained list of "models to register":

    from iquana_service_core import register_model
    from iquana_toolbox.ai.base_classes import InstanceSegmentationModel

    @register_model
    class Mask2Former(InstanceSegmentationModel):
        ...

``create_service_app(models_package="models", ...)`` auto-imports that package
at startup (see :mod:`iquana_service_core.discovery`), which runs the
decorators below, then registers everything collected here into MLflow.

The catalog is process-global on purpose: a service runs one app per process,
and tests can reset it with :func:`clear_catalog`.
"""
from __future__ import annotations

from logging import getLogger
from typing import Type, TypeVar

logger = getLogger(__name__)

_CATALOG: list[type] = []

_T = TypeVar("_T", bound=type)


def register_model(cls: _T | None = None):
    """Class decorator marking a model class for registration.

    Usable bare (``@register_model``) or called (``@register_model()``). The
    decorated class is only *collected* here; it is instantiated and written to
    MLflow later, at service startup.
    """

    def _add(target: _T) -> _T:
        if target not in _CATALOG:
            _CATALOG.append(target)
            logger.debug("Collected model class %s", getattr(target, "__name__", target))
        return target

    return _add if cls is None else _add(cls)


def collected_models() -> list[type]:
    """Return all model classes collected via :func:`register_model`."""
    return list(_CATALOG)


def clear_catalog() -> None:
    """Drop all collected classes. Intended for tests."""
    _CATALOG.clear()
