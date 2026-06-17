"""Model discovery for IQUANA services.

Two ways a model class ends up in the registry catalog
(:mod:`iquana_service_core.registry`):

* **In-tree:** importing a service's ``models`` package, so every
  ``@register_model`` decorator runs. Dropping a file in the package is enough
  -- no import has to be wired by hand.
* **Out-of-tree:** loading model packages advertised via the ``iquana.models``
  entry-point group. This is the seam for user-installed model packages; it
  stays dormant until someone ships one.
"""
from __future__ import annotations

import importlib
import pkgutil
from importlib.metadata import entry_points
from logging import getLogger

logger = getLogger(__name__)

ENTRY_POINT_GROUP = "iquana.models"


def import_models_package(package: str) -> None:
    """Import every submodule of ``package`` so ``@register_model`` fires.

    Removes the fragility of relying on some unrelated import (a router, say) to
    transitively pull the model modules in: the package contents alone decide
    what gets registered.
    """
    pkg = importlib.import_module(package)
    if not hasattr(pkg, "__path__"):
        # Plain module, not a package -- importing it already ran its decorators.
        return
    for _, name, _ in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        importlib.import_module(name)
        logger.debug("Imported model module %s", name)


def load_plugin_models(group: str = ENTRY_POINT_GROUP) -> None:
    """Import models contributed by installed packages via entry points.

    A third-party package adds models by declaring, in its packaging metadata::

        [project.entry-points."iquana.models"]
        my_models = "my_pkg.models"

    Loading the entry point imports the target, triggering ``@register_model``.
    A failing plugin is logged and skipped rather than taking the service down.
    """
    for ep in entry_points(group=group):
        try:
            ep.load()
            logger.info("Loaded model plugin '%s'", ep.name)
        except Exception:
            logger.exception("Failed to load model plugin '%s'", ep.name)
