# iquana-service-core

Shared FastAPI runtime for the IQUANA AI segmentation services
(`prompted-seg-service`, `instance-discovery-service`, future services).

It removes the per-service boilerplate that previously had to be copy-edited in
every repo: the app factory, CORS, startup lifecycle (HuggingFace login + model
registration), the `/health` endpoint, the `/models/*` registry routes and the
`/annotation_session/models/{key}/preload` route. A service now authors only its
own inference routes, model registry instance, and model registration config.

## Usage

```python
from iquana_service_core import create_service_app
from app.state import MODEL_REGISTRY
from app.routes.inference import router, session_router
from models.register_models import register_models

app = create_service_app(
    title="Instance Discovery API",
    task="instance-discovery",          # the registry `task` tag for this service
    registry=MODEL_REGISTRY,
    register_models=register_models,
    inference_routers=[router, session_router],
    hf_login=True,                       # log into HF for gated weights
)
```

See [`examples/instance_discovery_app.py`](examples/instance_discovery_app.py)
for a full before/after of `instance-discovery-service`.

## What it provides

| Surface | Source |
|---|---|
| `GET /health` (+ optional `health_extra`) | `routers/health.py` |
| `GET /models/all`, `/models/all/available`, `/models/{key}` | `routers/models.py` |
| `GET /annotation_session/models/{key}/preload` | `routers/models.py` |
| CORS, lifespan, HF login, model registration | `app.py`, `lifespan.py` |

## Contract / versioning

`create_service_app` is written against the **current `iquana-toolbox` API**:

- `MLFlowModelRegistry.get_model_infos_via_tags(tags)` (older services call the
  renamed `get_models_via_tags` — they must update).
- `registry.register_model(model)` deriving identifier/desc/tags from
  `model.model_info` (older `register_models.py` passes
  `model_identifier=/desc=/tags=` — they must update).

**All services and the backend must pin the same toolbox revision.** Cut a tag
on `iquana-toolbox`, pin it here and in every service, and bump in lockstep.
This package exists precisely so that contract changes touch one shared place
instead of N copies.

## Adoption checklist (per service)

1. Add `iquana-service-core` and bump `iquana-toolbox` to the canonical pin.
2. Replace `app/__init__.py` with a `create_service_app(...)` call.
3. Delete the now-redundant `app/routes/__init__.py` (health) and
   `app/routes/models.py`.
4. Update `models/register_models.py` to the current `register_model(model)` API.
5. Smoke-test: `GET /health`, `GET /models/all/available`, one inference call.
```
