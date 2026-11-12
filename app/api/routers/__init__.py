"""Production router composition, loaded only when ``api_router`` is requested.

Importing an individual read-only route does not initialize the chat dependency
tree. The lightweight mobile fixture can therefore use the same retrieval
contract with just FastAPI, while ``main:app`` retains every production route.

``api_router`` carries the two opt-in dependencies from app/api/security.py: the
API key check (METANAVIT_API_KEY, METANAVIT_API_KEYS_FILE), which also says who
the caller is, and the per-replica request budget (RATE_LIMIT_PER_MINUTE). Both
do nothing unless configured. An individual
router imported on its own, as the mobile fixture does, does not carry them.
"""

from importlib import import_module

from fastapi import APIRouter, Depends

_ROUTERS = {
    "chat_router": ("chat", "/chat"),
    "file_upload_router": ("upload", "/chat/upload"),
    "config_router": ("chat_config", "/chat/config"),
    "query_router": ("query", "/query"),
    "retrieve_router": ("retrieve", "/retrieve"),
    "plans_router": ("plans", "/plans"),
}


def __getattr__(name):
    if name in _ROUTERS:
        module, _ = _ROUTERS[name]
        router = getattr(import_module(f"{__name__}.{module}"), name)
        globals()[name] = router
        return router
    if name == "api_router":
        from app.api.security import enforce_request_budget, require_api_key

        router = APIRouter(dependencies=[Depends(require_api_key), Depends(enforce_request_budget)])
        for attribute, (_, prefix) in _ROUTERS.items():
            router.include_router(__getattr__(attribute), prefix=prefix)
        globals()[name] = router
        return router
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
