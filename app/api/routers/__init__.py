"""Production router composition, loaded only when ``api_router`` is requested.

Importing an individual read-only route does not initialize the chat dependency
tree. The lightweight mobile fixture can therefore use the same retrieval
contract with just FastAPI, while ``main:app`` retains every production route.
"""

from importlib import import_module

from fastapi import APIRouter

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
        router = APIRouter()
        for attribute, (_, prefix) in _ROUTERS.items():
            router.include_router(__getattr__(attribute), prefix=prefix)
        globals()[name] = router
        return router
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
