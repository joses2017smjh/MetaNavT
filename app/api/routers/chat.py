"""
Chat Router Module

Provides FastAPI endpoints for a streaming style chat response

Features:
- Message history management
- Document filtering
- Event handling
- Error tracking
"""
import asyncio
import contextlib
import logging
from typing import Awaitable, TypeVar

from fastapi import APIRouter, BackgroundTasks, Request, Response, status
from llama_index.core.llms import MessageRole

from app.api.errors import failure
from app.api.routers.events import EventCallbackHandler
from app.api.routers.models import (
    ChatData,
    Message,
    Result,
    SourceNodes,
)
from app.api.routers.vercel_response import VercelStreamResponse
from app.engine.engine import get_chat_engine
from app.engine.query_filter import generate_filters


# Initialize router - will be mounted in main app
chat_router = r = APIRouter()

logger = logging.getLogger("uvicorn")

# How often a chat route checks, while the agent runs, whether its client is still there (seconds).
DISCONNECT_POLL_S = 0.5
# The answer to a request whose client went away before the agent run ended; nobody reads it (nginx's code for this).
CLIENT_CLOSED_REQUEST = 499
DETAIL = "Error in chat engine"
T = TypeVar("T")


class ClientDisconnected(Exception):
    """The client went away before the agent run ended; the run was stopped."""


async def until_disconnected(request: Request, run: Awaitable[T], poll_s: float = DISCONNECT_POLL_S) -> T:
    """Await `run` (the agent's achat) unless the client disconnects first: then cancel it, which stops the workflow
    and the model call in flight (AgentWorkflow.achat), and raise ClientDisconnected. Before, the run went on after its
    client had left, and Ollama, which serves one request at a time, made later requests wait behind it."""
    task = asyncio.ensure_future(run)
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=poll_s)
            if done:
                return task.result()  # a failed run raises here
            if await request.is_disconnected():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                raise ClientDisconnected()
    finally:
        if not task.done():
            task.cancel()


def _gone(route: str) -> Response:
    logger.info("%s: the client disconnected before the answer; the agent run was stopped", route)
    return Response(status_code=CLIENT_CLOSED_REQUEST)

@r.post("")
async def chat(
    request: Request,
    data: ChatData,
    background_tasks: BackgroundTasks,
):
    """
    Streaming chat endpoint.
 
    Flow:
    1. Extract last message and history
    2. Apply document filters
    3. Initialize chat engine
    4. Stream response chunks
    """
    try:
        # Get latest message and conversation history
        last_message_content = data.get_last_message_content()
        messages = data.get_history_messages()

        # Set up document filtering
        doc_ids = data.get_chat_document_ids()

        # The only filter generated at the moment is public/private file type
        filters = generate_filters(doc_ids)
        params = data.data or {}
        
        # Initialize chat engine with:
        # - filters: Control document visibility (public/private)
        # - params: Custom configuration like temperature/tokens
        # - event_handlers: Track operations and stream chunks
        event_handler = EventCallbackHandler()

        # Chat engine is the agent runner
        chat_engine = get_chat_engine(
            filters=filters,
            params=params,
            event_handlers=[event_handler],
        )
        

            
        # Stream response using Vercel's streaming response
        # (Returns response chunks incrementally)
        response = await until_disconnected(request, chat_engine.achat(last_message_content, messages))

        #logger.info(f"Streaming response: {response.response} and {response.source_nodes}\nType: {type(response)}")
        response.is_dummy_stream = True
        return VercelStreamResponse(
            request=request,
            event_handler=event_handler,
            response=response,
            chat_data=data,
            background_tasks=background_tasks,
        )
        
    except ClientDisconnected:
        return _gone("POST /api/chat")
    except Exception as e:
        # The fixed text and the request id; the exception (a model server's or a database's error, a model call that
        # timed out) goes to the log.
        raise failure(e, "chat engine", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=DETAIL,
                      log=logger, traceback=True) from e


# non-streaming endpoint for testing
@r.post("/request")
async def chat_request(
    request: Request,
    data: ChatData,
) -> Result:
    last_message_content = data.get_last_message_content()
    messages = data.get_history_messages()

    doc_ids = data.get_chat_document_ids()
    filters = generate_filters(doc_ids)
    params = data.data or {}
    logger.info(
        f"Creating chat engine with filters: {str(filters)}",
    )

    try:
        chat_engine = get_chat_engine(filters=filters, params=params)
        response = await until_disconnected(request, chat_engine.achat(last_message_content, messages))
    except ClientDisconnected:
        return _gone("POST /api/chat/request")
    except Exception as e:
        # As POST /api/chat: a failed run is a 500 with the fixed text and the request id, logged under that id; it
        # used to come back as HTTP 200 with no text (the chat smoke committed in c9b4acd).
        raise failure(e, "chat engine", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=DETAIL,
                      log=logger, traceback=True) from e
    return Result(
        result=Message(role=MessageRole.ASSISTANT, content=response.response),
        nodes=SourceNodes.from_source_nodes(response.source_nodes),
        tools=response.sources
    )