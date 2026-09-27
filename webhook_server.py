"""
AutoStream Agent — FastAPI Webhook Server

Exposes REST endpoints for:
    - POST /chat          : Generic chat API
    - POST /chat/reset    : Reset an active session
    - GET  /whatsapp      : WhatsApp webhook verification
    - POST /whatsapp      : WhatsApp incoming message handler
    - GET  /health        : Health check
    - GET  /metrics       : Basic session metrics

WhatsApp integration uses the Meta Cloud API (Webhooks).
"""
import re
import time
import uuid

import hashlib
import hmac
import json

from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
)
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from sqlalchemy import text

from core.config import get_settings
from agent.graph import build_llm
from agent.state import AgentState
from utils.session_manager import get_session_store
from utils.logger import get_logger

from database.connection import SessionLocal, init_db
from database.repository import ConversationRepository, LeadRepository


# ---------------------------------------------------------------------------
# Application Configuration
# ---------------------------------------------------------------------------

load_dotenv()
settings = get_settings()

logger = get_logger("webhook_server")

# Initialize persistent database tables.
init_db()

app = FastAPI(
    title="AutoStream AI Agent API",
    description=(
        "Conversational AI agent for AutoStream — "
        "Social-to-Lead Workflow"
    ),
    version=settings.app_version,
)

# ---------------------------------------------------------------------------
# Request Observability
# ---------------------------------------------------------------------------

REQUEST_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9_.:-]{1,128}$"
)


def _get_request_id(request: Request) -> str:
    """
    Return a safe request correlation ID.

    A valid client-provided ID is preserved. Otherwise, generate
    a new UUID-based correlation ID.
    """
    incoming_request_id = request.headers.get(
        "X-Request-ID",
        "",
    ).strip()

    if incoming_request_id and REQUEST_ID_PATTERN.fullmatch(
        incoming_request_id
    ):
        return incoming_request_id

    return uuid.uuid4().hex


@app.middleware("http")
async def request_observability(
    request: Request,
    call_next,
):
    """
    Add request correlation and latency logging.

    The middleware intentionally logs only request metadata.
    It never logs request bodies or user message content.
    """
    request_id = _get_request_id(request)
    request.state.request_id = request_id

    started_at = time.perf_counter()

    try:
        response = await call_next(request)

    except Exception:
        duration_ms = (
            time.perf_counter() - started_at
        ) * 1000

        logger.exception(
            "Request failed | request_id=%s method=%s "
            "path=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            duration_ms,
        )

        raise

    duration_ms = (
        time.perf_counter() - started_at
    ) * 1000

    response.headers["X-Request-ID"] = request_id

    logger.info(
        "Request completed | request_id=%s method=%s "
        "path=%s status=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )

    return response

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LLM_PROVIDER = settings.llm_provider
LLM_MODEL = settings.llm_model

WA_VERIFY_TOKEN = settings.whatsapp_verify_token
WA_APP_SECRET = settings.whatsapp_app_secret or ""
WA_ACCESS_TOKEN = settings.whatsapp_access_token or ""
WA_PHONE_NUMBER_ID = settings.whatsapp_phone_number_id or ""

API_KEY = settings.api_key
ENVIRONMENT = settings.environment


# ---------------------------------------------------------------------------
# Shared Agent Graph
# ---------------------------------------------------------------------------

_llm = None
_graph = None


def get_shared_graph():
    """
    Return the shared LangGraph instance.

    The LLM and graph are initialized once and reused across requests.
    """
    global _llm, _graph

    if _graph is None:
        _llm = build_llm(
            provider=LLM_PROVIDER,
            model=LLM_MODEL,
        )

        from agent.graph import build_agent_graph

        _graph = build_agent_graph(_llm)

    return _graph


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """
    Request model for the generic chat endpoint.
    """

    session_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_.:-]+$",
    )

    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
    )


class ChatResponse(BaseModel):
    session_id: str
    response: str
    intent: str
    lead_captured: bool
    turn_count: int


# ---------------------------------------------------------------------------
# API Authentication
# ---------------------------------------------------------------------------


def _verify_api_key(
    x_api_key: str | None = Header(default=None),
) -> None:
    """
    Protect API endpoints when an API key is configured.

    Development environments may run without an API key.
    Production environments must have API_KEY configured.
    """
    if API_KEY:
        if not x_api_key or not hmac.compare_digest(
            x_api_key,
            API_KEY,
        ):
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key.",
            )

        return

    if ENVIRONMENT.lower() == "production":
        logger.error(
            "API_KEY is not configured in production."
        )

        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured.",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_or_create_state(session_id: str) -> AgentState:
    """
    Get an active session state from the in-memory store.

    If no active state exists, initialize a new agent state.
    """
    store = get_session_store()
    state = store.get(session_id)

    if state is None:
        logger.info(
            "New session created: %s",
            session_id,
        )

        state = {
            "messages": [],
            "current_intent": None,
            "lead_collection_active": False,
            "lead_collector_state": None,
            "lead_captured": False,
            "turn_count": 0,
            "rag_context": None,
            "response": None,
        }

    return state


def _run_agent(
    session_id: str,
    user_message: str,
) -> dict:
    """
    Run the LangGraph agent and persist conversation data.

    The in-memory SessionStore remains responsible for active
    LangGraph state, while SQLAlchemy provides durable
    conversation and lead persistence.
    """
    from langchain_core.messages import HumanMessage

    graph = get_shared_graph()
    store = get_session_store()

    state = _get_or_create_state(session_id)

    db = SessionLocal()

    try:
        conversation_repo = ConversationRepository(db)
        lead_repo = LeadRepository(db)

        # Ensure the conversation exists.
        conversation_repo.get_or_create_conversation(
            session_id
        )

        # Persist incoming user message.
        conversation_repo.add_message(
            session_id=session_id,
            role="user",
            content=user_message,
        )

        # Update in-memory LangGraph state.
        state["messages"] = list(
            state.get("messages", [])
        ) + [
            HumanMessage(content=user_message)
        ]

        # Execute the agent graph.
        result = graph.invoke(state)

        # Persist updated LangGraph state.
        store.set(
            session_id,
            result,
        )

        # Persist assistant response.
        response_text = result.get("response")

        if response_text:
            conversation_repo.add_message(
                session_id=session_id,
                role="assistant",
                content=response_text,
            )

        # Persist lead information once captured.
        if result.get("lead_captured"):
            lead_state = (
                result.get("lead_collector_state")
                or {}
            )

            name = lead_state.get("name")
            email = lead_state.get("email")
            platform = lead_state.get("platform")

            if name and email and platform:
                lead_repo.create_or_update_lead(
                    session_id=session_id,
                    name=name,
                    email=email,
                    platform=platform,
                )

        return result

    finally:
        db.close()


def _verify_whatsapp_signature(
    payload: bytes,
    signature_header: str,
) -> bool:
    """
    Verify that the request came from Meta using HMAC-SHA256.

    In development, signature verification can be skipped when
    no app secret is configured.

    In production, a missing app secret causes verification failure.
    """
    if not WA_APP_SECRET:
        return ENVIRONMENT.lower() != "production"

    expected = "sha256=" + hmac.new(
        WA_APP_SECRET.encode(),
        payload,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(
        expected,
        signature_header or "",
    )


async def _send_whatsapp_message(
    to: str,
    text: str,
):
    """
    Send a WhatsApp message via Meta Cloud API.

    Returns True when Meta accepts the request, otherwise False.
    """
    import httpx

    if not WA_ACCESS_TOKEN or not WA_PHONE_NUMBER_ID:
        logger.error(
            "WhatsApp credentials are not configured."
        )
        return False

    url = (
        "https://graph.facebook.com/v18.0/"
        f"{WA_PHONE_NUMBER_ID}/messages"
    )

    headers = {
        "Authorization": f"Bearer {WA_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {
            "body": text,
        },
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers=headers,
                json=payload,
                timeout=10,
            )

        if response.status_code != 200:
            logger.error(
                "WhatsApp send failed with status %s.",
                response.status_code,
            )
            return False

        logger.info(
            "WhatsApp message sent successfully."
        )
        return True

    except Exception as exc:
        logger.error(
            "WhatsApp send request failed: %s",
            exc,
            exc_info=True,
        )
        return False


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health_check():
    """
    Health check including database connectivity.

    This endpoint intentionally remains unauthenticated so
    deployment/load-balancer health probes can access it.
    """
    db = SessionLocal()

    try:
        db.execute(text("SELECT 1"))

        return {
            "status": "ok",
            "service": settings.app_name,
            "version": settings.app_version,
            "database": "ok",
        }

    except Exception as exc:
        logger.error(
            "Database health check failed: %s",
            exc,
        )

        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "service": settings.app_name,
                "version": settings.app_version,
                "database": "unavailable",
            },
        )

    finally:
        db.close()


@app.get(
    "/metrics",
    dependencies=[Depends(_verify_api_key)],
)
def metrics():
    """
    Basic application metrics.

    Protected because the endpoint exposes runtime information.
    """
    store = get_session_store()

    return {
        "active_sessions": store.active_sessions(),
        "llm_provider": LLM_PROVIDER,
        "model": LLM_MODEL or "default",
    }


@app.post(
    "/chat",
    response_model=ChatResponse,
    dependencies=[Depends(_verify_api_key)],
)
async def chat_endpoint(
    request: ChatRequest,
):
    """
    Generic chat endpoint.

    Intended for web/mobile integrations and protected by
    API-key authentication when configured.
    """
    logger.info(
        "[%s] Incoming chat request.",
        request.session_id,
    )

    try:
        result = _run_agent(
            session_id=request.session_id,
            user_message=request.message,
        )

    except Exception as exc:
        logger.error(
            "Agent error for session %s: %s",
            request.session_id,
            exc,
            exc_info=True,
        )

        raise HTTPException(
            status_code=500,
            detail="Agent encountered an internal error.",
        )

    response_text = (
        result.get("response")
        or "I'm sorry, I couldn't process that."
    )

    logger.info(
        "[%s] Agent response generated.",
        request.session_id,
    )

    return ChatResponse(
        session_id=request.session_id,
        response=response_text,
        intent=result.get(
            "current_intent",
            "UNKNOWN",
        ),
        lead_captured=result.get(
            "lead_captured",
            False,
        ),
        turn_count=result.get(
            "turn_count",
            0,
        ),
    )


@app.post(
    "/chat/reset",
    dependencies=[Depends(_verify_api_key)],
)
async def reset_session(
    session_id: str = Query(
        ...,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_.:-]+$",
    ),
):
    """
    Reset an active session.

    Durable database history is intentionally retained.
    """
    store = get_session_store()

    store.delete(session_id)

    logger.info(
        "Active session reset: %s",
        session_id,
    )

    return {
        "status": "reset",
        "session_id": session_id,
    }


# ---------------------------------------------------------------------------
# WhatsApp Webhook
# ---------------------------------------------------------------------------


@app.get("/whatsapp")
async def whatsapp_verify(
    hub_mode: str = Query(
        None,
        alias="hub.mode",
    ),
    hub_challenge: str = Query(
        None,
        alias="hub.challenge",
    ),
    hub_verify_token: str = Query(
        None,
        alias="hub.verify_token",
    ),
):
    """
    WhatsApp webhook verification endpoint.

    Meta calls this GET request when the webhook is configured
    in the Developer Console.
    """
    if not WA_VERIFY_TOKEN:
        logger.error(
            "WhatsApp verify token is not configured."
        )

        raise HTTPException(
            status_code=503,
            detail="WhatsApp webhook is not configured.",
        )

    if (
        hub_mode == "subscribe"
        and hub_verify_token
        and hmac.compare_digest(
            hub_verify_token,
            WA_VERIFY_TOKEN,
        )
        and hub_challenge is not None
    ):
        logger.info(
            "WhatsApp webhook verified successfully."
        )

        return PlainTextResponse(
            content=hub_challenge,
        )

    logger.warning(
        "WhatsApp webhook verification failed."
    )

    raise HTTPException(
        status_code=403,
        detail="Verification token mismatch.",
    )


@app.post("/whatsapp")
async def whatsapp_webhook(
    request: Request,
):
    """
    WhatsApp incoming message handler.

    Receives messages from Meta, runs them through the
    AutoStream agent, and replies using the Meta Cloud API.

    Message flow:
        Meta Cloud API
            ↓
        POST /whatsapp
            ↓
        Agent
            ↓
        Meta Cloud API /messages
    """

    # -----------------------------------------------------------------------
    # 1. Verify signature
    # -----------------------------------------------------------------------

    body_bytes = await request.body()

    signature = request.headers.get(
        "X-Hub-Signature-256",
        "",
    )

    if not _verify_whatsapp_signature(
        body_bytes,
        signature,
    ):
        logger.warning(
            "Invalid WhatsApp signature — rejecting request."
        )

        raise HTTPException(
            status_code=403,
            detail="Invalid signature.",
        )

    # -----------------------------------------------------------------------
    # 2. Parse payload
    # -----------------------------------------------------------------------

    try:
        payload = json.loads(body_bytes)

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON payload.",
        )

    # -----------------------------------------------------------------------
    # 3. Extract message
    # -----------------------------------------------------------------------

    try:
        entry = payload["entry"][0]
        changes = entry["changes"][0]
        value = changes["value"]

        # Ignore status updates.
        if "statuses" in value:
            return JSONResponse(
                content={
                    "status": "ignored",
                }
            )

        message_obj = value["messages"][0]

        from_number = message_obj["from"]
        message_type = message_obj.get(
            "type",
            "",
        )

        if message_type != "text":
            await _send_whatsapp_message(
                from_number,
                (
                    "I can only process text messages right now. "
                    "Please type your question!"
                ),
            )

            return JSONResponse(
                content={
                    "status": "non_text_ignored",
                }
            )

        user_text = message_obj["text"]["body"]

        session_id = f"wa_{from_number}"

        logger.info(
            "[WhatsApp] Incoming message from %s.",
            from_number,
        )

    except (KeyError, IndexError, TypeError) as exc:
        logger.error(
            "Malformed WhatsApp payload: %s",
            exc,
        )

        # Meta expects a successful response for webhook delivery.
        return JSONResponse(
            content={
                "status": "ok",
            }
        )

    # -----------------------------------------------------------------------
    # 4. Run agent
    # -----------------------------------------------------------------------

    try:
        result = _run_agent(
            session_id=session_id,
            user_message=user_text,
        )

        response_text = (
            result.get("response")
            or (
                "Sorry, I encountered an issue. "
                "Please try again."
            )
        )

    except Exception as exc:
        logger.error(
            "Agent error for WhatsApp session %s: %s",
            session_id,
            exc,
            exc_info=True,
        )

        response_text = (
            "Sorry, our AI is temporarily unavailable. "
            "Please try again shortly."
        )

    # -----------------------------------------------------------------------
    # 5. Send reply
    # -----------------------------------------------------------------------

    sent = await _send_whatsapp_message(
        from_number,
        response_text,
    )

    if not sent:
        logger.error(
            "Failed to send WhatsApp response."
        )

    # Meta requires a 200 OK response.
    return JSONResponse(
        content={
            "status": "ok",
        }
    )


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=settings.port,
        reload=settings.environment == "development",
    )