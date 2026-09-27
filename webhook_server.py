"""
AutoStream Agent — FastAPI Webhook Server
Exposes REST endpoints for:
  - POST /chat          : Generic chat API
  - GET  /whatsapp      : WhatsApp webhook verification
  - POST /whatsapp      : WhatsApp incoming message handler
  - GET  /health        : Health check
  - GET  /metrics       : Basic session metrics

WhatsApp integration uses the Meta Cloud API (Webhooks).
"""

import os
import hashlib
import hmac
import json

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy import text
from core.config import get_settings

load_dotenv()
settings = get_settings()

from agent.graph import AutoStreamAgent, build_llm
from agent.state import AgentState
from utils.session_manager import get_session_store
from utils.logger import get_logger 

from database.connection import SessionLocal, init_db
from database.repository import ConversationRepository, LeadRepository

logger = get_logger("webhook_server")
# Initialize persistent database tables.
init_db()

app = FastAPI(
    title="AutoStream AI Agent API",
    description="Conversational AI agent for AutoStream — Social-to-Lead Workflow",
    version="1.0.0",
)

# ---- Config ----
LLM_PROVIDER = settings.llm_provider
LLM_MODEL = settings.llm_model

WA_VERIFY_TOKEN = settings.whatsapp_verify_token
WA_APP_SECRET = settings.whatsapp_app_secret or ""
WA_ACCESS_TOKEN = settings.whatsapp_access_token or ""
WA_PHONE_NUMBER_ID = settings.whatsapp_phone_number_id or ""

# Shared LLM instance (expensive to init per request)
_llm = None
_graph = None

def get_shared_graph():
    global _llm, _graph
    if _graph is None:
        _llm = build_llm(provider=LLM_PROVIDER, model=LLM_MODEL)
        from agent.graph import build_agent_graph
        _graph = build_agent_graph(_llm)
    return _graph


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    response: str
    intent: str
    lead_captured: bool
    turn_count: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_or_create_state(session_id: str) -> AgentState:
    store = get_session_store()
    state = store.get(session_id)
    if state is None:
        logger.info(f"New session created: {session_id}")
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


def _run_agent(session_id: str, user_message: str) -> dict:
    """
    Run the LangGraph agent and persist conversation data.

    The existing in-memory SessionStore remains responsible for
    active LangGraph state, while SQLAlchemy provides durable
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
        conversation_repo.get_or_create_conversation(session_id)

        # Persist the incoming user message.
        conversation_repo.add_message(
            session_id=session_id,
            role="user",
            content=user_message,
        )

        # Update in-memory LangGraph state.
        state["messages"] = list(
            state.get("messages", [])
        ) + [HumanMessage(content=user_message)]

        # Execute the existing agent graph.
        result = graph.invoke(state)

        # Persist the updated LangGraph state for active sessions.
        store.set(session_id, result)

        # Persist the assistant response.
        response_text = result.get("response")

        if response_text:
            conversation_repo.add_message(
                session_id=session_id,
                role="assistant",
                content=response_text,
            )

        # Persist lead information once the agent has captured it.
        if result.get("lead_captured"):
            lead_state = result.get("lead_collector_state") or {}

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


def _verify_whatsapp_signature(payload: bytes, signature_header: str) -> bool:
    """Verify that the request came from Meta using HMAC-SHA256."""
    if not WA_APP_SECRET:
        return True  # Skip verification in dev if secret not set
    expected = "sha256=" + hmac.new(
        WA_APP_SECRET.encode(), payload, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature_header or "")


async def _send_whatsapp_message(to: str, text: str):
    """Send a WhatsApp message via Meta Cloud API."""
    import httpx
    url = f"https://graph.facebook.com/v18.0/{WA_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WA_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text},
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=headers, json=payload, timeout=10)
        if resp.status_code != 200:
            logger.error(f"WhatsApp send failed: {resp.status_code} — {resp.text}")
        else:
            logger.info(f"Message sent to {to}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    """Health check including database connectivity."""

    db = SessionLocal()

    try:
        db.execute(text("SELECT 1"))

        return {
            "status": "ok",
            "service": "AutoStream AI Agent",
            "version": "1.0.0",
            "database": "ok",
        }

    except Exception as exc:
        logger.error(f"Database health check failed: {exc}")

        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "service": "AutoStream AI Agent",
                "version": "1.0.0",
                "database": "unavailable",
            },
        )

    finally:
        db.close()


@app.get("/metrics")
def metrics():
    store = get_session_store()
    return {
        "active_sessions": store.active_sessions(),
        "llm_provider": LLM_PROVIDER,
        "model": LLM_MODEL or "default",
    }


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Generic chat endpoint. Use this for web/mobile integrations.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    logger.info(f"[{request.session_id}] User: {request.message}")

    try:
        result = _run_agent(session_id=request.session_id, user_message=request.message)
    except Exception as e:
        logger.error(f"Agent error for session {request.session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Agent encountered an internal error.")

    response_text = result.get("response") or "I'm sorry, I couldn't process that."
    logger.info(f"[{request.session_id}] Agent: {response_text}")

    return ChatResponse(
        session_id=request.session_id,
        response=response_text,
        intent=result.get("current_intent", "UNKNOWN"),
        lead_captured=result.get("lead_captured", False),
        turn_count=result.get("turn_count", 0),
    )


@app.post("/chat/reset")
async def reset_session(session_id: str = Query(...)):
    """Reset a session (clears conversation history and lead state)."""
    store = get_session_store()
    store.delete(session_id)
    return {"status": "reset", "session_id": session_id}


# ---------------------------------------------------------------------------
# WhatsApp Webhook
# ---------------------------------------------------------------------------

@app.get("/whatsapp")
async def whatsapp_verify(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
):
    """
    WhatsApp webhook verification endpoint.
    Meta calls this GET request when you configure the webhook in the Developer Console.
    """
    if hub_mode == "subscribe" and hub_verify_token == WA_VERIFY_TOKEN:
        logger.info("WhatsApp webhook verified successfully.")
        return PlainTextResponse(content=hub_challenge)
    logger.warning("WhatsApp webhook verification failed — token mismatch.")
    raise HTTPException(status_code=403, detail="Verification token mismatch.")


@app.post("/whatsapp")
async def whatsapp_webhook(request: Request):
    """
    WhatsApp incoming message handler.
    Receives messages from Meta, runs through AutoStream agent, and replies.

    Message flow:
      Meta Cloud API → POST /whatsapp → Agent → POST graph.facebook.com/messages
    """
    # 1. Verify signature
    body_bytes = await request.body()
    signature = request.headers.get("X-Hub-Signature-256", "")
    if not _verify_whatsapp_signature(body_bytes, signature):
        logger.warning("Invalid WhatsApp signature — rejecting request.")
        raise HTTPException(status_code=403, detail="Invalid signature.")

    # 2. Parse payload
    try:
        payload = json.loads(body_bytes)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    # 3. Extract message (Meta webhook format)
    try:
        entry = payload["entry"][0]
        changes = entry["changes"][0]
        value = changes["value"]

        # Ignore status updates (delivery/read receipts)
        if "statuses" in value:
            return JSONResponse(content={"status": "ignored"})

        message_obj = value["messages"][0]
        from_number = message_obj["from"]          # Sender's WhatsApp number
        message_type = message_obj.get("type", "")

        if message_type != "text":
            # For non-text messages (images, audio etc.), send a fallback
            await _send_whatsapp_message(
                from_number,
                "I can only process text messages right now. Please type your question!"
            )
            return JSONResponse(content={"status": "non_text_ignored"})

        user_text = message_obj["text"]["body"]
        session_id = f"wa_{from_number}"

        logger.info(f"[WhatsApp] From {from_number}: {user_text}")

    except (KeyError, IndexError) as e:
        logger.error(f"Malformed WhatsApp payload: {e}")
        return JSONResponse(content={"status": "ok"})  # Always 200 to Meta

    # 4. Run agent
    try:
        result = _run_agent(session_id=session_id, user_message=user_text)
        response_text = result.get("response") or "Sorry, I encountered an issue. Please try again."
    except Exception as e:
        logger.error(f"Agent error for WhatsApp session {session_id}: {e}", exc_info=True)
        response_text = "Sorry, our AI is temporarily unavailable. Please try again shortly."

    # 5. Send reply
    await _send_whatsapp_message(from_number, response_text)
    logger.info(f"[WhatsApp] To {from_number}: {response_text}")

    # Meta requires a 200 OK response
    return JSONResponse(content={"status": "ok"})


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
