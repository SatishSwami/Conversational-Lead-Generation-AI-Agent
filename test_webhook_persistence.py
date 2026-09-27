import os
import uuid

os.environ["LLM_PROVIDER"] = "google"


def test_database_repositories_are_available():
    from database.connection import SessionLocal
    from database.repository import ConversationRepository, LeadRepository

    db = SessionLocal()

    try:
        conversation_repo = ConversationRepository(db)
        lead_repo = LeadRepository(db)

        assert conversation_repo is not None
        assert lead_repo is not None
    finally:
        db.close()


def test_run_agent_persists_messages_and_lead(monkeypatch):
    from webhook_server import _run_agent
    from database.connection import SessionLocal
    from database.repository import ConversationRepository

    session_id = f"integration-test-session-{uuid.uuid4()}"

    class FakeGraph:
        def invoke(self, state):
            return {
                **state,
                "response": "The Pro plan costs $79/month.",
                "current_intent": "product_inquiry",
                "lead_collection_active": False,
                "lead_collector_state": None,
                "lead_captured": False,
                "turn_count": 1,
                "rag_context": None,
            }

    monkeypatch.setattr(
        "webhook_server.get_shared_graph",
        lambda: FakeGraph(),
    )

    result = _run_agent(
        session_id=session_id,
        user_message="How much is the Pro plan?",
    )

    assert result["response"] == "The Pro plan costs $79/month."

    db = SessionLocal()

    try:
        conversation_repo = ConversationRepository(db)
        messages = conversation_repo.get_messages(session_id)

        assert len(messages) == 2

        assert messages[0].role == "user"
        assert messages[0].content == "How much is the Pro plan?"

        assert messages[1].role == "assistant"
        assert messages[1].content == "The Pro plan costs $79/month."

    finally:
        db.close()


def test_run_agent_persists_captured_lead(monkeypatch):
    from webhook_server import _run_agent
    from database.connection import SessionLocal
    from database.repository import LeadRepository

    session_id = f"lead-integration-test-{uuid.uuid4()}"

    class FakeGraph:
        def invoke(self, state):
            return {
                **state,
                "response": "Thanks! Your information has been captured.",
                "current_intent": "high_intent_lead",
                "lead_collection_active": False,
                "lead_collector_state": {
                    "name": "Satish",
                    "email": "satish@example.com",
                    "platform": "YouTube",
                },
                "lead_captured": True,
                "turn_count": 4,
                "rag_context": None,
            }

    monkeypatch.setattr(
        "webhook_server.get_shared_graph",
        lambda: FakeGraph(),
    )

    result = _run_agent(
        session_id=session_id,
        user_message=(
            "My name is Satish, email is satish@example.com, "
            "and I use YouTube."
        ),
    )

    assert result["lead_captured"] is True

    db = SessionLocal()

    try:
        lead_repo = LeadRepository(db)
        lead = lead_repo.get_lead(session_id)

        assert lead is not None
        assert lead.name == "Satish"
        assert lead.email == "satish@example.com"
        assert lead.platform == "YouTube"

    finally:
        db.close()