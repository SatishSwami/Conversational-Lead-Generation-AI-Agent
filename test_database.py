from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import Base
from database.repository import ConversationRepository, LeadRepository


def create_test_db():
    engine = create_engine("sqlite:///:memory:")

    Base.metadata.create_all(engine)

    SessionLocal = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
    )

    return SessionLocal()


def test_conversation_and_messages():
    db = create_test_db()

    repo = ConversationRepository(db)

    conversation = repo.get_or_create_conversation("test-session")

    assert conversation.session_id == "test-session"

    repo.add_message(
        "test-session",
        "user",
        "What is the Pro plan?",
    )

    repo.add_message(
        "test-session",
        "assistant",
        "The Pro plan costs $79/month.",
    )

    messages = repo.get_messages("test-session")

    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"


def test_lead_persistence():
    db = create_test_db()

    conversation_repo = ConversationRepository(db)
    lead_repo = LeadRepository(db)

    conversation_repo.get_or_create_conversation("lead-session")

    lead = lead_repo.create_or_update_lead(
        session_id="lead-session",
        name="Satish",
        email="satish@example.com",
        platform="YouTube",
    )

    assert lead.name == "Satish"
    assert lead.email == "satish@example.com"
    assert lead.platform == "YouTube"

    stored_lead = lead_repo.get_lead("lead-session")

    assert stored_lead is not None
    assert stored_lead.name == "Satish"