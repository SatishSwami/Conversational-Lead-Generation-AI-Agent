"""
Repository layer for database operations.

The agent should interact with this layer rather than
containing raw database queries inside LangGraph nodes.
"""

from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import Conversation, Lead, Message


class ConversationRepository:
    """Persistence operations for conversations and messages."""

    def __init__(self, db: Session):
        self.db = db

    def get_or_create_conversation(
        self,
        session_id: str,
    ) -> Conversation:
        """Return an existing conversation or create a new one."""

        conversation = self.db.scalar(
            select(Conversation).where(
                Conversation.session_id == session_id
            )
        )

        if conversation:
            return conversation

        conversation = Conversation(session_id=session_id)

        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)

        return conversation

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> Message:
        """Persist a conversation message."""

        conversation = self.get_or_create_conversation(session_id)

        message = Message(
            session_id=conversation.session_id,
            role=role,
            content=content,
        )

        self.db.add(message)
        self.db.commit()
        self.db.refresh(message)

        return message

    def get_messages(
        self,
        session_id: str,
    ) -> list[Message]:
        """Return messages for a conversation."""

        return list(
            self.db.scalars(
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(Message.created_at)
            )
        )


class LeadRepository:
    """Persistence operations for leads."""

    def __init__(self, db: Session):
        self.db = db

    def create_or_update_lead(
        self,
        session_id: str,
        name: str,
        email: str,
        platform: str,
    ) -> Lead:
        """Create a lead or update an existing lead for the session."""

        conversation = self.db.scalar(
            select(Conversation).where(
                Conversation.session_id == session_id
            )
        )

        if not conversation:
            conversation = Conversation(session_id=session_id)
            self.db.add(conversation)
            self.db.flush()

        lead = self.db.scalar(
            select(Lead).where(
                Lead.session_id == session_id
            )
        )

        if lead:
            lead.name = name
            lead.email = email
            lead.platform = platform
        else:
            lead = Lead(
                session_id=session_id,
                name=name,
                email=email,
                platform=platform,
            )
            self.db.add(lead)

        self.db.commit()
        self.db.refresh(lead)

        return lead

    def get_lead(
        self,
        session_id: str,
    ) -> Optional[Lead]:
        """Return the lead associated with a session."""

        return self.db.scalar(
            select(Lead).where(
                Lead.session_id == session_id
            )
        )