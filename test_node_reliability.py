from langchain_core.messages import HumanMessage

from agent.nodes import (
    _safe_llm_invoke,
    node_generate_response,
    node_retrieve_context,
    node_handle_lead_collection,
)


class FailingLLM:
    """Fake LLM that always raises an exception."""

    def invoke(self, messages):
        raise RuntimeError("LLM provider unavailable")


class EmptyLLM:
    """Fake LLM that returns an invalid empty response."""

    def invoke(self, messages):
        return type("Response", (), {"content": ""})()


def test_safe_llm_invoke_returns_fallback_on_exception():
    llm = FailingLLM()

    result = _safe_llm_invoke(
        llm,
        [],
        fallback="Temporary fallback",
    )

    assert result == "Temporary fallback"


def test_safe_llm_invoke_returns_fallback_for_empty_response():
    llm = EmptyLLM()

    result = _safe_llm_invoke(
        llm,
        [],
        fallback="Temporary fallback",
    )

    assert result == "Temporary fallback"


def test_node_generate_response_survives_llm_failure():
    state = {
        "messages": [
            HumanMessage(content="Hello"),
        ],
        "current_intent": "greeting",
        "rag_context": None,
    }

    result = node_generate_response(
        state,
        FailingLLM(),
    )

    assert result["response"] == (
        "I'm temporarily unable to generate a response. "
        "Please try again in a moment."
    )


def test_node_retrieve_context_survives_rag_failure(monkeypatch):
    def failing_rag_pipeline():
        raise RuntimeError("RAG unavailable")

    monkeypatch.setattr(
        "agent.nodes.get_rag_pipeline",
        failing_rag_pipeline,
    )

    state = {
        "messages": [
            HumanMessage(content="What is the Pro plan?"),
        ],
        "current_intent": "product_inquiry",
    }

    result = node_retrieve_context(state)

    assert result["rag_context"] is None


def test_node_handle_lead_collection_survives_capture_failure(
    monkeypatch,
):
    from tools.lead_capture import LeadCollector

    original_execute_capture = LeadCollector.execute_capture

    def failing_execute_capture(self):
        raise RuntimeError("CRM unavailable")

    monkeypatch.setattr(
        LeadCollector,
        "execute_capture",
        failing_execute_capture,
    )

    state = {
        "messages": [
            HumanMessage(
                content="Everything is ready."
            ),
        ],
        "lead_collector_state": {
            "collected": {
                "name": "Satish",
                "email": "satish@example.com",
                "platform": "YouTube",
            },
            "is_captured": False,
        },
        "lead_collection_active": True,
        "lead_captured": False,
    }

    result = node_handle_lead_collection(
        state,
        FailingLLM(),
    )

    assert result["lead_captured"] is False
    assert result["lead_collection_active"] is True
    assert "temporary issue" in result["response"]