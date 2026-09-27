from core.config import Settings


def test_agent_uses_configured_provider(monkeypatch):
    from agent.graph import AutoStreamAgent

    class FakeLLM:
        pass

    monkeypatch.setattr(
        "agent.graph.build_llm",
        lambda provider, model: FakeLLM(),
    )

    monkeypatch.setattr(
        "agent.graph.build_agent_graph",
        lambda llm: "fake-graph",
    )

    monkeypatch.setattr(
        "agent.graph.get_settings",
        lambda: Settings(
            LLM_PROVIDER="google",
            LLM_MODEL="gemini-test",
        ),
    )

    agent = AutoStreamAgent()

    assert isinstance(agent.llm, FakeLLM)
    assert agent.graph == "fake-graph"