"""
LLM provider factory.

Keeps provider-specific model construction outside the agent graph.
"""

from typing import Any, Optional


def build_llm(
    provider: str,
    model: Optional[str] = None,
) -> Any:
    """
    Build and return the configured LangChain chat model.

    Supported providers:
    - anthropic
    - openai
    - google
    """

    provider = provider.lower().strip()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model or "claude-3-5-sonnet-20241022",
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model or "gpt-4o-mini",
        )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model or "gemini-2.5-flash",
        )

    raise ValueError(
        f"Unsupported LLM provider: {provider}"
    )