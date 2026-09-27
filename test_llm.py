import pytest

from core.llm import build_llm


def test_unsupported_provider_raises_error():
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        build_llm("unsupported-provider")