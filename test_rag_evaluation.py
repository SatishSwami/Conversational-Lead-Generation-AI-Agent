from agent.rag_pipeline import RAGPipeline


def _sources_for_query(pipeline, query):
    results = pipeline.retrieve(query, top_k=3)
    return [source for _, _, source in results]


def test_pricing_query_retrieves_pricing():
    pipeline = RAGPipeline()

    sources = _sources_for_query(
        pipeline,
        "How much does the Pro plan cost?",
    )

    assert "pricing" in sources


def test_video_limit_query_retrieves_pricing():
    pipeline = RAGPipeline()

    results = pipeline.retrieve(
        "How many videos can I process?",
        top_k=3,
    )

    assert results
    combined_text = " ".join(
        text for text, _, _ in results
    ).lower()

    assert "video" in combined_text


def test_refund_query_retrieves_policy():
    pipeline = RAGPipeline()

    sources = _sources_for_query(
        pipeline,
        "Can I get a refund?",
    )

    assert "policy" in sources


def test_support_query_retrieves_relevant_information():
    pipeline = RAGPipeline()

    results = pipeline.retrieve(
        "Do you provide 24/7 support?",
        top_k=3,
    )

    assert results

    combined_text = " ".join(
        text for text, _, _ in results
    ).lower()

    assert (
        "support" in combined_text
        or "24/7" in combined_text
    )


def test_faq_query_retrieves_faq_information():
    pipeline = RAGPipeline()

    results = pipeline.retrieve(
        "What is AutoStream?",
        top_k=3,
    )

    assert results


def test_empty_query_returns_no_results():
    pipeline = RAGPipeline()

    assert pipeline.retrieve("") == []
    assert pipeline.retrieve("   ") == []


def test_top_k_is_respected():
    pipeline = RAGPipeline()

    results = pipeline.retrieve(
        "AutoStream pricing features",
        top_k=2,
    )

    assert len(results) <= 2


def test_context_string_contains_source_marker():
    pipeline = RAGPipeline()

    context = pipeline.get_context_string(
        "What does the Pro plan cost?"
    )

    assert "[Source:" in context
    assert "pricing" in context.lower()


def test_unrelated_query_does_not_crash():
    pipeline = RAGPipeline()

    results = pipeline.retrieve(
        "quantum mechanics spacecraft agriculture",
    )

    assert isinstance(results, list)
    