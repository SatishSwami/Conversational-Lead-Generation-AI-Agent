"""
RAG Pipeline for AutoStream Agent.

Loads the local JSON knowledge base, converts it into retrievable
documents, and performs lightweight TF-IDF retrieval.

The implementation is intentionally self-contained:
- No external vector database
- No network dependency
- Deterministic retrieval
- Easy to evaluate and debug
"""

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KB_PATH = (
    Path(__file__).resolve().parent.parent
    / "autostream_kb.json"
)

DEFAULT_TOP_K = 3
MAX_TOP_K = 10
MIN_RELEVANCE_SCORE = 0.0


# ---------------------------------------------------------------------------
# Knowledge Base Loading
# ---------------------------------------------------------------------------


def _load_knowledge_base() -> dict:
    """
    Load the AutoStream knowledge base from the local JSON file.
    """
    with open(
        KB_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def _flatten_kb_to_documents(
    kb: dict,
) -> List[dict]:
    """
    Convert the nested KB JSON into a flat list of retrievable documents.

    Each document contains:
        {
            "id": str,
            "text": str,
            "source": str,
        }
    """
    documents = []

    # -----------------------------------------------------------------------
    # Company overview
    # -----------------------------------------------------------------------

    company = kb["company"]

    documents.append(
        {
            "id": "company_overview",
            "text": (
                f"{company['name']}: "
                f"{company['tagline']}. "
                f"{company['description']}"
            ),
            "source": "company",
        }
    )

    # -----------------------------------------------------------------------
    # Pricing plans
    # -----------------------------------------------------------------------

    for plan in kb["pricing"]["plans"]:
        features_text = ", ".join(
            plan["features"]
        )

        price_text = (
            f"{plan['name']} Plan costs "
            f"${plan['price_monthly']}/month "
            f"(or ${plan['price_annual']}/year). "
            f"Features include: {features_text}. "
            f"Ideal for: {plan['ideal_for']}."
        )

        documents.append(
            {
                "id": (
                    f"pricing_"
                    f"{plan['name'].lower()}"
                ),
                "text": price_text,
                "source": "pricing",
            }
        )

    # -----------------------------------------------------------------------
    # Policies
    # -----------------------------------------------------------------------

    for policy in kb["policies"]:
        documents.append(
            {
                "id": policy["id"],
                "text": (
                    f"{policy['title']}: "
                    f"{policy['content']}"
                ),
                "source": "policy",
            }
        )

    # -----------------------------------------------------------------------
    # FAQs
    # -----------------------------------------------------------------------

    for index, faq in enumerate(kb["faqs"]):
        documents.append(
            {
                "id": f"faq_{index}",
                "text": (
                    f"Q: {faq['question']} "
                    f"A: {faq['answer']}"
                ),
                "source": "faq",
            }
        )

    return documents


# ---------------------------------------------------------------------------
# Text Processing
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    """
    Normalize text and split it into word-like tokens.

    Keeps useful alphanumeric tokens such as:
        4k
        24/7
        29
    """
    if not text:
        return []

    normalized = text.lower()

    return re.findall(
        r"[a-z0-9]+(?:[./-][a-z0-9]+)*",
        normalized,
    )


def _compute_tf(
    tokens: List[str],
) -> Dict[str, float]:
    """
    Compute normalized term frequency for one document.
    """
    if not tokens:
        return {}

    frequencies: Dict[str, int] = {}

    for token in tokens:
        frequencies[token] = (
            frequencies.get(token, 0) + 1
        )

    total = len(tokens)

    return {
        token: count / total
        for token, count in frequencies.items()
    }


def _compute_idf(
    docs_tokens: List[List[str]],
) -> Dict[str, float]:
    """
    Compute smoothed inverse document frequency.
    """
    document_count = len(docs_tokens)

    if document_count == 0:
        return {}

    idf: Dict[str, float] = {}

    vocabulary = {
        token
        for document in docs_tokens
        for token in document
    }

    for token in vocabulary:
        document_frequency = sum(
            1
            for document in docs_tokens
            if token in document
        )

        idf[token] = (
            math.log(
                (document_count + 1)
                / (document_frequency + 1)
            )
            + 1
        )

    return idf


def _tfidf_score(
    query_tokens: List[str],
    doc_tf: Dict[str, float],
    idf: Dict[str, float],
) -> float:
    """
    Calculate a TF-IDF relevance score between a query
    and document.

    Query terms that appear in the document receive a
    relevance contribution weighted by IDF.
    """
    if not query_tokens:
        return 0.0

    score = 0.0

    for token in set(query_tokens):
        if token in doc_tf and token in idf:
            score += (
                doc_tf[token]
                * idf[token]
            )

    return score


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------


class RAGPipeline:
    """
    Lightweight TF-IDF retrieval pipeline over the
    AutoStream local knowledge base.

    The pipeline is deterministic and requires no
    external vector database or network connection.
    """

    def __init__(self):
        knowledge_base = _load_knowledge_base()

        self.documents = _flatten_kb_to_documents(
            knowledge_base
        )

        documents_tokens = [
            _tokenize(document["text"])
            for document in self.documents
        ]

        self.idf = _compute_idf(
            documents_tokens
        )

        self.doc_tfs = [
            _compute_tf(tokens)
            for tokens in documents_tokens
        ]

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> List[Tuple[str, float, str]]:
        """
        Retrieve the most relevant documents.

        Retrieval combines:
        - TF-IDF lexical scoring
        - Exact phrase matching
        - Important product/entity matching
        - Common product-question phrase matching

        Returns:
            List of tuples:
                (document_text, score, source)
        """
        if not isinstance(query, str):
            return []

        query = query.strip()

        if not query:
            return []

        top_k = max(
            1,
            min(top_k, MAX_TOP_K),
        )

        query_tokens = _tokenize(query)

        if not query_tokens:
            return []

        query_lower = query.lower()

        scored_documents = []

        for document, doc_tf in zip(
            self.documents,
            self.doc_tfs,
        ):
            document_text = document["text"]
            document_lower = document_text.lower()

            # ---------------------------------------------------------------
            # Base TF-IDF score
            # ---------------------------------------------------------------

            score = _tfidf_score(
                query_tokens,
                doc_tf,
                self.idf,
            )

            # ---------------------------------------------------------------
            # Exact phrase boost
            # ---------------------------------------------------------------

            normalized_query = " ".join(
                query_tokens
            )

            normalized_document = " ".join(
                _tokenize(document_text)
            )

            if (
                normalized_query
                and normalized_query in normalized_document
            ):
                score += 2.0

            # ---------------------------------------------------------------
            # Important product/entity matching
            # ---------------------------------------------------------------

            important_terms = [
                "pro",
                "basic",
                "pricing",
                "price",
                "cost",
                "refund",
                "support",
                "videos",
                "4k",
                "captions",
            ]

            for term in important_terms:
                if (
                    term in query_lower
                    and term in document_lower
                ):
                    score += 0.5

            # ---------------------------------------------------------------
            # Question phrase matching
            # ---------------------------------------------------------------

            query_phrases = [
                phrase
                for phrase in (
                    "pro plan",
                    "basic plan",
                    "how much",
                    "price",
                    "cost",
                    "refund",
                    "24/7 support",
                    "ai captions",
                    "unlimited videos",
                )
                if phrase in query_lower
            ]

            for phrase in query_phrases:
                if phrase in document_lower:
                    score += 1.5

            # ---------------------------------------------------------------
            # Store relevant documents
            # ---------------------------------------------------------------

            if score > MIN_RELEVANCE_SCORE:
                scored_documents.append(
                    (
                        document_text,
                        score,
                        document["source"],
                    )
                )

        # Highest relevance first
        scored_documents.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return scored_documents[:top_k]

    def get_context_string(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> str:
        """
        Return formatted retrieval context suitable
        for injection into an LLM prompt.
        """
        results = self.retrieve(
            query,
            top_k=top_k,
        )

        if not results:
            return (
                "No relevant information found "
                "in the knowledge base."
            )

        context_parts = []

        for index, (
            text,
            score,
            source,
        ) in enumerate(results, 1):
            context_parts.append(
                f"[Source: {source}]\n"
                f"{text}"
            )

        return "\n\n".join(context_parts)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


_rag_pipeline: RAGPipeline | None = None


def get_rag_pipeline() -> RAGPipeline:
    """
    Return the shared RAG pipeline instance.

    The knowledge base and TF-IDF statistics are loaded
    only once per application process.
    """
    global _rag_pipeline

    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()

    return _rag_pipeline