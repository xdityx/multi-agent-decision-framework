"""Tests for LLM-powered payment agents and RAG retriever."""
import pytest

from src.core.schemas import DecisionRequest
from src.core.workflow import DecisionWorkflow
from src.domains.payments.agents_llm import (
    LLMDecisionAgent,
    LLMExplanationAgent,
    LLMRiskAgent,
)
from src.domains.payments.rag_documents import (
    PAYMENT_POLICIES,
    get_policy_content,
    search_relevant_policies,
)
from src.domains.payments.rag_retriever import PaymentRAGRetriever


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rag_retriever():
    """Shared RAG retriever (ChromaDB client, one per test session)."""
    return PaymentRAGRetriever()


# ---------------------------------------------------------------------------
# RAG document tests
# ---------------------------------------------------------------------------


def test_rag_documents_loaded():
    """Policy corpus has the expected number of documents."""
    assert len(PAYMENT_POLICIES) == 5


def test_get_policy_content_found():
    """Fetching a known policy ID returns non-empty content."""
    content = get_policy_content("policy_fraud_001")
    assert "fraud" in content.lower()


def test_get_policy_content_missing():
    """Fetching an unknown policy ID returns the sentinel string."""
    result = get_policy_content("policy_does_not_exist")
    assert result == "Policy not found"


def test_search_relevant_policies_hits():
    """Keyword search finds at least one matching policy."""
    results = search_relevant_policies("fraud")
    assert len(results) >= 1
    assert "id" in results[0]
    assert "title" in results[0]
    assert "excerpt" in results[0]


def test_search_relevant_policies_no_hits():
    """Search with an irrelevant query returns an empty list."""
    results = search_relevant_policies("xyzzy_no_match_qwerty")
    assert results == []


# ---------------------------------------------------------------------------
# RAG retriever tests
# ---------------------------------------------------------------------------


def test_rag_retriever_initialization(rag_retriever):
    """ChromaDB client and collection are created on init."""
    assert rag_retriever.client is not None
    assert rag_retriever.collection is not None


def test_rag_retriever_population(rag_retriever):
    """Collection holds all policy documents after population."""
    assert rag_retriever.collection.count() == len(PAYMENT_POLICIES)


def test_rag_retriever_retrieve_returns_docs(rag_retriever):
    """retrieve() returns at least one document for a relevant query."""
    docs = rag_retriever.retrieve("fraud detection", top_k=3)
    assert len(docs) >= 1
    assert "title" in docs[0]
    assert "content" in docs[0]


def test_rag_retriever_top_k_respected(rag_retriever):
    """retrieve() never returns more than top_k documents."""
    docs = rag_retriever.retrieve("payment", top_k=2)
    assert len(docs) <= 2


def test_rag_retriever_format_context(rag_retriever):
    """format_context() produces a non-empty string containing policy titles."""
    docs = rag_retriever.retrieve("fraud", top_k=2)
    context = rag_retriever.format_context(docs)
    assert "Relevant Payment Policies" in context
    assert len(context) > 50


def test_rag_retriever_format_context_empty(rag_retriever):
    """format_context() handles an empty document list gracefully."""
    context = rag_retriever.format_context([])
    assert "No relevant policies found" in context


# ---------------------------------------------------------------------------
# LLM agent initialisation tests
# ---------------------------------------------------------------------------


def test_llm_risk_agent_init(rag_retriever):
    """LLMRiskAgent initialises with correct name and retriever."""
    agent = LLMRiskAgent(rag_retriever)
    assert agent.name == "llm_risk_agent"
    assert agent.rag_retriever is rag_retriever


def test_llm_decision_agent_init(rag_retriever):
    """LLMDecisionAgent initialises with correct name and retriever."""
    agent = LLMDecisionAgent(rag_retriever)
    assert agent.name == "llm_decision_agent"
    assert agent.rag_retriever is rag_retriever


def test_llm_explanation_agent_init():
    """LLMExplanationAgent initialises with correct name."""
    agent = LLMExplanationAgent()
    assert agent.name == "llm_explanation_agent"


# ---------------------------------------------------------------------------
# LLM agent execution tests (run without live API key — mock mode)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_risk_agent_run(rag_retriever):
    """LLMRiskAgent.run() produces expected output keys."""
    agent = LLMRiskAgent(rag_retriever)
    context = {
        "customer_id": "CUST_00000",
        "amount": 1500,
        "previous_agent_output": {
            "customer_profile": {
                "customer_id": "CUST_00000",
                "account_age_years": 3,
                "vip_status": "no",
            }
        },
    }
    output = await agent.run(context)

    assert output.agent_name == "llm_risk_agent"
    assert "risk_analysis" in output.analysis
    assert "retrieved_policies" in output.analysis
    assert output.analysis["rag_enabled"] is True
    assert isinstance(output.analysis["retrieved_policies"], list)


@pytest.mark.asyncio
async def test_llm_decision_agent_run(rag_retriever):
    """LLMDecisionAgent.run() maps to a valid decision."""
    agent = LLMDecisionAgent(rag_retriever)
    context = {
        "previous_agent_output": {
            "risk_analysis": "Medium risk. Score estimated at 35.",
            "customer_profile": {"vip_status": "no", "account_age_years": 2},
        }
    }
    output = await agent.run(context)

    assert output.agent_name == "llm_decision_agent"
    assert output.analysis["decision"] in {"APPROVE", "DECLINE", "REVIEW"}
    assert 0.0 <= output.analysis["score"] <= 1.0


@pytest.mark.asyncio
async def test_llm_explanation_agent_run():
    """LLMExplanationAgent.run() produces an explanation string."""
    agent = LLMExplanationAgent()
    context = {
        "previous_agent_output": {
            "decision": "APPROVE",
            "llm_reasoning": "Low risk customer with good history.",
            "score": 0.85,
        }
    }
    output = await agent.run(context)

    assert output.agent_name == "llm_explanation_agent"
    assert "explanation" in output.analysis
    assert isinstance(output.analysis["explanation"], str)
    assert len(output.analysis["explanation"]) > 10
    assert output.analysis["llm_generated"] is True


# ---------------------------------------------------------------------------
# End-to-end LLM workflow test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_workflow_end_to_end(rag_retriever):
    """Three-agent LLM workflow produces a valid DecisionResult."""
    agents = [
        LLMRiskAgent(rag_retriever),
        LLMDecisionAgent(rag_retriever),
        LLMExplanationAgent(),
    ]
    workflow = DecisionWorkflow(agents, "payment_llm_workflow")

    request = DecisionRequest(
        domain="payments",
        entity_id="CUST_00000",
        context={"customer_id": "CUST_00000", "amount": 2500},
    )
    result = await workflow.execute(request)

    assert result.domain == "payments"
    assert len(result.agent_outputs) == 3
    assert result.decision in {"APPROVE", "DECLINE", "REVIEW"}
    assert 0.0 <= result.decision_score <= 1.0
