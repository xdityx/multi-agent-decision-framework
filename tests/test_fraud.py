"""Tests for the fraud domain — heuristic agents, LLM agents, and workflows."""
import pytest

from src.core.schemas import DecisionRequest
from src.core.workflow import DecisionWorkflow
from src.domains.fraud.agents import (
    FraudDataAgent,
    FraudDecisionAgent,
    FraudExplanationAgent,
    FraudRiskAgent,
)
from src.domains.fraud.agents_llm import (
    LLMFraudAnalysisAgent,
    LLMFraudDecisionAgent,
    LLMFraudExplainerAgent,
)
from src.domains.fraud.data import FraudDataGenerator
from src.domains.fraud.rag_documents import (
    FRAUD_POLICIES,
    get_policy_content,
    search_relevant_policies,
)
from src.domains.fraud.rag_retriever import FraudRAGRetriever
from src.domains.fraud.tools import create_tools

VALID_ACTIONS = {"BLOCK", "CHALLENGE", "MONITOR", "APPROVE"}
FIRST_TXN = "TXN_00000000"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fraud_transactions():
    """Generate 100 deterministic transactions (seed=42)."""
    return FraudDataGenerator(seed=42).generate_transactions(count=100)


@pytest.fixture(scope="module")
def fraud_tools(fraud_transactions):
    return create_tools(fraud_transactions)


@pytest.fixture(scope="module")
def rag_retriever():
    return FraudRAGRetriever()


# ---------------------------------------------------------------------------
# RAG document tests
# ---------------------------------------------------------------------------


def test_fraud_policies_count():
    """Policy corpus contains exactly 5 documents."""
    assert len(FRAUD_POLICIES) == 5


def test_fraud_policy_ids_unique():
    """Every policy has a unique ID."""
    ids = [p["id"] for p in FRAUD_POLICIES]
    assert len(ids) == len(set(ids))


def test_get_policy_content_found():
    """Known policy ID returns its content."""
    content = get_policy_content("policy_red_flags_001")
    assert "fraud" in content.lower()


def test_get_policy_content_missing():
    """Unknown ID returns the sentinel string."""
    assert get_policy_content("nonexistent") == "Policy not found"


def test_search_policies_hit():
    """Keyword search finds at least one match for 'fraud'."""
    results = search_relevant_policies("fraud")
    assert len(results) >= 1
    assert all("id" in r and "title" in r and "excerpt" in r for r in results)


def test_search_policies_no_hit():
    """Irrelevant query returns empty list."""
    assert search_relevant_policies("xyzzy_no_match") == []


# ---------------------------------------------------------------------------
# RAG retriever tests
# ---------------------------------------------------------------------------


def test_rag_retriever_init(rag_retriever):
    """Retriever loads all five policies."""
    assert len(rag_retriever.policies) == 5


def test_rag_retriever_returns_docs(rag_retriever):
    """retrieve() returns documents for a relevant query."""
    docs = rag_retriever.retrieve("fraud detection red flags", top_k=3)
    assert len(docs) >= 1
    assert "title" in docs[0]
    assert "content" in docs[0]


def test_rag_retriever_top_k(rag_retriever):
    """retrieve() never exceeds top_k."""
    docs = rag_retriever.retrieve("fraud", top_k=2)
    assert len(docs) <= 2


def test_rag_format_context(rag_retriever):
    """format_context() produces a labelled block."""
    docs = rag_retriever.retrieve("fraud detection", top_k=2)
    ctx = rag_retriever.format_context(docs)
    assert "Relevant Fraud Detection Policies" in ctx


def test_rag_format_context_empty(rag_retriever):
    """format_context([]) returns the sentinel string."""
    assert "No relevant fraud policies found" in rag_retriever.format_context([])


# ---------------------------------------------------------------------------
# Data + tool tests
# ---------------------------------------------------------------------------


def test_fraud_data_generation(fraud_transactions):
    """Generator produces the right count with correct first ID."""
    assert len(fraud_transactions) == 100
    assert fraud_transactions[0].transaction_id == FIRST_TXN


def test_transaction_ids_unique(fraud_transactions):
    """All transaction IDs are unique."""
    ids = [t.transaction_id for t in fraud_transactions]
    assert len(ids) == len(set(ids))


def test_fraud_tools_transaction_details(fraud_tools):
    """get_transaction_details returns core fields."""
    details = fraud_tools.get_transaction_details(FIRST_TXN)
    assert "amount" in details
    assert "merchant" in details
    assert "vpn_detected" in details


def test_fraud_tools_unknown_transaction(fraud_tools):
    """Unknown transaction ID returns an error dict."""
    result = fraud_tools.get_transaction_details("TXN_XXXXX")
    assert "error" in result


def test_fraud_tools_fraud_score(fraud_tools):
    """calculate_fraud_score returns score in [0, 100]."""
    result = fraud_tools.calculate_fraud_score(FIRST_TXN)
    assert 0 <= result["fraud_score"] <= 100
    assert "reasoning" in result


def test_fraud_tools_blocklist(fraud_tools):
    """check_blocklist returns a deterministic result."""
    result = fraud_tools.check_blocklist(FIRST_TXN)
    assert "is_blocklisted" in result
    assert result["status"] in {"flagged", "clear"}
    # Deterministic — same call returns same result
    assert fraud_tools.check_blocklist(FIRST_TXN)["is_blocklisted"] == result["is_blocklisted"]


def test_fraud_tools_geo_check(fraud_tools):
    """check_geographic_consistency returns expected fields."""
    result = fraud_tools.check_geographic_consistency(FIRST_TXN)
    assert "is_consistent" in result
    assert result["risk"] in {"low", "high"}


def test_determine_action_block_high_score(fraud_tools):
    """Fraud score > 85 → BLOCK."""
    result = fraud_tools.determine_fraud_action(90, False, {"risk": "low"})
    assert result["action"] == "BLOCK"


def test_determine_action_block_blocklist(fraud_tools):
    """Blocklist flag → BLOCK regardless of score."""
    result = fraud_tools.determine_fraud_action(30, True, {"risk": "low"})
    assert result["action"] == "BLOCK"


def test_determine_action_challenge(fraud_tools):
    """Score 61-85 → CHALLENGE."""
    result = fraud_tools.determine_fraud_action(70, False, {"risk": "low"})
    assert result["action"] == "CHALLENGE"


def test_determine_action_monitor(fraud_tools):
    """Score 41-60 → MONITOR."""
    result = fraud_tools.determine_fraud_action(50, False, {"risk": "low"})
    assert result["action"] == "MONITOR"


def test_determine_action_approve(fraud_tools):
    """Score ≤ 40 → APPROVE."""
    result = fraud_tools.determine_fraud_action(25, False, {"risk": "low"})
    assert result["action"] == "APPROVE"


# ---------------------------------------------------------------------------
# Heuristic agent tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fraud_data_agent(fraud_tools):
    """FraudDataAgent populates transaction_details."""
    agent = FraudDataAgent(fraud_tools)
    output = await agent.run({"transaction_id": FIRST_TXN})

    assert output.agent_name == "fraud_data_agent"
    assert "transaction_details" in output.analysis


@pytest.mark.asyncio
async def test_fraud_risk_agent(fraud_tools):
    """FraudRiskAgent produces fraud_score, blocklist_flag, geo consistency."""
    agent = FraudRiskAgent(fraud_tools)
    output = await agent.run({"transaction_id": FIRST_TXN})

    assert output.agent_name == "fraud_risk_agent"
    assert "fraud_score" in output.analysis
    assert output.analysis["risk_category"] in {"HIGH", "MEDIUM", "LOW"}


@pytest.mark.asyncio
async def test_fraud_decision_agent(fraud_tools):
    """FraudDecisionAgent maps risk to a valid action."""
    agent = FraudDecisionAgent(fraud_tools)
    context = {
        "previous_agent_output": {
            "fraud_score": 65,
            "blocklist_flag": False,
            "geographic_consistency": True,
        }
    }
    output = await agent.run(context)

    assert output.agent_name == "fraud_decision_agent"
    assert output.analysis["decision"] in VALID_ACTIONS
    assert 0.0 <= output.analysis["score"] <= 1.0


@pytest.mark.asyncio
async def test_fraud_explanation_agent(fraud_tools):
    """FraudExplanationAgent produces a non-empty explanation."""
    agent = FraudExplanationAgent(fraud_tools)
    context = {
        "previous_agent_output": {
            "decision": "BLOCK",
            "reason": "High fraud probability.",
        }
    }
    output = await agent.run(context)

    assert output.agent_name == "fraud_explanation_agent"
    assert "BLOCK" in output.analysis["explanation"] or "blocked" in output.analysis["explanation"].lower()


# ---------------------------------------------------------------------------
# LLM agent tests (mock mode)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_fraud_analysis_agent(rag_retriever, fraud_tools):
    """LLMFraudAnalysisAgent produces expected output keys in mock mode."""
    agent = LLMFraudAnalysisAgent(rag_retriever)
    txn_details = fraud_tools.get_transaction_details(FIRST_TXN)
    context = {"previous_agent_output": {"transaction_details": txn_details}}
    output = await agent.run(context)

    assert output.agent_name == "llm_fraud_analysis_agent"
    assert "fraud_analysis" in output.analysis
    assert output.analysis["rag_enabled"] is True
    assert isinstance(output.analysis["retrieved_policies"], list)


@pytest.mark.asyncio
async def test_llm_fraud_decision_agent(rag_retriever):
    """LLMFraudDecisionAgent maps to a valid action in mock mode."""
    agent = LLMFraudDecisionAgent(rag_retriever)
    context = {
        "previous_agent_output": {
            "fraud_analysis": "High risk — VPN detected, geo mismatch, failed attempts.",
        }
    }
    output = await agent.run(context)

    assert output.agent_name == "llm_fraud_decision_agent"
    assert output.analysis["decision"] in VALID_ACTIONS
    assert 0.0 <= output.analysis["score"] <= 1.0


@pytest.mark.asyncio
async def test_llm_fraud_explainer_agent():
    """LLMFraudExplainerAgent produces a non-empty explanation in mock mode."""
    agent = LLMFraudExplainerAgent()
    context = {
        "previous_agent_output": {
            "decision": "CHALLENGE",
            "llm_reasoning": "Medium risk — unusual location.",
        }
    }
    output = await agent.run(context)

    assert output.agent_name == "llm_fraud_explainer_agent"
    assert len(output.analysis["explanation"]) > 10
    assert output.analysis["llm_generated"] is True


# ---------------------------------------------------------------------------
# Workflow tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_heuristic_workflow_end_to_end(fraud_tools):
    """Full 4-agent heuristic fraud workflow produces a valid DecisionResult."""
    agents = [
        FraudDataAgent(fraud_tools),
        FraudRiskAgent(fraud_tools),
        FraudDecisionAgent(fraud_tools),
        FraudExplanationAgent(fraud_tools),
    ]
    workflow = DecisionWorkflow(agents, "fraud_heuristic_workflow")

    request = DecisionRequest(
        domain="fraud",
        entity_id=FIRST_TXN,
        context={"transaction_id": FIRST_TXN},
    )
    result = await workflow.execute(request)

    assert result.domain == "fraud"
    assert len(result.agent_outputs) == 4
    assert result.decision in VALID_ACTIONS


@pytest.mark.asyncio
async def test_llm_workflow_end_to_end(rag_retriever):
    """Full 3-agent LLM fraud workflow produces a valid DecisionResult."""
    agents = [
        LLMFraudAnalysisAgent(rag_retriever),
        LLMFraudDecisionAgent(rag_retriever),
        LLMFraudExplainerAgent(),
    ]
    workflow = DecisionWorkflow(agents, "fraud_llm_workflow")

    request = DecisionRequest(
        domain="fraud",
        entity_id=FIRST_TXN,
        context={"transaction_id": FIRST_TXN},
    )
    result = await workflow.execute(request)

    assert result.domain == "fraud"
    assert len(result.agent_outputs) == 3
    assert result.decision in VALID_ACTIONS
