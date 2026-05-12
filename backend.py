"""FastAPI backend for the multi-agent decision framework.

Exposes three POST endpoints (/analyze/payments, /analyze/churn,
/analyze/fraud) that accept an entity ID and an agent_type flag
(``heuristic`` or ``llm``), run the corresponding workflow, and
return the full chain-of-agent reasoning.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Domain imports — Payments
# ---------------------------------------------------------------------------
from src.domains.payments.agents import (
    PaymentDataAgent,
    PaymentDecisionAgent,
    PaymentExplanationAgent,
    PaymentRiskAgent,
)
from src.domains.payments.agents_llm import (
    LLMDecisionAgent as PaymentLLMDecisionAgent,
    LLMExplanationAgent as PaymentLLMExplanationAgent,
    LLMRiskAgent as PaymentLLMRiskAgent,
)
from src.domains.payments.data import PaymentDataGenerator
from src.domains.payments.rag_retriever import PaymentRAGRetriever
from src.domains.payments.tools import create_tools as create_payment_tools

# ---------------------------------------------------------------------------
# Domain imports — Churn
# ---------------------------------------------------------------------------
from src.domains.churn.agents import (
    ChurnDataAgent,
    ChurnDecisionAgent,
    ChurnExplanationAgent,
    ChurnRiskAgent,
)
from src.domains.churn.agents_llm import (
    LLMChurnAnalysisAgent,
    LLMRetentionDecisionAgent,
    LLMRetentionExplainerAgent,
)
from src.domains.churn.data import ChurnDataGenerator
from src.domains.churn.rag_retriever import ChurnRAGRetriever
from src.domains.churn.tools import create_tools as create_churn_tools

# ---------------------------------------------------------------------------
# Domain imports — Fraud
# ---------------------------------------------------------------------------
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
from src.domains.fraud.rag_retriever import FraudRAGRetriever
from src.domains.fraud.tools import create_tools as create_fraud_tools

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
from src.core.schemas import DecisionRequest
from src.core.workflow import DecisionWorkflow


# ============================================================================
# FastAPI app
# ============================================================================

app = FastAPI(
    title="Multi-Agent Decision Framework",
    description=(
        "Unified REST API for multi-agent decision-making across "
        "Payments, Churn, and Fraud domains.  Choose between fast "
        "heuristic agents or LLM-powered agents with RAG."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request / Response schemas
# ============================================================================


class PaymentAnalysisRequest(BaseModel):
    customer_id: str = "CUST_00000"
    amount: float = 2500.0
    agent_type: Optional[str] = "heuristic"  # "heuristic" | "llm"


class ChurnAnalysisRequest(BaseModel):
    customer_id: str = "CUST_00000"
    agent_type: Optional[str] = "heuristic"


class FraudAnalysisRequest(BaseModel):
    transaction_id: str = "TXN_00000000"
    agent_type: Optional[str] = "heuristic"


class AgentTrace(BaseModel):
    agent: str
    analysis: Dict[str, Any]


class DecisionResponse(BaseModel):
    domain: str
    decision: str
    decision_score: float
    reasoning: str
    agent_outputs: List[AgentTrace]
    agent_type: str


# ============================================================================
# One-time data initialisation (module-level singletons)
# ============================================================================

_payment_gen = PaymentDataGenerator(seed=42)
_payment_customers = _payment_gen.generate_customers(50)
_payment_transactions = _payment_gen.generate_transactions(_payment_customers, 100)
_payment_tools = create_payment_tools(_payment_customers, _payment_transactions)
_payment_rag = PaymentRAGRetriever()

_churn_gen = ChurnDataGenerator(seed=42)
_churn_customers = _churn_gen.generate_customers(50)
_churn_tools = create_churn_tools(_churn_customers)
_churn_rag = ChurnRAGRetriever()

_fraud_gen = FraudDataGenerator(seed=42)
_fraud_transactions = _fraud_gen.generate_transactions(100)
_fraud_tools = create_fraud_tools(_fraud_transactions)
_fraud_rag = FraudRAGRetriever()


# ============================================================================
# Input Validation
# ============================================================================


def _validate_customer_id(customer_id: str) -> None:
    """Raise HTTPException 400 if *customer_id* is not in CUST_00000..CUST_00049."""
    if not customer_id.startswith("CUST_"):
        raise HTTPException(
            status_code=400,
            detail="Invalid customer ID format. Must start with 'CUST_' (e.g., CUST_00000).",
        )
    try:
        num = int(customer_id[5:])
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid customer ID format. Use CUST_XXXXX where X is a digit.",
        )
    if not (0 <= num < 50):
        raise HTTPException(
            status_code=400,
            detail=f"Customer ID out of range. Valid: CUST_00000 to CUST_00049 (got {customer_id}).",
        )


def _validate_transaction_id(transaction_id: str) -> None:
    """Raise HTTPException 400 if *transaction_id* is not in TXN_00000000..TXN_00000099."""
    if not transaction_id.startswith("TXN_"):
        raise HTTPException(
            status_code=400,
            detail="Invalid transaction ID format. Must start with 'TXN_' (e.g., TXN_00000000).",
        )
    try:
        num = int(transaction_id[4:])
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid transaction ID format. Use TXN_XXXXXXXX where X is a digit.",
        )
    if not (0 <= num < 100):
        raise HTTPException(
            status_code=400,
            detail=f"Transaction ID out of range. Valid: TXN_00000000 to TXN_00000099 (got {transaction_id}).",
        )


# ============================================================================
# Endpoints
# ============================================================================


@app.get("/health")
async def health_check():
    """Quick health / readiness probe."""
    return {
        "status": "healthy",
        "domains": ["payments", "churn", "fraud"],
        "agent_types": ["heuristic", "llm"],
    }


@app.post("/analyze/payments", response_model=DecisionResponse)
async def analyze_payment(req: PaymentAnalysisRequest):
    """Analyze a payment for fraud using heuristic or LLM agents."""
    _validate_customer_id(req.customer_id)
    try:
        if req.agent_type == "llm":
            agents = [
                PaymentDataAgent(_payment_tools),
                PaymentLLMRiskAgent(_payment_rag),
                PaymentLLMDecisionAgent(_payment_rag),
                PaymentLLMExplanationAgent(),
            ]
        else:
            agents = [
                PaymentDataAgent(_payment_tools),
                PaymentRiskAgent(_payment_tools),
                PaymentDecisionAgent(_payment_tools),
                PaymentExplanationAgent(_payment_tools),
            ]

        result = await DecisionWorkflow(agents, "payment_api").execute(
            DecisionRequest(
                domain="payments",
                entity_id=req.customer_id,
                context={"customer_id": req.customer_id, "amount": req.amount},
            )
        )

        return DecisionResponse(
            domain="payments",
            decision=result.decision,
            decision_score=result.decision_score,
            reasoning=result.reasoning,
            agent_outputs=[
                AgentTrace(agent=o.agent_name, analysis=o.analysis)
                for o in result.agent_outputs
            ],
            agent_type=req.agent_type or "heuristic",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/analyze/churn", response_model=DecisionResponse)
async def analyze_churn(req: ChurnAnalysisRequest):
    """Analyze customer churn risk using heuristic or LLM agents."""
    _validate_customer_id(req.customer_id)
    try:
        if req.agent_type == "llm":
            agents = [
                ChurnDataAgent(_churn_tools),
                LLMChurnAnalysisAgent(_churn_rag),
                LLMRetentionDecisionAgent(_churn_rag),
                LLMRetentionExplainerAgent(),
            ]
        else:
            agents = [
                ChurnDataAgent(_churn_tools),
                ChurnRiskAgent(_churn_tools),
                ChurnDecisionAgent(_churn_tools),
                ChurnExplanationAgent(_churn_tools),
            ]

        result = await DecisionWorkflow(agents, "churn_api").execute(
            DecisionRequest(
                domain="churn",
                entity_id=req.customer_id,
                context={"customer_id": req.customer_id},
            )
        )

        return DecisionResponse(
            domain="churn",
            decision=result.decision,
            decision_score=result.decision_score,
            reasoning=result.reasoning,
            agent_outputs=[
                AgentTrace(agent=o.agent_name, analysis=o.analysis)
                for o in result.agent_outputs
            ],
            agent_type=req.agent_type or "heuristic",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/analyze/fraud", response_model=DecisionResponse)
async def analyze_fraud(req: FraudAnalysisRequest):
    """Analyze a transaction for fraud using heuristic or LLM agents."""
    _validate_transaction_id(req.transaction_id)
    try:
        if req.agent_type == "llm":
            agents = [
                FraudDataAgent(_fraud_tools),
                LLMFraudAnalysisAgent(_fraud_rag),
                LLMFraudDecisionAgent(_fraud_rag),
                LLMFraudExplainerAgent(),
            ]
        else:
            agents = [
                FraudDataAgent(_fraud_tools),
                FraudRiskAgent(_fraud_tools),
                FraudDecisionAgent(_fraud_tools),
                FraudExplanationAgent(_fraud_tools),
            ]

        result = await DecisionWorkflow(agents, "fraud_api").execute(
            DecisionRequest(
                domain="fraud",
                entity_id=req.transaction_id,
                context={"transaction_id": req.transaction_id},
            )
        )

        return DecisionResponse(
            domain="fraud",
            decision=result.decision,
            decision_score=result.decision_score,
            reasoning=result.reasoning,
            agent_outputs=[
                AgentTrace(agent=o.agent_name, analysis=o.analysis)
                for o in result.agent_outputs
            ],
            agent_type=req.agent_type or "heuristic",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ============================================================================
# Entry-point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
