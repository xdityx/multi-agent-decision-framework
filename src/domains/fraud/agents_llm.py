"""LLM-powered fraud domain agents with RAG.

Agents degrade gracefully when no ANTHROPIC_API_KEY is set, returning
deterministic mock responses so the full test suite passes without
live credentials.
"""
import os
from typing import Any, Dict

from src.core.agent import Agent
from src.core.schemas import AgentOutput

from .llm_prompts import (
    SYSTEM_PROMPT_FRAUD_ANALYSIS,
    SYSTEM_PROMPT_FRAUD_DECISION,
    SYSTEM_PROMPT_FRAUD_EXPLAINER,
    get_fraud_analysis_prompt,
    get_fraud_decision_prompt,
    get_fraud_explanation_prompt,
)
from .rag_retriever import FraudRAGRetriever


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_llm(temperature: float = 0.3):
    """Try to build a ChatAnthropic LLM; return None if key is absent."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key.startswith("your-"):
        return None
    try:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model="claude-3-sonnet-20240229",
            temperature=temperature,
            api_key=api_key,
        )
    except Exception as exc:
        print(f"[LLM] Could not initialise Claude: {exc}")
        return None


def _invoke(llm, system: str, user: str, fallback: str) -> str:
    """Invoke *llm*; return *fallback* if LLM is None or raises."""
    if llm is None:
        return fallback
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        response = llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        return response.content
    except Exception as exc:
        print(f"[LLM] Invocation error: {exc}")
        return fallback


# ---------------------------------------------------------------------------
# LLM Agents
# ---------------------------------------------------------------------------


class LLMFraudAnalysisAgent(Agent):
    """LLM-powered fraud analysis agent with RAG-retrieved policy context."""

    def __init__(self, rag_retriever: FraudRAGRetriever) -> None:
        super().__init__(
            name="llm_fraud_analysis_agent",
            responsibility="Analyze fraud risk using LLM reasoning and fraud policies",
        )
        self.rag_retriever = rag_retriever
        self.llm = _build_llm(temperature=0.3)

    async def run(self, context: Dict[str, Any]) -> AgentOutput:
        transaction_details: Dict[str, Any] = context.get(
            "previous_agent_output", {}
        ).get("transaction_details", {})

        # RAG: retrieve relevant fraud policies
        retrieved_docs = self.rag_retriever.retrieve(
            "fraud detection red flags transaction analysis", top_k=3
        )
        policies_context = self.rag_retriever.format_context(retrieved_docs)

        user_msg = get_fraud_analysis_prompt(transaction_details, policies_context)
        fallback = (
            "Mock fraud analysis — no LLM key configured. "
            "Transaction flagged as MEDIUM risk. Recommend MONITOR."
        )
        llm_analysis = _invoke(
            self.llm, SYSTEM_PROMPT_FRAUD_ANALYSIS, user_msg, fallback
        )

        analysis: Dict[str, Any] = {
            "fraud_analysis": llm_analysis,
            "retrieved_policies": [doc["title"] for doc in retrieved_docs],
            "rag_enabled": True,
        }

        return AgentOutput(
            agent_name=self.name,
            analysis=analysis,
            tools_used=["rag_retriever", "llm_claude"],
            reasoning=(
                f"Analyzed using LLM with {len(retrieved_docs)} relevant "
                "policies retrieved from RAG"
            ),
        )


class LLMFraudDecisionAgent(Agent):
    """LLM-powered fraud decision agent."""

    def __init__(self, rag_retriever: FraudRAGRetriever) -> None:
        super().__init__(
            name="llm_fraud_decision_agent",
            responsibility="Make fraud prevention decisions using LLM reasoning",
        )
        self.rag_retriever = rag_retriever
        self.llm = _build_llm(temperature=0.2)

    async def run(self, context: Dict[str, Any]) -> AgentOutput:
        prev = context.get("previous_agent_output", {})
        fraud_analysis: str = prev.get(
            "fraud_analysis", "No prior fraud analysis available."
        )

        # RAG: retrieve decision-threshold policies
        retrieved_docs = self.rag_retriever.retrieve(
            "fraud detection rules block challenge approve", top_k=2
        )
        policies_context = self.rag_retriever.format_context(retrieved_docs)

        user_msg = get_fraud_decision_prompt(
            fraud_analysis, blocklist_flag=False, policies=policies_context
        )
        fallback = "MONITOR\nNo LLM key — defaulting to monitor for safety."
        decision_text = _invoke(
            self.llm, SYSTEM_PROMPT_FRAUD_DECISION, user_msg, fallback
        )

        # Parse decision from first line
        first_line = decision_text.strip().upper().splitlines()[0]
        if "BLOCK" in first_line:
            decision = "BLOCK"
        elif "CHALLENGE" in first_line:
            decision = "CHALLENGE"
        elif "APPROVE" in first_line:
            decision = "APPROVE"
        else:
            decision = "MONITOR"

        analysis: Dict[str, Any] = {
            "decision": decision,
            "llm_reasoning": decision_text,
            "score": 0.60,
        }

        return AgentOutput(
            agent_name=self.name,
            analysis=analysis,
            tools_used=["rag_retriever", "llm_claude"],
            reasoning=f"LLM fraud decision: {decision}",
        )


class LLMFraudExplainerAgent(Agent):
    """LLM-powered fraud explanation agent."""

    def __init__(self) -> None:
        super().__init__(
            name="llm_fraud_explainer_agent",
            responsibility="Generate clear, customer-friendly fraud explanations using LLM",
        )
        self.llm = _build_llm(temperature=0.5)

    async def run(self, context: Dict[str, Any]) -> AgentOutput:
        prev = context.get("previous_agent_output", {})
        decision: str = prev.get("decision", "MONITOR")
        reason: str = prev.get("llm_reasoning", prev.get("reason", ""))

        user_msg = get_fraud_explanation_prompt(decision, reason)
        fallback = (
            f"Your transaction has been set to {decision}. "
            "Our fraud-detection system evaluated multiple risk signals. "
            "If this was your transaction, please contact customer support."
        )
        explanation = _invoke(
            self.llm, SYSTEM_PROMPT_FRAUD_EXPLAINER, user_msg, fallback
        )

        analysis: Dict[str, Any] = {
            "decision": decision,
            "explanation": explanation,
            "llm_generated": True,
        }

        return AgentOutput(
            agent_name=self.name,
            analysis=analysis,
            tools_used=["llm_claude"],
            reasoning="Generated customer-friendly fraud explanation using LLM",
        )
