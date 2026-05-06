"""LLM-powered payment agents with RAG.

These agents extend the core Agent base class and use:
- ChromaDB-backed RAG (PaymentRAGRetriever) for policy context
- Claude via langchain-anthropic for reasoning

When no API key is configured, agents degrade gracefully to a mock response
so that the full test suite continues to pass without live credentials.
"""
import os
from typing import Any, Dict

from src.core.agent import Agent
from src.core.schemas import AgentOutput

from .llm_prompts import (
    SYSTEM_PROMPT_DECISION_MAKER,
    SYSTEM_PROMPT_EXPLAINER,
    SYSTEM_PROMPT_RISK_ANALYSIS,
    get_decision_prompt,
    get_explanation_prompt,
    get_risk_analysis_prompt,
)
from .rag_retriever import PaymentRAGRetriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_llm(temperature: float = 0.3):
    """Try to construct a ChatAnthropic LLM; return None on failure."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key.startswith("your-"):
        return None  # No real key — run in mock mode

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
    """Invoke *llm* with a system + user message; return *fallback* on error."""
    if llm is None:
        return fallback
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return response.content
    except Exception as exc:
        print(f"[LLM] Invocation error: {exc}")
        return fallback


# ---------------------------------------------------------------------------
# LLM Agents
# ---------------------------------------------------------------------------


class LLMRiskAgent(Agent):
    """LLM-powered risk analysis agent with RAG-retrieved policy context."""

    def __init__(self, rag_retriever: PaymentRAGRetriever) -> None:
        super().__init__(
            name="llm_risk_agent",
            responsibility="Analyze fraud risk using LLM reasoning and payment policies",
        )
        self.rag_retriever = rag_retriever
        self.llm = _build_llm(temperature=0.3)

    async def run(self, context: Dict[str, Any]) -> AgentOutput:
        customer_id = context.get("customer_id", "CUST_00001")
        amount = context.get("amount", 1000)

        # Customer profile may have been enriched by a preceding DataAgent
        customer_profile: Dict[str, Any] = context.get(
            "previous_agent_output", {}
        ).get("customer_profile", {"customer_id": customer_id})

        # RAG: retrieve policies relevant to this transaction
        rag_query = f"fraud detection payment approval amount {amount}"
        retrieved_docs = self.rag_retriever.retrieve(rag_query, top_k=3)
        policies_context = self.rag_retriever.format_context(retrieved_docs)

        # Build prompt and call LLM
        transaction = {"amount": amount, "merchant": context.get("merchant", "Unknown")}
        user_msg = get_risk_analysis_prompt(customer_profile, transaction, policies_context)
        fallback = (
            f"Mock risk analysis for ${amount} transaction by {customer_id}. "
            "No LLM key configured — risk treated as MEDIUM."
        )
        llm_analysis = _invoke(self.llm, SYSTEM_PROMPT_RISK_ANALYSIS, user_msg, fallback)

        analysis: Dict[str, Any] = {
            "risk_analysis": llm_analysis,
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


class LLMDecisionAgent(Agent):
    """LLM-powered decision agent — reads risk analysis from previous agent."""

    def __init__(self, rag_retriever: PaymentRAGRetriever) -> None:
        super().__init__(
            name="llm_decision_agent",
            responsibility="Make approval decision using LLM reasoning",
        )
        self.rag_retriever = rag_retriever
        self.llm = _build_llm(temperature=0.2)  # Lower temperature for decisions

    async def run(self, context: Dict[str, Any]) -> AgentOutput:
        prev = context.get("previous_agent_output", {})
        risk_analysis: str = prev.get("risk_analysis", "No prior risk analysis available.")
        customer_profile: Dict[str, Any] = prev.get(
            "customer_profile", {}
        ) or context.get("customer_profile", {})

        # RAG: retrieve approval/tier policies
        retrieved_docs = self.rag_retriever.retrieve("approval rules customer tier", top_k=2)
        policies_context = self.rag_retriever.format_context(retrieved_docs)

        user_msg = get_decision_prompt(risk_analysis, customer_profile, policies_context)
        fallback = "REVIEW\nNo LLM key configured — defaulting to human review."
        decision_text = _invoke(
            self.llm, SYSTEM_PROMPT_DECISION_MAKER, user_msg, fallback
        )

        # Parse the decision from the first line of the LLM response
        first_line = decision_text.strip().upper().splitlines()[0]
        if "APPROVE" in first_line:
            decision = "APPROVE"
            score = 0.80
        elif "DECLINE" in first_line:
            decision = "DECLINE"
            score = 0.15
        else:
            decision = "REVIEW"
            score = 0.50

        analysis: Dict[str, Any] = {
            "decision": decision,
            "llm_reasoning": decision_text,
            "score": score,
        }

        return AgentOutput(
            agent_name=self.name,
            analysis=analysis,
            tools_used=["rag_retriever", "llm_claude"],
            reasoning=f"LLM decision: {decision}",
        )


class LLMExplanationAgent(Agent):
    """LLM-powered explanation agent — produces plain-English decision summaries."""

    def __init__(self) -> None:
        super().__init__(
            name="llm_explanation_agent",
            responsibility="Generate clear explanations using LLM",
        )
        self.llm = _build_llm(temperature=0.5)

    async def run(self, context: Dict[str, Any]) -> AgentOutput:
        prev = context.get("previous_agent_output", {})
        decision: str = prev.get("decision", "PENDING")
        reasoning: str = prev.get("llm_reasoning", prev.get("reason", ""))

        user_msg = get_explanation_prompt(decision, reasoning)
        fallback = (
            f"Your payment has been set to {decision}. "
            "Our system evaluated your transaction history and applied payment policies. "
            "Please contact support for more details."
        )
        explanation = _invoke(self.llm, SYSTEM_PROMPT_EXPLAINER, user_msg, fallback)

        analysis: Dict[str, Any] = {
            "decision": decision,
            "explanation": explanation,
            "llm_generated": True,
        }

        return AgentOutput(
            agent_name=self.name,
            analysis=analysis,
            tools_used=["llm_claude"],
            reasoning="Generated plain-English explanation using LLM",
        )
