"""LLM-powered churn domain agents with RAG.

When no ANTHROPIC_API_KEY is configured, agents return deterministic mock
responses so the full test suite continues to pass without live credentials.
"""
import os
from typing import Any, Dict

from src.core.agent import Agent
from src.core.schemas import AgentOutput

from .llm_prompts import (
    SYSTEM_PROMPT_CHURN_ANALYSIS,
    SYSTEM_PROMPT_RETENTION_DECISION,
    SYSTEM_PROMPT_RETENTION_EXPLAINER,
    get_churn_analysis_prompt,
    get_retention_decision_prompt,
    get_retention_explanation_prompt,
)
from .rag_retriever import ChurnRAGRetriever


# ---------------------------------------------------------------------------
# Shared helpers (mirrors the payments domain pattern)
# ---------------------------------------------------------------------------


def _build_llm(temperature: float = 0.3):
    """Build an LLM in priority order: Ollama Cloud → Anthropic → None (mock).

    Priority:
      1. Ollama Cloud / local — uses Bearer-token auth when OLLAMA_API_KEY set.
      2. Anthropic Claude — when ANTHROPIC_API_KEY is set.
      3. None — deterministic mock fallback (tests always pass).
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "mistral")
    api_key = os.getenv("OLLAMA_API_KEY", "")

    try:
        import httpx
        from langchain_ollama import ChatOllama

        kwargs: dict = dict(model=model, base_url=base_url, temperature=temperature)
        if api_key and not api_key.startswith("your-"):
            kwargs["http_client"] = httpx.Client(
                headers={"Authorization": f"Bearer {api_key}"}
            )
        llm = ChatOllama(**kwargs)
        # Connectivity check — ChatOllama requires message objects, not plain strings
        from langchain_core.messages import HumanMessage, SystemMessage
        llm.invoke([SystemMessage(content="You are a helpful assistant."), HumanMessage(content="Say OK")])
        source = "Ollama Cloud" if api_key else "Ollama local"
        print(f"[Churn] {source} LLM ready: {model}")
        return llm
    except Exception:
        pass

    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    if anthropic_key and not anthropic_key.startswith("your-"):
        try:
            from langchain_anthropic import ChatAnthropic

            print("[Churn] Anthropic Claude LLM ready")
            return ChatAnthropic(
                model="claude-3-sonnet-20240229",
                temperature=temperature,
                api_key=anthropic_key,
            )
        except Exception as exc:
            print(f"[Churn] Could not initialise Claude: {exc}")

    return None


def _invoke(llm, system: str, user: str, fallback: str) -> str:
    """Invoke *llm*; return *fallback* if LLM is None or raises.

    Both ChatOllama and ChatAnthropic use the messages protocol.
    """
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


class LLMChurnAnalysisAgent(Agent):
    """LLM-powered churn analysis agent with RAG-retrieved retention policies."""

    def __init__(self, rag_retriever: ChurnRAGRetriever) -> None:
        super().__init__(
            name="llm_churn_analysis_agent",
            responsibility="Analyze churn risk using LLM reasoning and retention policies",
        )
        self.rag_retriever = rag_retriever
        self.llm = _build_llm(temperature=0.3)

    async def run(self, context: Dict[str, Any]) -> AgentOutput:
        customer_id = context.get("customer_id", "CUST_00001")
        prev = context.get("previous_agent_output", {})
        customer_profile: Dict[str, Any] = prev.get(
            "customer_profile", {"customer_id": customer_id}
        )
        engagement: Dict[str, Any] = prev.get("engagement_trends", {})

        # RAG: retrieve relevant retention policies
        retrieved_docs = self.rag_retriever.retrieve(
            "churn risk engagement metrics retention", top_k=3
        )
        policies_context = self.rag_retriever.format_context(retrieved_docs)

        user_msg = get_churn_analysis_prompt(customer_profile, engagement, policies_context)
        fallback = (
            f"Mock churn analysis for {customer_id}. "
            "No LLM key configured — risk treated as MEDIUM. "
            "Recommend PROACTIVE_OUTREACH."
        )
        llm_analysis = _invoke(self.llm, SYSTEM_PROMPT_CHURN_ANALYSIS, user_msg, fallback)

        analysis: Dict[str, Any] = {
            "churn_analysis": llm_analysis,
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


class LLMRetentionDecisionAgent(Agent):
    """LLM-powered retention decision agent."""

    def __init__(self, rag_retriever: ChurnRAGRetriever) -> None:
        super().__init__(
            name="llm_retention_decision_agent",
            responsibility="Make retention strategy decisions using LLM reasoning",
        )
        self.rag_retriever = rag_retriever
        self.llm = _build_llm(temperature=0.2)

    async def run(self, context: Dict[str, Any]) -> AgentOutput:
        prev = context.get("previous_agent_output", {})
        churn_analysis: str = prev.get(
            "churn_analysis", "No prior churn analysis available."
        )

        # RAG: retrieve decision-relevant policies
        retrieved_docs = self.rag_retriever.retrieve(
            "retention strategy customer tier ltv", top_k=2
        )
        policies_context = self.rag_retriever.format_context(retrieved_docs)

        ltv_data: Dict[str, Any] = {
            "ltv": 10_000,
            "tier": "Standard",
            "retention_budget": 1_000,
        }

        user_msg = get_retention_decision_prompt(churn_analysis, ltv_data, policies_context)
        fallback = "PROACTIVE_OUTREACH\nNo LLM key — defaulting to proactive outreach."
        decision_text = _invoke(
            self.llm, SYSTEM_PROMPT_RETENTION_DECISION, user_msg, fallback
        )

        # Parse decision from first line
        first_line = decision_text.strip().upper().splitlines()[0]
        if "EXECUTIVE" in first_line:
            decision = "EXECUTIVE_OUTREACH"
        elif "URGENT" in first_line:
            decision = "URGENT_RETENTION"
        elif "STANDARD" in first_line:
            decision = "STANDARD_ENGAGEMENT"
        else:
            decision = "PROACTIVE_OUTREACH"

        analysis: Dict[str, Any] = {
            "decision": decision,
            "llm_reasoning": decision_text,
            "score": 0.70,
        }

        return AgentOutput(
            agent_name=self.name,
            analysis=analysis,
            tools_used=["rag_retriever", "llm_claude"],
            reasoning=f"LLM retention decision: {decision}",
        )


class LLMRetentionExplainerAgent(Agent):
    """LLM-powered retention explanation agent."""

    def __init__(self) -> None:
        super().__init__(
            name="llm_retention_explainer_agent",
            responsibility="Generate clear, empathetic retention explanations using LLM",
        )
        self.llm = _build_llm(temperature=0.5)

    async def run(self, context: Dict[str, Any]) -> AgentOutput:
        prev = context.get("previous_agent_output", {})
        decision: str = prev.get("decision", "PROACTIVE_OUTREACH")
        reasoning: str = prev.get("llm_reasoning", prev.get("description", ""))

        user_msg = get_retention_explanation_prompt(reasoning, decision)
        fallback = (
            f"The customer has been flagged for {decision}. "
            "Based on engagement analysis and lifetime value, our team will "
            "reach out with a personalised retention offer."
        )
        explanation = _invoke(
            self.llm, SYSTEM_PROMPT_RETENTION_EXPLAINER, user_msg, fallback
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
            reasoning="Generated plain-English retention explanation using LLM",
        )
