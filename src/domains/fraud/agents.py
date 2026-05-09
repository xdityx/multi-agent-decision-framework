"""Fraud domain heuristic agents — inherit from core Agent."""
from typing import Any, Dict

from src.core.agent import Agent
from src.core.schemas import AgentOutput

from .tools import FraudTools

VALID_ACTIONS = {"BLOCK", "CHALLENGE", "MONITOR", "APPROVE"}


class FraudDataAgent(Agent):
    """Agent that gathers raw transaction context."""

    def __init__(self, tools: FraudTools) -> None:
        super().__init__(
            name="fraud_data_agent",
            responsibility="Gather transaction context and patterns",
        )
        self.fraud_tools = tools

    async def run(self, context: Dict[str, Any]) -> AgentOutput:
        transaction_id = context.get("transaction_id", "TXN_00000000")
        txn_details = self.fraud_tools.get_transaction_details(transaction_id)

        return AgentOutput(
            agent_name=self.name,
            analysis={"transaction_details": txn_details},
            tools_used=["get_transaction_details"],
            reasoning=f"Gathered transaction context for {transaction_id}",
        )


class FraudRiskAgent(Agent):
    """Agent that scores fraud risk and runs signal checks."""

    def __init__(self, tools: FraudTools) -> None:
        super().__init__(
            name="fraud_risk_agent",
            responsibility="Analyze fraud score and risk indicators",
        )
        self.fraud_tools = tools

    async def run(self, context: Dict[str, Any]) -> AgentOutput:
        transaction_id = context.get("transaction_id", "TXN_00000000")

        fraud_result = self.fraud_tools.calculate_fraud_score(transaction_id)
        blocklist = self.fraud_tools.check_blocklist(transaction_id)
        geo = self.fraud_tools.check_geographic_consistency(transaction_id)

        fraud_score = fraud_result["fraud_score"]
        risk_category = (
            "HIGH" if fraud_score > 70 else "MEDIUM" if fraud_score > 40 else "LOW"
        )

        analysis: Dict[str, Any] = {
            "fraud_score": fraud_score,
            "blocklist_flag": blocklist["is_blocklisted"],
            "geographic_consistency": geo["is_consistent"],
            "risk_category": risk_category,
        }

        return AgentOutput(
            agent_name=self.name,
            analysis=analysis,
            tools_used=[
                "calculate_fraud_score",
                "check_blocklist",
                "check_geographic_consistency",
            ],
            reasoning=(
                f"Fraud score: {fraud_score} ({risk_category}), "
                f"Blocklist: {blocklist['status']}, "
                f"Geo: {'consistent' if geo['is_consistent'] else 'inconsistent'}"
            ),
        )


class FraudDecisionAgent(Agent):
    """Agent that maps risk signals to a fraud prevention action."""

    def __init__(self, tools: FraudTools) -> None:
        super().__init__(
            name="fraud_decision_agent",
            responsibility="Determine the appropriate fraud prevention action",
        )
        self.fraud_tools = tools

    async def run(self, context: Dict[str, Any]) -> AgentOutput:
        risk = context.get("previous_agent_output", {})
        fraud_score: float = risk.get("fraud_score", 50)
        blocklist_flag: bool = risk.get("blocklist_flag", False)
        geo_check: Dict[str, Any] = {
            "is_consistent": risk.get("geographic_consistency", True),
            "risk": "low" if risk.get("geographic_consistency", True) else "high",
        }

        action = self.fraud_tools.determine_fraud_action(
            fraud_score, blocklist_flag, geo_check
        )

        confidence = max(0.0, 100.0 - fraud_score) / 100.0

        analysis: Dict[str, Any] = {
            "decision": action["action"],
            "score": confidence,
            "reason": action["reason"],
        }

        return AgentOutput(
            agent_name=self.name,
            analysis=analysis,
            tools_used=["determine_fraud_action"],
            reasoning=action["reason"],
        )


class FraudExplanationAgent(Agent):
    """Agent that generates customer-facing explanations."""

    def __init__(self, tools: FraudTools) -> None:
        super().__init__(
            name="fraud_explanation_agent",
            responsibility="Generate clear, non-accusatory fraud decision explanations",
        )
        self.fraud_tools = tools

    async def run(self, context: Dict[str, Any]) -> AgentOutput:
        prev = context.get("previous_agent_output", {})
        decision: str = prev.get("decision", "MONITOR")
        reason: str = prev.get("reason", "")

        templates = {
            "BLOCK": f"We blocked this transaction due to elevated fraud risk. {reason} "
                     "If this was your transaction, please contact support.",
            "CHALLENGE": f"We're requesting additional verification to protect your account. "
                         f"{reason} Please complete the verification step.",
            "MONITOR": f"This transaction has been approved and is being monitored. "
                       f"{reason}",
            "APPROVE": f"Transaction approved. {reason}",
        }
        explanation = templates.get(decision, f"Decision: {decision}. {reason}")

        analysis: Dict[str, Any] = {
            "decision": decision,
            "explanation": explanation,
        }

        return AgentOutput(
            agent_name=self.name,
            analysis=analysis,
            tools_used=[],
            reasoning="Generated customer-facing fraud decision explanation",
        )
