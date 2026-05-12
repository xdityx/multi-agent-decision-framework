from typing import Any, Dict, List

from .agent import Agent
from .schemas import AgentOutput, DecisionRequest, DecisionResult


class DecisionWorkflow:
    """Orchestrates a sequence of agents to produce a final decision."""

    def __init__(self, agents: List[Agent], workflow_name: str) -> None:
        self.agents = agents
        self.agent_map = {agent.name: agent for agent in agents}
        self.workflow_name = workflow_name

    async def execute(self, request: DecisionRequest) -> DecisionResult:
        """
        Execute all agents in sequence.

        Each agent receives the evolving context, including the previous
        agent's analysis to support chained reasoning.
        """
        outputs: List[AgentOutput] = []
        context: Dict[str, Any] = request.context.copy()

        for agent in self.agents:
            agent_output = await agent.run(context)
            outputs.append(agent_output)
            context["previous_agent_output"] = agent_output.analysis

        final_output = outputs[-1] if outputs else None
        final_analysis = final_output.analysis if final_output else {}
        final_reasoning = (
            final_output.reasoning
            if final_output
            else "No agents were configured for this workflow"
        )

        # Find the score from whichever agent produced it (decision agent, not
        # explanation agent which is last but doesn't emit a score).
        decision_score = 0.5
        for output in reversed(outputs):
            if isinstance(output.analysis, dict) and "score" in output.analysis:
                decision_score = float(output.analysis["score"])
                break

        return DecisionResult(
            decision=final_analysis.get("decision", "PENDING"),
            decision_score=decision_score,
            reasoning=final_reasoning,
            agent_outputs=outputs,
            domain=request.domain,
        )

    def __repr__(self) -> str:
        agent_names = ", ".join(agent.name for agent in self.agents)
        return f"DecisionWorkflow({agent_names})"
