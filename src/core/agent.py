from typing import Any, Callable, Dict, Optional

from .schemas import AgentOutput


class Agent:
    """Generic agent that can be specialized for any domain."""

    def __init__(
        self,
        name: str,
        responsibility: str,
        tools: Optional[Dict[str, Callable[..., Any]]] = None,
    ) -> None:
        self.name = name
        self.responsibility = responsibility
        self.tools = tools or {}

    async def run(self, context: Dict[str, Any]) -> AgentOutput:
        """
        Execute the agent against the provided context.

        Subclasses can override this with domain-specific logic.
        """
        return AgentOutput(
            agent_name=self.name,
            analysis=context.copy(),
            tools_used=list(self.tools.keys()),
            reasoning=f"{self.name} analyzed the context",
        )

    def add_tool(self, name: str, func: Callable[..., Any]) -> None:
        """Register a tool this agent can use."""
        self.tools[name] = func

    def __repr__(self) -> str:
        return f"Agent(name={self.name}, responsibility={self.responsibility})"
