from typing import Any, Callable, Dict, List, Optional


class ToolRegistry:
    """Registry for tools that agents can use across domains."""

    def __init__(self) -> None:
        self.tools: Dict[str, Callable[..., Any]] = {}

    def register(
        self,
        name: str,
        func: Callable[..., Any],
        description: str = "",
    ) -> Callable[..., Any]:
        """Register a tool and optionally set a short description."""
        self.tools[name] = func
        if description:
            func.__doc__ = description
        return func

    def get_tool(self, name: str) -> Optional[Callable[..., Any]]:
        """Return a registered tool by name, if present."""
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self.tools.keys())
