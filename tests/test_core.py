import asyncio

from src.core.agent import Agent
from src.core.schemas import AgentOutput, DecisionRequest, DecisionResult
from src.core.tools import ToolRegistry
from src.core.workflow import DecisionWorkflow


def test_agent_creation():
    """Test agent can be created."""
    agent = Agent(name="test_agent", responsibility="test")
    assert agent.name == "test_agent"
    assert len(agent.tools) == 0


def test_agent_add_tool():
    """Test adding a tool to an agent."""
    agent = Agent(name="test_agent", responsibility="test")

    def dummy_tool(x):
        return x * 2

    agent.add_tool("double", dummy_tool)
    assert "double" in agent.tools
    assert agent.tools["double"](5) == 10


def test_agent_execution():
    """Test agent runs without error."""
    agent = Agent(name="test_agent", responsibility="test")
    context = {"data": "test"}
    output = asyncio.run(agent.run(context))

    assert output.agent_name == "test_agent"
    assert output.analysis == context
    assert isinstance(output, AgentOutput)


def test_workflow_execution():
    """Test workflow executes multiple agents."""
    agent1 = Agent(name="data_agent", responsibility="gather data")
    agent2 = Agent(name="analysis_agent", responsibility="analyze data")
    agent3 = Agent(name="decision_agent", responsibility="make decision")

    workflow = DecisionWorkflow([agent1, agent2, agent3], "test_workflow")

    request = DecisionRequest(
        domain="payments",
        entity_id="CUST_123",
        context={"amount": 1000},
    )

    result = asyncio.run(workflow.execute(request))

    assert isinstance(result, DecisionResult)
    assert result.domain == "payments"
    assert len(result.agent_outputs) == 3


def test_decision_request_schema():
    """Test request validation."""
    request = DecisionRequest(
        domain="payments",
        entity_id="CUST_123",
        context={"amount": 100},
    )
    assert request.domain == "payments"
    assert request.entity_id == "CUST_123"


def test_decision_result_schema():
    """Test result validation."""
    result = DecisionResult(
        decision="APPROVE",
        decision_score=0.95,
        reasoning="Low risk",
        agent_outputs=[],
        domain="payments",
    )
    assert result.decision == "APPROVE"
    assert result.decision_score == 0.95


def test_tool_registry():
    """Test tool registry registration and lookup."""
    registry = ToolRegistry()

    def add_one(value):
        return value + 1

    registry.register("add_one", add_one, description="Increment a value")

    assert registry.get_tool("add_one") is add_one
    assert registry.list_tools() == ["add_one"]
    assert registry.get_tool("add_one")(2) == 3
