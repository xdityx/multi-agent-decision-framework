from enum import Enum
from typing import Any, Dict, List
import uuid

from pydantic import BaseModel, ConfigDict, Field


class DecisionStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class DecisionRequest(BaseModel):
    """Generic request for a decision across any supported domain."""

    domain: str
    entity_id: str
    context: Dict[str, Any]
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "domain": "payments",
                "entity_id": "CUST_123",
                "context": {"amount": 1500, "merchant": "Amazon"},
                "request_id": "req_xyz",
            }
        }
    )


class AgentOutput(BaseModel):
    """Structured output from a single agent."""

    agent_name: str
    analysis: Dict[str, Any]
    tools_used: List[str] = Field(default_factory=list)
    reasoning: str


class DecisionResult(BaseModel):
    """Final decision and the chain of agent outputs that led to it."""

    decision: str
    decision_score: float
    reasoning: str
    agent_outputs: List[AgentOutput] = Field(default_factory=list)
    domain: str
