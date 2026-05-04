# Multi-Agent Decision Framework

A generalized framework for multi-agent AI decision-making. The core architecture stays the same while different domains plug in their own agents, tools, and business rules.

## Architecture

- **Agent**: Generic blueprint for specialized decision-makers
- **Workflow**: Orchestrates multiple agents in sequence
- **ToolRegistry**: Holds reusable domain-specific tools
- **Schemas**: Domain-agnostic request and response models

## Domains (To Be Implemented)

1. **Payments**: Payment risk assessment
2. **Churn**: Customer churn prediction
3. **Fraud**: Fraud detection

## Getting Started

```bash
pip install -r requirements.txt
pytest tests/ -v
```

## Usage

```python
from src.core import Agent, DecisionRequest, DecisionWorkflow

data_agent = Agent("data_agent", "gather data")
analysis_agent = Agent("analysis_agent", "analyze data")
decision_agent = Agent("decision_agent", "make decision")

workflow = DecisionWorkflow(
    [data_agent, analysis_agent, decision_agent],
    "payment_flow",
)

request = DecisionRequest(
    domain="payments",
    entity_id="CUST_123",
    context={"amount": 1500, "merchant": "Amazon"},
)

# Inside an async function:
result = await workflow.execute(request)
```
