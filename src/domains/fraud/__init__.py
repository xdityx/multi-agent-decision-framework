"""Fraud domain — heuristic and LLM agents for fraud detection and prevention."""

# Heuristic agents
from .agents import (
    FraudDataAgent,
    FraudDecisionAgent,
    FraudExplanationAgent,
    FraudRiskAgent,
)

# LLM agents
from .agents_llm import (
    LLMFraudAnalysisAgent,
    LLMFraudDecisionAgent,
    LLMFraudExplainerAgent,
)

# Data, tools, RAG
from .data import FraudDataGenerator
from .rag_documents import FRAUD_POLICIES
from .rag_retriever import FraudRAGRetriever
from .tools import FraudTools, create_tools

__all__ = [
    # Heuristic
    "FraudDataAgent",
    "FraudRiskAgent",
    "FraudDecisionAgent",
    "FraudExplanationAgent",
    # LLM
    "LLMFraudAnalysisAgent",
    "LLMFraudDecisionAgent",
    "LLMFraudExplainerAgent",
    # Tools & data
    "FraudTools",
    "create_tools",
    "FraudDataGenerator",
    # RAG
    "FraudRAGRetriever",
    "FRAUD_POLICIES",
]
