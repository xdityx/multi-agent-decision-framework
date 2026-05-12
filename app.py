"""Streamlit dashboard for the multi-agent decision framework.

Provides an interactive UI that talks to the FastAPI backend, with
domain-specific forms for Payments, Churn, and Fraud, plus a full
agent-trace expander so users can inspect every agent's reasoning.

Can also run in **standalone mode** (no backend required) by calling
the domain workflows directly when the API is unreachable.
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()  # Load .env before domain agents read env vars

import streamlit as st


# ── Standalone imports (used when API is down) ──────────────────────────────
from src.core.schemas import DecisionRequest
from src.core.workflow import DecisionWorkflow

from src.domains.payments.agents import (
    PaymentDataAgent, PaymentDecisionAgent, PaymentExplanationAgent, PaymentRiskAgent,
)
from src.domains.payments.data import PaymentDataGenerator
from src.domains.payments.tools import create_tools as create_payment_tools

from src.domains.churn.agents import (
    ChurnDataAgent, ChurnDecisionAgent, ChurnExplanationAgent, ChurnRiskAgent,
)
from src.domains.churn.data import ChurnDataGenerator
from src.domains.churn.tools import create_tools as create_churn_tools

from src.domains.fraud.agents import (
    FraudDataAgent, FraudDecisionAgent, FraudExplanationAgent, FraudRiskAgent,
)
from src.domains.fraud.data import FraudDataGenerator
from src.domains.fraud.tools import create_tools as create_fraud_tools


# ============================================================================
# Input Validation
# ============================================================================


def _validate_customer_id(customer_id: str) -> tuple[bool, str]:
    """Return (True, "") if valid, else (False, error_message).

    Valid range: CUST_00000 to CUST_00049 (matches the 50 synthetic customers).
    """
    if not customer_id:
        return False, "Customer ID cannot be empty."
    if not customer_id.startswith("CUST_"):
        return False, "Invalid format — must start with 'CUST_' (e.g. CUST_00005)."
    try:
        num = int(customer_id[5:])
    except ValueError:
        return False, "Invalid format — use CUST_00000 to CUST_00049."
    if not (0 <= num < 50):
        return False, f"Out of range — valid IDs are CUST_00000 to CUST_00049 (got {customer_id})."
    return True, ""


def _validate_transaction_id(transaction_id: str) -> tuple[bool, str]:
    """Return (True, "") if valid, else (False, error_message).

    Valid range: TXN_00000000 to TXN_00000099 (matches the 100 synthetic transactions).
    """
    if not transaction_id:
        return False, "Transaction ID cannot be empty."
    if not transaction_id.startswith("TXN_"):
        return False, "Invalid format — must start with 'TXN_' (e.g. TXN_00000005)."
    try:
        num = int(transaction_id[4:])
    except ValueError:
        return False, "Invalid format — use TXN_00000000 to TXN_00000099."
    if not (0 <= num < 100):
        return False, f"Out of range — valid IDs are TXN_00000000 to TXN_00000099 (got {transaction_id})."
    return True, ""


# ============================================================================
# Page config
# ============================================================================

st.set_page_config(
    page_title="Multi-Agent Decision Framework",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# Custom CSS — dark-themed, premium look
# ============================================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Header gradient */
    .header-gradient {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        color: white;
    }
    .header-gradient h1 { color: white; margin: 0 0 0.25rem 0; font-weight: 700; }
    .header-gradient p  { color: rgba(255,255,255,0.85); margin: 0; font-size: 1.05rem; }

    /* Decision badges */
    .decision-badge {
        display: inline-block;
        padding: 0.5rem 1.4rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.03em;
    }
    .badge-approve  { background: #d4edda; color: #155724; }
    .badge-decline  { background: #f8d7da; color: #721c24; }
    .badge-review   { background: #fff3cd; color: #856404; }
    .badge-block    { background: #f8d7da; color: #721c24; }
    .badge-challenge{ background: #fff3cd; color: #856404; }
    .badge-monitor  { background: #d1ecf1; color: #0c5460; }
    .badge-default  { background: #e2e3e5; color: #383d41; }

    /* Stat cards */
    .stat-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .stat-card h3 { margin: 0; font-size: 0.85rem; color: #555; font-weight: 500; }
    .stat-card p  { margin: 0.3rem 0 0; font-size: 1.6rem; font-weight: 700; color: #333; }

    /* Agent trace cards */
    .agent-card {
        border-left: 4px solid #667eea;
        background: #f8f9fa;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
    }
    .agent-card h4 { margin: 0 0 0.5rem; color: #333; font-size: 0.95rem; }

    /* Footer */
    .footer {
        text-align: center;
        color: #888;
        font-size: 0.82rem;
        padding-top: 2rem;
        border-top: 1px solid #e5e5e5;
        margin-top: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# Cached data singletons (for standalone mode)
# ============================================================================


@st.cache_resource
def _get_payment_tools():
    gen = PaymentDataGenerator(seed=42)
    custs = gen.generate_customers(50)
    txns = gen.generate_transactions(custs, 100)
    return create_payment_tools(custs, txns)


@st.cache_resource
def _get_churn_tools():
    gen = ChurnDataGenerator(seed=42)
    custs = gen.generate_customers(50)
    return create_churn_tools(custs)


@st.cache_resource
def _get_fraud_tools():
    gen = FraudDataGenerator(seed=42)
    txns = gen.generate_transactions(100)
    return create_fraud_tools(txns)


# ============================================================================
# Helper: run workflow directly (standalone mode)
# ============================================================================


def _run_standalone(domain: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a heuristic workflow in-process — no backend needed."""
    if domain == "payments":
        tools = _get_payment_tools()
        agents = [
            PaymentDataAgent(tools), PaymentRiskAgent(tools),
            PaymentDecisionAgent(tools), PaymentExplanationAgent(tools),
        ]
    elif domain == "churn":
        tools = _get_churn_tools()
        agents = [
            ChurnDataAgent(tools), ChurnRiskAgent(tools),
            ChurnDecisionAgent(tools), ChurnExplanationAgent(tools),
        ]
    else:
        tools = _get_fraud_tools()
        agents = [
            FraudDataAgent(tools), FraudRiskAgent(tools),
            FraudDecisionAgent(tools), FraudExplanationAgent(tools),
        ]

    workflow = DecisionWorkflow(agents, f"{domain}_standalone")
    entity_id = context.get("customer_id") or context.get("transaction_id", "")
    request = DecisionRequest(domain=domain, entity_id=entity_id, context=context)

    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(workflow.execute(request))
    loop.close()

    return {
        "domain": result.domain,
        "decision": result.decision,
        "decision_score": result.decision_score,
        "reasoning": result.reasoning,
        "agent_outputs": [
            {"agent": o.agent_name, "analysis": o.analysis}
            for o in result.agent_outputs
        ],
        "agent_type": context.get("agent_type", "heuristic"),
    }


# ============================================================================
# Helper: call the FastAPI backend
# ============================================================================


def _call_api(url: str, payload: Dict[str, Any]) -> Dict[str, Any] | None:
    """POST to the FastAPI backend; return None if unreachable."""
    try:
        import requests

        resp = requests.post(url, json=payload, timeout=60)
        if resp.status_code == 200:
            return resp.json()
        st.warning(f"API returned {resp.status_code}: {resp.text}")
        return None
    except Exception:
        return None


# ============================================================================
# Decision badge helper
# ============================================================================

_BADGE_MAP = {
    "APPROVE": "badge-approve",
    "DECLINE": "badge-decline",
    "REVIEW": "badge-review",
    "BLOCK": "badge-block",
    "CHALLENGE": "badge-challenge",
    "MONITOR": "badge-monitor",
    "EXECUTIVE_OUTREACH": "badge-decline",
    "URGENT_RETENTION": "badge-challenge",
    "PROACTIVE_OUTREACH": "badge-monitor",
    "STANDARD_ENGAGEMENT": "badge-approve",
}

_EMOJI = {
    "APPROVE": "✅", "DECLINE": "❌", "REVIEW": "⚠️",
    "BLOCK": "🚫", "CHALLENGE": "⚠️", "MONITOR": "👁️",
    "EXECUTIVE_OUTREACH": "🔴", "URGENT_RETENTION": "🟠",
    "PROACTIVE_OUTREACH": "🟡", "STANDARD_ENGAGEMENT": "🟢",
}


def _render_badge(decision: str):
    css = _BADGE_MAP.get(decision, "badge-default")
    emoji = _EMOJI.get(decision, "ℹ️")
    st.markdown(
        f'<span class="decision-badge {css}">{emoji} {decision}</span>',
        unsafe_allow_html=True,
    )


# ============================================================================
# Render results helper
# ============================================================================


def _render_results(result: Dict[str, Any]):
    """Render the decision, stats, and agent trace."""

    # ── Top row: badge + stats ──
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        _render_badge(result["decision"])
    with c2:
        score = result.get("decision_score", 0)
        st.markdown(
            f'<div class="stat-card"><h3>Confidence</h3><p>{score:.0%}</p></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="stat-card"><h3>Agent Type</h3><p>{result["agent_type"].upper()}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Reasoning ──
    st.markdown("#### 💡 Reasoning")
    st.info(result.get("reasoning", "—"))

    # ── Agent trace ──
    with st.expander("🔍 Full Agent Trace", expanded=False):
        for i, output in enumerate(result.get("agent_outputs", []), 1):
            st.markdown(
                f'<div class="agent-card"><h4>Step {i} — {output["agent"]}</h4></div>',
                unsafe_allow_html=True,
            )
            st.json(output["analysis"])


# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    domain = st.selectbox(
        "Domain",
        ["Payments", "Churn", "Fraud"],
        help="Select the business domain to analyse.",
    )

    agent_type = st.radio(
        "Agent Type",
        ["Heuristic", "LLM"],
        help="Heuristic: fast & deterministic. LLM: Claude-powered reasoning.",
    )

    use_api = st.toggle(
        "Use FastAPI backend",
        value=False,
        help="When OFF the dashboard runs agents directly in-process.",
    )

    if use_api:
        api_url = st.text_input("Backend URL", value="http://localhost:8000")
    else:
        api_url = ""

    st.markdown("---")
    st.markdown(
        "**Domains:** Payments · Churn · Fraud\n\n"
        "**Tests:** 103 passing\n\n"
        "**Stack:** LangChain · Claude · ChromaDB · FastAPI · Streamlit"
    )


# ============================================================================
# Header
# ============================================================================

st.markdown(
    '<div class="header-gradient">'
    "<h1>🤖 Multi-Agent Decision Framework</h1>"
    "<p>Intelligent, explainable decision-making across Payments, Churn, and Fraud</p>"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================================
# Domain UIs
# ============================================================================

if domain == "Payments":
    st.markdown("### 💳 Payment Fraud Detection")

    col_a, col_b = st.columns([3, 1])
    with col_a:
        customer_id = st.text_input(
            "Customer ID",
            value="CUST_00000",
            key="pay_cid",
            help="Format: CUST_00000 to CUST_00049",
        )
    with col_b:
        st.info("**Available IDs**\nCUST_00000\nto\nCUST_00049")

    amount = st.number_input(
        "Amount ($)", min_value=0.0, value=2500.0, step=100.0, key="pay_amt"
    )

    is_valid, err = _validate_customer_id(customer_id)
    if not is_valid:
        st.error(f"Invalid Customer ID: {err}")
        st.stop()

    if st.button("Analyze Payment", key="pay_btn", use_container_width=True):
        with st.spinner("Running payment agents..."):
            payload = {
                "customer_id": customer_id,
                "amount": amount,
                "agent_type": agent_type.lower(),
            }
            result = (
                _call_api(f"{api_url}/analyze/payments", payload) if use_api else None
            )
            if result is None:
                result = _run_standalone("payments", payload)

        _render_results(result)


elif domain == "Churn":
    st.markdown("### 📉 Customer Churn Prediction")

    col_a, col_b = st.columns([3, 1])
    with col_a:
        customer_id = st.text_input(
            "Customer ID",
            value="CUST_00000",
            key="churn_cid",
            help="Format: CUST_00000 to CUST_00049",
        )
    with col_b:
        st.info("**Available IDs**\nCUST_00000\nto\nCUST_00049")

    is_valid, err = _validate_customer_id(customer_id)
    if not is_valid:
        st.error(f"Invalid Customer ID: {err}")
        st.stop()

    if st.button("Analyze Churn Risk", key="churn_btn", use_container_width=True):
        with st.spinner("Running churn agents..."):
            payload = {
                "customer_id": customer_id,
                "agent_type": agent_type.lower(),
            }
            result = (
                _call_api(f"{api_url}/analyze/churn", payload) if use_api else None
            )
            if result is None:
                result = _run_standalone("churn", payload)

        _render_results(result)


elif domain == "Fraud":
    st.markdown("### 🛡️ Transaction Fraud Prevention")

    col_a, col_b = st.columns([3, 1])
    with col_a:
        transaction_id = st.text_input(
            "Transaction ID",
            value="TXN_00000000",
            key="fraud_tid",
            help="Format: TXN_00000000 to TXN_00000099",
        )
    with col_b:
        st.info("**Available IDs**\nTXN_00000000\nto\nTXN_00000099")

    is_valid, err = _validate_transaction_id(transaction_id)
    if not is_valid:
        st.error(f"Invalid Transaction ID: {err}")
        st.stop()

    if st.button("Analyze Fraud Risk", key="fraud_btn", use_container_width=True):
        with st.spinner("Running fraud agents..."):
            payload = {
                "transaction_id": transaction_id,
                "agent_type": agent_type.lower(),
            }
            result = (
                _call_api(f"{api_url}/analyze/fraud", payload) if use_api else None
            )
            if result is None:
                result = _run_standalone("fraud", payload)

        _render_results(result)


# ============================================================================
# Footer
# ============================================================================

st.markdown(
    '<div class="footer">'
    "Multi-Agent Decision Framework v1.0 · "
    "3 Domains · 103+ Tests · LLM + RAG Powered · "
    f"Built by <b>Aditya</b> · {datetime.now().year}"
    "</div>",
    unsafe_allow_html=True,
)
