"""LLM prompt templates for payment agents."""


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_RISK_ANALYSIS = """You are a payment risk analysis expert. \
Analyze the provided payment transaction and relevant policies.

Use chain-of-thought reasoning:
1. Identify the key transaction details (amount, merchant, customer history).
2. Check those details against the retrieved payment policies and fraud patterns.
3. Synthesize a fraud risk score (0-100) with a concise rationale.

Be thorough but concise. Consider context and legitimate use cases."""

SYSTEM_PROMPT_DECISION_MAKER = """You are a payment approval decision-maker. \
Given a risk analysis and policy context, decide whether to:
- APPROVE  — transaction is safe to proceed
- DECLINE  — transaction should be blocked
- REVIEW   — transaction needs human review

Consider: customer tier and history, fraud risk score, and retrieved policies.
Respond with exactly one of APPROVE / DECLINE / REVIEW on the first line, \
followed by a 2-3 sentence rationale."""

SYSTEM_PROMPT_EXPLAINER = """You are a payment decision communicator. \
Explain payment decisions clearly to non-technical stakeholders.

Format:
1. **Decision**: APPROVE / DECLINE / REVIEW
2. **Key Factors**: 2-3 most important factors
3. **Summary**: 2-3 plain-English sentences a customer would understand.

Avoid jargon."""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def get_risk_analysis_prompt(
    customer_profile: dict, transaction: dict, policies: str
) -> str:
    """Compose the risk-analysis user message."""
    return (
        "Analyze this payment transaction:\n\n"
        "**Customer Profile:**\n"
        f"- ID: {customer_profile.get('customer_id', 'N/A')}\n"
        f"- Account Age: {customer_profile.get('account_age_years', 'N/A')} years\n"
        f"- Transaction Count: {customer_profile.get('total_transactions', 'N/A')}\n"
        f"- Average Transaction: ${customer_profile.get('average_transaction_amount', 'N/A')}\n"
        f"- VIP Status: {customer_profile.get('vip_status', 'no')}\n\n"
        "**Current Transaction:**\n"
        f"- Amount: ${transaction.get('amount', 'N/A')}\n"
        f"- Merchant: {transaction.get('merchant', 'N/A')}\n"
        f"- MCC Code: {transaction.get('mcc_code', 'N/A')}\n\n"
        f"**Payment Policies:**\n{policies}\n\n"
        "Analyze fraud risk using chain-of-thought reasoning. "
        "Provide a fraud risk score (0-100) and detailed explanation."
    )


def get_decision_prompt(
    risk_analysis: str, customer_profile: dict, policies: str
) -> str:
    """Compose the decision-making user message."""
    return (
        "Make an approval decision for this payment.\n\n"
        f"**Risk Analysis:**\n{risk_analysis}\n\n"
        "**Customer Info:**\n"
        f"- VIP: {customer_profile.get('vip_status', 'no')}\n"
        f"- Account Age: {customer_profile.get('account_age_years', 'N/A')} years\n\n"
        f"**Relevant Policies:**\n{policies}\n\n"
        "Respond with APPROVE, DECLINE, or REVIEW on the first line, "
        "then explain your reasoning in 2-3 sentences."
    )


def get_explanation_prompt(decision: str, reasoning: str) -> str:
    """Compose the explanation user message."""
    return (
        "Explain this payment decision in simple, clear language:\n\n"
        f"**Decision**: {decision}\n"
        f"**Reasoning**: {reasoning}\n\n"
        "Write a clear, non-technical explanation (2-3 sentences) "
        "that a customer would understand. "
        "Include the key factors that led to this decision."
    )
