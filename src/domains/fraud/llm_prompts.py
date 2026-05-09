"""LLM prompt templates for fraud domain agents."""


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_FRAUD_ANALYSIS = """You are a fraud detection expert. \
Analyze the provided transaction details and retrieved fraud policies.

Use chain-of-thought reasoning:
1. Identify transaction characteristics (amount, merchant, geography, device).
2. Map those characteristics to known fraud red flags and patterns.
3. Synthesize a fraud risk score (0-100) and recommended action.

Be specific about which risk factors are present and cite the relevant policy."""

SYSTEM_PROMPT_FRAUD_DECISION = """You are a fraud prevention decision-maker. \
Given a transaction risk analysis and relevant policies, decide:
- BLOCK    — transaction is fraudulent or high-risk; reject immediately
- CHALLENGE — request additional verification (2FA/step-up auth)
- MONITOR  — allow but flag for real-time monitoring
- APPROVE  — transaction is safe; proceed normally

Respond with exactly one action label on the first line, \
followed by a 2-3 sentence justification citing specific risk factors."""

SYSTEM_PROMPT_FRAUD_EXPLAINER = """You are a fraud security communicator. \
Explain fraud prevention decisions clearly and helpfully to the customer.

Format:
1. **Fraud Risk**: Simple explanation of the risk level
2. **Reason**: Why this transaction was flagged
3. **Action**: What we did (blocked / challenged / approved)
4. **Next Steps**: How the customer can resolve it if legitimate

Be professional, clear, and not accusatory."""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def get_fraud_analysis_prompt(transaction_context: dict, policies: str) -> str:
    """Compose the fraud analysis user message."""
    return (
        "Analyze this transaction for fraud risk:\n\n"
        "**Transaction Details:**\n"
        f"- Amount: ${transaction_context.get('amount', 'N/A')}\n"
        f"- Merchant: {transaction_context.get('merchant', 'Unknown')}\n"
        f"- Merchant Country: {transaction_context.get('merchant_country', 'Unknown')}\n"
        f"- Customer Location: {transaction_context.get('customer_location', 'Unknown')}\n"
        f"- Card Country: {transaction_context.get('card_country', 'Unknown')}\n"
        f"- Device: {transaction_context.get('device', 'Unknown')}\n"
        f"- VPN Detected: {transaction_context.get('vpn_detected', False)}\n"
        f"- Failed Attempts Before: {transaction_context.get('failed_attempts', 0)}\n"
        f"- Test Amount: {transaction_context.get('test_amount', False)}\n"
        f"- Account Age: {transaction_context.get('account_age_days', 0)} days\n\n"
        f"**Fraud Policies:**\n{policies}\n\n"
        "Analyze fraud risk using chain-of-thought reasoning. "
        "Provide a fraud score (0-100) and a detailed risk assessment."
    )


def get_fraud_decision_prompt(
    fraud_analysis: str, blocklist_flag: bool, policies: str
) -> str:
    """Compose the fraud decision user message."""
    blocklist_status = "FLAGGED" if blocklist_flag else "CLEAR"
    return (
        "Make a fraud prevention decision for this transaction.\n\n"
        f"**Fraud Analysis:**\n{fraud_analysis}\n\n"
        f"**Blocklist Status:** {blocklist_status}\n\n"
        f"**Fraud Policies:**\n{policies}\n\n"
        "Decide: BLOCK, CHALLENGE, MONITOR, or APPROVE\n"
        "Explain your reasoning in 2-3 sentences."
    )


def get_fraud_explanation_prompt(decision: str, reason: str) -> str:
    """Compose the customer-facing explanation user message."""
    return (
        "Explain this fraud prevention action to the customer:\n\n"
        f"**Decision**: {decision}\n"
        f"**Reason**: {reason}\n\n"
        "Write a clear, helpful explanation (2-3 sentences) covering:\n"
        "1. Why the transaction was flagged\n"
        "2. What action we took\n"
        "3. How to resolve it if it was their legitimate transaction\n\n"
        "Be professional but friendly — do not accuse the customer."
    )
