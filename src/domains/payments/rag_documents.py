"""RAG documents for the payments domain — payment policies and fraud patterns."""
from typing import Any, Dict, List


PAYMENT_POLICIES: List[Dict[str, str]] = [
    {
        "id": "policy_fraud_001",
        "title": "Fraud Detection Thresholds",
        "content": (
            "Payment fraud detection thresholds:\n"
            "- Low risk: fraud probability < 20%\n"
            "- Medium risk: fraud probability 20-50%\n"
            "- High risk: fraud probability > 50%\n\n"
            "For high-risk transactions, require additional verification.\n"
            "For medium-risk, monitor and alert.\n"
            "For low-risk, approve automatically."
        ),
    },
    {
        "id": "policy_velocity_001",
        "title": "Velocity Rules",
        "content": (
            "Transaction velocity rules:\n"
            "- Single transaction limit: 50% of account limit\n"
            "- Daily limit: 150% of account limit\n"
            "- Unusual velocity: 3x normal frequency in 1 hour\n\n"
            "Block transactions exceeding single transaction limit.\n"
            "Flag transactions exceeding daily limit."
        ),
    },
    {
        "id": "policy_customer_001",
        "title": "Customer Tiers and Approval Rules",
        "content": (
            "Customer approval rules by tier:\n"
            "- VIP: Auto-approve up to $50,000 if fraud score < 30%\n"
            "- Regular: Auto-approve up to account limit if fraud score < 20%\n"
            "- New: Require review if transaction > $5,000\n"
            "- Flagged: All transactions require manual review\n\n"
            "Consider account age and transaction history when setting tier."
        ),
    },
    {
        "id": "pattern_fraud_001",
        "title": "Common Fraud Patterns",
        "content": (
            "Common payment fraud patterns:\n"
            "1. Structuring: Multiple small transactions to avoid limits\n"
            "2. Unusual merchant: Transaction with never-before-used merchant\n"
            "3. Location mismatch: Transaction from unusual geographic location\n"
            "4. Account takeover: Sudden change in transaction patterns\n"
            "5. Velocity surge: Abnormal number of transactions in short time\n\n"
            "Monitor for these patterns and flag accordingly."
        ),
    },
    {
        "id": "pattern_legitimate_001",
        "title": "Legitimate High-Risk Indicators",
        "content": (
            "Transactions that appear risky but are often legitimate:\n"
            "- Travel: Large international purchases when customer is traveling\n"
            "- Seasonal: Spike in shopping during holiday seasons\n"
            "- Business: Regular large purchases for business accounts\n"
            "- Weddings/Events: Planned large expenses\n"
            "- New merchants: Legitimate first-time merchants\n\n"
            "Context matters. Consider customer's history and lifecycle."
        ),
    },
]


def get_rag_documents() -> List[Dict[str, Any]]:
    """Return all RAG policy documents."""
    return PAYMENT_POLICIES


def get_policy_content(policy_id: str) -> str:
    """Fetch a specific policy's content by its ID."""
    for policy in PAYMENT_POLICIES:
        if policy["id"] == policy_id:
            return policy["content"]
    return "Policy not found"


def search_relevant_policies(query: str) -> List[Dict[str, Any]]:
    """Keyword search over policy titles and content."""
    query_lower = query.lower()
    results = []

    for policy in PAYMENT_POLICIES:
        title_match = query_lower in policy["title"].lower()
        content_match = query_lower in policy["content"].lower()

        if title_match or content_match:
            results.append(
                {
                    "id": policy["id"],
                    "title": policy["title"],
                    "excerpt": policy["content"][:200] + "...",
                }
            )

    return results
