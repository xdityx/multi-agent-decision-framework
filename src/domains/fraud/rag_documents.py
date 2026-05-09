"""RAG documents for the fraud domain — detection policies and fraud patterns."""
from typing import Any, Dict, List


FRAUD_POLICIES: List[Dict[str, str]] = [
    {
        "id": "policy_fraud_types_001",
        "title": "Types of Transaction Fraud",
        "content": (
            "Common fraud types:\n"
            "1. Card Present Fraud: Stolen card used in-person\n"
            "2. Card Not Present (CNP): Stolen card used online\n"
            "3. Account Takeover: Attacker gains login credentials\n"
            "4. Identity Theft: Using someone else's identity\n"
            "5. Synthetic Fraud: Creating fake identity with mix of real/fake info\n"
            "6. Friendly Fraud: Legitimate customer claims unauthorized transaction\n"
            "7. First-Party Fraud: Customer lies about transaction\n"
            "8. Money Mule: Using third-party to move stolen funds\n\n"
            "Each type has distinct patterns and indicators."
        ),
    },
    {
        "id": "policy_red_flags_001",
        "title": "Fraud Red Flags and Indicators",
        "content": (
            "Critical fraud indicators:\n\n"
            "TRANSACTION LEVEL:\n"
            "- Large amount for low-risk merchant\n"
            "- Unusual merchant type for customer\n"
            "- Multiple declined transactions then approval\n"
            "- Rush delivery address different from billing\n"
            "- Using multiple cards in short time\n\n"
            "CUSTOMER LEVEL:\n"
            "- New account with immediate large purchases\n"
            "- Changed billing/shipping address\n"
            "- Multiple failed login attempts\n"
            "- Device change without prior pattern\n"
            "- VPN/proxy detected\n\n"
            "BEHAVIORAL:\n"
            "- Time zone mismatch (card in US, transaction in foreign country)\n"
            "- Velocity: Multiple transactions in short window\n"
            "- Structuring: Small transactions to avoid limits\n"
            "- Test transactions: $0.99 or small amounts then large"
        ),
    },
    {
        "id": "policy_detection_rules_001",
        "title": "Fraud Detection Rules and Thresholds",
        "content": (
            "Fraud decision rules:\n\n"
            "BLOCK IMMEDIATELY if:\n"
            "- Fraud score > 85%\n"
            "- Known fraud list match\n"
            "- Blocked country/IP\n"
            "- Impossible travel (2 transactions, different countries, <2 hours)\n"
            "- Multiple simultaneous transactions\n\n"
            "CHALLENGE (2FA) if:\n"
            "- Fraud score 60-85%\n"
            "- Unusual amount or merchant\n"
            "- Device/location change\n"
            "- Suspicious behavior pattern\n\n"
            "APPROVE if:\n"
            "- Fraud score < 40%\n"
            "- Matches historical behavior\n"
            "- Consistent device/location\n"
            "- Customer verified\n"
            "- Low-risk merchant"
        ),
    },
    {
        "id": "policy_prevention_001",
        "title": "Fraud Prevention Best Practices",
        "content": (
            "Prevention strategies:\n\n"
            "PREVENTION:\n"
            "- Require 2FA for new cards\n"
            "- Limit daily/monthly transaction amounts\n"
            "- Restrict to customer's home country initially\n"
            "- Monitor velocity (transactions per hour)\n"
            "- Implement 3D Secure for online\n\n"
            "DETECTION:\n"
            "- Real-time transaction scoring\n"
            "- ML models trained on historical fraud\n"
            "- Behavioral biometrics\n"
            "- Network analysis (who else uses this IP/device)\n"
            "- Consortium data (cross-institution sharing)\n\n"
            "RESPONSE:\n"
            "- Fast block (< 1 second decision)\n"
            "- Customer notification\n"
            "- Containment (freeze card)\n"
            "- Investigation\n"
            "- Law enforcement coordination"
        ),
    },
    {
        "id": "pattern_synthetic_001",
        "title": "Synthetic Fraud Patterns",
        "content": (
            "Synthetic fraud (fake identity fraud) indicators:\n\n"
            "EARLY STAGE (Account Creation):\n"
            "- New address in high-fraud area\n"
            "- Unusual SSN pattern\n"
            "- No credit history\n"
            "- Thin file (minimal credit data)\n"
            "- Inconsistent address history\n\n"
            "BUILDUP STAGE (Credit History):\n"
            "- Rapid credit line opens\n"
            "- Multiple inquiries in short timeframe\n"
            "- Credit limits increasing rapidly\n"
            "- Timely payments (artificially good behavior)\n"
            "- No late payments (unrealistic)\n\n"
            "EXPLOITATION STAGE:\n"
            "- Sudden spending surge\n"
            "- Maxing out new credit lines\n"
            "- Cash advances\n"
            "- Transfers to other accounts\n"
            "- Dormancy then exploitation\n\n"
            "Response: Freeze account, law enforcement notification, credential verification"
        ),
    },
]


def get_rag_documents() -> List[Dict[str, Any]]:
    """Return all fraud policy documents."""
    return FRAUD_POLICIES


def get_policy_content(policy_id: str) -> str:
    """Fetch a specific policy's content by ID."""
    for policy in FRAUD_POLICIES:
        if policy["id"] == policy_id:
            return policy["content"]
    return "Policy not found"


def search_relevant_policies(query: str) -> List[Dict[str, Any]]:
    """Keyword search over fraud policy titles and content."""
    query_lower = query.lower()
    results = []

    for policy in FRAUD_POLICIES:
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
