"""Fraud detection tools for agents."""
import random
from typing import Any, Dict, List

from .data import Transaction, get_transaction_context

VALID_ACTIONS = {"BLOCK", "CHALLENGE", "MONITOR", "APPROVE"}


class FraudTools:
    """Tools for fraud detection and prevention agents."""

    def __init__(self, transactions: List[Transaction]) -> None:
        self.transactions = transactions
        # Index by ID for O(1) look-up
        self._index: Dict[str, Transaction] = {
            t.transaction_id: t for t in transactions
        }

    def get_transaction_details(self, transaction_id: str) -> Dict[str, Any]:
        """Return transaction context dict for a given ID."""
        txn = self._index.get(transaction_id)
        if txn is None:
            return {"error": f"Transaction {transaction_id} not found"}
        return get_transaction_context(txn)

    def calculate_fraud_score(self, transaction_id: str) -> Dict[str, Any]:
        """
        Calculate a fraud probability score (0-100).

        Higher = more suspicious. Uses a heuristic model; production would
        use an ML model trained on labelled transaction data.
        """
        txn = self._index.get(transaction_id)
        if txn is None:
            return {"fraud_score": 50, "reasoning": "Transaction not found — defaulting to medium risk"}

        score = 30  # Baseline

        # Risk-increasing factors
        if txn.amount > 1_000:
            score += 15
        if txn.is_test_amount:
            score += 20
        if txn.merchant_country != txn.customer_location:
            score += 20
        if txn.merchant_country != txn.card_country:
            score += 15
        if txn.is_vpn:
            score += 25
        if txn.failed_attempts_before > 0:
            score += 20
        if txn.time_since_account_open_days < 7:
            score += 15
        if txn.device_type == "unknown":
            score += 10

        # Trust-increasing factors
        if txn.merchant in {"Amazon", "Walmart", "Apple"}:
            score -= 10
        if txn.card_country == txn.customer_location:
            score -= 10

        return {
            "fraud_score": max(0, min(100, score)),
            "reasoning": "Based on transaction patterns and risk indicators",
        }

    def check_blocklist(self, transaction_id: str) -> Dict[str, Any]:
        """
        Check whether the transaction matches a known fraud pattern.

        Uses a deterministic per-ID random check so tests are reproducible.
        """
        rng = random.Random(transaction_id)
        is_flagged = rng.random() < 0.05  # ~5% flagged

        return {
            "is_blocklisted": is_flagged,
            "status": "flagged" if is_flagged else "clear",
        }

    def check_geographic_consistency(self, transaction_id: str) -> Dict[str, Any]:
        """Check whether card country matches customer location."""
        txn = self._index.get(transaction_id)
        if txn is None:
            return {"is_consistent": True, "risk": "low"}

        is_consistent = txn.card_country == txn.customer_location
        return {
            "is_consistent": is_consistent,
            "card_country": txn.card_country,
            "customer_location": txn.customer_location,
            "risk": "low" if is_consistent else "high",
        }

    def determine_fraud_action(
        self,
        fraud_score: float,
        blocklist_flag: bool,
        geo_check: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Map fraud score and signals to a prevention action."""
        if fraud_score > 85 or blocklist_flag:
            return {
                "action": "BLOCK",
                "reason": "High fraud probability or blocklist match",
            }
        elif fraud_score > 60:
            return {
                "action": "CHALLENGE",
                "reason": "Medium-high fraud risk — request step-up authentication",
            }
        elif fraud_score > 40:
            return {
                "action": "MONITOR",
                "reason": "Moderate risk — allow but flag for real-time monitoring",
            }
        else:
            return {
                "action": "APPROVE",
                "reason": "Low fraud risk — transaction consistent with history",
            }


def create_tools(transactions: List[Transaction]) -> FraudTools:
    """Factory function to create a configured FraudTools instance."""
    return FraudTools(transactions)
