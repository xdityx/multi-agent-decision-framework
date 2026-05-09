"""Synthetic fraud transaction data for testing and demo."""
import random
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Transaction:
    transaction_id: str
    amount: float
    merchant: str
    merchant_country: str
    customer_location: str
    card_country: str
    device_type: str
    is_vpn: bool
    failed_attempts_before: int
    is_test_amount: bool
    time_since_account_open_days: int
    is_fraudulent: bool


class FraudDataGenerator:
    """Generate synthetic fraud transaction data."""

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def generate_transactions(self, count: int = 100) -> List[Transaction]:
        """Generate *count* synthetic transactions (seeded, deterministic)."""
        merchants = [
            "Amazon", "Walmart", "PayPal", "Apple",
            "Google", "Crypto Exchange", "Wire Transfer", "Cash Advance",
        ]
        countries = ["US", "UK", "CN", "RU", "NG", "IN", "BR", "MX"]
        devices = ["mobile", "desktop", "unknown"]

        transactions: List[Transaction] = []

        for i in range(count):
            is_fraud = random.random() < 0.10  # 10% fraud rate

            if is_fraud:
                amount = random.choice(
                    [random.uniform(5_000, 50_000), random.uniform(100, 500)]
                )
                merchant = random.choice(merchants)
                merchant_country = random.choice(countries)
                customer_location = random.choice(countries)
                card_country = random.choice(countries)
                is_vpn = random.random() < 0.3
                failed_attempts = random.randint(2, 5)
                is_test = random.random() < 0.2
            else:
                amount = random.uniform(10, 500)
                merchant = random.choice(merchants[:4])  # Known/safe merchants
                merchant_country = "US"
                customer_location = "US"
                card_country = "US"
                is_vpn = random.random() < 0.01
                failed_attempts = random.randint(0, 1)
                is_test = False

            transactions.append(
                Transaction(
                    transaction_id=f"TXN_{i:08d}",
                    amount=amount,
                    merchant=merchant,
                    merchant_country=merchant_country,
                    customer_location=customer_location,
                    card_country=card_country,
                    device_type=random.choice(devices),
                    is_vpn=is_vpn,
                    failed_attempts_before=failed_attempts,
                    is_test_amount=is_test,
                    time_since_account_open_days=random.randint(0, 1825),
                    is_fraudulent=is_fraud,
                )
            )

        return transactions


def get_transaction_context(transaction: Transaction) -> Dict[str, Any]:
    """Flatten a Transaction dataclass into a plain dict for agent consumption."""
    return {
        "amount": transaction.amount,
        "merchant": transaction.merchant,
        "merchant_country": transaction.merchant_country,
        "customer_location": transaction.customer_location,
        "card_country": transaction.card_country,
        "device": transaction.device_type,
        "vpn_detected": transaction.is_vpn,
        "failed_attempts": transaction.failed_attempts_before,
        "test_amount": transaction.is_test_amount,
        "account_age_days": transaction.time_since_account_open_days,
    }
