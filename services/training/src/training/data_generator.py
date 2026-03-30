"""
Synthetic transaction dataset generator for training and load testing.

Generates realistic fraud patterns:
- Base fraud rate: ~2%
- Elevated fraud signals: high velocity, unusual hour, cross-country, high-risk merchant
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Iterator

import numpy as np
import pandas as pd

MERCHANT_CATEGORIES = ["retail", "food", "travel", "entertainment", "online", "atm", "utility", "healthcare", "other"]
POS_TYPES = ["chip", "swipe", "contactless", "online"]
COUNTRY_CODES = ["US", "GB", "DE", "FR", "CA", "AU", "JP", "BR", "NG", "RU"]
HIGH_RISK_COUNTRIES = {"NG", "RU", "BR"}

RNG = np.random.default_rng(42)


def generate_training_dataset(
    n_samples: int = 500_000,
    fraud_rate: float = 0.02,
    start_date: datetime | None = None,
) -> pd.DataFrame:
    """Generate a labelled training dataset with realistic fraud patterns."""
    if start_date is None:
        start_date = datetime.now(timezone.utc) - timedelta(days=90)

    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    legit = _generate_legitimate(n_legit, start_date)
    fraud = _generate_fraudulent(n_fraud, start_date)

    df = pd.concat([legit, fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def _generate_legitimate(n: int, start_date: datetime) -> pd.DataFrame:
    rows = []
    for _ in range(n):
        amount = float(RNG.lognormal(mean=3.5, sigma=1.2))  # log-normal spend
        hour = int(RNG.choice(range(24), p=_hour_prob_legit()))
        rows.append({
            "amount": round(min(amount, 10000), 2),
            "hour_of_day": hour,
            "day_of_week": int(RNG.integers(0, 7)),
            "merchant_category_encoded": int(RNG.integers(0, 9)),
            "pos_type_encoded": int(RNG.choice([0, 1, 2, 3], p=[0.4, 0.2, 0.3, 0.1])),
            "tx_count_1m": int(RNG.poisson(0.3)),
            "tx_count_5m": int(RNG.poisson(0.8)),
            "tx_count_1h": int(RNG.poisson(3)),
            "tx_count_24h": int(RNG.poisson(12)),
            "amount_sum_1h": float(RNG.lognormal(4.5, 1.5)),
            "amount_avg_1h": float(RNG.lognormal(3.5, 1.2)),
            "amount_max_1h": float(RNG.lognormal(4.0, 1.5)),
            "amount_sum_24h": float(RNG.lognormal(5.5, 1.5)),
            "distinct_merchants_1h": int(RNG.poisson(1.5)),
            "distinct_countries_1h": 1,
            "card_risk_score": float(RNG.beta(1, 10)),
            "merchant_fraud_rate_30d": float(RNG.beta(1, 50)),
            "merchant_avg_amount": float(RNG.lognormal(3.5, 1.0)),
            "card_avg_spend_30d": float(RNG.lognormal(4.0, 1.0)),
            "amount_vs_avg_ratio": float(RNG.lognormal(0, 0.5)),
            "amount_vs_merchant_ratio": float(RNG.lognormal(0, 0.5)),
            "is_fraud": 0,
        })
    return pd.DataFrame(rows)


def _generate_fraudulent(n: int, start_date: datetime) -> pd.DataFrame:
    rows = []
    fraud_patterns = ["velocity_burst", "high_amount", "cross_country", "high_risk_merchant"]
    for _ in range(n):
        pattern = RNG.choice(fraud_patterns)
        row = _base_fraud_row()
        if pattern == "velocity_burst":
            row["tx_count_1m"] = int(RNG.integers(5, 20))
            row["tx_count_5m"] = int(RNG.integers(15, 50))
            row["distinct_merchants_1h"] = int(RNG.integers(5, 15))
        elif pattern == "high_amount":
            row["amount"] = float(RNG.uniform(2000, 10000))
            row["amount_vs_avg_ratio"] = float(RNG.uniform(10, 50))
        elif pattern == "cross_country":
            row["distinct_countries_1h"] = int(RNG.integers(2, 5))
            row["card_risk_score"] = float(RNG.uniform(0.4, 0.9))
        elif pattern == "high_risk_merchant":
            row["merchant_fraud_rate_30d"] = float(RNG.uniform(0.05, 0.3))
            row["card_risk_score"] = float(RNG.uniform(0.3, 0.8))
        row["is_fraud"] = 1
        rows.append(row)
    return pd.DataFrame(rows)


def _base_fraud_row() -> dict:
    amount = float(RNG.lognormal(4.5, 1.5))
    return {
        "amount": round(min(amount, 10000), 2),
        "hour_of_day": int(RNG.choice([0, 1, 2, 3, 23], p=[0.2, 0.2, 0.2, 0.2, 0.2])),
        "day_of_week": int(RNG.integers(0, 7)),
        "merchant_category_encoded": int(RNG.integers(0, 9)),
        "pos_type_encoded": int(RNG.choice([1, 3], p=[0.5, 0.5])),  # swipe or online
        "tx_count_1m": int(RNG.poisson(2)),
        "tx_count_5m": int(RNG.poisson(5)),
        "tx_count_1h": int(RNG.poisson(10)),
        "tx_count_24h": int(RNG.poisson(25)),
        "amount_sum_1h": float(RNG.lognormal(6, 1)),
        "amount_avg_1h": float(RNG.lognormal(4.5, 1)),
        "amount_max_1h": float(RNG.lognormal(5, 1)),
        "amount_sum_24h": float(RNG.lognormal(7, 1)),
        "distinct_merchants_1h": int(RNG.poisson(3)),
        "distinct_countries_1h": 1,
        "card_risk_score": float(RNG.uniform(0.2, 0.7)),
        "merchant_fraud_rate_30d": float(RNG.uniform(0.01, 0.1)),
        "merchant_avg_amount": float(RNG.lognormal(3.5, 1.0)),
        "card_avg_spend_30d": float(RNG.lognormal(4.0, 1.0)),
        "amount_vs_avg_ratio": float(RNG.uniform(2, 20)),
        "amount_vs_merchant_ratio": float(RNG.uniform(1, 15)),
    }


def _hour_prob_legit() -> list[float]:
    """Realistic hour distribution for legitimate transactions (peak 10am-8pm)."""
    probs = np.zeros(24)
    probs[8:23] = np.array([0.02, 0.04, 0.07, 0.08, 0.08, 0.07, 0.07, 0.07, 0.07, 0.06, 0.05, 0.05, 0.04, 0.04, 0.04])
    probs[0:8] = 0.01
    probs[22:24] = 0.02
    return (probs / probs.sum()).tolist()


def transaction_stream(rate_per_sec: int = 100) -> Iterator[dict]:
    """Infinite generator of synthetic transactions for load testing / local dev."""
    import time
    card_ids = [f"card_{i:06d}" for i in range(10_000)]
    merchant_ids = [f"merch_{i:04d}" for i in range(1_000)]
    while True:
        for _ in range(rate_per_sec):
            yield {
                "event_id": str(RNG.integers(10**12, 10**13)),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "card_id": random.choice(card_ids),
                "merchant_id": random.choice(merchant_ids),
                "merchant_category": random.choice(MERCHANT_CATEGORIES),
                "amount": round(float(RNG.lognormal(3.5, 1.2)), 2),
                "currency": "USD",
                "country_code": random.choice(COUNTRY_CODES),
                "pos_type": random.choice(POS_TYPES),
                "mcc": int(RNG.integers(1000, 9999)),
            }
        time.sleep(1)
