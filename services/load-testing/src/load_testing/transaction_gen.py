"""Lightweight transaction generator for load testing (avoids heavy dependencies)."""

from __future__ import annotations

import random
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np

CATEGORIES = ["retail", "food", "travel", "entertainment", "online", "atm", "utility", "healthcare", "other"]
POS_TYPES = ["chip", "swipe", "contactless", "online"]
COUNTRIES = ["US", "GB", "DE", "FR", "CA"]

RNG = np.random.default_rng(42)
_CARD_IDS = [f"card_{i:06d}" for i in range(10_000)]
_MERCHANT_IDS = [f"merch_{i:04d}" for i in range(1_000)]


class TransactionGenerator:
    def generate_batch(self, n: int) -> list[dict[str, Any]]:
        return [self._generate_one() for _ in range(n)]

    def _generate_one(self) -> dict[str, Any]:
        amount = float(RNG.lognormal(3.5, 1.2))
        now = datetime.now(timezone.utc)
        velocity = {
            "tx_count_1m": random.randint(0, 5),
            "tx_count_5m": random.randint(0, 10),
            "tx_count_1h": random.randint(0, 30),
            "tx_count_24h": random.randint(0, 80),
            "amount_sum_1h": round(float(RNG.lognormal(4.5, 1.5)), 2),
            "amount_avg_1h": round(float(RNG.lognormal(3.5, 1.0)), 2),
            "amount_max_1h": round(float(RNG.lognormal(4.0, 1.2)), 2),
            "amount_sum_24h": round(float(RNG.lognormal(5.5, 1.5)), 2),
            "distinct_merchants_1h": random.randint(1, 5),
            "distinct_countries_1h": 1,
        }
        return {
            "event_id": str(int(time.time() * 1e6) + random.randint(0, 999)),
            "timestamp": now.isoformat(),
            "timestamp_unix_ms": int(now.timestamp() * 1000),
            "hour_of_day": now.hour,
            "day_of_week": now.weekday(),
            "card_id": random.choice(_CARD_IDS),
            "merchant_id": random.choice(_MERCHANT_IDS),
            "merchant_category": random.choice(CATEGORIES),
            "amount": round(min(amount, 5000), 2),
            "currency": "USD",
            "country_code": random.choice(COUNTRIES),
            "pos_type": random.choice(POS_TYPES),
            "mcc": random.randint(1000, 9999),
            "velocity": velocity,
            "velocity_tx_count_1m": velocity["tx_count_1m"],
            "velocity_tx_count_5m": velocity["tx_count_5m"],
            "velocity_tx_count_1h": velocity["tx_count_1h"],
            "velocity_tx_count_24h": velocity["tx_count_24h"],
            "velocity_amount_sum_1h": velocity["amount_sum_1h"],
            "velocity_amount_avg_1h": velocity["amount_avg_1h"],
            "velocity_amount_max_1h": velocity["amount_max_1h"],
            "velocity_amount_sum_24h": velocity["amount_sum_24h"],
            "velocity_distinct_merchants_1h": velocity["distinct_merchants_1h"],
            "velocity_distinct_countries_1h": velocity["distinct_countries_1h"],
            "card_risk_score": round(float(RNG.beta(1, 10)), 4),
            "merchant_fraud_rate_30d": round(float(RNG.beta(1, 50)), 4),
            "merchant_avg_amount": round(float(RNG.lognormal(3.5, 1.0)), 2),
            "card_avg_spend_30d": round(float(RNG.lognormal(4.0, 1.0)), 2),
        }
