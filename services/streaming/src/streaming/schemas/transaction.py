"""Pydantic schemas for transaction events flowing through the pipeline."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class MerchantCategory(str, Enum):
    RETAIL = "retail"
    FOOD = "food"
    TRAVEL = "travel"
    ENTERTAINMENT = "entertainment"
    ONLINE = "online"
    ATM = "atm"
    UTILITY = "utility"
    HEALTHCARE = "healthcare"
    OTHER = "other"


class RawTransaction(BaseModel):
    """Event emitted by the transaction producer → transactions-raw topic."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime
    card_id: str
    merchant_id: str
    merchant_category: MerchantCategory
    amount: float
    currency: str = "USD"
    country_code: str
    pos_type: str  # "chip", "swipe", "contactless", "online"
    mcc: int  # Merchant Category Code

    @field_validator("amount")
    @classmethod
    def amount_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("amount must be positive")
        return round(v, 2)


class VelocityFeatures(BaseModel):
    """Computed by Flink velocity operator (keyed by card_id)."""

    card_id: str
    tx_count_1m: int = 0
    tx_count_5m: int = 0
    tx_count_1h: int = 0
    tx_count_24h: int = 0
    amount_sum_1h: float = 0.0
    amount_avg_1h: float = 0.0
    amount_max_1h: float = 0.0
    amount_sum_24h: float = 0.0
    distinct_merchants_1h: int = 0
    distinct_countries_1h: int = 0


class EnrichedTransaction(BaseModel):
    """Raw transaction + velocity features + Feast features → transactions-enriched topic."""

    # Original event
    event_id: str
    timestamp: datetime
    card_id: str
    merchant_id: str
    merchant_category: MerchantCategory
    amount: float
    currency: str
    country_code: str
    pos_type: str
    mcc: int

    # Velocity features (computed by Flink)
    velocity: VelocityFeatures

    # Feast online features
    card_risk_score: float = 0.0
    merchant_fraud_rate_30d: float = 0.0
    merchant_avg_amount: float = 0.0
    card_avg_spend_30d: float = 0.0
    card_typical_countries: list[str] = Field(default_factory=list)


class ScoredTransaction(BaseModel):
    """Inference result → transactions-scored topic."""

    event_id: str
    timestamp: datetime
    card_id: str
    fraud_score: float  # 0.0 – 1.0
    is_fraud: bool
    model_version: str
    inference_latency_ms: float
    online_model_score: float | None = None  # SGD adapter score
    ensemble_score: float | None = None  # Weighted combo
