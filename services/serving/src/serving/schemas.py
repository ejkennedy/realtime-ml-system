from __future__ import annotations

from typing import Any

import msgspec


class VelocityFeatures(msgspec.Struct, omit_defaults=True):
    tx_count_1m: float = 0.0
    tx_count_5m: float = 0.0
    tx_count_1h: float = 0.0
    tx_count_24h: float = 0.0
    amount_sum_1h: float = 0.0
    amount_avg_1h: float = 0.0
    amount_max_1h: float = 0.0
    amount_sum_24h: float = 0.0
    distinct_merchants_1h: float = 0.0
    distinct_countries_1h: float = 0.0


class TransactionRequest(msgspec.Struct, omit_defaults=True):
    event_id: str = ""
    timestamp: str | None = None
    timestamp_unix_ms: int | None = None
    timestamp_epoch_ms: int | None = None
    hour_of_day: int | None = None
    day_of_week: int | None = None
    card_id: str = ""
    merchant_id: str = ""
    merchant_category: str = "other"
    amount: float = 0.0
    currency: str = "USD"
    country_code: str = "US"
    pos_type: str = "chip"
    mcc: int = 0
    velocity: VelocityFeatures | None = None
    velocity_tx_count_1m: float = 0.0
    velocity_tx_count_5m: float = 0.0
    velocity_tx_count_1h: float = 0.0
    velocity_tx_count_24h: float = 0.0
    velocity_amount_sum_1h: float = 0.0
    velocity_amount_avg_1h: float = 0.0
    velocity_amount_max_1h: float = 0.0
    velocity_amount_sum_24h: float = 0.0
    velocity_distinct_merchants_1h: float = 0.0
    velocity_distinct_countries_1h: float = 0.0
    card_risk_score: float = 0.0
    merchant_fraud_rate_30d: float = 0.0
    merchant_avg_amount: float = 0.0
    card_avg_spend_30d: float = 0.0


class ScoreResponse(msgspec.Struct):
    event_id: str
    fraud_score: float
    is_fraud: bool
    model_version: str
    inference_latency_ms: float


_decoder = msgspec.json.Decoder(type=TransactionRequest)


def decode_transaction_request(payload: bytes | bytearray | memoryview) -> TransactionRequest:
    return _decoder.decode(payload)


def score_response_to_dict(response: ScoreResponse) -> dict[str, Any]:
    return {
        "event_id": response.event_id,
        "fraud_score": response.fraud_score,
        "is_fraud": response.is_fraud,
        "model_version": response.model_version,
        "inference_latency_ms": response.inference_latency_ms,
    }
