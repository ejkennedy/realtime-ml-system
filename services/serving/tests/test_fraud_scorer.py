from __future__ import annotations

import numpy as np

from serving.deployments.fraud_scorer import FraudScorer
from serving.schemas import TransactionRequest, VelocityFeatures

RAW_FRAUD_SCORER = FraudScorer.func_or_class


def _build_scorer() -> object:
    scorer = RAW_FRAUD_SCORER.__new__(RAW_FRAUD_SCORER)
    scorer._input_buffer = np.zeros((1, 21), dtype=np.float32)
    return scorer


def test_extract_calendar_features_prefers_explicit_fields() -> None:
    payload = TransactionRequest(hour_of_day=14, day_of_week=2)

    assert RAW_FRAUD_SCORER._extract_calendar_features(payload) == (14, 2)


def test_extract_calendar_features_reads_unix_timestamp() -> None:
    payload = TransactionRequest(timestamp_unix_ms=1_714_554_000_000)

    hour_of_day, day_of_week = RAW_FRAUD_SCORER._extract_calendar_features(payload)

    assert hour_of_day == 9
    assert day_of_week == 2


def test_prepare_features_uses_nested_velocity_when_available() -> None:
    scorer = _build_scorer()
    payload = TransactionRequest(
        amount=120.0,
        merchant_category="online",
        pos_type="online",
        card_risk_score=0.4,
        merchant_fraud_rate_30d=0.05,
        merchant_avg_amount=60.0,
        card_avg_spend_30d=80.0,
        velocity=VelocityFeatures(
            tx_count_1m=3,
            tx_count_5m=5,
            tx_count_1h=9,
            tx_count_24h=20,
            amount_sum_1h=240.0,
            amount_avg_1h=80.0,
            amount_max_1h=120.0,
            amount_sum_24h=900.0,
            distinct_merchants_1h=2,
            distinct_countries_1h=1,
        ),
    )

    features = scorer._prepare_features(payload)

    assert features.shape == (1, 21)
    assert np.isclose(features[0, 0], 120.0)
    assert np.isclose(features[0, 5], 3.0)
    assert np.isclose(features[0, 15], 0.4)
    assert np.isclose(features[0, 19], 1.5)
    assert np.isclose(features[0, 20], 2.0)
