from serving.schemas import ScoreResponse, decode_transaction_request, score_response_to_dict


def test_decode_transaction_request_supports_nested_velocity_features() -> None:
    payload = b"""
    {
      "event_id": "evt-1",
      "amount": 42.5,
      "merchant_category": "online",
      "pos_type": "online",
      "velocity": {
        "tx_count_1m": 2,
        "amount_sum_1h": 120.0
      }
    }
    """

    decoded = decode_transaction_request(payload)

    assert decoded.event_id == "evt-1"
    assert decoded.velocity is not None
    assert decoded.velocity.tx_count_1m == 2
    assert decoded.velocity.amount_sum_1h == 120.0


def test_score_response_to_dict_keeps_public_api_shape() -> None:
    response = ScoreResponse(
        event_id="evt-2",
        fraud_score=0.91,
        is_fraud=True,
        model_version="local",
        inference_latency_ms=8.7,
    )

    assert score_response_to_dict(response) == {
        "event_id": "evt-2",
        "fraud_score": 0.91,
        "is_fraud": True,
        "model_version": "local",
        "inference_latency_ms": 8.7,
    }
