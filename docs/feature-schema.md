# Feature Schema

This document makes the serving contract explicit: what flows through the Kafka
topics, what the 12 partitions carry, and which model features are actually fed
into ONNX.

## Partitioning And Topic Layout

The high-throughput topics are provisioned with **12 partitions**:

- `transactions-raw`
- `transactions-enriched`
- `transactions-scored`

The producer writes `transactions-raw` with `card_id` as the Kafka key in
[scripts/produce_transactions.py](/Users/ethan/Dev/realtime-ml-system/scripts/produce_transactions.py),
and the Flink velocity operator is keyed by `card_id` in
[services/streaming/src/streaming/job.py](/Users/ethan/Dev/realtime-ml-system/services/streaming/src/streaming/job.py).

That means the 12 partitions carry sharded transaction streams by `card_id`,
preserving per-card ordering for velocity state while allowing parallelism
across many cards. The `shadow-results` topic is intentionally smaller at
6 partitions because it is used for offline comparison traffic rather than the
main request path.

## Raw Transaction Schema

Fields on `transactions-raw`:

| Field | Type | Notes |
|---|---|---|
| `event_id` | `str` | Unique transaction event ID |
| `timestamp` | `datetime` | Event timestamp |
| `card_id` | `str` | Partitioning and state key |
| `merchant_id` | `str` | Merchant identifier |
| `merchant_category` | `enum` | Retail, food, travel, online, etc. |
| `amount` | `float` | Positive transaction amount |
| `currency` | `str` | Default `USD` |
| `country_code` | `str` | ISO country-like code in synthetic data |
| `pos_type` | `str` | `chip`, `swipe`, `contactless`, `online` |
| `mcc` | `int` | Merchant category code |

Source:
[services/streaming/src/streaming/schemas/transaction.py](/Users/ethan/Dev/realtime-ml-system/services/streaming/src/streaming/schemas/transaction.py)

## Enriched Serving Payload

By the time the request reaches serving, the payload contains:

- original transaction fields
- Flink velocity aggregates
- Feast online features
- convenience top-level `velocity_*` fields used by the msgspec decoder

Top-level enrichment fields:

| Field | Type | Source |
|---|---|---|
| `velocity_tx_count_1m` | `float` | Flink velocity state |
| `velocity_tx_count_5m` | `float` | Flink velocity state |
| `velocity_tx_count_1h` | `float` | Flink velocity state |
| `velocity_tx_count_24h` | `float` | Flink velocity state |
| `velocity_amount_sum_1h` | `float` | Flink velocity state |
| `velocity_amount_avg_1h` | `float` | Flink velocity state |
| `velocity_amount_max_1h` | `float` | Flink velocity state |
| `velocity_amount_sum_24h` | `float` | Flink velocity state |
| `velocity_distinct_merchants_1h` | `float` | Flink velocity state |
| `velocity_distinct_countries_1h` | `float` | Flink velocity state |
| `card_risk_score` | `float` | Feast lookup |
| `merchant_fraud_rate_30d` | `float` | Feast lookup |
| `merchant_avg_amount` | `float` | Feast lookup |
| `card_avg_spend_30d` | `float` | Feast lookup |

Source:
[services/serving/src/serving/schemas.py](/Users/ethan/Dev/realtime-ml-system/services/serving/src/serving/schemas.py)

## ONNX Input Feature Vector

The model consumes **21 numeric features** in this exact order:

| Index | Feature | Description |
|---:|---|---|
| 0 | `amount` | Transaction amount |
| 1 | `hour_of_day` | Event hour |
| 2 | `day_of_week` | Event weekday |
| 3 | `merchant_category_encoded` | Integer-encoded category |
| 4 | `pos_type_encoded` | Integer-encoded POS type |
| 5 | `tx_count_1m` | Velocity count over 1 minute |
| 6 | `tx_count_5m` | Velocity count over 5 minutes |
| 7 | `tx_count_1h` | Velocity count over 1 hour |
| 8 | `tx_count_24h` | Velocity count over 24 hours |
| 9 | `amount_sum_1h` | Amount sum over 1 hour |
| 10 | `amount_avg_1h` | Amount average over 1 hour |
| 11 | `amount_max_1h` | Amount max over 1 hour |
| 12 | `amount_sum_24h` | Amount sum over 24 hours |
| 13 | `distinct_merchants_1h` | Distinct merchants over 1 hour |
| 14 | `distinct_countries_1h` | Distinct countries over 1 hour |
| 15 | `card_risk_score` | Card-level risk prior |
| 16 | `merchant_fraud_rate_30d` | Merchant fraud prior |
| 17 | `merchant_avg_amount` | Merchant spend baseline |
| 18 | `card_avg_spend_30d` | Cardholder spend baseline |
| 19 | `amount_vs_avg_ratio` | `amount / card_avg_spend_30d` |
| 20 | `amount_vs_merchant_ratio` | `amount / merchant_avg_amount` |

Source:
[services/serving/src/serving/models/onnx_runner.py](/Users/ethan/Dev/realtime-ml-system/services/serving/src/serving/models/onnx_runner.py)
