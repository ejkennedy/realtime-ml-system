"""
Feast feature definitions for the fraud detection system.

Entities:
  - card: identified by card_id
  - merchant: identified by merchant_id

Feature Views:
  - card_stats: risk score, avg spend, typical countries (materialised from offline)
  - merchant_stats: fraud rate, avg amount (materialised from offline)
  - card_velocity: real-time velocity features (stream-materialised from Flink output)
"""

from datetime import timedelta
from pathlib import Path

import pandas as pd
from feast import Entity, FeatureStore, FeatureView, Field, FileSource
from feast.types import Float32, Int32, String

# ── Entities ──────────────────────────────────────────────────────────────────

card = Entity(
    name="card",
    join_keys=["card_id"],
    description="Payment card entity",
)

merchant = Entity(
    name="merchant",
    join_keys=["merchant_id"],
    description="Merchant entity",
)

# ── Data Sources ──────────────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).parent.parent.parent.parent.parent.parent / "data"

card_stats_source = FileSource(
    name="card_stats_source",
    path=str(_DATA_DIR / "synthetic/card_stats.parquet"),
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

merchant_stats_source = FileSource(
    name="merchant_stats_source",
    path=str(_DATA_DIR / "synthetic/merchant_stats.parquet"),
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

# ── Feature Views ─────────────────────────────────────────────────────────────

card_stats_view = FeatureView(
    name="card_stats",
    entities=[card],
    ttl=timedelta(days=30),
    schema=[
        Field(name="risk_score", dtype=Float32),
        Field(name="avg_spend_30d", dtype=Float32),
        Field(name="typical_countries", dtype=String),
    ],
    source=card_stats_source,
    online=True,
    tags={"team": "fraud"},
)

merchant_stats_view = FeatureView(
    name="merchant_stats",
    entities=[merchant],
    ttl=timedelta(days=30),
    schema=[
        Field(name="fraud_rate_30d", dtype=Float32),
        Field(name="avg_amount", dtype=Float32),
        Field(name="transaction_count_30d", dtype=Int32),
    ],
    source=merchant_stats_source,
    online=True,
    tags={"team": "fraud"},
)
