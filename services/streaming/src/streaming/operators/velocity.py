"""
Velocity feature operator: stateful sliding window counts/aggregates per card.

Uses Flink keyed state (MapState + ListState) backed by RocksDB for fault-tolerance.
All state is keyed by card_id and scoped to TTL windows to prevent state explosion.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from pyflink.common import Types
from pyflink.common.watermark_strategy import WatermarkStrategy
from pyflink.datastream import KeyedProcessFunction, RuntimeContext
from pyflink.datastream.state import (
    ListState,
    ListStateDescriptor,
    StateTtlConfig,
)

from streaming.schemas.transaction import RawTransaction, VelocityFeatures

# Window boundaries in milliseconds
WINDOW_1M = 60_000
WINDOW_5M = 300_000
WINDOW_1H = 3_600_000
WINDOW_24H = 86_400_000


class VelocityOperator(KeyedProcessFunction):
    """
    Stateful Flink ProcessFunction that computes per-card velocity features.

    State layout (all keyed by card_id):
    - _tx_timestamps: ListState[int]  — event timestamps (ms) in last 24h
    - _tx_amounts: ListState[float]   — amounts paired with timestamps
    - _tx_countries: ListState[str]   — (timestamp, country) pairs as JSON
    - _tx_merchants: ListState[str]   — (timestamp, merchant_id) pairs as JSON

    TTL: 25h (slightly over 24h window to handle late arrivals up to 1h)
    """

    def open(self, runtime_context: RuntimeContext) -> None:
        ttl = (
            StateTtlConfig.new_builder(Time.hours(25))
            .set_update_type(StateTtlConfig.UpdateType.OnCreateAndWrite)
            .set_state_visibility(
                StateTtlConfig.StateVisibility.NeverReturnExpired
            )
            .build()
        )

        ts_descriptor = ListStateDescriptor("tx_timestamps", Types.LONG())
        ts_descriptor.enable_time_to_live(ttl)
        self._tx_timestamps: ListState = runtime_context.get_list_state(ts_descriptor)

        amt_descriptor = ListStateDescriptor("tx_amounts", Types.FLOAT())
        amt_descriptor.enable_time_to_live(ttl)
        self._tx_amounts: ListState = runtime_context.get_list_state(amt_descriptor)

        country_descriptor = ListStateDescriptor("tx_countries", Types.STRING())
        country_descriptor.enable_time_to_live(ttl)
        self._tx_countries: ListState = runtime_context.get_list_state(country_descriptor)

        merchant_descriptor = ListStateDescriptor("tx_merchants", Types.STRING())
        merchant_descriptor.enable_time_to_live(ttl)
        self._tx_merchants: ListState = runtime_context.get_list_state(merchant_descriptor)

    def process_element(self, value: dict[str, Any], ctx: KeyedProcessFunction.Context):
        now_ms = int(ctx.timestamp()) if ctx.timestamp() else int(datetime.now(timezone.utc).timestamp() * 1000)
        cutoff_24h = now_ms - WINDOW_24H

        # Load existing state into memory for this invocation
        timestamps = list(self._tx_timestamps.get() or [])
        amounts = list(self._tx_amounts.get() or [])
        countries_raw = list(self._tx_countries.get() or [])
        merchants_raw = list(self._tx_merchants.get() or [])

        # Prune entries older than 24h (state TTL handles expiry, this prevents unbounded reads)
        valid_mask = [ts >= cutoff_24h for ts in timestamps]
        timestamps = [ts for ts, v in zip(timestamps, valid_mask) if v]
        amounts = [a for a, v in zip(amounts, valid_mask) if v]
        countries_raw = [c for c, v in zip(countries_raw, valid_mask) if v]
        merchants_raw = [m for m, v in zip(merchants_raw, valid_mask) if v]

        countries = [json.loads(c) for c in countries_raw]
        merchants = [json.loads(m) for m in merchants_raw]

        # Append current transaction
        timestamps.append(now_ms)
        amounts.append(value["amount"])
        countries.append({"ts": now_ms, "country": value["country_code"]})
        merchants.append({"ts": now_ms, "merchant": value["merchant_id"]})

        # Persist back
        self._tx_timestamps.update(timestamps)
        self._tx_amounts.update(amounts)
        self._tx_countries.update([json.dumps(c) for c in countries])
        self._tx_merchants.update([json.dumps(m) for m in merchants])

        # Compute features over windows
        features = self._compute_features(
            now_ms, timestamps, amounts, countries, merchants
        )

        value["velocity"] = features.model_dump()
        yield value

    def _compute_features(
        self,
        now_ms: int,
        timestamps: list[int],
        amounts: list[float],
        countries: list[dict],
        merchants: list[dict],
    ) -> VelocityFeatures:
        """Compute velocity features over 1m, 5m, 1h, 24h windows."""

        def in_window(ts: int, window_ms: int) -> bool:
            return ts >= now_ms - window_ms

        # Transaction counts per window
        tx_count_1m = sum(1 for ts in timestamps if in_window(ts, WINDOW_1M))
        tx_count_5m = sum(1 for ts in timestamps if in_window(ts, WINDOW_5M))
        tx_count_1h = sum(1 for ts in timestamps if in_window(ts, WINDOW_1H))
        tx_count_24h = len(timestamps)

        # Amount aggregates over 1h
        amounts_1h = [a for ts, a in zip(timestamps, amounts) if in_window(ts, WINDOW_1H)]
        amount_sum_1h = sum(amounts_1h) if amounts_1h else 0.0
        amount_avg_1h = amount_sum_1h / len(amounts_1h) if amounts_1h else 0.0
        amount_max_1h = max(amounts_1h) if amounts_1h else 0.0

        # Amount sum over 24h
        amount_sum_24h = sum(
            a for ts, a in zip(timestamps, amounts) if in_window(ts, WINDOW_24H)
        )

        # Distinct merchants in 1h (anomaly: card used at many merchants quickly)
        merchants_1h = {
            m["merchant"] for m in merchants if in_window(m["ts"], WINDOW_1H)
        }

        # Distinct countries in 1h (anomaly: geographically impossible travel)
        countries_1h = {
            c["country"] for c in countries if in_window(c["ts"], WINDOW_1H)
        }

        return VelocityFeatures(
            card_id="",  # filled by caller
            tx_count_1m=tx_count_1m,
            tx_count_5m=tx_count_5m,
            tx_count_1h=tx_count_1h,
            tx_count_24h=tx_count_24h,
            amount_sum_1h=round(amount_sum_1h, 2),
            amount_avg_1h=round(amount_avg_1h, 2),
            amount_max_1h=round(amount_max_1h, 2),
            amount_sum_24h=round(amount_sum_24h, 2),
            distinct_merchants_1h=len(merchants_1h),
            distinct_countries_1h=len(countries_1h),
        )
