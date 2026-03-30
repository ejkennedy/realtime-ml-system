"""Generate synthetic offline feature tables for Feast local development."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
DATA_DIR = Path(__file__).resolve().parent


def generate_card_stats(n_cards: int = 10_000) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    return pd.DataFrame(
        {
            "card_id": [f"card_{i:06d}" for i in range(n_cards)],
            "risk_score": RNG.beta(1.2, 8.0, size=n_cards).astype(np.float32),
            "avg_spend_30d": RNG.lognormal(mean=4.0, sigma=0.7, size=n_cards).astype(np.float32),
            "typical_countries": RNG.choice(
                ["US", "GB", "DE", "FR", "CA", "AU"], size=n_cards
            ),
            "event_timestamp": [now - timedelta(days=1)] * n_cards,
            "created": [now] * n_cards,
        }
    )


def generate_merchant_stats(n_merchants: int = 1_000) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    return pd.DataFrame(
        {
            "merchant_id": [f"merch_{i:04d}" for i in range(n_merchants)],
            "fraud_rate_30d": RNG.beta(1.0, 45.0, size=n_merchants).astype(np.float32),
            "avg_amount": RNG.lognormal(mean=3.8, sigma=0.8, size=n_merchants).astype(np.float32),
            "transaction_count_30d": RNG.integers(100, 50_000, size=n_merchants, dtype=np.int32),
            "event_timestamp": [now - timedelta(days=1)] * n_merchants,
            "created": [now] * n_merchants,
        }
    )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    card_stats = generate_card_stats()
    merchant_stats = generate_merchant_stats()

    card_path = DATA_DIR / "card_stats.parquet"
    merchant_path = DATA_DIR / "merchant_stats.parquet"

    card_stats.to_parquet(card_path, index=False)
    merchant_stats.to_parquet(merchant_path, index=False)

    print(f"Wrote {len(card_stats)} card rows to {card_path}")
    print(f"Wrote {len(merchant_stats)} merchant rows to {merchant_path}")


if __name__ == "__main__":
    main()
