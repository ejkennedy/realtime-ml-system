"""
LinUCB contextual bandit for adaptive fraud threshold selection.

Instead of a fixed 0.5 threshold, the bandit learns per-context thresholds
that balance false positive rate (customer friction) vs fraud catch rate.

Contexts: merchant_category × hour_bucket (9 × 4 = 36 contexts)
Arms: threshold buckets [0.3, 0.4, 0.5, 0.6, 0.7] (5 arms)

Reward:
  +1  if is_fraud=True and we flagged it (true positive)
  +0  if is_fraud=False and we didn't flag it (true negative)
  -0.5 if is_fraud=False and we flagged it (false positive — customer friction)
  -2  if is_fraud=True and we missed it (false negative — fraud loss)
"""

from __future__ import annotations

import os
import pickle
import threading
import time
from typing import Optional

import numpy as np
import redis
import structlog

log = structlog.get_logger()

REDIS_BANDIT_KEY = "online_model:bandit:linucb"
THRESHOLDS = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
N_ARMS = len(THRESHOLDS)
N_CONTEXTS = 36  # 9 merchant categories × 4 hour buckets
ALPHA = 1.5      # UCB exploration parameter


class LinUCBBandit:
    """
    LinUCB bandit with disjoint linear models (one per arm).

    Each arm maintains A (d×d matrix) and b (d-vector) for linear regression
    with UCB exploration bonus.

    Context vector (d=4):
      [hour_sin, hour_cos, merchant_category_norm, fraud_score_raw]
    """

    D = 4  # context dimension

    def __init__(self) -> None:
        # A[arm] = D×D identity (ridge regression init)
        self.A = np.stack([np.eye(self.D) for _ in range(N_ARMS)])
        self.b = np.zeros((N_ARMS, self.D))
        self._lock = threading.Lock()

    def select_threshold(self, fraud_score: float, merchant_category: int, hour: int) -> float:
        """Select the threshold with highest UCB score for this context."""
        ctx = self._build_context(fraud_score, merchant_category, hour)
        ucb_scores = np.zeros(N_ARMS)
        with self._lock:
            for arm in range(N_ARMS):
                A_inv = np.linalg.inv(self.A[arm])
                theta = A_inv @ self.b[arm]
                ucb_scores[arm] = theta @ ctx + ALPHA * np.sqrt(ctx @ A_inv @ ctx)
        best_arm = int(np.argmax(ucb_scores))
        return THRESHOLDS[best_arm]

    def update(
        self,
        fraud_score: float,
        merchant_category: int,
        hour: int,
        chosen_threshold: float,
        actual_label: int,
        predicted_fraud: bool,
    ) -> None:
        """Update LinUCB with observed reward."""
        arm = self._threshold_to_arm(chosen_threshold)
        ctx = self._build_context(fraud_score, merchant_category, hour)
        reward = self._compute_reward(actual_label, predicted_fraud)
        with self._lock:
            self.A[arm] += np.outer(ctx, ctx)
            self.b[arm] += reward * ctx

    def _build_context(self, fraud_score: float, merchant_category: int, hour: int) -> np.ndarray:
        """Encode context as a float32 vector."""
        angle = 2 * np.pi * hour / 24
        return np.array([
            np.sin(angle),
            np.cos(angle),
            merchant_category / 8.0,   # normalise to [0,1]
            fraud_score,
        ], dtype=np.float32)

    @staticmethod
    def _threshold_to_arm(threshold: float) -> int:
        diffs = [abs(threshold - t) for t in THRESHOLDS]
        return int(np.argmin(diffs))

    @staticmethod
    def _compute_reward(actual_label: int, predicted_fraud: bool) -> float:
        if actual_label == 1 and predicted_fraud:
            return 1.0    # true positive — caught fraud
        if actual_label == 0 and not predicted_fraud:
            return 0.1    # true negative — no friction
        if actual_label == 0 and predicted_fraud:
            return -0.5   # false positive — unnecessary friction
        return -2.0        # false negative — missed fraud (worst outcome)

    def to_redis(self, redis_client: redis.Redis) -> None:
        data = pickle.dumps({"A": self.A, "b": self.b})
        redis_client.set(REDIS_BANDIT_KEY, data, ex=86400)

    @classmethod
    def from_redis(cls, redis_client: redis.Redis) -> "LinUCBBandit":
        data = redis_client.get(REDIS_BANDIT_KEY)
        bandit = cls()
        if data:
            try:
                state = pickle.loads(data)
                bandit.A = state["A"]
                bandit.b = state["b"]
                log.info("LinUCB bandit loaded from Redis")
            except Exception as e:
                log.warning("Failed to load bandit from Redis", error=str(e))
        return bandit
