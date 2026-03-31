# Fraud Detector Model Card

## Summary

- Model family: `XGBoost` binary classifier exported to ONNX for serving.
- Primary use case: score card-present and online transactions for fraud risk in
  the real-time serving path.
- Training data in this portfolio repo: synthetic transactions with fraud
  patterns such as velocity bursts, high-amount anomalies, cross-country use,
  and high-risk merchants.
- Serving threshold: `0.5` fraud probability by default.

## Published Evaluation

The committed quick-eval artifact bundle was generated on **March 31, 2026**
with:

```bash
.venv/bin/python scripts/eval_quick.py --n-samples 6000 --artifact-dir docs/assets/model_eval
```

Results:

| Metric | Value |
|---|---:|
| Samples | 6000 |
| Fraud rate | 2.00% |
| Validation ROC AUC | 1.0000 |
| Validation PR AUC | 1.0000 |

These values are intentionally labelled as synthetic-data validation, not as a
claim about real production fraud performance.

## Feature Importance

Source: [docs/assets/model_eval/feature_importance.csv](/Users/ethan/Dev/realtime-ml-system/docs/assets/model_eval/feature_importance.csv)

| Rank | Feature | Importance |
|---|---|---:|
| 1 | `tx_count_24h` | 0.5948 |
| 2 | `amount_vs_avg_ratio` | 0.1823 |
| 3 | `distinct_countries_1h` | 0.0471 |
| 4 | `card_risk_score` | 0.0373 |
| 5 | `distinct_merchants_1h` | 0.0357 |
| 6 | `amount_vs_merchant_ratio` | 0.0288 |
| 7 | `tx_count_1h` | 0.0182 |
| 8 | `tx_count_5m` | 0.0180 |

The feature ranking is consistent with the fraud patterns encoded in the
synthetic generator: long-window velocity, amount spikes relative to baseline,
cross-country activity, and risk priors dominate the decision boundary.

## ROC And PR Operating Points

Source files:

- [docs/assets/model_eval/roc_curve.csv](/Users/ethan/Dev/realtime-ml-system/docs/assets/model_eval/roc_curve.csv)
- [docs/assets/model_eval/precision_recall_curve.csv](/Users/ethan/Dev/realtime-ml-system/docs/assets/model_eval/precision_recall_curve.csv)

Selected ROC checkpoints:

| Threshold | TPR | FPR |
|---|---:|---:|
| 0.9907 | 0.5417 | 0.0000 |
| 0.6974 | 1.0000 | 0.0000 |
| 0.0824 | 1.0000 | 0.0026 |

Selected precision-recall checkpoints:

| Threshold | Recall | Precision |
|---|---:|---:|
| 0.0091 | 1.0000 | 0.0200 |
| 0.6974 | 1.0000 | 1.0000 |
| 0.9907 | 0.5417 | 1.0000 |

## Known Limits

- The portfolio dataset is synthetic. It is designed to exercise platform
  choices, feature flow, and evaluation wiring, not to approximate a bank's
  real fraud distribution.
- The repo publishes both latency and model-quality numbers, but they should be
  interpreted as local validation artifacts rather than audited production
  benchmarks.
