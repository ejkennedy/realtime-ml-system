from __future__ import annotations

from pathlib import Path

import click
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from training.data_generator import generate_training_dataset
from training.xgboost_trainer import FEATURE_COLS, TARGET_COL, XGBOOST_PARAMS

load_dotenv()


@click.command()
@click.option("--n-samples", default=10000, show_default=True, help="Synthetic sample size for CI/local checks")
@click.option("--fraud-rate", default=0.02, show_default=True)
@click.option("--min-auc", default=0.97, show_default=True, help="Minimum validation ROC AUC")
@click.option("--min-aucpr", default=0.80, show_default=True, help="Minimum validation PR AUC")
@click.option(
    "--artifact-dir",
    default="reports/ci_eval",
    show_default=True,
    help="Directory for evaluation artifacts",
)
def main(
    n_samples: int,
    fraud_rate: float,
    min_auc: float,
    min_aucpr: float,
    artifact_dir: str,
) -> None:
    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = generate_training_dataset(n_samples=n_samples, fraud_rate=fraud_rate)
    X_train, X_val, y_train, y_val = train_test_split(
        df[FEATURE_COLS].values,
        df[TARGET_COL].values,
        test_size=0.2,
        stratify=df[TARGET_COL].values,
        random_state=42,
    )
    params = {
        **XGBOOST_PARAMS,
        "n_estimators": 80,
        "max_depth": 5,
        "n_jobs": 1,
        "early_stopping_rounds": 10,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    val_preds = model.predict_proba(X_val)[:, 1]

    auc = float(roc_auc_score(y_val, val_preds))
    aucpr = float(average_precision_score(y_val, val_preds))
    _write_artifacts(
        output_dir=output_dir,
        model=model,
        auc=auc,
        aucpr=aucpr,
        y_val=y_val,
        val_preds=val_preds,
        n_samples=n_samples,
        fraud_rate=fraud_rate,
    )
    if auc < min_auc or aucpr < min_aucpr:
        raise SystemExit(
            "Evaluation gate failed: "
            f"val_auc={auc:.4f} (min {min_auc:.4f}), "
            f"val_aucpr={aucpr:.4f} (min {min_aucpr:.4f})"
        )

    print(
        "Evaluation gate passed: "
        f"val_auc={auc:.4f}, val_aucpr={aucpr:.4f}, "
        f"artifact_dir={output_dir}"
    )


def _write_artifacts(
    output_dir: Path,
    model: xgb.XGBClassifier,
    auc: float,
    aucpr: float,
    y_val: object,
    val_preds: object,
    n_samples: int,
    fraud_rate: float,
) -> None:
    feature_df = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    feature_df.to_csv(output_dir / "feature_importance.csv", index=False)

    fpr, tpr, roc_thresholds = roc_curve(y_val, val_preds)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds}).to_csv(
        output_dir / "roc_curve.csv",
        index=False,
    )

    precision, recall, pr_thresholds = precision_recall_curve(y_val, val_preds)
    pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "threshold": list(pr_thresholds) + [None],
        }
    ).to_csv(output_dir / "precision_recall_curve.csv", index=False)

    top_features = [
        f"- `{row.feature}`: `{row.importance:.4f}`"
        for row in feature_df.head(5).itertuples(index=False)
    ]
    (output_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Quick Evaluation Summary",
                "",
                f"- Samples: `{n_samples}`",
                f"- Fraud rate: `{fraud_rate:.4f}`",
                f"- Validation ROC AUC: `{auc:.4f}`",
                f"- Validation PR AUC: `{aucpr:.4f}`",
                "",
                "## Top Features",
                *top_features,
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
