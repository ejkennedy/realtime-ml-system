from training.data_generator import generate_training_dataset
from training.xgboost_trainer import FEATURE_COLS


def test_generate_training_dataset_has_expected_feature_columns() -> None:
    df = generate_training_dataset(n_samples=1000, fraud_rate=0.05)

    assert set(FEATURE_COLS).issubset(df.columns)
    assert "is_fraud" in df.columns
    assert len(df) == 1000


def test_generate_training_dataset_roughly_matches_requested_fraud_rate() -> None:
    df = generate_training_dataset(n_samples=2000, fraud_rate=0.03)

    observed = float(df["is_fraud"].mean())
    assert 0.025 <= observed <= 0.035
