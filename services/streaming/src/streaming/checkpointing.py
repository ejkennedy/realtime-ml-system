"""Flink exactly-once checkpointing configuration."""

from datetime import timedelta

from pyflink.common import RestartStrategies
from pyflink.datastream import (
    CheckpointingMode,
    ExternalizedCheckpointCleanup,
    StreamExecutionEnvironment,
)
from pyflink.datastream.state_backend import EmbeddedRocksDBStateBackend


def configure_exactly_once(env: StreamExecutionEnvironment) -> None:
    """
    Configure exactly-once semantics with incremental RocksDB checkpointing.

    Key decisions:
    - 10s checkpoint interval: balance between recovery granularity and overhead
    - Incremental RocksDB: reduces checkpoint latency from ~2s to ~200ms at steady state
    - Tolerate 1 consecutive failure: prevents transient checkpoint timeouts from failing the job
    - RETAIN_ON_CANCELLATION: enables manual recovery inspection
    """
    env.enable_checkpointing(10_000, CheckpointingMode.EXACTLY_ONCE)

    config = env.get_checkpoint_config()
    config.set_min_pause_between_checkpoints(5_000)     # 5s min between checkpoints
    config.set_checkpoint_timeout(60_000)               # fail if checkpoint takes > 60s
    config.set_max_concurrent_checkpoints(1)            # never overlap checkpoints
    config.set_tolerable_checkpoint_failure_number(1)   # tolerate 1 transient failure
    config.enable_externalized_checkpoints(
        ExternalizedCheckpointCleanup.RETAIN_ON_CANCELLATION
    )

    # RocksDB with incremental snapshots
    backend = EmbeddedRocksDBStateBackend(enable_incremental_checkpointing=True)
    env.set_state_backend(backend)
    env.set_restart_strategy(
        RestartStrategies.fixed_delay_restart(
            restart_attempts=3,
            delay_between_attempts=timedelta(seconds=10),
        )
    )
