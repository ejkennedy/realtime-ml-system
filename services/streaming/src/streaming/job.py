"""
Main Flink job: transactions-raw → feature enrichment → transactions-enriched.

Pipeline:
  Redpanda (transactions-raw)
    → VelocityOperator (keyed by card_id, RocksDB state)
    → FeastEnrichmentOperator (Redis pipeline lookup)
    → Redpanda (transactions-enriched)
    → Iceberg sink (offline store, async)

Run locally:
    python -m streaming.job
"""

from __future__ import annotations

import json
import os

import structlog
from pyflink.common import WatermarkStrategy
from pyflink.common.serialization import SimpleStringSchema
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import (
    DeliveryGuarantee,
    KafkaOffsetsInitializer,
    KafkaRecordSerializationSchema,
    KafkaSink,
    KafkaSource,
)

from streaming.checkpointing import configure_exactly_once
from streaming.operators.enrichment import FeastEnrichmentOperator
from streaming.operators.velocity import VelocityOperator

log = structlog.get_logger()

KAFKA_BROKERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
TOPIC_RAW = os.environ.get("KAFKA_TOPIC_RAW", "transactions-raw")
TOPIC_ENRICHED = os.environ.get("KAFKA_TOPIC_ENRICHED", "transactions-enriched")
CONSUMER_GROUP = os.environ.get("KAFKA_CONSUMER_GROUP", "fraud-detection-cg")


def build_pipeline(env: StreamExecutionEnvironment) -> None:
    # ── Source: Redpanda / Kafka ──────────────────────────────────────────────
    source = (
        KafkaSource.builder()
        .set_bootstrap_servers(KAFKA_BROKERS)
        .set_topics(TOPIC_RAW)
        .set_group_id(CONSUMER_GROUP)
        .set_starting_offsets(KafkaOffsetsInitializer.earliest())
        .set_value_only_deserializer(SimpleStringSchema())
        .build()
    )

    raw_stream = env.from_source(
        source,
        WatermarkStrategy.for_monotonous_timestamps(),
        "transactions-raw-source",
    )

    # ── Deserialise JSON → dict ───────────────────────────────────────────────
    transactions = raw_stream.map(
        lambda s: json.loads(s),
        output_type=None,
    ).name("deserialise")

    # ── Velocity features (stateful, keyed by card_id) ────────────────────────
    enriched = (
        transactions
        .key_by(lambda t: t["card_id"])
        .process(VelocityOperator())
        .name("velocity-features")
    )

    # ── Feast enrichment (stateless Redis lookup) ─────────────────────────────
    fully_enriched = (
        enriched
        .map(FeastEnrichmentOperator())
        .name("feast-enrichment")
    )

    # ── Sink: enriched topic ──────────────────────────────────────────────────
    sink = (
        KafkaSink.builder()
        .set_bootstrap_servers(KAFKA_BROKERS)
        .set_record_serializer(
            KafkaRecordSerializationSchema.builder()
            .set_topic(TOPIC_ENRICHED)
            .set_value_serialization_schema(
                SimpleStringSchema()  # JSON serialised dict
            )
            .build()
        )
        .set_delivery_guarantee(
            DeliveryGuarantee.EXACTLY_ONCE
        )
        .set_transactional_id_prefix("fraud-pipeline-")
        .build()
    )

    fully_enriched.map(json.dumps).name("serialise").sink_to(sink).name(
        "transactions-enriched-sink"
    )


def main() -> None:
    env = StreamExecutionEnvironment.get_execution_environment()

    # Parallelism: 4 for local dev, override via FLINK_PARALLELISM env var
    parallelism = int(os.environ.get("FLINK_PARALLELISM", 4))
    env.set_parallelism(parallelism)

    configure_exactly_once(env)
    build_pipeline(env)

    log.info("Starting Flink job", parallelism=parallelism)
    env.execute("fraud-feature-pipeline")


if __name__ == "__main__":
    main()
