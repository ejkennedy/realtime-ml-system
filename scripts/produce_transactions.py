"""
Synthetic transaction producer — publishes to Redpanda at a configurable rate.

Usage:
    python scripts/produce_transactions.py --rate 1000
"""

import json
import os
import time

import click
import structlog
from confluent_kafka import Producer

log = structlog.get_logger()


@click.command()
@click.option("--rate", default=100, show_default=True, help="Events per second")
@click.option("--duration", default=0, show_default=True, help="Duration in seconds (0=infinite)")
def main(rate: int, duration: int) -> None:
    import sys
    sys.path.insert(0, "services/training/src")
    from training.data_generator import transaction_stream

    producer = Producer({
        "bootstrap.servers": os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092"),
        "linger.ms": 5,
        "batch.num.messages": 500,
        "compression.type": "lz4",
    })
    topic = os.environ.get("KAFKA_TOPIC_RAW", "transactions-raw")

    log.info("Starting producer", rate=rate, topic=topic)
    count = 0
    start = time.time()
    for tx in transaction_stream(rate_per_sec=rate):
        producer.produce(topic, key=tx["card_id"], value=json.dumps(tx).encode())
        producer.poll(0)
        count += 1
        if count % 10_000 == 0:
            log.info("Produced", count=count, elapsed_s=round(time.time() - start, 1))
        if duration > 0 and time.time() - start >= duration:
            break

    producer.flush()
    log.info("Producer done", total=count)


if __name__ == "__main__":
    main()
