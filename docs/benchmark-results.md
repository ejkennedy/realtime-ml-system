# Benchmark Results

These are committed local benchmark results from artifacts generated on
**March 31, 2026** with the repo's laptop-friendly perf path:

```bash
make serve-perf
bash scripts/benchmark_matrix.sh
make perf-breakdown
```

They are useful for relative tradeoff analysis and regression detection. They
are not presented as proof of the production SLA target.

## Local Benchmark Matrix

Source artifacts:

- `reports/load_test_summary_benchmark_r1_s1.md`
- `reports/load_test_summary_benchmark_r1_s2.md`
- `reports/load_test_summary_benchmark_r2_s2.md`
- `reports/benchmark_matrix.md`

| Config | Replicas | ONNX Sessions | Requests | Failure Rate | Avg RPS | p50 | p95 | p99 | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `r1_s1` | 1 | 1 | 4703 | 0.00% | 174.05 | 141.44 ms | 231.77 ms | 273.11 ms | 132.62 ms |
| `r1_s2` | 1 | 2 | 4190 | 0.00% | 154.98 | 153.02 ms | 275.67 ms | 315.10 ms | 148.36 ms |
| `r2_s2` | 2 | 2 | 4284 | 0.00% | 158.50 | 144.34 ms | 266.40 ms | 335.71 ms | 144.97 ms |

## Tradeoffs

- `r1_s1` is the best published local profile in this repo. A single scorer
  replica with a single ONNX session avoids CPU oversubscription on a laptop
  and delivered both the lowest p95 and the highest RPS.
- `r1_s2` increases the ONNX pool without increasing useful parallelism on the
  same host. That adds contention and degrades p95 from `231.77 ms` to
  `275.67 ms`.
- `r2_s2` adds a second scorer replica, but on local hardware it still loses to
  `r1_s1`. That supports the design choice in `make serve-perf` to benchmark
  with bounded concurrency instead of "more replicas by default".

## Perf Snapshot

The latest perf breakdown artifact on March 31, 2026 reported:

- local profile label: `local`
- total requests: `13799`
- failure rate: `0.00%`
- average RPS: `167.63`
- p50: `152.89 ms`
- p95: `260.84 ms`
- p99: `347.94 ms`

See [README.md](/Users/ethan/Dev/realtime-ml-system/README.md) for the short
summary and [docs/latency-sla.md](/Users/ethan/Dev/realtime-ml-system/docs/latency-sla.md)
for how these local numbers relate to the architecture target.
