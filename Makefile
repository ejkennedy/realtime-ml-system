.PHONY: help up down logs topics feast-apply feast-materialize train train-quick eval-quick test serve serve-perf serve-perf-quantized load-test load-test-local load-test-stress perf-breakdown drift-check rollback clean

COMPOSE = docker compose -f infra/docker/docker-compose.yml
UV = uv

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Infrastructure ────────────────────────────────────────────────────────────

up: ## Start all local services
	$(COMPOSE) up -d --build
	@echo "Waiting for services..."
	@sleep 15
	@$(MAKE) topics
	@echo "Stack ready. Services:"
	@echo "  Redpanda Console:  http://localhost:8080"
	@echo "  Flink UI:          http://localhost:8081"
	@echo "  Ray Dashboard:     http://localhost:8265"
	@echo "  MLflow:            http://localhost:5001"
	@echo "  Grafana:           http://localhost:3000 (admin/admin)"
	@echo "  Prometheus:        http://localhost:9090"
	@echo "  MinIO:             http://localhost:9001 (minioadmin/minioadmin)"
	@echo "  Evidently:         http://localhost:8085"

down: ## Stop all local services
	$(COMPOSE) down

logs: ## Follow logs for all services
	$(COMPOSE) logs -f

topics: ## Create Redpanda topics
	$(COMPOSE) exec redpanda rpk topic create transactions-raw --partitions 12 --replicas 1 || true
	$(COMPOSE) exec redpanda rpk topic create transactions-enriched --partitions 12 --replicas 1 || true
	$(COMPOSE) exec redpanda rpk topic create transactions-scored --partitions 12 --replicas 1 || true
	$(COMPOSE) exec redpanda rpk topic create shadow-results --partitions 6 --replicas 1 || true
	$(COMPOSE) exec redpanda rpk topic create retraining-triggers --partitions 1 --replicas 1 || true
	@echo "Topics created."

# ── Feature Store ─────────────────────────────────────────────────────────────

feast-apply: ## Apply Feast feature definitions
	$(UV) run --package feature-store feast -c services/feature-store/src/feature_store/repo apply

feast-materialize: ## Materialise features to Redis online store
	$(UV) run --package feature-store feast -c services/feature-store/src/feature_store/repo materialize-incremental $$(date -u +%Y-%m-%dT%H:%M:%S)

seed-data: ## Generate synthetic feature store seed data
	$(UV) run python data/synthetic/generate_feature_data.py

# ── Model Training ────────────────────────────────────────────────────────────

train: ## Train XGBoost model and export to ONNX
	$(UV) run --package training python -m training.pipeline
	@echo "Model saved to models/registry/fraud_detector/latest/model.onnx"

train-quick: ## Quick training run (50k samples)
	$(UV) run --package training python -m training.pipeline --n-samples 50000

eval-quick: ## Fast model quality gate for CI/local checks
	mkdir -p reports/ci_eval
	$(UV) run --package training python scripts/eval_quick.py --artifact-dir reports/ci_eval

test: ## Run the fast unit test suite
	$(UV) run pytest

# ── Serving ───────────────────────────────────────────────────────────────────

serve: ## Start Ray Serve locally (without Docker)
	$(UV) run --package serving python -m serving.app

serve-perf: ## Start Ray Serve with lighter local perf settings
	ray stop --force >/dev/null 2>&1 || true
	mkdir -p $(HOME)/.raytmp
	RAY_TMPDIR=$(HOME)/.raytmp \
	SHADOW_ENABLED=false \
	VERSION_MANAGER_ENABLED=false \
	ONLINE_UPDATES_ENABLED=false \
	SERVE_USE_ROUTER=false \
	SERVE_SCORER_REPLICAS=1 \
	SERVE_ROUTER_REPLICAS=1 \
	SERVE_ROUTER_MAX_ONGOING_REQUESTS=32 \
	ONNX_SESSION_POOL_SIZE=1 \
	ONNX_INTRA_OP_THREADS=1 \
	ONNX_INTER_OP_THREADS=1 \
	$(UV) run --package serving python -m serving.app

serve-perf-quantized: ## Start Ray Serve in local perf mode using the quantized ONNX model when available
	ray stop --force >/dev/null 2>&1 || true
	mkdir -p $(HOME)/.raytmp
	RAY_TMPDIR=$(HOME)/.raytmp \
	SHADOW_ENABLED=false \
	VERSION_MANAGER_ENABLED=false \
	ONLINE_UPDATES_ENABLED=false \
	SERVE_USE_ROUTER=false \
	SERVE_USE_QUANTIZED_MODEL=true \
	SERVE_SCORER_REPLICAS=1 \
	SERVE_ROUTER_REPLICAS=1 \
	SERVE_ROUTER_MAX_ONGOING_REQUESTS=32 \
	ONNX_SESSION_POOL_SIZE=1 \
	ONNX_INTRA_OP_THREADS=1 \
	ONNX_INTER_OP_THREADS=1 \
	$(UV) run --package serving python -m serving.app

serve-docker: ## Deploy serving via docker-compose ray-head
	$(COMPOSE) restart ray-head

# ── Streaming ─────────────────────────────────────────────────────────────────

produce: ## Start synthetic transaction producer
	$(UV) run python scripts/produce_transactions.py

flink-job: ## Submit Flink feature pipeline job
	$(COMPOSE) exec flink-jobmanager flink run \
		-py /app/streaming/src/streaming/job.py \
		-pyfs /app/streaming/src

# ── Load Testing ──────────────────────────────────────────────────────────────

load-test: ## Run stress load test (aggressive local saturation profile)
	mkdir -p reports
	LOAD_TEST_LABEL=stress \
	$(UV) run --package load-testing locust \
		-f services/load-testing/src/load_testing/locustfile.py \
		--host http://localhost:8000 \
		--users 200 --spawn-rate 20 --run-time 5m \
		--headless --html reports/load_test_$$(date +%Y%m%d_%H%M%S).html

load-test-stress: load-test ## Alias for the aggressive stress profile

load-test-local: ## Run lighter local load test profile
	mkdir -p reports
	LOAD_TEST_LABEL=local \
	LOAD_TEST_STAGE1_DURATION_S=20 \
	LOAD_TEST_STAGE1_USERS=10 \
	LOAD_TEST_STAGE1_SPAWN_RATE=2 \
	LOAD_TEST_STAGE2_DURATION_S=80 \
	LOAD_TEST_STAGE2_USERS=30 \
	LOAD_TEST_STAGE2_SPAWN_RATE=5 \
	LOAD_TEST_STAGE3_DURATION_S=100 \
	LOAD_TEST_STAGE3_USERS=0 \
	LOAD_TEST_STAGE3_SPAWN_RATE=10 \
	$(UV) run --package load-testing locust \
		-f services/load-testing/src/load_testing/locustfile.py \
		--host http://localhost:8000 \
		--headless --html reports/load_test_local_$$(date +%Y%m%d_%H%M%S).html

load-test-ui: ## Run Locust with web UI
	mkdir -p reports
	$(UV) run --package load-testing locust \
		-f services/load-testing/src/load_testing/locustfile.py \
		--host http://localhost:8000

perf-breakdown: ## Snapshot live scorer stage timings into a Markdown artifact
	mkdir -p reports
	$(UV) run python scripts/perf_breakdown.py --finalize-profiles

latency-plot: ## Latency plots are generated automatically during load-test
	@echo "Run 'make load-test' to generate HTML and latency plots in ./reports."

# ── Monitoring ────────────────────────────────────────────────────────────────

drift-check: ## Run drift detection check
	$(UV) run --package monitoring python -m monitoring.drift_detector --window-hours 1

# ── MLOps ─────────────────────────────────────────────────────────────────────

rollback: ## One-click model rollback to previous production version
	$(UV) run python scripts/rollback.py
	@echo "Rollback complete. Check http://localhost:5001 for model versions."

shadow-status: ## Show shadow deployment comparison metrics
	$(UV) run python scripts/shadow_status.py

# ── Utilities ────────────────────────────────────────────────────────────────

smoke-test: ## End-to-end smoke test
	bash scripts/smoke_test.sh

install: ## Install all Python dependencies
	$(UV) sync

clean: ## Remove local artifacts
	rm -rf models/registry/fraud_detector/*/model.onnx
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

clean-docker: ## Remove all Docker volumes (DESTRUCTIVE)
	$(COMPOSE) down -v
