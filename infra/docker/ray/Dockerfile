FROM python:3.12-slim

RUN apt-get update && apt-get install -y curl gcc g++ && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:/root/.local/bin:$PATH"

# Install Ray and serving dependencies
RUN pip install --no-cache-dir \
    "ray[serve]==2.32.0" \
    "onnxruntime==1.18.1" \
    "numpy>=1.26,<2.0" \
    "feast[redis]==0.40.0" \
    "redis>=5.0" \
    "pydantic>=2.7" \
    "prometheus-client>=0.20" \
    "structlog>=24.0" \
    "mlflow>=2.14,<3.0" \
    "python-dotenv>=1.0" \
    "httpx>=0.27" \
    "fastapi>=0.111" \
    "uvicorn[standard]>=0.30" \
    "confluent-kafka>=2.4"

WORKDIR /app
