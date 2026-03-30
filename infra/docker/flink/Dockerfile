FROM flink:1.19.1-scala_2.12-java11

# Install Python 3.12
RUN apt-get update && apt-get install -y \
    python3.12 python3.12-dev python3-pip curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:/root/.local/bin:$PATH"

# Set Flink Python environment
ENV PYFLINK_PYTHON=/usr/bin/python3

# Download required JARs into Flink lib
RUN curl -fSL https://repo1.maven.org/maven2/org/apache/flink/flink-connector-kafka/3.2.0-1.19/flink-connector-kafka-3.2.0-1.19.jar \
    -o /opt/flink/lib/flink-connector-kafka.jar

RUN curl -fSL https://repo1.maven.org/maven2/org/apache/kafka/kafka-clients/3.6.2/kafka-clients-3.6.2.jar \
    -o /opt/flink/lib/kafka-clients.jar

RUN curl -fSL https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-flink-runtime-1.19/1.6.0/iceberg-flink-runtime-1.19-1.6.0.jar \
    -o /opt/flink/lib/iceberg-flink-runtime.jar

RUN curl -fSL https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-common/3.3.6/hadoop-common-3.3.6.jar \
    -o /opt/flink/lib/hadoop-common.jar

# Install PyFlink and Python deps
WORKDIR /app/streaming
COPY services/streaming/pyproject.toml ./pyproject.toml
RUN pip3 install apache-flink==1.19.0 confluent-kafka pydantic redis pyarrow structlog python-dotenv feast

COPY services/streaming/src ./src

WORKDIR /opt/flink
