FROM flink:1.19.1-scala_2.12-java11

# Install Python 3, pip, and JDK (full JDK needed for pemja — apache-flink's
# Java-Python bridge — which requires JDK headers to compile from source)
RUN apt-get update && apt-get install -y \
    python3 python3-dev python3-pip curl default-jdk \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

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

# Install PyFlink + deps. JAVA_HOME must point to the JDK (not JRE) so that
# pemja can find the include/ headers. Derive it dynamically from javac.
RUN export JAVA_HOME=$(dirname $(dirname $(readlink -f /usr/bin/javac))) && \
    pip3 install --no-cache-dir \
        apache-flink==1.19.0 \
        confluent-kafka \
        pydantic \
        redis \
        structlog \
        python-dotenv

WORKDIR /opt/flink
