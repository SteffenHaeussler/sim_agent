import base64
import os

from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor


def setup_tracing(config: dict):
    telemetry_enabled = config.get("telemetry_enabled", "False")
    os.environ["TELEMETRY_ENABLED"] = telemetry_enabled

    if telemetry_enabled == "true":
        LANGFUSE_AUTH = base64.b64encode(
            f"{config['langfuse_public_key']}:{config['langfuse_secret_key']}".encode()
        ).decode()

        trace_provider = TracerProvider()
        trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))

        SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)

        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = config[
            "otel_exporter_otlp_endpoint"
        ]
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = (
            f"Authorization=Basic {LANGFUSE_AUTH}"
        )

        os.environ["LANGFUSE_HOST"] = config["langfuse_host"]
        os.environ["LANGFUSE_PROJECT_ID"] = config["langfuse_project_id"]
        os.environ["LANGFUSE_PUBLIC_KEY"] = config["langfuse_public_key"]
        os.environ["LANGFUSE_SECRET_KEY"] = config["langfuse_secret_key"]

    else:
        if "LANGFUSE_HOST" in os.environ:
            del os.environ["LANGFUSE_HOST"]
        if "LANGFUSE_PROJECT_ID" in os.environ:
            del os.environ["LANGFUSE_PROJECT_ID"]
        if "LANGFUSE_PUBLIC_KEY" in os.environ:
            del os.environ["LANGFUSE_PUBLIC_KEY"]
        if "LANGFUSE_SECRET_KEY" in os.environ:
            del os.environ["LANGFUSE_SECRET_KEY"]
        if "OTEL_EXPORTER_OTLP_ENDPOINT" in os.environ:
            del os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]
        if "OTEL_EXPORTER_OTLP_HEADERS" in os.environ:
            del os.environ["OTEL_EXPORTER_OTLP_HEADERS"]
