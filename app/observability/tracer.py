# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpenTelemetry distributed tracing for user turns, agent delegation, and tools.

Implements Requirement 3 of Category 4: Observability & Tracing:
- Integrates `opentelemetry-api` and `opentelemetry-sdk`.
- Creates structured spans for each `user_turn`, `agent_delegation`, and
  `tool_execution` with PII-redacted span attributes.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Span, Status, StatusCode

from app.observability.pii_scrubber import scrub_pii

SERVICE_NAME = "retail-returns-concierge-agent"

# Shared in-memory span exporter so tests and local diagnostics can inspect spans
in_memory_span_exporter = InMemorySpanExporter()
_processor_registered = False


def configure_tracer_provider() -> TracerProvider:
    """Ensures an OpenTelemetry SDK `TracerProvider` is active and exports spans."""
    global _processor_registered
    current_provider = trace.get_tracer_provider()

    if isinstance(current_provider, TracerProvider):
        provider = current_provider
    else:
        resource = Resource.create({"service.name": SERVICE_NAME})
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)

    if not _processor_registered:
        provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
        _processor_registered = True

    return provider


def get_tracer(instrumenting_module_name: str = SERVICE_NAME) -> trace.Tracer:
    """Returns a configured OpenTelemetry `Tracer` instance."""
    configure_tracer_provider()
    return trace.get_tracer(instrumenting_module_name)


def get_recorded_spans() -> list[Any]:
    """Returns all finished OpenTelemetry spans captured by `in_memory_span_exporter`."""
    return list(in_memory_span_exporter.get_finished_spans())


def clear_recorded_spans() -> None:
    """Clears the in-memory finished spans buffer."""
    in_memory_span_exporter.clear()


@contextlib.contextmanager
def trace_user_turn(
    session_id: str = "default_session",
    user_message: str = "",
    user_id: str = "default_user",
    agent_name: str = "TriageCoordinator",
) -> Iterator[Span]:
    """Creates an OpenTelemetry span for a single user conversation turn.

    Args:
        session_id: Conversation session ID.
        user_message: Raw user prompt text (scrubbed for PII before attaching).
        user_id: User identifier.
        agent_name: Entrypoint agent handling the turn.

    Yields:
        The active OpenTelemetry `Span`.
    """
    tracer = get_tracer()
    safe_message = str(scrub_pii(user_message))
    with tracer.start_as_current_span("user_turn") as span:
        span.set_attribute("span.category", "user_turn")
        span.set_attribute("session.id", str(session_id))
        span.set_attribute("user.id", str(scrub_pii(user_id)))
        span.set_attribute("agent.name", str(agent_name))
        span.set_attribute("turn.user_message", safe_message)
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as exc:
            span.set_status(Status(StatusCode.ERROR, str(scrub_pii(str(exc)))))
            span.record_exception(exc)
            raise


@contextlib.contextmanager
def trace_agent_delegation(
    source_agent: str,
    target_agent: str,
    session_id: str = "default_session",
    reason: str = "",
) -> Iterator[Span]:
    """Creates an OpenTelemetry span representing delegation between agents.

    Args:
        source_agent: Name of the delegating coordinator/agent.
        target_agent: Name of the specialist sub-agent receiving control.
        session_id: Conversation session ID.
        reason: Reason or task context for the delegation.

    Yields:
        The active OpenTelemetry `Span`.
    """
    tracer = get_tracer()
    safe_reason = str(scrub_pii(reason))
    with tracer.start_as_current_span("agent_delegation") as span:
        span.set_attribute("span.category", "agent_delegation")
        span.set_attribute("session.id", str(session_id))
        span.set_attribute("agent.source", str(source_agent))
        span.set_attribute("agent.target", str(target_agent))
        span.set_attribute("delegation.reason", safe_reason)
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as exc:
            span.set_status(Status(StatusCode.ERROR, str(scrub_pii(str(exc)))))
            span.record_exception(exc)
            raise


@contextlib.contextmanager
def trace_tool_execution(
    tool_name: str,
    parameters: dict[str, Any] | None = None,
    session_id: str = "default_session",
    agent_name: str = "retail_returns_concierge",
    predicted_purpose: str = "",
) -> Iterator[Span]:
    """Creates an OpenTelemetry span for a tool execution with PII-redacted attributes.

    Args:
        tool_name: Name of the tool being executed.
        parameters: Input arguments passed to the tool.
        session_id: Conversation session ID.
        agent_name: Name of the agent invoking the tool.
        predicted_purpose: Predicted intent/purpose of the tool call.

    Yields:
        The active OpenTelemetry `Span`.
    """
    tracer = get_tracer()
    safe_params = scrub_pii(parameters or {})
    with tracer.start_as_current_span("tool_execution") as span:
        span.set_attribute("span.category", "tool_execution")
        span.set_attribute("session.id", str(session_id))
        span.set_attribute("agent.name", str(agent_name))
        span.set_attribute("tool.name", str(tool_name))
        span.set_attribute("tool.predicted_purpose", str(scrub_pii(predicted_purpose)))
        span.set_attribute(
            "tool.parameters",
            json.dumps(safe_params, default=str),
        )
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as exc:
            span.set_status(Status(StatusCode.ERROR, str(scrub_pii(str(exc)))))
            span.record_exception(exc)
            raise
