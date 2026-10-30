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

"""Observability, tracing, and PII redaction package."""

from app.observability.logger import (
    StructuredJsonFormatter,
    StructuredObservabilityLogger,
    clear_emitted_log_records,
    get_emitted_log_records,
    get_structured_logger,
    log_tool_intent,
    log_tool_intent_and_outcome,
    log_tool_outcome,
    set_observability_context,
)
from app.observability.pii_scrubber import (
    REDACTED_CREDIT_CARD,
    REDACTED_CVV,
    REDACTED_EMAIL,
    REDACTED_PHONE,
    PIIScrubber,
    scrub_pii,
    scrub_text,
)
from app.observability.tracer import (
    clear_recorded_spans,
    configure_tracer_provider,
    get_recorded_spans,
    get_tracer,
    in_memory_span_exporter,
    trace_agent_delegation,
    trace_tool_execution,
    trace_user_turn,
)

__all__ = [
    "REDACTED_CREDIT_CARD",
    "REDACTED_CVV",
    "REDACTED_EMAIL",
    "REDACTED_PHONE",
    "PIIScrubber",
    "StructuredJsonFormatter",
    "StructuredObservabilityLogger",
    "clear_emitted_log_records",
    "clear_recorded_spans",
    "configure_tracer_provider",
    "get_emitted_log_records",
    "get_recorded_spans",
    "get_structured_logger",
    "get_tracer",
    "in_memory_span_exporter",
    "log_tool_intent",
    "log_tool_intent_and_outcome",
    "log_tool_outcome",
    "scrub_pii",
    "scrub_text",
    "set_observability_context",
    "trace_agent_delegation",
    "trace_tool_execution",
    "trace_user_turn",
]
