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

"""Structured JSON logging and Intent-vs-Outcome tool instrumentation.

Implements Requirement 1 & Requirement 2 of Category 4: Observability & Tracing:
1. Structured JSON Logging via Python's `logging` module with `timestamp`,
   `session_id`, `log_level`, `event_type`, `message`, and `context` metadata
   (all actively scrubbed for PII via `scrub_pii`, with no bare print statements).
2. Intent vs. Outcome Logging via `@log_tool_intent_and_outcome` decorator and
   callback helpers that log the agent's `TOOL_INTENT` (tool name, parameters,
   predicted purpose) immediately before execution and `TOOL_OUTCOME` (actual
   return value, latency_ms, status) immediately after execution.
"""

from __future__ import annotations

import contextvars
import datetime
import functools
import inspect
import json
import logging
import sys
import time
from collections.abc import Callable
from typing import Any, TypeVar

from app.observability.pii_scrubber import scrub_pii
from app.observability.tracer import trace_tool_execution

_F = TypeVar("_F", bound=Callable[..., Any])

# Context variables for propagating session_id and agent_name across tool calls
current_session_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_session_id", default="default_session"
)
current_agent_name_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_agent_name", default="TriageCoordinator"
)

# In-memory ring buffer of parsed JSON log payloads for audit and test verification
_EMITTED_JSON_LOGS: list[dict[str, Any]] = []
_MAX_BUFFER_SIZE: int = 500


def set_observability_context(
    session_id: str | None = None,
    agent_name: str | None = None,
) -> None:
    """Sets the active `session_id` and `agent_name` in contextvars for logging/tracing."""
    if session_id:
        current_session_id_var.set(str(session_id))
    if agent_name:
        current_agent_name_var.set(str(agent_name))


def get_emitted_log_records() -> list[dict[str, Any]]:
    """Returns a copy of all structured JSON log records emitted in this process."""
    return list(_EMITTED_JSON_LOGS)


def clear_emitted_log_records() -> None:
    """Clears the in-memory structured JSON log buffer."""
    _EMITTED_JSON_LOGS.clear()


class StructuredJsonFormatter(logging.Formatter):
    """Formats Python `LogRecord` instances as single-line PII-scrubbed JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Serializes `record` into a structured JSON string with required metadata."""
        session_id = getattr(record, "session_id", None) or current_session_id_var.get()
        event_type = getattr(record, "event_type", "APPLICATION_LOG")
        raw_context = getattr(record, "context_metadata", {}) or {}

        payload: dict[str, Any] = {
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "session_id": str(scrub_pii(str(session_id))),
            "log_level": record.levelname,
            "logger_name": record.name,
            "event_type": str(event_type),
            "message": str(scrub_pii(record.getMessage())),
            "context": scrub_pii(raw_context),
        }

        if len(_EMITTED_JSON_LOGS) >= _MAX_BUFFER_SIZE:
            _EMITTED_JSON_LOGS.pop(0)
        _EMITTED_JSON_LOGS.append(payload)

        return json.dumps(payload, default=str)


class StructuredObservabilityLogger:
    """Wrapper around `logging.Logger` emitting PII-scrubbed structured JSON logs."""

    def __init__(self, name: str = "retail_returns_concierge") -> None:
        """Initializes the structured JSON logger without duplicate handlers."""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if not any(
            isinstance(getattr(h, "formatter", None), StructuredJsonFormatter)
            for h in self.logger.handlers
        ):
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(StructuredJsonFormatter())
            self.logger.addHandler(handler)

    def log_event(
        self,
        message: str,
        *,
        level: int = logging.INFO,
        session_id: str | None = None,
        event_type: str = "APPLICATION_LOG",
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Emits a structured JSON log record and returns the emitted payload dict."""
        resolved_session = session_id or current_session_id_var.get()
        self.logger.log(
            level,
            message,
            extra={
                "session_id": resolved_session,
                "event_type": event_type,
                "context_metadata": context or {},
            },
        )
        return _EMITTED_JSON_LOGS[-1]

    def info(
        self,
        message: str,
        *,
        session_id: str | None = None,
        event_type: str = "APPLICATION_LOG",
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Emits an INFO-level structured JSON log."""
        return self.log_event(
            message,
            level=logging.INFO,
            session_id=session_id,
            event_type=event_type,
            context=context,
        )

    def warning(
        self,
        message: str,
        *,
        session_id: str | None = None,
        event_type: str = "APPLICATION_LOG",
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Emits a WARNING-level structured JSON log."""
        return self.log_event(
            message,
            level=logging.WARNING,
            session_id=session_id,
            event_type=event_type,
            context=context,
        )

    def error(
        self,
        message: str,
        *,
        session_id: str | None = None,
        event_type: str = "APPLICATION_LOG",
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Emits an ERROR-level structured JSON log."""
        return self.log_event(
            message,
            level=logging.ERROR,
            session_id=session_id,
            event_type=event_type,
            context=context,
        )


_DEFAULT_LOGGER = StructuredObservabilityLogger("retail_returns_concierge")


def get_structured_logger(
    name: str = "retail_returns_concierge",
) -> StructuredObservabilityLogger:
    """Returns a configured `StructuredObservabilityLogger` instance."""
    if name == "retail_returns_concierge":
        return _DEFAULT_LOGGER
    return StructuredObservabilityLogger(name)


def _extract_outcome_status(result: Any) -> str:
    """Extracts the execution status string from a structured tool return value."""
    if hasattr(result, "status"):
        return str(result.status)
    if isinstance(result, dict):
        if "status" in result:
            return str(result["status"])
        if "error" in result:
            return "error"
    return "success"


def log_tool_intent(
    tool_name: str,
    parameters: dict[str, Any],
    predicted_purpose: str,
    *,
    session_id: str | None = None,
    agent_name: str | None = None,
) -> dict[str, Any]:
    """Logs the agent's INTENT immediately before executing a tool."""
    resolved_session = session_id or current_session_id_var.get()
    resolved_agent = agent_name or current_agent_name_var.get()
    return _DEFAULT_LOGGER.info(
        f"INTENT: Agent '{resolved_agent}' intends to call tool '{tool_name}' ({predicted_purpose}).",
        session_id=resolved_session,
        event_type="TOOL_INTENT",
        context={
            "phase": "INTENT",
            "agent_name": resolved_agent,
            "tool_name": tool_name,
            "predicted_purpose": predicted_purpose,
            "parameters": scrub_pii(parameters),
        },
    )


def log_tool_outcome(
    tool_name: str,
    return_value: Any,
    latency_ms: float,
    status: str,
    *,
    session_id: str | None = None,
    agent_name: str | None = None,
) -> dict[str, Any]:
    """Logs the tool execution OUTCOME immediately after execution."""
    resolved_session = session_id or current_session_id_var.get()
    resolved_agent = agent_name or current_agent_name_var.get()
    serialized_return = (
        return_value.model_dump()
        if hasattr(return_value, "model_dump")
        else return_value
    )
    level = (
        logging.WARNING
        if status in ("error", "ineligible", "denied", "HITL_PENDING", "HITL_REJECTED")
        else logging.INFO
    )
    return _DEFAULT_LOGGER.log_event(
        f"OUTCOME: Tool '{tool_name}' finished with status='{status}' in {latency_ms:.2f}ms.",
        level=level,
        session_id=resolved_session,
        event_type="TOOL_OUTCOME",
        context={
            "phase": "OUTCOME",
            "agent_name": resolved_agent,
            "tool_name": tool_name,
            "status": status,
            "latency_ms": round(latency_ms, 3),
            "return_value": scrub_pii(serialized_return),
        },
    )


def log_tool_intent_and_outcome(
    predicted_purpose: str | None = None,
) -> Callable[[_F], _F]:
    """Decorator that logs `TOOL_INTENT` before and `TOOL_OUTCOME` after tool execution.

    Also wraps the execution in an OpenTelemetry `"tool_execution"` span via
    `trace_tool_execution`, recording `tool.status` and `tool.latency_ms`.

    Args:
        predicted_purpose: Human-readable description of why the tool is called.
            Defaults to the first summary line of the function's docstring.

    Returns:
        Decorated callable with identical signature, docstring, and type annotations.
    """

    def decorator(func: _F) -> _F:
        doc_first_line = (
            inspect.cleandoc(func.__doc__).splitlines()[0]
            if func.__doc__
            else f"Execute {func.__name__}"
        )
        purpose = predicted_purpose or doc_first_line
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                bound = sig.bind_partial(*args, **kwargs)
                bound_args = {
                    k: (v.model_dump() if hasattr(v, "model_dump") else v)
                    for k, v in bound.arguments.items()
                }
            except TypeError:
                bound_args = {"args": list(args), "kwargs": dict(kwargs)}

            session_id = current_session_id_var.get()
            agent_name = current_agent_name_var.get()

            log_tool_intent(
                tool_name=func.__name__,
                parameters=bound_args,
                predicted_purpose=purpose,
                session_id=session_id,
                agent_name=agent_name,
            )

            start_ts = time.perf_counter()
            with trace_tool_execution(
                tool_name=func.__name__,
                parameters=bound_args,
                session_id=session_id,
                agent_name=agent_name,
                predicted_purpose=purpose,
            ) as span:
                try:
                    result = func(*args, **kwargs)
                    latency_ms = (time.perf_counter() - start_ts) * 1000.0
                    status = _extract_outcome_status(result)
                    span.set_attribute("tool.status", status)
                    span.set_attribute("tool.latency_ms", round(latency_ms, 3))
                    log_tool_outcome(
                        tool_name=func.__name__,
                        return_value=result,
                        latency_ms=latency_ms,
                        status=status,
                        session_id=session_id,
                        agent_name=agent_name,
                    )
                    return result
                except Exception as exc:
                    latency_ms = (time.perf_counter() - start_ts) * 1000.0
                    span.set_attribute("tool.status", "exception")
                    span.set_attribute("tool.latency_ms", round(latency_ms, 3))
                    log_tool_outcome(
                        tool_name=func.__name__,
                        return_value={"error": str(exc)},
                        latency_ms=latency_ms,
                        status="exception",
                        session_id=session_id,
                        agent_name=agent_name,
                    )
                    raise

        return wrapper  # type: ignore[return-value]

    return decorator
