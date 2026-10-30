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

"""Unit tests for Category 4: Observability & Tracing (`app/observability/`)."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from google.adk.models.llm_request import LlmRequest
from google.genai import types

from app.agent import (
    input_validation_guardrail_callback,
    observability_before_agent_callback,
)
from app.memory.session_store import PersistentSessionStore
from app.observability.logger import (
    clear_emitted_log_records,
    get_emitted_log_records,
    set_observability_context,
)
from app.observability.pii_scrubber import (
    REDACTED_CREDIT_CARD,
    REDACTED_CVV,
    REDACTED_EMAIL,
    REDACTED_PHONE,
    scrub_pii,
)
from app.observability.tracer import (
    clear_recorded_spans,
    get_recorded_spans,
)
from app.tools.return_tools import lookup_order_details


def test_pii_scrubber_redacts_cards_cvvs_phones_and_emails() -> None:
    """Verifies regex and key-aware PII redaction across strings and nested dicts."""
    raw_payload = {
        "note": (
            "Customer email is alice.rivera@example.com, phone (555) 234-5678 "
            "or local 555-0101, card 4111-2222-3333-4444 with CVV: 842."
        ),
        "cvv": "919",
        "customer_email": "david.kim@example.com",
        "customer_phone": "+1-555-876-5432",
        "order_id": "ORD-1001",
    }
    scrubbed = scrub_pii(raw_payload)
    assert REDACTED_EMAIL in scrubbed["note"]
    assert REDACTED_PHONE in scrubbed["note"]
    assert REDACTED_CREDIT_CARD in scrubbed["note"]
    assert REDACTED_CVV in scrubbed["note"]
    assert "alice.rivera@example.com" not in scrubbed["note"]
    assert "4111-2222-3333-4444" not in scrubbed["note"]
    assert scrubbed["cvv"] == REDACTED_CVV
    assert scrubbed["customer_email"] == REDACTED_EMAIL
    assert scrubbed["customer_phone"] == REDACTED_PHONE
    assert scrubbed["order_id"] == "ORD-1001"


def test_structured_json_intent_and_outcome_logging_and_tool_tracing() -> None:
    """Verifies TOOL_INTENT and TOOL_OUTCOME JSON logs and OpenTelemetry tool spans."""
    clear_emitted_log_records()
    clear_recorded_spans()
    set_observability_context(
        session_id="obs-session-101", agent_name="TriageCoordinator"
    )

    res = lookup_order_details(
        order_id="ORD-1001",
        customer_email="alice.rivera@example.com",
    )
    assert res.status == "success"

    logs = get_emitted_log_records()
    intent_logs = [entry for entry in logs if entry["event_type"] == "TOOL_INTENT"]
    outcome_logs = [entry for entry in logs if entry["event_type"] == "TOOL_OUTCOME"]

    assert len(intent_logs) >= 1
    assert len(outcome_logs) >= 1

    intent_entry = intent_logs[-1]
    assert intent_entry["session_id"] == "obs-session-101"
    assert intent_entry["log_level"] == "INFO"
    assert "timestamp" in intent_entry
    assert intent_entry["context"]["tool_name"] == "lookup_order_details"
    assert intent_entry["context"]["predicted_purpose"]
    # Verify PII in tool parameters was scrubbed before logging
    assert intent_entry["context"]["parameters"]["customer_email"] == REDACTED_EMAIL

    outcome_entry = outcome_logs[-1]
    assert outcome_entry["context"]["tool_name"] == "lookup_order_details"
    assert outcome_entry["context"]["status"] == "success"
    assert isinstance(outcome_entry["context"]["latency_ms"], float)
    assert (
        outcome_entry["context"]["return_value"]["order"]["customer_email"]
        == REDACTED_EMAIL
    )

    spans = get_recorded_spans()
    tool_spans = [s for s in spans if s.name == "tool_execution"]
    assert len(tool_spans) >= 1
    assert tool_spans[-1].attributes["tool.name"] == "lookup_order_details"
    assert tool_spans[-1].attributes["tool.status"] == "success"


def test_distributed_tracing_user_turn_and_agent_delegation() -> None:
    """Verifies OpenTelemetry spans for user_turn and agent_delegation."""
    clear_recorded_spans()
    shared_state: dict[str, Any] = {"session_id": "trace-session-202"}

    # 1. TriageCoordinator starts turn
    ctx_triage: Any = SimpleNamespace(
        agent_name="TriageCoordinator",
        state=shared_state,
    )
    observability_before_agent_callback(ctx_triage)

    llm_req = LlmRequest(
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text="My email is alice.rivera@example.com, please check ORD-1001."
                    )
                ],
            )
        ]
    )
    input_validation_guardrail_callback(ctx_triage, llm_req)

    # 2. TriageCoordinator delegates to ReturnPolicyEvaluator
    ctx_policy: Any = SimpleNamespace(
        agent_name="ReturnPolicyEvaluator",
        state=shared_state,
    )
    observability_before_agent_callback(ctx_policy)

    spans = get_recorded_spans()
    span_names = {s.name for s in spans}
    assert "user_turn" in span_names
    assert "agent_delegation" in span_names

    delegation_span = [s for s in spans if s.name == "agent_delegation"][-1]
    assert delegation_span.attributes["agent.source"] == "TriageCoordinator"
    assert delegation_span.attributes["agent.target"] == "ReturnPolicyEvaluator"

    user_turn_span = [s for s in spans if s.name == "user_turn"][-1]
    assert REDACTED_EMAIL in str(user_turn_span.attributes["turn.user_message"])
    assert "alice.rivera@example.com" not in str(
        user_turn_span.attributes["turn.user_message"]
    )


def test_session_store_scrubs_pii_before_disk_persistence(tmp_path: Path) -> None:
    """Verifies PersistentSessionStore scrubs PII from state and turns on disk."""
    store = PersistentSessionStore(storage_path=tmp_path / "pii_sessions.db")
    store.save_session(
        session_id="sess-pii-1",
        state={
            "customer_email": "alice.rivera@example.com",
            "customer_phone": "555-0101",
            "cvv": "321",
        },
        turns=[
            {
                "role": "user",
                "content": "Refund to card 4111 2222 3333 4444, CVV: 321, email alice.rivera@example.com",
            }
        ],
    )
    loaded = store.load_session("sess-pii-1")
    assert loaded is not None
    assert loaded["state"]["customer_email"] == REDACTED_EMAIL
    assert loaded["state"]["customer_phone"] == REDACTED_PHONE
    assert loaded["state"]["cvv"] == REDACTED_CVV
    assert "4111 2222 3333 4444" not in loaded["turns"][0]["content"]
    assert REDACTED_CREDIT_CARD in loaded["turns"][0]["content"]
    assert REDACTED_CVV in loaded["turns"][0]["content"]
    assert REDACTED_EMAIL in loaded["turns"][0]["content"]
