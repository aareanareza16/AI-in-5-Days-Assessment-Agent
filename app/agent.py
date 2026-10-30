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

"""Multi-agent orchestration, context/memory, HITL, guardrails, and observability.

Integrates Categories 1-4 of the Omnichannel Retail Return & Exchange Concierge:
- Category 1: Typed Pydantic v2 tools from `app.tools.return_tools`.
- Category 2: System Constitution (`app.prompts`), History Compaction (`app.memory.compactor`),
  and Persistent Session Store (`app.memory.session_store`) with async background tasks.
- Category 3: Coordinator / Specialist Multi-Agent Pattern (`TriageCoordinator`,
  `ReturnPolicyEvaluator`, `FulfillmentAgent`), Strategic Model Routing
  (`gemini-2.0-flash` and `gemini-2.0-pro`), HITL Checkpoint, and Guardrails.
- Category 4: Structured JSON Logging, Intent-vs-Outcome Logging, OpenTelemetry
  Distributed Tracing (`user_turn`, `agent_delegation`, `tool_execution`), and
  Active PII Redaction (`app.observability`).
"""

from __future__ import annotations

import datetime
import re
import time
from typing import Any

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.apps import App, ResumabilityConfig
from google.adk.apps.app import EventsCompactionConfig
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools import BaseTool, FunctionTool, ToolContext
from google.genai import types

from app.config import AgentSettings, get_secret, get_settings, settings
from app.memory.compactor import (
    HistoryCompactor,
    async_compact_conversation_history,
    compact_conversation_history,
    estimate_token_count,
)
from app.memory.session_store import PersistentSessionStore
from app.observability.logger import (
    get_structured_logger,
    log_tool_intent,
    log_tool_intent_and_outcome,
    log_tool_outcome,
    set_observability_context,
)
from app.observability.pii_scrubber import scrub_pii
from app.observability.tracer import (
    trace_agent_delegation,
    trace_tool_execution,
    trace_user_turn,
)
from app.prompts import (
    CONSTITUTION_PROMPT,
    FULFILLMENT_AGENT_PROMPT,
    RETURN_POLICY_EVALUATOR_PROMPT,
    SYSTEM_CONSTITUTION,
    TRIAGE_COORDINATOR_PROMPT,
    build_agent_instruction,
)
from app.tools.return_tools import (
    MOCK_ORDERS_DB,
    lookup_order_details,
    process_refund_and_generate_label,
    request_manager_override_approval,
    verify_item_return_eligibility,
)

# ============================================================================
# Strategic Model Routing Constants (Loaded via app.config.settings)
# ============================================================================

FAST_TRIAGE_MODEL = settings.fast_triage_model
DEEP_POLICY_MODEL = settings.deep_policy_model

# ============================================================================
# Human-in-the-Loop (HITL) Threshold Constants
# ============================================================================

HITL_HIGH_VALUE_REFUND_THRESHOLD_USD: float = settings.hitl_refund_threshold_usd
HITL_RETURN_WINDOW_DAYS: int = settings.hitl_return_window_days
RESTOCKING_FEE_RATE_OPENED: float = 0.15

# ============================================================================
# Shared Memory, Compaction, and Observability Services
# ============================================================================

history_compactor = HistoryCompactor()
session_store = PersistentSessionStore(compactor=history_compactor)
agent_logger = get_structured_logger("retail_returns_concierge.agent")

# ============================================================================
# Guardrails: Prompt Injection & Adversarial Request Detection
# ============================================================================

PROMPT_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"ignore\s+(all\s+)?(previous|prior|above|system)\s+(instructions|prompts|rules)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(disregard|forget|override)\s+(your\s+)?(system\s+)?(instructions|rules|policy|guardrails)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(system\s+override|developer\s+mode|jailbreak|admin\s+mode|sudo\s+refund|bypass\s+hitl|skip\s+supervisor)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(reveal|print|show|output)\s+(your\s+)?(system\s+prompt|hidden\s+instructions|fraud\s+threshold)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(you\s+are\s+now|act\s+as\s+an?\s+unrestricted|pretend\s+you\s+have\s+no\s+rules)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(\[system\]|<system>|hitl_approved\s*=\s*true|supervisor_approved\s*=\s*true)",
        re.IGNORECASE,
    ),
)

ADVERSARIAL_UNPURCHASED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"refund\s+.*(never\s+bought|never\s+purchased|didn'?t\s+buy|did\s+not\s+buy|did\s+not\s+purchase|unpurchased)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(never\s+bought|never\s+purchased|didn'?t\s+buy|did\s+not\s+buy|unpurchased\s+item|not\s+on\s+my\s+order).*refund",
        re.IGNORECASE,
    ),
    re.compile(
        r"(demand|give\s+me|issue)\s+a?\s*refund\s+without\s+(an\s+order|buying|purchasing)",
        re.IGNORECASE,
    ),
)


def _extract_latest_user_text(llm_request: LlmRequest) -> str:
    """Extracts the text content of the most recent user message in `llm_request`."""
    if not llm_request.contents:
        return ""
    for content in reversed(llm_request.contents):
        if content.role == "user" and content.parts:
            text_parts = [p.text for p in content.parts if p.text is not None]
            if text_parts:
                return " ".join(text_parts)
    return ""


def _resolve_session_id(context: Any) -> str:
    """Extracts a session identifier from an ADK `CallbackContext` or `ToolContext`."""
    inv_ctx = getattr(context, "_invocation_context", None)
    if inv_ctx is not None:
        session = getattr(inv_ctx, "session", None)
        if session is not None and getattr(session, "id", None):
            return str(session.id)
    state = getattr(context, "state", None)
    if isinstance(state, dict) and state.get("session_id"):
        return str(state["session_id"])
    if hasattr(state, "get") and state.get("session_id"):
        return str(state.get("session_id"))
    return "default_session"


def _resolve_agent_name(context: Any) -> str:
    """Extracts the active agent name from an ADK context object."""
    agent_name = getattr(context, "agent_name", None)
    if agent_name:
        return str(agent_name)
    inv_ctx = getattr(context, "_invocation_context", None)
    if inv_ctx is not None and getattr(inv_ctx, "agent", None) is not None:
        return str(getattr(inv_ctx.agent, "name", "TriageCoordinator"))
    return "TriageCoordinator"


def observability_before_agent_callback(
    callback_context: CallbackContext,
) -> types.Content | None:
    """ADK `before_agent_callback` tracking agent delegations and setting trace context."""
    session_id = _resolve_session_id(callback_context)
    agent_name = _resolve_agent_name(callback_context)
    set_observability_context(session_id=session_id, agent_name=agent_name)

    state = callback_context.state
    previous_agent = state.get("active_agent_name")
    if previous_agent and previous_agent != agent_name:
        with trace_agent_delegation(
            source_agent=str(previous_agent),
            target_agent=agent_name,
            session_id=session_id,
            reason=f"Coordinator/specialist handoff from {previous_agent} to {agent_name}",
        ):
            agent_logger.info(
                f"Delegated control from '{previous_agent}' to '{agent_name}'.",
                session_id=session_id,
                event_type="AGENT_DELEGATION",
                context={
                    "source_agent": str(previous_agent),
                    "target_agent": agent_name,
                },
            )
    state["active_agent_name"] = agent_name
    return None


def detect_adversarial_or_injection_input(user_text: str) -> tuple[bool, str, str]:
    """Evaluates user input for prompt injection or unpurchased-item refund demands.

    Args:
        user_text: Raw user message string.

    Returns:
        A 3-tuple of `(is_blocked, violation_type, reason_message)`.
    """
    if not user_text or not user_text.strip():
        return False, "", ""

    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern.search(user_text):
            return (
                True,
                "prompt_injection",
                (
                    "Guardrail triggered: Prompt injection or instruction-override "
                    "attempt detected. I cannot bypass return policies, supervisor "
                    "approvals, or system instructions."
                ),
            )

    for pattern in ADVERSARIAL_UNPURCHASED_PATTERNS:
        if pattern.search(user_text):
            return (
                True,
                "unpurchased_item_refund_demand",
                (
                    "Guardrail triggered: Adversarial refund request for an unpurchased "
                    "item detected. Refunds and exchanges can only be processed for "
                    "verified items present on a valid customer order."
                ),
            )

    # Check if the user mentions both a known order ID and an explicit SKU that was
    # never purchased on that order while demanding a refund.
    order_match = re.search(r"\b(ORD-\d{4})\b", user_text, re.IGNORECASE)
    sku_match = re.search(r"\b(SKU-[A-Z0-9-]+)\b", user_text, re.IGNORECASE)
    if order_match and sku_match and re.search(r"\brefund\b", user_text, re.IGNORECASE):
        ord_id = order_match.group(1).upper()
        sku_id = sku_match.group(1).upper()
        order = MOCK_ORDERS_DB.get(ord_id)
        if order is not None:
            valid_skus = {item.sku.upper() for item in order.items}
            if sku_id not in valid_skus:
                return (
                    True,
                    "unpurchased_item_refund_demand",
                    (
                        f"Guardrail triggered: Item '{sku_id}' was not purchased in "
                        f"order '{ord_id}' (purchased SKUs: {', '.join(sorted(valid_skus))}). "
                        "Refunds cannot be issued for unpurchased merchandise."
                    ),
                )

    return False, "", ""


def input_validation_guardrail_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> LlmResponse | None:
    """ADK `before_model_callback` enforcing guardrails, tracing user turns, and compacting history."""
    session_id = _resolve_session_id(callback_context)
    agent_name = _resolve_agent_name(callback_context)
    set_observability_context(session_id=session_id, agent_name=agent_name)
    state = callback_context.state

    # Hydrate persisted session state on first turn of a reloaded session
    if not state.get("_session_hydrated"):
        persisted = session_store.load_session(session_id)
        if persisted and isinstance(persisted.get("state"), dict):
            for k, v in persisted["state"].items():
                if k not in state:
                    state[k] = v
            if persisted.get("compacted_summary") and not state.get(
                "compacted_history_summary"
            ):
                state["compacted_history_summary"] = persisted["compacted_summary"]
        state["_session_hydrated"] = True

    user_text = _extract_latest_user_text(llm_request)

    with trace_user_turn(
        session_id=session_id,
        user_message=user_text,
        agent_name=agent_name,
    ) as turn_span:
        agent_logger.info(
            f"Processing user turn in '{agent_name}'.",
            session_id=session_id,
            event_type="USER_TURN",
            context={"agent_name": agent_name, "user_message": scrub_pii(user_text)},
        )

        is_blocked, violation_type, reason = detect_adversarial_or_injection_input(
            user_text
        )
        if is_blocked:
            turn_span.set_attribute("guardrail.triggered", True)
            turn_span.set_attribute("guardrail.violation_type", violation_type)
            state["guardrail_triggered"] = True
            state["guardrail_violation_type"] = violation_type
            state["guardrail_reason"] = reason
            agent_logger.warning(
                reason,
                session_id=session_id,
                event_type="GUARDRAIL_EVENT",
                context={
                    "agent_name": agent_name,
                    "violation_type": violation_type,
                    "blocked_input": scrub_pii(user_text),
                },
            )
            session_store.schedule_background_save_and_compact(
                session_id=session_id,
                state=dict(state.to_dict())
                if hasattr(state, "to_dict")
                else dict(state),
            )
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=reason)],
                )
            )

        turn_span.set_attribute("guardrail.triggered", False)
        state["guardrail_triggered"] = False

        # Compact older conversation turns in llm_request.contents when thresholds are exceeded
        if llm_request.contents:
            existing_summary = state.get("compacted_history_summary")
            compacted_contents, updated_summary, was_compacted = (
                history_compactor.compact_adk_contents(
                    list(llm_request.contents),
                    existing_summary=str(existing_summary)
                    if existing_summary
                    else None,
                )
            )
            if was_compacted and updated_summary:
                llm_request.contents = compacted_contents
                state["compacted_history_summary"] = updated_summary
                state["history_was_compacted"] = True

        # Schedule non-blocking background persistence with PII-scrubbed turns
        state_dict = dict(state.to_dict()) if hasattr(state, "to_dict") else dict(state)
        turns_snapshot: list[dict[str, Any]] = []
        if llm_request.contents:
            for content in llm_request.contents:
                if content.parts:
                    txt = " ".join(p.text for p in content.parts if p.text is not None)
                    if txt:
                        turns_snapshot.append(
                            {
                                "role": str(content.role or "user"),
                                "content": str(scrub_pii(txt)),
                            }
                        )
        session_store.schedule_background_save_and_compact(
            session_id=session_id,
            state=state_dict,
            turns=turns_snapshot if turns_snapshot else None,
        )
        return None


def graceful_model_error_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
    error: Exception,
) -> LlmResponse | None:
    """ADK `on_model_error_callback` providing a graceful fallback in unauthenticated test environments."""
    session_id = _resolve_session_id(callback_context)
    agent_name = _resolve_agent_name(callback_context)
    user_text = _extract_latest_user_text(llm_request)
    agent_logger.warning(
        f"LLM invocation fallback triggered in '{agent_name}': {type(error).__name__}",
        session_id=session_id,
        event_type="MODEL_FALLBACK_EVENT",
        context={
            "agent_name": agent_name,
            "error_type": type(error).__name__,
            "user_message": scrub_pii(user_text),
        },
    )
    fallback_text = (
        "Hello! Welcome to the Omnichannel Retail Return & Exchange Concierge. "
        "Please share your order ID (for example, ORD-1001) or loyalty email "
        "address so I can look up your order and assist with your return or exchange."
    )
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text=fallback_text)],
        ),
        turn_complete=True,
    )


def async_memory_persistence_after_agent_callback(
    callback_context: CallbackContext,
) -> types.Content | None:
    """ADK `after_agent_callback` that schedules non-blocking session persistence and compaction."""
    session_id = _resolve_session_id(callback_context)
    state = callback_context.state
    state_dict = dict(state.to_dict()) if hasattr(state, "to_dict") else dict(state)
    session_store.schedule_background_save_and_compact(
        session_id=session_id,
        state=state_dict,
    )
    return None


# ============================================================================
# Human-in-the-Loop (HITL) Checkpoint Logic
# ============================================================================


def evaluate_hitl_requirement(
    tool_name: str,
    args: dict[str, Any],
) -> tuple[bool, list[str], dict[str, Any]]:
    """Determines whether a tool invocation requires Human-in-the-Loop supervisor approval.

    High-stakes actions requiring human supervisor confirmation:
    1. High-value refund: Any refund or return action exceeding $75.00 (`> 75.00`).
    2. Policy exception: Any return attempted past the 30-day window (`> 30` days).

    Args:
        tool_name: Name of the tool being invoked.
        args: Dictionary of arguments passed to the tool.

    Returns:
        A 3-tuple `(requires_hitl, trigger_reasons, metadata)` describing whether
        the HITL checkpoint must pause execution.
    """
    high_stakes_tools = {
        "process_refund_and_generate_label",
        "request_manager_override_approval",
    }
    if tool_name not in high_stakes_tools:
        return False, [], {}

    payload: dict[str, Any] = dict(args)
    nested_req = payload.get("request")
    if nested_req is not None:
        if hasattr(nested_req, "model_dump"):
            payload.update(nested_req.model_dump())
        elif isinstance(nested_req, dict):
            payload.update(nested_req)

    order_id = str(payload.get("order_id") or "").strip().upper()
    item_sku = str(payload.get("item_sku") or "").strip().upper()
    resolution_type = str(
        payload.get("resolution_type")
        or payload.get("requested_resolution")
        or "refund"
    ).lower()

    order = MOCK_ORDERS_DB.get(order_id)
    unit_price: float = float(payload.get("refund_amount") or 0.0)
    days_elapsed: int = int(payload.get("days_since_purchase") or 0)
    return_window_days: int = HITL_RETURN_WINDOW_DAYS

    if order is not None:
        purchase_date = datetime.date.fromisoformat(order.purchase_date)
        days_elapsed = max(0, (datetime.date.today() - purchase_date).days)
        for item in order.items:
            if item.sku.upper() == item_sku or item.item_id.upper() == item_sku:
                unit_price = item.unit_price
                return_window_days = item.return_window_days
                break

    trigger_reasons: list[str] = []
    if unit_price > HITL_HIGH_VALUE_REFUND_THRESHOLD_USD:
        trigger_reasons.append(
            f"High-value refund (${unit_price:.2f} exceeds ${HITL_HIGH_VALUE_REFUND_THRESHOLD_USD:.2f} supervisor threshold)"
        )

    if days_elapsed > return_window_days:
        trigger_reasons.append(
            f"Policy exception (return attempted {days_elapsed} days after purchase, past the {return_window_days}-day window)"
        )

    metadata = {
        "tool_name": tool_name,
        "order_id": order_id,
        "item_sku": item_sku,
        "resolution_type": resolution_type,
        "refund_amount": unit_price,
        "days_since_purchase": days_elapsed,
        "return_window_days": return_window_days,
    }
    return bool(trigger_reasons), trigger_reasons, metadata


def _is_hitl_approved(tool_context: ToolContext, order_id: str, item_sku: str) -> bool:
    """Checks whether human supervisor confirmation has been granted for this action."""
    confirmation = getattr(tool_context, "tool_confirmation", None)
    if confirmation is not None and getattr(confirmation, "confirmed", False) is True:
        return True

    state = getattr(tool_context, "state", {}) or {}
    if state.get("hitl_approved") is True or state.get("supervisor_approved") is True:
        return True
    if state.get("hitl_status") == "approved":
        return True
    if state.get(f"hitl_approved:{order_id}:{item_sku}") is True:
        return True
    return False


def _record_intercepted_tool_observability(
    tool_name: str,
    args: dict[str, Any],
    outcome_payload: dict[str, Any],
    status: str,
    start_ts: float,
    session_id: str,
    agent_name: str,
) -> None:
    """Logs Intent/Outcome and creates an OpenTelemetry span when a tool call is intercepted."""
    purpose = f"Intercepted tool call '{tool_name}' evaluated by guardrail/HITL hook"
    log_tool_intent(
        tool_name=tool_name,
        parameters=args,
        predicted_purpose=purpose,
        session_id=session_id,
        agent_name=agent_name,
    )
    latency_ms = (time.perf_counter() - start_ts) * 1000.0
    with trace_tool_execution(
        tool_name=tool_name,
        parameters=args,
        session_id=session_id,
        agent_name=agent_name,
        predicted_purpose=purpose,
    ) as span:
        span.set_attribute("tool.status", status)
        span.set_attribute("tool.latency_ms", round(latency_ms, 3))
    log_tool_outcome(
        tool_name=tool_name,
        return_value=outcome_payload,
        latency_ms=latency_ms,
        status=status,
        session_id=session_id,
        agent_name=agent_name,
    )


def hitl_and_guardrail_before_tool_callback(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
) -> dict[str, Any] | None:
    """ADK `before_tool_callback` enforcing tool guardrails, HITL checkpoint, and observability."""
    start_ts = time.perf_counter()
    tool_name = getattr(tool, "name", "")
    state = getattr(tool_context, "state", None)
    session_id = _resolve_session_id(tool_context)
    agent_name = _resolve_agent_name(tool_context)
    set_observability_context(session_id=session_id, agent_name=agent_name)

    # 1. Check string arguments for prompt injection payloads
    serialized_args = " ".join(str(v) for v in args.values())
    is_blocked, violation_type, reason = detect_adversarial_or_injection_input(
        serialized_args
    )
    if is_blocked:
        if state is not None:
            state["guardrail_triggered"] = True
            state["guardrail_violation_type"] = violation_type
            state["guardrail_reason"] = reason
        blocked_resp = {
            "status": "error",
            "error_code": "GUARDRAIL_VIOLATION",
            "violation_type": violation_type,
            "message": reason,
            "recovery_instructions": (
                "Refuse the adversarial or unpurchased-item request and explain "
                "that only verified items on a valid order can be processed."
            ),
        }
        _record_intercepted_tool_observability(
            tool_name,
            args,
            blocked_resp,
            "GUARDRAIL_VIOLATION",
            start_ts,
            session_id,
            agent_name,
        )
        return blocked_resp

    # 2. Check for unpurchased SKU manipulation on known orders
    order_id = str(args.get("order_id") or "").strip().upper()
    item_sku = str(args.get("item_sku") or "").strip().upper()
    if order_id in MOCK_ORDERS_DB and item_sku:
        order = MOCK_ORDERS_DB[order_id]
        valid_skus = {item.sku.upper() for item in order.items} | {
            item.item_id.upper() for item in order.items
        }
        if item_sku not in valid_skus:
            if state is not None:
                state["guardrail_triggered"] = True
                state["guardrail_violation_type"] = "unpurchased_item"
            if tool_name == "process_refund_and_generate_label":
                available_str = ", ".join(i.sku for i in order.items)
                unp_resp = {
                    "status": "error",
                    "processed": False,
                    "order_id": order.order_id,
                    "item_sku": item_sku,
                    "error_code": "GUARDRAIL_UNPURCHASED_ITEM",
                    "message": (
                        f"Guardrail blocked refund for SKU '{item_sku}': item was not "
                        f"purchased in order '{order.order_id}' (purchased SKUs: {available_str})."
                    ),
                    "recovery_instructions": (
                        f"Inform the customer that '{item_sku}' is not part of order "
                        f"'{order.order_id}'. Only purchased items ({available_str}) are eligible."
                    ),
                }
                _record_intercepted_tool_observability(
                    tool_name,
                    args,
                    unp_resp,
                    "GUARDRAIL_UNPURCHASED_ITEM",
                    start_ts,
                    session_id,
                    agent_name,
                )
                return unp_resp

    # 3. Evaluate Human-in-the-Loop (HITL) checkpoint requirements
    requires_hitl, trigger_reasons, metadata = evaluate_hitl_requirement(
        tool_name, args
    )
    if not requires_hitl:
        if state is not None:
            state["hitl_pending"] = False
        return None

    # Check if supervisor explicitly rejected the confirmation
    confirmation = getattr(tool_context, "tool_confirmation", None)
    if (
        confirmation is not None and getattr(confirmation, "confirmed", None) is False
    ) or (state is not None and state.get("hitl_status") == "rejected"):
        if state is not None:
            state["hitl_pending"] = False
            state["hitl_status"] = "rejected"
        rejected_resp = {
            "status": "HITL_REJECTED",
            "hitl_pending": False,
            "approved": False,
            "order_id": metadata["order_id"],
            "item_sku": metadata["item_sku"],
            "message": (
                f"Human supervisor rejected the high-stakes action for order "
                f"'{metadata['order_id']}' (item '{metadata['item_sku']}')."
            ),
            "recovery_instructions": (
                "Politely inform the customer that the supervisor could not approve "
                "the exception or high-value refund request."
            ),
        }
        _record_intercepted_tool_observability(
            tool_name,
            args,
            rejected_resp,
            "HITL_REJECTED",
            start_ts,
            session_id,
            agent_name,
        )
        return rejected_resp

    # Check if supervisor has already approved this action
    if _is_hitl_approved(tool_context, metadata["order_id"], metadata["item_sku"]):
        if state is not None:
            state["hitl_pending"] = False
            state["hitl_status"] = "approved"
            state_dict = (
                dict(state.to_dict()) if hasattr(state, "to_dict") else dict(state)
            )
            session_store.schedule_background_save_and_compact(
                session_id=session_id,
                state=state_dict,
            )
        return None

    # Pause execution and emit HITL pending state & confirmation event
    hitl_payload = {
        "checkpoint": "SUPERVISOR_HITL_APPROVAL",
        "status": "HITL_PENDING",
        "tool_name": tool_name,
        "order_id": metadata["order_id"],
        "item_sku": metadata["item_sku"],
        "refund_amount": metadata["refund_amount"],
        "days_since_purchase": metadata["days_since_purchase"],
        "trigger_reasons": trigger_reasons,
    }
    if state is not None:
        state["hitl_pending"] = True
        state["hitl_status"] = "pending_approval"
        state["hitl_request"] = hitl_payload
        state_dict = dict(state.to_dict()) if hasattr(state, "to_dict") else dict(state)
        session_store.schedule_background_save_and_compact(
            session_id=session_id,
            state=state_dict,
        )

    hint_message = (
        f"HUMAN SUPERVISOR APPROVAL REQUIRED before executing '{tool_name}' for "
        f"order '{metadata['order_id']}' (item '{metadata['item_sku']}', "
        f"amount: ${metadata['refund_amount']:.2f}, days since purchase: "
        f"{metadata['days_since_purchase']}). Triggers: {'; '.join(trigger_reasons)}."
    )

    if getattr(tool_context, "function_call_id", None):
        tool_context.request_confirmation(
            hint=hint_message,
            payload=hitl_payload,
        )

    pending_resp = {
        "status": "HITL_PENDING",
        "hitl_pending": True,
        "requires_human_approval": True,
        "approved": False,
        "order_id": metadata["order_id"],
        "item_sku": metadata["item_sku"],
        "refund_amount": metadata["refund_amount"],
        "days_since_purchase": metadata["days_since_purchase"],
        "trigger_reasons": trigger_reasons,
        "message": (
            f"HITL CHECKPOINT PAUSED: {hint_message} Execution halted pending "
            "human supervisor confirmation."
        ),
        "recovery_instructions": (
            "Stop and inform the customer that their request has been paused for "
            "human supervisor approval because it involves a refund over $75.00 "
            "or a return past the 30-day window. Do not retry the tool until "
            "supervisor approval is provided."
        ),
    }
    _record_intercepted_tool_observability(
        tool_name,
        args,
        pending_resp,
        "HITL_PENDING",
        start_ts,
        session_id,
        agent_name,
    )
    return pending_resp


@log_tool_intent_and_outcome(
    predicted_purpose="Calculate applicable restocking fee and net refund amount based on item condition and return window"
)
def calculate_restocking_fee(
    unit_price: float,
    item_condition: str = "unopened",
    days_since_purchase: int = 0,
    return_window_days: int = 30,
) -> dict[str, Any]:
    """Calculates applicable restocking fees and net refund amount during policy evaluation.

    Policy rules:
    - 'unopened', 'like_new', or 'defective' items within the 30-day window: $0.00 fee (0%).
    - 'opened' non-defective items or late returns (>30 days): 15% restocking fee.

    Args:
        unit_price: Original purchase price of the item in USD.
        item_condition: Physical condition ('unopened', 'like_new', 'opened',
            'worn', 'damaged', 'defective').
        days_since_purchase: Number of days elapsed since the order purchase date.
        return_window_days: Standard return window in days (default 30).

    Returns:
        Dictionary containing `unit_price`, `restocking_fee_rate`,
        `restocking_fee_usd`, `net_refund_usd`, and `explanation`.
    """
    condition_norm = item_condition.strip().lower()
    if condition_norm == "defective":
        rate = 0.0
        reason = "Defective items are exempt from restocking fees."
    elif condition_norm == "opened" or days_since_purchase > return_window_days:
        rate = RESTOCKING_FEE_RATE_OPENED
        reason = (
            "15% restocking fee applies to opened merchandise or late returns "
            f"past the {return_window_days}-day window."
        )
    else:
        rate = 0.0
        reason = (
            "No restocking fee for unopened or like-new items within the 30-day window."
        )

    fee_usd = round(unit_price * rate, 2)
    net_usd = round(max(0.0, unit_price - fee_usd), 2)
    return {
        "unit_price": unit_price,
        "restocking_fee_rate": rate,
        "restocking_fee_usd": fee_usd,
        "net_refund_usd": net_usd,
        "explanation": reason,
    }


def _needs_confirmation_predicate(**kwargs: Any) -> bool:
    """Predicate for ADK `FunctionTool(require_confirmation=...)`."""
    requires_hitl, _, _ = evaluate_hitl_requirement(
        "process_refund_and_generate_label", kwargs
    )
    return requires_hitl


# ============================================================================
# Specialist & Coordinator Agent Definitions
# ============================================================================

ReturnPolicyEvaluator = Agent(
    name="ReturnPolicyEvaluator",
    model=DEEP_POLICY_MODEL,
    description=(
        "Specialist agent for deep return policy evaluation, item condition and "
        "restocking fee calculation, fraud risk assessment, and manager override "
        "eligibility checks."
    ),
    instruction=RETURN_POLICY_EVALUATOR_PROMPT,
    tools=[
        FunctionTool(verify_item_return_eligibility),
        FunctionTool(calculate_restocking_fee),
        FunctionTool(request_manager_override_approval),
    ],
    before_agent_callback=observability_before_agent_callback,
    before_model_callback=input_validation_guardrail_callback,
    on_model_error_callback=graceful_model_error_callback,
    before_tool_callback=hitl_and_guardrail_before_tool_callback,
    after_agent_callback=async_memory_persistence_after_agent_callback,
)

FulfillmentAgent = Agent(
    name="FulfillmentAgent",
    model=FAST_TRIAGE_MODEL,
    description=(
        "Specialist agent that executes approved refunds, exchanges, or store credits "
        "and generates prepaid return shipping labels or in-store QR drop-off passes."
    ),
    instruction=FULFILLMENT_AGENT_PROMPT,
    tools=[
        FunctionTool(
            process_refund_and_generate_label,
            require_confirmation=_needs_confirmation_predicate,
        ),
    ],
    before_agent_callback=observability_before_agent_callback,
    before_model_callback=input_validation_guardrail_callback,
    on_model_error_callback=graceful_model_error_callback,
    before_tool_callback=hitl_and_guardrail_before_tool_callback,
    after_agent_callback=async_memory_persistence_after_agent_callback,
)

TriageCoordinator = Agent(
    name="TriageCoordinator",
    model=FAST_TRIAGE_MODEL,
    description=(
        "Omnichannel Retail Return & Exchange Concierge coordinator that greets the "
        "customer, extracts order intent and item SKUs, looks up order records, and "
        "routes to ReturnPolicyEvaluator and FulfillmentAgent."
    ),
    instruction=TRIAGE_COORDINATOR_PROMPT,
    tools=[FunctionTool(lookup_order_details)],
    sub_agents=[ReturnPolicyEvaluator, FulfillmentAgent],
    before_agent_callback=observability_before_agent_callback,
    before_model_callback=input_validation_guardrail_callback,
    on_model_error_callback=graceful_model_error_callback,
    before_tool_callback=hitl_and_guardrail_before_tool_callback,
    after_agent_callback=async_memory_persistence_after_agent_callback,
)

# Snake-case aliases and root_agent export for ADK runtime & tests
triage_coordinator = TriageCoordinator
return_policy_evaluator = ReturnPolicyEvaluator
fulfillment_agent = FulfillmentAgent
root_agent = TriageCoordinator

app = App(
    root_agent=root_agent,
    name="app",
    resumability_config=ResumabilityConfig(is_resumable=True),
    events_compaction_config=EventsCompactionConfig(
        token_threshold=32000,
        event_retention_size=4,
    ),
)

__all__ = [
    "CONSTITUTION_PROMPT",
    "DEEP_POLICY_MODEL",
    "FAST_TRIAGE_MODEL",
    "FULFILLMENT_AGENT_PROMPT",
    "HITL_HIGH_VALUE_REFUND_THRESHOLD_USD",
    "HITL_RETURN_WINDOW_DAYS",
    "RETURN_POLICY_EVALUATOR_PROMPT",
    "SYSTEM_CONSTITUTION",
    "TRIAGE_COORDINATOR_PROMPT",
    "AgentSettings",
    "FulfillmentAgent",
    "HistoryCompactor",
    "PersistentSessionStore",
    "ReturnPolicyEvaluator",
    "TriageCoordinator",
    "agent_logger",
    "app",
    "async_compact_conversation_history",
    "async_memory_persistence_after_agent_callback",
    "build_agent_instruction",
    "calculate_restocking_fee",
    "compact_conversation_history",
    "detect_adversarial_or_injection_input",
    "estimate_token_count",
    "evaluate_hitl_requirement",
    "fulfillment_agent",
    "get_secret",
    "get_settings",
    "history_compactor",
    "hitl_and_guardrail_before_tool_callback",
    "input_validation_guardrail_callback",
    "observability_before_agent_callback",
    "return_policy_evaluator",
    "root_agent",
    "session_store",
    "settings",
    "triage_coordinator",
]
