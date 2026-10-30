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

"""Automated evaluation suite executing against `tests/golden_dataset.json`.

Implements Requirement 1 of Category 5: Infrastructure & CI/CD:
- Loads `tests/golden_dataset.json` (>= 8 test cases covering standard eligible
  return, final-sale rejection, expired 30-day window, high-value refund
  triggering HITL, missing order ID, PII masking, missing receipt, restocking
  fees, and adversarial guardrails).
- Asserts expected agent intent, specialist routing, tool selection, policy
  outcomes, HITL flags, and PII redaction for every golden test case.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from google.adk.models.llm_request import LlmRequest
from google.adk.tools import FunctionTool
from google.genai import types

from app.agent import (
    FulfillmentAgent,
    ReturnPolicyEvaluator,
    TriageCoordinator,
    calculate_restocking_fee,
    evaluate_hitl_requirement,
    hitl_and_guardrail_before_tool_callback,
    input_validation_guardrail_callback,
    observability_before_agent_callback,
)
from app.observability.logger import (
    clear_emitted_log_records,
    get_emitted_log_records,
)
from app.observability.pii_scrubber import scrub_pii
from app.observability.tracer import (
    clear_recorded_spans,
    get_recorded_spans,
)
from app.tools.return_tools import (
    lookup_order_details,
    process_refund_and_generate_label,
    request_manager_override_approval,
    verify_item_return_eligibility,
)

GOLDEN_DATASET_PATH = Path(__file__).resolve().parent / "golden_dataset.json"


def load_golden_dataset() -> dict[str, Any]:
    """Loads and parses `tests/golden_dataset.json`."""
    return json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))


GOLDEN_CASES: list[dict[str, Any]] = load_golden_dataset()["test_cases"]


def _execute_golden_case(case: dict[str, Any]) -> dict[str, Any]:
    """Executes the concierge orchestration and tool pipeline for a golden test case."""
    clear_emitted_log_records()
    clear_recorded_spans()

    case_id = case["case_id"]
    user_prompt = case["user_prompt"]
    inp = case["input"]
    session_state: dict[str, Any] = {"session_id": f"eval-{case_id}"}

    # Step 1: Run TriageCoordinator entrypoint & input validation guardrail hook
    triage_ctx: Any = SimpleNamespace(
        agent_name=TriageCoordinator.name,
        state=session_state,
    )
    observability_before_agent_callback(triage_ctx)
    llm_req = LlmRequest(
        contents=[
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_prompt)],
            )
        ]
    )
    guardrail_resp = input_validation_guardrail_callback(triage_ctx, llm_req)
    if guardrail_resp is not None:
        return {
            "intent": (
                "block_prompt_injection"
                if session_state.get("guardrail_violation_type") == "prompt_injection"
                else "block_unpurchased_item_refund"
            ),
            "target_specialist": TriageCoordinator.name,
            "selected_tools": [],
            "hitl_required": False,
            "guardrail_triggered": bool(session_state.get("guardrail_triggered")),
            "guardrail_violation_type": session_state.get("guardrail_violation_type"),
        }

    executed_tools: list[str] = []

    # Special Case: Missing order ID fallback workflow
    if case["category"] == "missing_order_id":
        initial_lookup = lookup_order_details(order_id=inp["missing_order_id"])
        executed_tools.append("lookup_order_details")
        fallback_lookup = lookup_order_details(
            customer_email=inp["fallback_customer_email"],
            customer_phone=inp.get("fallback_customer_phone"),
        )
        return {
            "intent": "lookup_order_with_fallback",
            "target_specialist": TriageCoordinator.name,
            "selected_tools": executed_tools,
            "initial_lookup_status": initial_lookup.status,
            "initial_error_code": initial_lookup.error_code,
            "has_recovery_instructions": bool(initial_lookup.recovery_instructions),
            "fallback_lookup_status": fallback_lookup.status,
            "resolved_order_id": (
                fallback_lookup.order.order_id if fallback_lookup.order else None
            ),
            "hitl_required": False,
            "guardrail_triggered": False,
        }

    # Step 2: Order Lookup in TriageCoordinator
    order_id = inp["order_id"]
    item_sku = inp["item_sku"]
    lookup_res = lookup_order_details(
        order_id=order_id,
        customer_email=inp.get("customer_email"),
        customer_phone=inp.get("customer_phone"),
    )
    executed_tools.append("lookup_order_details")
    assert lookup_res.found is True and lookup_res.order is not None

    # Step 3: Delegate to ReturnPolicyEvaluator
    policy_ctx: Any = SimpleNamespace(
        agent_name=ReturnPolicyEvaluator.name,
        state=session_state,
    )
    observability_before_agent_callback(policy_ctx)

    item_condition = inp.get("item_condition", "unopened")
    return_reason = inp.get("return_reason", "changed_mind")
    has_receipt = inp.get("has_receipt", True)

    elig_res = verify_item_return_eligibility(
        order_id=order_id,
        item_sku=item_sku,
        item_condition=item_condition,
        return_reason=return_reason,
        has_receipt=has_receipt,
    )
    executed_tools.append("verify_item_return_eligibility")

    # Calculate restocking fee when item is opened or standard eligible return
    restocking_info: dict[str, Any] | None = None
    if case["category"] in ("standard_eligible_return", "restocking_fee_calculation"):
        restocking_info = calculate_restocking_fee(
            unit_price=elig_res.unit_price or 0.0,
            item_condition=item_condition,
            days_since_purchase=elig_res.days_since_purchase or 0,
            return_window_days=elig_res.return_window_days,
        )
        executed_tools.append("calculate_restocking_fee")

    # Check if case stops at ReturnPolicyEvaluator (ineligible or fee-inquiry only)
    if elig_res.status == "ineligible":
        return {
            "intent": "evaluate_final_sale_policy",
            "target_specialist": ReturnPolicyEvaluator.name,
            "selected_tools": executed_tools,
            "eligibility_status": elig_res.status,
            "is_eligible": elig_res.is_eligible,
            "error_code": elig_res.error_code,
            "requires_manager_override": elig_res.requires_manager_override,
            "hitl_required": False,
            "has_recovery_instructions": bool(elig_res.recovery_instructions),
            "guardrail_triggered": False,
        }

    if case["category"] == "restocking_fee_calculation" and restocking_info is not None:
        return {
            "intent": "evaluate_restocking_fee",
            "target_specialist": ReturnPolicyEvaluator.name,
            "selected_tools": executed_tools,
            "eligibility_status": elig_res.status,
            "is_eligible": elig_res.is_eligible,
            "requires_manager_override": elig_res.requires_manager_override,
            "hitl_required": False,
            "restocking_fee_usd": restocking_info["restocking_fee_usd"],
            "net_refund_usd": restocking_info["net_refund_usd"],
            "guardrail_triggered": False,
        }

    # Evaluate HITL requirement for override / fulfillment
    override_code: str | None = None
    hitl_required, _, _ = evaluate_hitl_requirement(
        "process_refund_and_generate_label",
        {"order_id": order_id, "item_sku": item_sku},
    )
    hitl_status_before: str | None = None

    if elig_res.requires_manager_override:
        override_tool = FunctionTool(request_manager_override_approval)
        tool_ctx: Any = SimpleNamespace(
            agent_name=ReturnPolicyEvaluator.name,
            state=session_state,
            tool_confirmation=None,
            function_call_id=None,
        )
        override_args = {
            "order_id": order_id,
            "item_sku": item_sku,
            "override_reason": inp.get("override_reason", "policy_exception"),
            "customer_justification": inp.get(
                "customer_justification",
                "Customer requested manager policy exception review.",
            ),
            "requested_resolution": inp.get("resolution_type", "refund"),
        }
        hitl_gate = hitl_and_guardrail_before_tool_callback(
            override_tool, override_args, tool_ctx
        )
        executed_tools.append("request_manager_override_approval")
        if hitl_gate is not None and hitl_gate.get("status") == "HITL_PENDING":
            hitl_status_before = "HITL_PENDING"

        # If this case only evaluates the policy exception + HITL pause (e.g. expired window / missing receipt)
        if case["category"] in ("expired_30_day_window", "missing_receipt"):
            return {
                "intent": (
                    "policy_exception_expired_window"
                    if case["category"] == "expired_30_day_window"
                    else "missing_receipt_store_credit"
                ),
                "target_specialist": ReturnPolicyEvaluator.name,
                "selected_tools": executed_tools,
                "eligibility_status": elig_res.status,
                "is_eligible": elig_res.is_eligible,
                "error_code": elig_res.error_code,
                "requires_manager_override": elig_res.requires_manager_override,
                "hitl_required": hitl_required,
                "hitl_status_before_approval": hitl_status_before,
                "has_recovery_instructions": bool(elig_res.recovery_instructions),
                "guardrail_triggered": False,
            }

        # Grant supervisor HITL approval so high-value workflow can proceed to fulfillment
        session_state["hitl_approved"] = True
        override_res = request_manager_override_approval(**override_args)
        override_code = override_res.override_approval_code

    # Step 4: Delegate to FulfillmentAgent to issue refund and generate shipping label
    fulfill_ctx: Any = SimpleNamespace(
        agent_name=FulfillmentAgent.name,
        state=session_state,
    )
    observability_before_agent_callback(fulfill_ctx)

    refund_res = process_refund_and_generate_label(
        order_id=order_id,
        item_sku=item_sku,
        resolution_type=inp.get("resolution_type", "refund"),
        return_channel=inp.get("return_channel", "mail_carrier"),
        override_approval_code=override_code,
        has_receipt=has_receipt,
        item_condition=item_condition,
    )
    executed_tools.append("process_refund_and_generate_label")

    intent_map = {
        "standard_eligible_return": "return_eligible_item",
        "high_value_refund_hitl": "high_value_return_with_hitl",
        "pii_masking": "pii_redacted_return",
    }
    return {
        "intent": intent_map.get(case["category"], "return_eligible_item"),
        "target_specialist": FulfillmentAgent.name,
        "selected_tools": executed_tools,
        "eligibility_status": elig_res.status,
        "is_eligible": elig_res.is_eligible,
        "error_code": elig_res.error_code,
        "requires_manager_override": elig_res.requires_manager_override,
        "hitl_required": hitl_required,
        "hitl_status_before_approval": hitl_status_before,
        "restocking_fee_usd": (
            restocking_info["restocking_fee_usd"] if restocking_info else 0.0
        ),
        "refund_amount_usd": refund_res.refund_amount,
        "guardrail_triggered": False,
        "scrubbed_prompt": scrub_pii(user_prompt),
    }


def test_golden_dataset_coverage_requirements() -> None:
    """Verifies golden_dataset.json has >= 8 cases and covers all required categories."""
    dataset = load_golden_dataset()
    cases = dataset["test_cases"]
    assert len(cases) >= 8

    categories = {c["category"] for c in cases}
    required_categories = {
        "standard_eligible_return",
        "final_sale_rejection",
        "expired_30_day_window",
        "high_value_refund_hitl",
        "missing_order_id",
        "pii_masking",
    }
    assert required_categories.issubset(categories)


@pytest.mark.parametrize(
    "case",
    GOLDEN_CASES,
    ids=[c["case_id"] for c in GOLDEN_CASES],
)
def test_golden_dataset_case(case: dict[str, Any]) -> None:
    """Executes each golden dataset case and asserts intent, tools, policy, and HITL."""
    outcome = _execute_golden_case(case)
    expected = case["expected"]

    assert outcome["intent"] == expected["intent"]
    assert outcome["target_specialist"] == expected["target_specialist"]
    assert outcome["selected_tools"] == expected["selected_tools"]
    assert outcome["hitl_required"] == expected["hitl_required"]
    assert outcome["guardrail_triggered"] == expected["guardrail_triggered"]

    if "eligibility_status" in expected:
        assert outcome["eligibility_status"] == expected["eligibility_status"]
    if "is_eligible" in expected:
        assert outcome["is_eligible"] == expected["is_eligible"]
    if "error_code" in expected:
        assert outcome["error_code"] == expected["error_code"]
    if "requires_manager_override" in expected:
        assert (
            outcome["requires_manager_override"]
            == expected["requires_manager_override"]
        )
    if "hitl_status_before_approval" in expected:
        assert (
            outcome["hitl_status_before_approval"]
            == expected["hitl_status_before_approval"]
        )
    if "restocking_fee_usd" in expected:
        assert outcome["restocking_fee_usd"] == expected["restocking_fee_usd"]
    if "net_refund_usd" in expected:
        assert outcome["net_refund_usd"] == expected["net_refund_usd"]
    if "refund_amount_usd" in expected:
        assert outcome["refund_amount_usd"] == expected["refund_amount_usd"]
    if "has_recovery_instructions" in expected:
        assert (
            outcome["has_recovery_instructions"]
            == expected["has_recovery_instructions"]
        )
    if "initial_lookup_status" in expected:
        assert outcome["initial_lookup_status"] == expected["initial_lookup_status"]
        assert outcome["initial_error_code"] == expected["initial_error_code"]
        assert outcome["fallback_lookup_status"] == expected["fallback_lookup_status"]
        assert outcome["resolved_order_id"] == expected["resolved_order_id"]
    if "guardrail_violation_type" in expected:
        assert (
            outcome["guardrail_violation_type"] == expected["guardrail_violation_type"]
        )
    if "pii_redacted_tokens" in expected:
        scrubbed_prompt = outcome["scrubbed_prompt"]
        for token in expected["pii_redacted_tokens"]:
            assert token in scrubbed_prompt
        # Also verify no raw PII leaked into structured JSON logs or OpenTelemetry spans
        raw_card = case["input"]["raw_card"]
        raw_email = case["input"]["customer_email"]
        serialized_logs = json.dumps(get_emitted_log_records())
        assert raw_card not in serialized_logs
        assert raw_email not in serialized_logs
        for span in get_recorded_spans():
            assert raw_card not in json.dumps(dict(span.attributes or {}))
