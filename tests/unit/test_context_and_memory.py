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

"""Unit tests for Category 2: Context & Memory (`prompts.py`, `compactor.py`, `session_store.py`)."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from google.adk.models.llm_request import LlmRequest
from google.genai import types

from app.agent import (
    FulfillmentAgent,
    ReturnPolicyEvaluator,
    TriageCoordinator,
    input_validation_guardrail_callback,
)
from app.memory.compactor import (
    HistoryCompactor,
    async_compact_conversation_history,
)
from app.memory.session_store import PersistentSessionStore
from app.prompts import (
    FULFILLMENT_AGENT_PROMPT,
    RETURN_POLICY_EVALUATOR_PROMPT,
    SYSTEM_CONSTITUTION,
    TRIAGE_COORDINATOR_PROMPT,
)


def test_system_constitution_and_agent_prompts() -> None:
    """Verifies the System Constitution covers persona, policies, and constraints."""
    assert "empathetic" in SYSTEM_CONSTITUTION.lower()
    assert (
        "30 calendar days" in SYSTEM_CONSTITUTION
        or "30-day" in SYSTEM_CONSTITUTION.lower()
    )
    assert "original retail tags" in SYSTEM_CONSTITUTION.lower()
    assert "final sale" in SYSTEM_CONSTITUTION.lower()
    assert "$75.00" in SYSTEM_CONSTITUTION
    assert "never disclose internal fraud thresholds" in SYSTEM_CONSTITUTION.lower()

    assert TriageCoordinator.instruction == TRIAGE_COORDINATOR_PROMPT
    assert ReturnPolicyEvaluator.instruction == RETURN_POLICY_EVALUATOR_PROMPT
    assert FulfillmentAgent.instruction == FULFILLMENT_AGENT_PROMPT


def test_history_compactor_turn_and_token_thresholds() -> None:
    """Verifies HistoryCompactor summarizes older turns while preserving entities."""
    compactor = HistoryCompactor(max_turns=5, max_tokens=500, retain_recent_turns=2)

    turns = [
        {
            "role": "user",
            "content": "Hi, my email is alice.rivera@example.com and my order is ORD-1001.",
        },
        {"role": "model", "content": "I found order ORD-1001 with SKU-TSHIRT-BLK-M."},
        {
            "role": "user",
            "content": "I want to return SKU-TSHIRT-BLK-M because it is the wrong size.",
        },
        {
            "role": "model",
            "content": "Checking eligibility for SKU-TSHIRT-BLK-M under the 30-day window.",
        },
        {"role": "user", "content": "It is unopened with original tags attached."},
        {
            "role": "model",
            "content": "Great, SKU-TSHIRT-BLK-M is eligible for a full refund of $35.00.",
        },
        {"role": "user", "content": "Please generate a UPS return label."},
    ]
    assert compactor.should_compact(turns) is True

    result = compactor.compact(turns)
    assert result["compacted"] is True
    assert result["final_turn_count"] == 3  # 1 summary turn + 2 retained recent turns
    assert "ORD-1001" in result["summary"]
    assert "SKU-TSHIRT-BLK-M" in result["summary"]
    assert "alice.rivera@example.com" in result["summary"]


@pytest.mark.asyncio
async def test_persistent_session_store_and_async_operations(tmp_path: Path) -> None:
    """Verifies SQLite + JSON disk persistence across reloads and async background tasks."""
    db_file = tmp_path / "test_sessions.db"
    json_file = tmp_path / "test_sessions.json"
    compactor = HistoryCompactor(max_turns=4, max_tokens=400, retain_recent_turns=2)

    store = PersistentSessionStore(
        storage_path=db_file,
        json_mirror_path=json_file,
        compactor=compactor,
    )

    turns = [
        {"role": "user", "content": "Order ORD-1004 for david.kim@example.com"},
        {"role": "model", "content": "Found SKU-CASHMERE-COAT-M ($349.00)"},
        {"role": "user", "content": "Requesting return for SKU-CASHMERE-COAT-M"},
        {"role": "model", "content": "Manager override MGR-OVR-ORD1004-OATM issued"},
        {"role": "user", "content": "Proceed with refund"},
    ]

    task = store.schedule_background_save_and_compact(
        session_id="sess-1004",
        state={"active_order_id": "ORD-1004", "hitl_pending": True},
        turns=turns,
        user_id="user-david",
    )
    assert task is not None
    await store.flush_background_tasks()

    # Reload from a brand-new PersistentSessionStore instance to verify disk persistence
    reloaded_store = PersistentSessionStore(
        storage_path=db_file,
        json_mirror_path=json_file,
        compactor=compactor,
    )
    loaded = await reloaded_store.load_session_async("sess-1004")
    assert loaded is not None
    assert loaded["state"]["active_order_id"] == "ORD-1004"
    assert loaded["state"]["hitl_pending"] is True
    assert loaded["state"]["history_was_compacted"] is True
    assert "ORD-1004" in loaded["compacted_summary"]
    assert "SKU-CASHMERE-COAT-M" in loaded["compacted_summary"]
    assert json_file.exists()

    # Also test async_compact_conversation_history helper directly
    async_res = await async_compact_conversation_history(
        turns, max_turns=4, retain_recent_turns=2
    )
    assert async_res["compacted"] is True


def test_agent_callback_compacts_adk_contents() -> None:
    """Verifies input_validation_guardrail_callback compacts long ADK LlmRequest history."""
    cb_ctx: Any = SimpleNamespace(state={"session_id": "test-callback-compaction"})
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=f"Turn {i}: checking order ORD-1001 SKU-TSHIRT-BLK-M"
                )
            ],
        )
        for i in range(10)
    ]
    llm_req = LlmRequest(contents=contents)
    resp = input_validation_guardrail_callback(cb_ctx, llm_req)
    assert resp is None
    assert cb_ctx.state.get("history_was_compacted") is True
    assert "ORD-1001" in cb_ctx.state.get("compacted_history_summary", "")
    assert len(llm_req.contents) < 10
