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

"""Unit tests for Category 1: Tool & Interface Design (`schemas.py` and `return_tools.py`)."""

from google.adk.tools import FunctionTool

from app.models.schemas import (
    LookupOrderDetailsInput,
    LookupOrderDetailsOutput,
    ProcessRefundAndGenerateLabelInput,
    ProcessRefundAndGenerateLabelOutput,
    RequestManagerOverrideApprovalInput,
    RequestManagerOverrideApprovalOutput,
    VerifyItemReturnEligibilityInput,
    VerifyItemReturnEligibilityOutput,
)
from app.tools.return_tools import (
    MOCK_ORDERS_DB,
    lookup_order_details,
    process_refund_and_generate_label,
    request_manager_override_approval,
    verify_item_return_eligibility,
)


def test_mock_orders_db_scenarios() -> None:
    """Verifies the mock database includes all required return scenarios."""
    assert "ORD-1001" in MOCK_ORDERS_DB  # Standard eligible return
    assert "ORD-1002" in MOCK_ORDERS_DB  # Final-sale item
    assert "ORD-1003" in MOCK_ORDERS_DB  # Past 30-day window
    assert "ORD-1004" in MOCK_ORDERS_DB  # High-value item > $100
    assert "ORD-1005" in MOCK_ORDERS_DB  # Missing receipt


def test_lookup_order_details_success_and_guided_errors() -> None:
    """Tests order lookup by ID, fallback email/phone, and guided error recovery."""
    # Direct lookup by order_id
    res = lookup_order_details(order_id="ORD-1001")
    assert isinstance(res, LookupOrderDetailsOutput)
    assert res.status == "success"
    assert res.found is True
    assert res.order is not None
    assert res.order.order_id == "ORD-1001"

    # Lookup via Pydantic input model
    res_model = lookup_order_details(
        request=LookupOrderDetailsInput(customer_email="alice.rivera@example.com")
    )
    assert res_model.status == "success"
    assert res_model.found is True

    # Missing order guided error
    not_found = lookup_order_details(order_id="ORD-9999")
    assert not_found.status == "error"
    assert not_found.found is False
    assert not_found.error_code == "ORDER_NOT_FOUND"
    assert "Recovery:" in not_found.message
    assert not_found.recovery_instructions is not None

    # Validation error when no identifiers are provided
    invalid = lookup_order_details()
    assert invalid.status == "error"
    assert invalid.error_code == "VALIDATION_ERROR"
    assert "Recovery:" in invalid.message


def test_verify_item_return_eligibility_conditions() -> None:
    """Tests return eligibility across standard, final-sale, >30 days, >$100, and missing receipt."""
    # 1. Standard eligible item (ORD-1001)
    std = verify_item_return_eligibility(
        order_id="ORD-1001",
        item_sku="SKU-TSHIRT-BLK-M",
        item_condition="unopened",
    )
    assert isinstance(std, VerifyItemReturnEligibilityOutput)
    assert std.status == "eligible"
    assert std.is_eligible is True
    assert std.requires_manager_override is False

    # 2. Final-sale item (ORD-1002)
    final_sale = verify_item_return_eligibility(
        order_id="ORD-1002",
        item_sku="SKU-SILK-SCARF-FS",
        item_condition="unopened",
    )
    assert final_sale.status == "ineligible"
    assert final_sale.is_eligible is False
    assert final_sale.is_final_sale is True
    assert final_sale.error_code == "FINAL_SALE_ITEM_INELIGIBLE"
    assert "Recovery:" in final_sale.message

    # 3. Past 30-day window (ORD-1003, 45 days old)
    expired = verify_item_return_eligibility(
        order_id="ORD-1003",
        item_sku="SKU-DENIM-JKT-L",
    )
    assert expired.status == "requires_override"
    assert expired.requires_manager_override is True
    assert expired.error_code == "OUTSIDE_RETURN_WINDOW"
    assert "Recovery:" in expired.message

    # 4. High-value item over $100 (ORD-1004, $349.00)
    high_val = verify_item_return_eligibility(
        request=VerifyItemReturnEligibilityInput(
            order_id="ORD-1004",
            item_sku="SKU-CASHMERE-COAT-M",
        )
    )
    assert high_val.status == "requires_override"
    assert high_val.is_eligible is True
    assert high_val.requires_manager_override is True
    assert high_val.is_high_value_item is True
    assert high_val.error_code == "HIGH_VALUE_OVERRIDE_REQUIRED"
    assert "Recovery:" in high_val.message

    # 5. Missing receipt (ORD-1005)
    no_receipt = verify_item_return_eligibility(
        order_id="ORD-1005",
        item_sku="SKU-SNEAKER-RUN-9",
    )
    assert no_receipt.status == "requires_override"
    assert no_receipt.requires_manager_override is True
    assert no_receipt.error_code == "MISSING_RECEIPT"
    assert "Recovery:" in no_receipt.message


def test_manager_override_and_refund_label_workflow() -> None:
    """Tests manager override approval and refund/label generation end-to-end."""
    # Attempting to process high-value item without override fails with guided recovery
    blocked = process_refund_and_generate_label(
        order_id="ORD-1004",
        item_sku="SKU-CASHMERE-COAT-M",
        resolution_type="refund",
    )
    assert isinstance(blocked, ProcessRefundAndGenerateLabelOutput)
    assert blocked.status == "error"
    assert blocked.processed is False
    assert blocked.error_code == "HIGH_VALUE_OVERRIDE_REQUIRED"
    assert "Recovery:" in blocked.message

    # Request manager override for high-value item
    override = request_manager_override_approval(
        request=RequestManagerOverrideApprovalInput(
            order_id="ORD-1004",
            item_sku="SKU-CASHMERE-COAT-M",
            override_reason="high_value_item",
            customer_justification="Platinum customer returning unworn coat within 12 days.",
            requested_resolution="refund",
        )
    )
    assert isinstance(override, RequestManagerOverrideApprovalOutput)
    assert override.status == "approved"
    assert override.approved is True
    assert override.override_approval_code is not None

    # Process refund with override code
    processed = process_refund_and_generate_label(
        request=ProcessRefundAndGenerateLabelInput(
            order_id="ORD-1004",
            item_sku="SKU-CASHMERE-COAT-M",
            resolution_type="refund",
            return_channel="mail_carrier",
            override_approval_code=override.override_approval_code,
        )
    )
    assert processed.status == "success"
    assert processed.processed is True
    assert processed.refund_amount == 349.00
    assert processed.shipping_label is not None
    assert processed.shipping_label.tracking_number.startswith("1Z")


def test_adk_function_tool_compatibility() -> None:
    """Ensures all four tools wrap cleanly as ADK FunctionTools with valid declarations."""
    for fn in (
        lookup_order_details,
        verify_item_return_eligibility,
        request_manager_override_approval,
        process_refund_and_generate_label,
    ):
        tool = FunctionTool(fn)
        decl = tool._get_declaration()
        assert decl is not None
        assert decl.name == fn.__name__
        assert (
            decl.description
            and "Args:" in decl.description
            and "Returns:" in decl.description
        )
