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

"""Omnichannel Retail Return & Exchange Concierge tools and mock database.

This module implements the core tool suite for the Omnichannel Retail Return &
Exchange Concierge agent, adhering to Category 1: Tool & Interface Design:
1. Descriptive Naming (`lookup_order_details`, `verify_item_return_eligibility`,
   `request_manager_override_approval`, `process_refund_and_generate_label`).
2. Explicit Schemas using strict Pydantic v2 BaseModel classes for both inputs
   and outputs.
3. Comprehensive Google-style docstrings detailing purpose, parameters, and
   structured return payloads.
4. Guided Error Handling returning structured error responses with actionable
   `Recovery:` instructions for the LLM on any validation or business logic error.
"""

from __future__ import annotations

import datetime
import re
from typing import Any

from pydantic import ValidationError

from app.models.schemas import (
    ErrorDetail,
    LookupOrderDetailsInput,
    LookupOrderDetailsOutput,
    OrderItem,
    OrderRecord,
    ProcessRefundAndGenerateLabelInput,
    ProcessRefundAndGenerateLabelOutput,
    RequestManagerOverrideApprovalInput,
    RequestManagerOverrideApprovalOutput,
    ReturnShippingLabel,
    VerifyItemReturnEligibilityInput,
    VerifyItemReturnEligibilityOutput,
)
from app.observability.logger import log_tool_intent_and_outcome

HIGH_VALUE_THRESHOLD_USD: float = 100.0
DEFAULT_RETURN_WINDOW_DAYS: int = 30
MAX_OVERRIDE_WINDOW_DAYS: int = 90


def _days_ago_iso(days: int, base_date: datetime.date | None = None) -> str:
    """Returns an ISO-8601 date string (`YYYY-MM-DD`) `days` before `base_date`."""
    anchor = base_date or datetime.date.today()
    return (anchor - datetime.timedelta(days=days)).isoformat()


def _build_mock_orders_db() -> dict[str, OrderRecord]:
    """Creates the in-memory mock database of omnichannel retail orders.

    Includes orders covering varying return conditions:
    - ORD-1001: Standard eligible return (10 days old, under $100, non-final sale).
    - ORD-1002: Final-sale item (`SKU-SILK-SCARF-FS`) + standard item (`SKU-BELT-LTHR-34`).
    - ORD-1003: Order past the 30-day return window (45 days old).
    - ORD-1004: High-value items over $100 (`SKU-CASHMERE-COAT-M` at $349.00,
      `SKU-HEADPHONES-PRO` at $249.00).
    - ORD-1005: Missing receipt on file (`has_receipt_on_file=False`, in-store cash order).
    - ORD-1006: Severely expired order (120 days old, outside manager override window).
    """
    return {
        "ORD-1001": OrderRecord(
            order_id="ORD-1001",
            customer_id="CUST-101",
            customer_name="Alice Rivera",
            customer_email="alice.rivera@example.com",
            customer_phone="555-0101",
            loyalty_tier="silver",
            purchase_date=_days_ago_iso(10),
            purchase_channel="online",
            has_receipt_on_file=True,
            payment_method="Visa ending in 4242",
            order_total=59.00,
            items=[
                OrderItem(
                    item_id="ITEM-1001-1",
                    sku="SKU-TSHIRT-BLK-M",
                    name="Classic Organic Cotton Crewneck T-Shirt - Black (M)",
                    category="Apparel",
                    quantity=1,
                    unit_price=35.00,
                    is_final_sale=False,
                    return_window_days=30,
                    available_exchange_sizes=["S", "M", "L", "XL"],
                ),
                OrderItem(
                    item_id="ITEM-1001-2",
                    sku="SKU-SOCKS-WOOL-L",
                    name="Merino Wool Everyday Crew Socks (3-Pack)",
                    category="Accessories",
                    quantity=1,
                    unit_price=24.00,
                    is_final_sale=False,
                    return_window_days=30,
                    available_exchange_sizes=["M", "L"],
                ),
            ],
        ),
        "ORD-1002": OrderRecord(
            order_id="ORD-1002",
            customer_id="CUST-102",
            customer_name="Brian Chen",
            customer_email="brian.chen@example.com",
            customer_phone="555-0102",
            loyalty_tier="standard",
            purchase_date=_days_ago_iso(8),
            purchase_channel="online",
            has_receipt_on_file=True,
            payment_method="Mastercard ending in 8819",
            order_total=103.00,
            items=[
                OrderItem(
                    item_id="ITEM-1002-1",
                    sku="SKU-SILK-SCARF-FS",
                    name="Limited Edition Printed Silk Scarf [Final Sale]",
                    category="Accessories",
                    quantity=1,
                    unit_price=48.00,
                    is_final_sale=True,
                    return_window_days=30,
                    available_exchange_sizes=[],
                ),
                OrderItem(
                    item_id="ITEM-1002-2",
                    sku="SKU-BELT-LTHR-34",
                    name="Full-Grain Italian Leather Belt (Size 34)",
                    category="Accessories",
                    quantity=1,
                    unit_price=55.00,
                    is_final_sale=False,
                    return_window_days=30,
                    available_exchange_sizes=["32", "34", "36", "38"],
                ),
            ],
        ),
        "ORD-1003": OrderRecord(
            order_id="ORD-1003",
            customer_id="CUST-103",
            customer_name="Carla Mendez",
            customer_email="carla.mendez@example.com",
            customer_phone="555-0103",
            loyalty_tier="gold",
            purchase_date=_days_ago_iso(45),
            purchase_channel="in_store",
            has_receipt_on_file=True,
            payment_method="Amex ending in 1005",
            order_total=89.00,
            items=[
                OrderItem(
                    item_id="ITEM-1003-1",
                    sku="SKU-DENIM-JKT-L",
                    name="Vintage Wash Selvedge Denim Jacket (L)",
                    category="Outerwear",
                    quantity=1,
                    unit_price=89.00,
                    is_final_sale=False,
                    return_window_days=30,
                    available_exchange_sizes=["M", "L", "XL"],
                ),
            ],
        ),
        "ORD-1004": OrderRecord(
            order_id="ORD-1004",
            customer_id="CUST-104",
            customer_name="David Kim",
            customer_email="david.kim@example.com",
            customer_phone="555-0104",
            loyalty_tier="platinum",
            purchase_date=_days_ago_iso(12),
            purchase_channel="mobile_app",
            has_receipt_on_file=True,
            payment_method="Apple Pay (Visa ending in 9012)",
            order_total=598.00,
            items=[
                OrderItem(
                    item_id="ITEM-1004-1",
                    sku="SKU-CASHMERE-COAT-M",
                    name="Italian Double-Faced Cashmere Overcoat (M)",
                    category="Outerwear",
                    quantity=1,
                    unit_price=349.00,
                    is_final_sale=False,
                    return_window_days=30,
                    available_exchange_sizes=["S", "M", "L"],
                ),
                OrderItem(
                    item_id="ITEM-1004-2",
                    sku="SKU-HEADPHONES-PRO",
                    name="Studio Noise-Canceling Wireless Over-Ear Headphones",
                    category="Electronics",
                    quantity=1,
                    unit_price=249.00,
                    is_final_sale=False,
                    return_window_days=30,
                    available_exchange_sizes=["Matte Black", "Silver Frost"],
                ),
            ],
        ),
        "ORD-1005": OrderRecord(
            order_id="ORD-1005",
            customer_id="CUST-105",
            customer_name="Elena Rostova",
            customer_email="elena.rostova@example.com",
            customer_phone="555-0105",
            loyalty_tier="gold",
            purchase_date=_days_ago_iso(14),
            purchase_channel="in_store",
            has_receipt_on_file=False,
            payment_method="In-Store Cash",
            order_total=95.00,
            items=[
                OrderItem(
                    item_id="ITEM-1005-1",
                    sku="SKU-SNEAKER-RUN-9",
                    name="CloudStride Performance Running Shoes (Size 9)",
                    category="Footwear",
                    quantity=1,
                    unit_price=95.00,
                    is_final_sale=False,
                    return_window_days=30,
                    available_exchange_sizes=["8.5", "9", "9.5", "10"],
                ),
            ],
        ),
        "ORD-1006": OrderRecord(
            order_id="ORD-1006",
            customer_id="CUST-106",
            customer_name="Frank Wright",
            customer_email="frank.wright@example.com",
            customer_phone="555-0106",
            loyalty_tier="standard",
            purchase_date=_days_ago_iso(120),
            purchase_channel="online",
            has_receipt_on_file=True,
            payment_method="Discover ending in 3310",
            order_total=68.00,
            items=[
                OrderItem(
                    item_id="ITEM-1006-1",
                    sku="SKU-LINEN-SHIRT-XL",
                    name="Coastal Breeze Linen Button-Down Shirt (XL)",
                    category="Apparel",
                    quantity=1,
                    unit_price=68.00,
                    is_final_sale=False,
                    return_window_days=30,
                    available_exchange_sizes=["L", "XL"],
                ),
            ],
        ),
    }


MOCK_ORDERS_DB: dict[str, OrderRecord] = _build_mock_orders_db()

# Tracks issued manager override codes: code -> metadata dict
MANAGER_OVERRIDE_APPROVALS: dict[str, dict[str, Any]] = {}

# Tracks completed RMA transactions: rma_number -> ProcessRefundAndGenerateLabelOutput
PROCESSED_RETURNS_LOG: dict[str, ProcessRefundAndGenerateLabelOutput] = {}


def _normalize_phone(phone: str) -> str:
    """Strips non-digit characters from a phone string for comparison."""
    return re.sub(r"\D", "", phone)


def _find_item_in_order(order: OrderRecord, item_sku: str) -> OrderItem | None:
    """Finds an OrderItem in `order` by SKU or item_id (case-insensitive)."""
    target = item_sku.strip().upper()
    for item in order.items:
        if item.sku.upper() == target or item.item_id.upper() == target:
            return item
    return None


def _calculate_days_since_purchase(
    purchase_date_str: str,
    reference_date_str: str | None = None,
) -> int:
    """Computes the number of calendar days between `purchase_date_str` and `reference_date_str`."""
    purchase_date = datetime.date.fromisoformat(purchase_date_str)
    ref_date = (
        datetime.date.fromisoformat(reference_date_str)
        if reference_date_str
        else datetime.date.today()
    )
    return max(0, (ref_date - purchase_date).days)


@log_tool_intent_and_outcome(
    predicted_purpose="Look up customer order record by order ID, email, or phone number"
)
def lookup_order_details(
    order_id: str | None = None,
    customer_email: str | None = None,
    customer_phone: str | None = None,
    request: LookupOrderDetailsInput | None = None,
) -> LookupOrderDetailsOutput:
    """Looks up omnichannel retail order details by order ID, customer email, or phone number.

    Queries the in-memory retail orders database (`MOCK_ORDERS_DB`) to retrieve
    order metadata, purchase channel, receipt availability, loyalty tier, and
    line-item SKUs. Supports direct lookup by `order_id` or fallback search via
    `customer_email` and `customer_phone`.

    Args:
        order_id: The unique order identifier to look up (e.g., 'ORD-1001').
            Can be omitted if `customer_email` or `customer_phone` is provided.
        customer_email: Customer's loyalty email address (e.g.,
            'alice.rivera@example.com') used when the customer does not have
            their order ID.
        customer_phone: Customer's phone number (e.g., '555-0101') used as a
            secondary fallback identifier.
        request: Optional pre-constructed `LookupOrderDetailsInput` Pydantic
            model containing the lookup parameters.

    Returns:
        LookupOrderDetailsOutput: A strict Pydantic v2 model containing:
            - status: 'success' if matching order(s) are found, otherwise 'error'.
            - found: Boolean indicating whether any order matched the query.
            - message: Human-readable status summary or error message with
              embedded `Recovery:` instructions.
            - order: The primary `OrderRecord` matched (if found).
            - matching_orders: List of all `OrderRecord` instances matching the
              customer query.
            - error_code: Machine-readable error code (e.g., 'VALIDATION_ERROR',
              'ORDER_NOT_FOUND', 'CUSTOMER_ORDERS_NOT_FOUND') on failure.
            - recovery_instructions: Actionable guidance for the LLM to recover
              from lookup failures.
            - error_detail: Structured `ErrorDetail` object when `status` is 'error'.
    """
    try:
        if isinstance(order_id, LookupOrderDetailsInput):
            validated = order_id
        elif isinstance(request, LookupOrderDetailsInput):
            validated = request
        elif isinstance(request, dict):
            validated = LookupOrderDetailsInput.model_validate(request)
        else:
            validated = LookupOrderDetailsInput(
                order_id=order_id,
                customer_email=customer_email,
                customer_phone=customer_phone,
            )
    except ValidationError as exc:
        recovery = (
            "Ask the customer to provide a valid order ID (e.g., 'ORD-1001'), "
            "a valid loyalty email address containing '@', or their phone number, "
            "then call lookup_order_details again."
        )
        error_msg = (
            f"Input validation failed for lookup_order_details: {exc.errors()[0]['msg']} "
            f"Recovery: {recovery}"
        )
        return LookupOrderDetailsOutput(
            status="error",
            found=False,
            message=error_msg,
            order=None,
            matching_orders=[],
            error_code="VALIDATION_ERROR",
            recovery_instructions=recovery,
            error_detail=ErrorDetail(
                error_code="VALIDATION_ERROR",
                error_message=error_msg,
                recovery_instructions=recovery,
                failed_field="order_id | customer_email | customer_phone",
            ),
        )

    # 1. Primary lookup by order_id if provided
    if validated.order_id:
        normalized_id = validated.order_id.upper()
        matched_order = MOCK_ORDERS_DB.get(normalized_id)
        if matched_order is not None:
            return LookupOrderDetailsOutput(
                status="success",
                found=True,
                message=(
                    f"Successfully retrieved order '{matched_order.order_id}' for "
                    f"{matched_order.customer_name} ({len(matched_order.items)} item(s), "
                    f"purchased on {matched_order.purchase_date} via {matched_order.purchase_channel})."
                ),
                order=matched_order,
                matching_orders=[matched_order],
            )

        # If order_id was not found, check if fallback email/phone were also supplied
        if not validated.customer_email and not validated.customer_phone:
            recovery = (
                "Prompt customer to provide their phone number or loyalty email "
                "to re-query the database."
            )
            error_msg = f"Order '{validated.order_id}' not found. Recovery: {recovery}"
            return LookupOrderDetailsOutput(
                status="error",
                found=False,
                message=error_msg,
                order=None,
                matching_orders=[],
                error_code="ORDER_NOT_FOUND",
                recovery_instructions=recovery,
                error_detail=ErrorDetail(
                    error_code="ORDER_NOT_FOUND",
                    error_message=error_msg,
                    recovery_instructions=recovery,
                    failed_field="order_id",
                ),
            )

    # 2. Fallback lookup by customer_email and/or customer_phone
    matched_orders: list[OrderRecord] = []
    target_email = (
        validated.customer_email.lower() if validated.customer_email else None
    )
    target_phone = (
        _normalize_phone(validated.customer_phone) if validated.customer_phone else None
    )

    for order in MOCK_ORDERS_DB.values():
        email_match = (
            target_email is not None and order.customer_email.lower() == target_email
        )
        phone_match = (
            target_phone is not None
            and bool(target_phone)
            and _normalize_phone(order.customer_phone) == target_phone
        )
        if email_match or phone_match:
            matched_orders.append(order)

    if matched_orders:
        primary = matched_orders[0]
        order_ids_str = ", ".join(o.order_id for o in matched_orders)
        return LookupOrderDetailsOutput(
            status="success",
            found=True,
            message=(
                f"Found {len(matched_orders)} order(s) ({order_ids_str}) for customer "
                f"{primary.customer_name}. Primary order '{primary.order_id}' loaded."
            ),
            order=primary,
            matching_orders=matched_orders,
        )

    criteria_parts: list[str] = []
    if validated.order_id:
        criteria_parts.append(f"order_id='{validated.order_id}'")
    if validated.customer_email:
        criteria_parts.append(f"customer_email='{validated.customer_email}'")
    if validated.customer_phone:
        criteria_parts.append(f"customer_phone='{validated.customer_phone}'")
    criteria_str = ", ".join(criteria_parts)

    recovery = (
        "Prompt customer to verify the spelling of their loyalty email address or "
        "phone number, or ask for the exact order ID from their confirmation email "
        "to re-query the database."
    )
    error_msg = f"No orders found matching {criteria_str}. Recovery: {recovery}"
    return LookupOrderDetailsOutput(
        status="error",
        found=False,
        message=error_msg,
        order=None,
        matching_orders=[],
        error_code="CUSTOMER_ORDERS_NOT_FOUND",
        recovery_instructions=recovery,
        error_detail=ErrorDetail(
            error_code="CUSTOMER_ORDERS_NOT_FOUND",
            error_message=error_msg,
            recovery_instructions=recovery,
            failed_field="customer_email"
            if validated.customer_email
            else "customer_phone",
        ),
    )


@log_tool_intent_and_outcome(
    predicted_purpose="Verify item return eligibility against 30-day window, condition, final sale, receipt, and high-value rules"
)
def verify_item_return_eligibility(
    order_id: str = "",
    item_sku: str = "",
    item_condition: str = "unopened",
    return_reason: str = "changed_mind",
    has_receipt: bool = True,
    reference_date: str | None = None,
    request: VerifyItemReturnEligibilityInput | None = None,
) -> VerifyItemReturnEligibilityOutput:
    """Verifies whether a specific item in an order is eligible for return or exchange.

    Evaluates omnichannel retail return rules against the target item:
    1. Order & SKU existence in the order database.
    2. Final Sale restriction (`is_final_sale=True` items are ineligible unless defective).
    3. Receipt verification (`has_receipt` or `order.has_receipt_on_file`).
    4. 30-day return window (`days_since_purchase <= item.return_window_days`).
    5. Item physical condition (`unopened`, `like_new`, `opened`, `defective` vs `worn`/`damaged`).
    6. High-value threshold (`unit_price > $100.00` requires manager override approval).

    Args:
        order_id: Unique order identifier containing the item (e.g., 'ORD-1001').
        item_sku: SKU (or line-item ID) of the product to evaluate (e.g., 'SKU-TSHIRT-BLK-M').
        item_condition: Physical state of the item: 'unopened', 'like_new',
            'opened', 'worn', 'damaged', or 'defective'. Defaults to 'unopened'.
        return_reason: Customer's reason for returning the item (e.g.,
            'changed_mind', 'wrong_size', 'defective', 'damaged_in_transit').
        has_receipt: Whether the customer provided a receipt or proof of purchase.
            Defaults to True.
        reference_date: Optional ISO-8601 date string ('YYYY-MM-DD') for calculating
            days since purchase; defaults to current date.
        request: Optional pre-constructed `VerifyItemReturnEligibilityInput`
            Pydantic model.

    Returns:
        VerifyItemReturnEligibilityOutput: A strict Pydantic v2 model containing:
            - status: 'eligible', 'ineligible', 'requires_override', or 'error'.
            - is_eligible: True if eligible immediately or eligible pending manager override.
            - requires_manager_override: True if `request_manager_override_approval`
              must be called before processing the return.
            - order_id: Evaluated order ID.
            - item_sku: Evaluated item SKU.
            - item_name: Product name if found.
            - unit_price: Item unit price in USD.
            - days_since_purchase: Elapsed days since purchase date.
            - return_window_days: Allowed return window (default 30 days).
            - is_final_sale: Whether the item is marked Final Sale.
            - is_high_value_item: True if unit_price > $100.00.
            - eligible_resolutions: Allowed resolutions ('refund', 'exchange', 'store_credit').
            - policy_findings: Audit messages for each policy rule checked.
            - message: Summary message or error description with `Recovery:` instructions.
            - error_code: Policy or validation error code if not standard-eligible.
            - recovery_instructions: Actionable next steps for the LLM.
            - error_detail: Structured `ErrorDetail` when an error or block occurs.
    """
    try:
        if isinstance(order_id, VerifyItemReturnEligibilityInput):
            validated = order_id
        elif isinstance(request, VerifyItemReturnEligibilityInput):
            validated = request
        elif isinstance(request, dict):
            validated = VerifyItemReturnEligibilityInput.model_validate(request)
        else:
            validated = VerifyItemReturnEligibilityInput(
                order_id=order_id,
                item_sku=item_sku,
                item_condition=item_condition,  # type: ignore[arg-type]
                return_reason=return_reason,
                has_receipt=has_receipt,
                reference_date=reference_date,
            )
        if validated.reference_date is not None:
            datetime.date.fromisoformat(validated.reference_date)
    except (ValidationError, ValueError) as exc:
        raw_order_id = order_id if isinstance(order_id, str) else ""
        raw_sku = item_sku if isinstance(item_sku, str) else ""
        recovery = (
            "Ensure 'order_id' and 'item_sku' are non-empty strings, 'item_condition' "
            "is one of ['unopened', 'like_new', 'opened', 'worn', 'damaged', 'defective'], "
            "and 'reference_date' (if provided) is in YYYY-MM-DD format, then retry "
            "verify_item_return_eligibility."
        )
        error_msg = f"Validation error in verify_item_return_eligibility: {exc}. Recovery: {recovery}"
        return VerifyItemReturnEligibilityOutput(
            status="error",
            is_eligible=False,
            requires_manager_override=False,
            order_id=raw_order_id,
            item_sku=raw_sku,
            message=error_msg,
            error_code="VALIDATION_ERROR",
            recovery_instructions=recovery,
            error_detail=ErrorDetail(
                error_code="VALIDATION_ERROR",
                error_message=error_msg,
                recovery_instructions=recovery,
                failed_field="item_condition | order_id | item_sku | reference_date",
            ),
        )

    normalized_order_id = validated.order_id.upper()
    order = MOCK_ORDERS_DB.get(normalized_order_id)
    if order is None:
        recovery = (
            "Prompt customer to provide their phone number or loyalty email to "
            "re-query the database via lookup_order_details before verifying eligibility."
        )
        error_msg = f"Order '{validated.order_id}' not found. Recovery: {recovery}"
        return VerifyItemReturnEligibilityOutput(
            status="error",
            is_eligible=False,
            requires_manager_override=False,
            order_id=validated.order_id,
            item_sku=validated.item_sku,
            message=error_msg,
            error_code="ORDER_NOT_FOUND",
            recovery_instructions=recovery,
            error_detail=ErrorDetail(
                error_code="ORDER_NOT_FOUND",
                error_message=error_msg,
                recovery_instructions=recovery,
                failed_field="order_id",
            ),
        )

    item = _find_item_in_order(order, validated.item_sku)
    if item is None:
        available_skus = ", ".join(f"{i.sku} ({i.name})" for i in order.items)
        recovery = (
            f"Ask the customer to confirm which item from order '{order.order_id}' "
            f"they want to return using one of the valid SKUs: [{available_skus}], "
            "then call verify_item_return_eligibility again."
        )
        error_msg = (
            f"Item SKU '{validated.item_sku}' not found in order '{order.order_id}'. "
            f"Available items: {available_skus}. Recovery: {recovery}"
        )
        return VerifyItemReturnEligibilityOutput(
            status="error",
            is_eligible=False,
            requires_manager_override=False,
            order_id=order.order_id,
            item_sku=validated.item_sku,
            message=error_msg,
            error_code="ITEM_NOT_FOUND_IN_ORDER",
            recovery_instructions=recovery,
            error_detail=ErrorDetail(
                error_code="ITEM_NOT_FOUND_IN_ORDER",
                error_message=error_msg,
                recovery_instructions=recovery,
                failed_field="item_sku",
            ),
        )

    days_elapsed = _calculate_days_since_purchase(
        order.purchase_date, validated.reference_date
    )
    is_high_value = item.unit_price > HIGH_VALUE_THRESHOLD_USD
    receipt_verified = validated.has_receipt and order.has_receipt_on_file
    is_defective = (
        validated.item_condition == "defective"
        or "defect" in validated.return_reason.lower()
        or "damaged_in_transit" in validated.return_reason.lower()
    )

    findings: list[str] = [
        f"Order '{order.order_id}' purchased {days_elapsed} days ago on {order.purchase_date} (window: {item.return_window_days} days).",
        f"Item '{item.name}' (SKU: {item.sku}) unit price: ${item.unit_price:.2f} (high-value >${HIGH_VALUE_THRESHOLD_USD:.2f}: {is_high_value}).",
        f"Item condition: '{validated.item_condition}', return reason: '{validated.return_reason}', receipt verified: {receipt_verified}.",
    ]

    # Rule 1: Final Sale items
    if item.is_final_sale:
        if is_defective:
            recovery = (
                f"Because Final Sale item '{item.sku}' arrived defective/damaged, call "
                f"request_manager_override_approval with order_id='{order.order_id}', "
                f"item_sku='{item.sku}', override_reason='defective_final_sale', and "
                "requested_resolution='exchange' or 'store_credit'."
            )
            msg = (
                f"Item '{item.name}' (SKU: '{item.sku}') is marked Final Sale, but qualifies "
                f"for a defective-item exception with manager override. Recovery: {recovery}"
            )
            findings.append(
                "Final Sale item reported as defective; manager override required."
            )
            return VerifyItemReturnEligibilityOutput(
                status="requires_override",
                is_eligible=True,
                requires_manager_override=True,
                order_id=order.order_id,
                item_sku=item.sku,
                item_name=item.name,
                unit_price=item.unit_price,
                days_since_purchase=days_elapsed,
                return_window_days=item.return_window_days,
                is_final_sale=True,
                is_high_value_item=is_high_value,
                eligible_resolutions=["exchange", "store_credit"],
                policy_findings=findings,
                message=msg,
                error_code="FINAL_SALE_DEFECTIVE_OVERRIDE_REQUIRED",
                recovery_instructions=recovery,
                error_detail=ErrorDetail(
                    error_code="FINAL_SALE_DEFECTIVE_OVERRIDE_REQUIRED",
                    error_message=msg,
                    recovery_instructions=recovery,
                    failed_field="is_final_sale",
                ),
            )

        recovery = (
            "Inform the customer that Final Sale clearance items cannot be returned for "
            "a refund. Ask if the item arrived damaged or defective (and if so, re-verify "
            "with item_condition='defective'), or offer to check if another non-final-sale "
            "item from their order needs to be returned."
        )
        msg = (
            f"Item '{item.name}' (SKU: '{item.sku}') is a Final Sale item and is ineligible "
            f"for return or exchange. Recovery: {recovery}"
        )
        findings.append("Policy block: Item is Final Sale and not reported defective.")
        return VerifyItemReturnEligibilityOutput(
            status="ineligible",
            is_eligible=False,
            requires_manager_override=False,
            order_id=order.order_id,
            item_sku=item.sku,
            item_name=item.name,
            unit_price=item.unit_price,
            days_since_purchase=days_elapsed,
            return_window_days=item.return_window_days,
            is_final_sale=True,
            is_high_value_item=is_high_value,
            eligible_resolutions=[],
            policy_findings=findings,
            message=msg,
            error_code="FINAL_SALE_ITEM_INELIGIBLE",
            recovery_instructions=recovery,
            error_detail=ErrorDetail(
                error_code="FINAL_SALE_ITEM_INELIGIBLE",
                error_message=msg,
                recovery_instructions=recovery,
                failed_field="is_final_sale",
            ),
        )

    # Rule 2: Worn or customer-damaged condition (non-defective)
    if validated.item_condition in ("worn", "damaged") and not is_defective:
        recovery = (
            "Clarify with the customer whether the item arrived defective or damaged in "
            "transit (if so, re-run verify_item_return_eligibility with item_condition='defective'). "
            "Otherwise, inform the customer that worn or user-damaged merchandise is ineligible "
            "for standard return, or request a courtesy store-credit override for loyalty members "
            "via request_manager_override_approval."
        )
        msg = (
            f"Item '{item.name}' (SKU: '{item.sku}') is in '{validated.item_condition}' condition "
            f"and does not meet unworn/like-new return policy criteria. Recovery: {recovery}"
        )
        findings.append(
            f"Policy block: Item condition '{validated.item_condition}' is not eligible without defect claim."
        )
        return VerifyItemReturnEligibilityOutput(
            status="ineligible",
            is_eligible=False,
            requires_manager_override=False,
            order_id=order.order_id,
            item_sku=item.sku,
            item_name=item.name,
            unit_price=item.unit_price,
            days_since_purchase=days_elapsed,
            return_window_days=item.return_window_days,
            is_final_sale=False,
            is_high_value_item=is_high_value,
            eligible_resolutions=[],
            policy_findings=findings,
            message=msg,
            error_code="INVALID_ITEM_CONDITION",
            recovery_instructions=recovery,
            error_detail=ErrorDetail(
                error_code="INVALID_ITEM_CONDITION",
                error_message=msg,
                recovery_instructions=recovery,
                failed_field="item_condition",
            ),
        )

    # Rule 3: Outside 30-day return window
    if days_elapsed > item.return_window_days:
        if days_elapsed <= MAX_OVERRIDE_WINDOW_DAYS or is_defective:
            recovery = (
                f"Inform the customer that order '{order.order_id}' is {days_elapsed} days old "
                f"(past the {item.return_window_days}-day return window). Offer an exchange or "
                f"store credit by calling request_manager_override_approval with "
                f"order_id='{order.order_id}', item_sku='{item.sku}', "
                "override_reason='expired_return_window', and requested_resolution='store_credit' or 'exchange'."
            )
            msg = (
                f"Item '{item.name}' (SKU: '{item.sku}') is outside the {item.return_window_days}-day "
                f"return window ({days_elapsed} days since purchase on {order.purchase_date}). "
                f"Recovery: {recovery}"
            )
            findings.append(
                f"Return window expired ({days_elapsed} > {item.return_window_days} days); eligible for manager override (exchange/store_credit)."
            )
            return VerifyItemReturnEligibilityOutput(
                status="requires_override",
                is_eligible=False,
                requires_manager_override=True,
                order_id=order.order_id,
                item_sku=item.sku,
                item_name=item.name,
                unit_price=item.unit_price,
                days_since_purchase=days_elapsed,
                return_window_days=item.return_window_days,
                is_final_sale=False,
                is_high_value_item=is_high_value,
                eligible_resolutions=["exchange", "store_credit"],
                policy_findings=findings,
                message=msg,
                error_code="OUTSIDE_RETURN_WINDOW",
                recovery_instructions=recovery,
                error_detail=ErrorDetail(
                    error_code="OUTSIDE_RETURN_WINDOW",
                    error_message=msg,
                    recovery_instructions=recovery,
                    failed_field="purchase_date",
                ),
            )

        recovery = (
            f"Explain to the customer that purchases over {MAX_OVERRIDE_WINDOW_DAYS} days old "
            f"({days_elapsed} days elapsed) cannot be returned or overridden in the automated system. "
            "If the item has a manufacturer warranty defect, offer to escalate to a specialist or "
            "re-verify with item_condition='defective'."
        )
        msg = (
            f"Item '{item.name}' (SKU: '{item.sku}') was purchased {days_elapsed} days ago, "
            f"exceeding both the {item.return_window_days}-day standard window and the "
            f"{MAX_OVERRIDE_WINDOW_DAYS}-day maximum override window. Recovery: {recovery}"
        )
        findings.append(
            f"Policy block: {days_elapsed} days elapsed exceeds {MAX_OVERRIDE_WINDOW_DAYS}-day maximum ceiling."
        )
        return VerifyItemReturnEligibilityOutput(
            status="ineligible",
            is_eligible=False,
            requires_manager_override=False,
            order_id=order.order_id,
            item_sku=item.sku,
            item_name=item.name,
            unit_price=item.unit_price,
            days_since_purchase=days_elapsed,
            return_window_days=item.return_window_days,
            is_final_sale=False,
            is_high_value_item=is_high_value,
            eligible_resolutions=[],
            policy_findings=findings,
            message=msg,
            error_code="OUTSIDE_RETURN_WINDOW",
            recovery_instructions=recovery,
            error_detail=ErrorDetail(
                error_code="OUTSIDE_RETURN_WINDOW",
                error_message=msg,
                recovery_instructions=recovery,
                failed_field="purchase_date",
            ),
        )

    # Rule 4: Missing receipt check
    if not receipt_verified:
        recovery = (
            f"Order '{order.order_id}' does not have a verified receipt on file. "
            "Ask the customer if they can provide a receipt number, or request a manager "
            f"override for store credit or exchange by calling request_manager_override_approval "
            f"with order_id='{order.order_id}', item_sku='{item.sku}', "
            "override_reason='missing_receipt', and requested_resolution='store_credit' or 'exchange'."
        )
        msg = (
            f"Missing receipt for item '{item.name}' (SKU: '{item.sku}') in order '{order.order_id}'. "
            f"Original payment refunds require a valid receipt. Recovery: {recovery}"
        )
        findings.append(
            "Missing receipt: Original-payment refund blocked; eligible for store credit or exchange via manager override."
        )
        return VerifyItemReturnEligibilityOutput(
            status="requires_override",
            is_eligible=False,
            requires_manager_override=True,
            order_id=order.order_id,
            item_sku=item.sku,
            item_name=item.name,
            unit_price=item.unit_price,
            days_since_purchase=days_elapsed,
            return_window_days=item.return_window_days,
            is_final_sale=False,
            is_high_value_item=is_high_value,
            eligible_resolutions=["exchange", "store_credit"],
            policy_findings=findings,
            message=msg,
            error_code="MISSING_RECEIPT",
            recovery_instructions=recovery,
            error_detail=ErrorDetail(
                error_code="MISSING_RECEIPT",
                error_message=msg,
                recovery_instructions=recovery,
                failed_field="has_receipt",
            ),
        )

    # Rule 5: High-value item over $100 requires manager override approval
    if is_high_value:
        recovery = (
            f"Call request_manager_override_approval with order_id='{order.order_id}', "
            f"item_sku='{item.sku}', override_reason='high_value_item', and the customer's "
            "justification to obtain an override_approval_code before calling "
            "process_refund_and_generate_label."
        )
        msg = (
            f"Item '{item.name}' (SKU: '{item.sku}', price: ${item.unit_price:.2f}) meets "
            f"the 30-day return window and condition requirements, but exceeds the "
            f"${HIGH_VALUE_THRESHOLD_USD:.2f} high-value threshold and requires manager "
            f"override approval. Recovery: {recovery}"
        )
        findings.append(
            f"High-value item (${item.unit_price:.2f} > ${HIGH_VALUE_THRESHOLD_USD:.2f}): requires manager override approval code."
        )
        return VerifyItemReturnEligibilityOutput(
            status="requires_override",
            is_eligible=True,
            requires_manager_override=True,
            order_id=order.order_id,
            item_sku=item.sku,
            item_name=item.name,
            unit_price=item.unit_price,
            days_since_purchase=days_elapsed,
            return_window_days=item.return_window_days,
            is_final_sale=False,
            is_high_value_item=True,
            eligible_resolutions=["refund", "exchange", "store_credit"],
            policy_findings=findings,
            message=msg,
            error_code="HIGH_VALUE_OVERRIDE_REQUIRED",
            recovery_instructions=recovery,
            error_detail=ErrorDetail(
                error_code="HIGH_VALUE_OVERRIDE_REQUIRED",
                error_message=msg,
                recovery_instructions=recovery,
                failed_field="unit_price",
            ),
        )

    # Standard eligible return
    findings.append(
        "All standard return policy checks passed; no manager override required."
    )
    return VerifyItemReturnEligibilityOutput(
        status="eligible",
        is_eligible=True,
        requires_manager_override=False,
        order_id=order.order_id,
        item_sku=item.sku,
        item_name=item.name,
        unit_price=item.unit_price,
        days_since_purchase=days_elapsed,
        return_window_days=item.return_window_days,
        is_final_sale=False,
        is_high_value_item=False,
        eligible_resolutions=["refund", "exchange", "store_credit"],
        policy_findings=findings,
        message=(
            f"Item '{item.name}' (SKU: '{item.sku}', ${item.unit_price:.2f}) in order "
            f"'{order.order_id}' is fully eligible for a full refund, exchange, or store credit. "
            "You may proceed directly to process_refund_and_generate_label."
        ),
    )


@log_tool_intent_and_outcome(
    predicted_purpose="Request manager override approval code for high-value items or policy exceptions"
)
def request_manager_override_approval(
    order_id: str = "",
    item_sku: str = "",
    override_reason: str = "",
    customer_justification: str = "",
    requested_resolution: str = "refund",
    request: RequestManagerOverrideApprovalInput | None = None,
) -> RequestManagerOverrideApprovalOutput:
    """Requests manager override authorization for high-value items or policy exceptions.

    Evaluates exception criteria for:
    - High-value items over $100.00 (`high_value_item`).
    - Returns past the 30-day window up to 90 days (`expired_return_window`).
    - Missing receipt returns (`missing_receipt`) for verified loyalty customers.
    - Defective Final Sale items (`defective_final_sale`).

    When approved, generates and registers a unique `override_approval_code`
    (e.g., `'MGR-OVR-ORD1004-1249'`) that unlocks `process_refund_and_generate_label`.

    Args:
        order_id: Unique order identifier requiring manager override (e.g., 'ORD-1004').
        item_sku: SKU of the item requiring override approval (e.g., 'SKU-CASHMERE-COAT-M').
        override_reason: Reason category for the override request (e.g.,
            'high_value_item', 'expired_return_window', 'missing_receipt',
            'defective_final_sale').
        customer_justification: Explanation of the customer's circumstances
            (at least 5 characters) supporting the override request.
        requested_resolution: Desired resolution type: 'refund', 'exchange', or
            'store_credit'. Defaults to 'refund'.
        request: Optional pre-constructed `RequestManagerOverrideApprovalInput`
            Pydantic model.

    Returns:
        RequestManagerOverrideApprovalOutput: A strict Pydantic v2 model containing:
            - status: 'approved', 'denied', or 'error'.
            - approved: True if the override was granted.
            - override_approval_code: Authorization token to pass to
              `process_refund_and_generate_label` when approved.
            - order_id: Order ID evaluated.
            - item_sku: Item SKU evaluated.
            - approved_resolution: Resolution type approved ('refund', 'exchange', or 'store_credit').
            - max_approved_amount: Maximum USD amount authorized.
            - manager_notes: Explanation of the manager policy decision.
            - message: Outcome summary or guided error with `Recovery:` instructions.
            - error_code: Error or denial code if not approved.
            - recovery_instructions: Actionable guidance for the LLM on denial or error.
            - error_detail: Structured `ErrorDetail` object on denial or error.
    """
    try:
        if isinstance(order_id, RequestManagerOverrideApprovalInput):
            validated = order_id
        elif isinstance(request, RequestManagerOverrideApprovalInput):
            validated = request
        elif isinstance(request, dict):
            validated = RequestManagerOverrideApprovalInput.model_validate(request)
        else:
            validated = RequestManagerOverrideApprovalInput(
                order_id=order_id,
                item_sku=item_sku,
                override_reason=override_reason,
                customer_justification=customer_justification,
                requested_resolution=requested_resolution,  # type: ignore[arg-type]
            )
    except ValidationError as exc:
        raw_order_id = order_id if isinstance(order_id, str) else ""
        raw_sku = item_sku if isinstance(item_sku, str) else ""
        recovery = (
            "Provide non-empty 'order_id', 'item_sku', 'override_reason' (>=3 chars), "
            "a detailed 'customer_justification' (>=5 chars), and a valid "
            "'requested_resolution' ('refund', 'exchange', or 'store_credit'), "
            "then call request_manager_override_approval again."
        )
        error_msg = (
            f"Validation error in request_manager_override_approval: {exc.errors()[0]['msg']} "
            f"Recovery: {recovery}"
        )
        return RequestManagerOverrideApprovalOutput(
            status="error",
            approved=False,
            override_approval_code=None,
            order_id=raw_order_id,
            item_sku=raw_sku,
            manager_notes="Validation failed before manager review.",
            message=error_msg,
            error_code="VALIDATION_ERROR",
            recovery_instructions=recovery,
            error_detail=ErrorDetail(
                error_code="VALIDATION_ERROR",
                error_message=error_msg,
                recovery_instructions=recovery,
                failed_field="override_reason | customer_justification | requested_resolution",
            ),
        )

    normalized_order_id = validated.order_id.upper()
    order = MOCK_ORDERS_DB.get(normalized_order_id)
    if order is None:
        recovery = (
            "Prompt customer to provide their phone number or loyalty email to "
            "re-query the database via lookup_order_details before requesting a manager override."
        )
        error_msg = f"Order '{validated.order_id}' not found. Recovery: {recovery}"
        return RequestManagerOverrideApprovalOutput(
            status="error",
            approved=False,
            override_approval_code=None,
            order_id=validated.order_id,
            item_sku=validated.item_sku,
            manager_notes="Order not found in database.",
            message=error_msg,
            error_code="ORDER_NOT_FOUND",
            recovery_instructions=recovery,
            error_detail=ErrorDetail(
                error_code="ORDER_NOT_FOUND",
                error_message=error_msg,
                recovery_instructions=recovery,
                failed_field="order_id",
            ),
        )

    item = _find_item_in_order(order, validated.item_sku)
    if item is None:
        available_skus = ", ".join(i.sku for i in order.items)
        recovery = (
            f"Verify the item SKU against order '{order.order_id}' (available SKUs: "
            f"{available_skus}) and retry request_manager_override_approval."
        )
        error_msg = (
            f"Item SKU '{validated.item_sku}' not found in order '{order.order_id}'. "
            f"Recovery: {recovery}"
        )
        return RequestManagerOverrideApprovalOutput(
            status="error",
            approved=False,
            override_approval_code=None,
            order_id=order.order_id,
            item_sku=validated.item_sku,
            manager_notes="Item SKU does not belong to the specified order.",
            message=error_msg,
            error_code="ITEM_NOT_FOUND_IN_ORDER",
            recovery_instructions=recovery,
            error_detail=ErrorDetail(
                error_code="ITEM_NOT_FOUND_IN_ORDER",
                error_message=error_msg,
                recovery_instructions=recovery,
                failed_field="item_sku",
            ),
        )

    days_elapsed = _calculate_days_since_purchase(order.purchase_date)
    reason_lower = validated.override_reason.lower()
    justification_lower = validated.customer_justification.lower()
    claims_defect = (
        "defect" in reason_lower
        or "defect" in justification_lower
        or "damaged" in reason_lower
        or "damaged" in justification_lower
        or "broken" in justification_lower
    )

    # Check 1: Final sale items
    if item.is_final_sale and not claims_defect:
        recovery = (
            "Inform the customer that manager overrides cannot be granted for non-defective "
            "Final Sale items. If the item arrived defective or damaged in transit, ask the "
            "customer for details of the defect and retry request_manager_override_approval "
            "with override_reason='defective_final_sale' and requested_resolution='store_credit' or 'exchange'."
        )
        error_msg = (
            f"Manager override denied for Final Sale item '{item.sku}' in order '{order.order_id}': "
            f"Final Sale items are non-returnable unless defective. Recovery: {recovery}"
        )
        return RequestManagerOverrideApprovalOutput(
            status="denied",
            approved=False,
            override_approval_code=None,
            order_id=order.order_id,
            item_sku=item.sku,
            manager_notes="Denied: Final Sale merchandise without a verified defect claim.",
            message=error_msg,
            error_code="OVERRIDE_DENIED_FINAL_SALE",
            recovery_instructions=recovery,
            error_detail=ErrorDetail(
                error_code="OVERRIDE_DENIED_FINAL_SALE",
                error_message=error_msg,
                recovery_instructions=recovery,
                failed_field="override_reason",
            ),
        )

    # Check 2: Severely expired orders (>90 days) without defect claim
    if days_elapsed > MAX_OVERRIDE_WINDOW_DAYS and not claims_defect:
        recovery = (
            f"Apologize to the customer and explain that orders over {MAX_OVERRIDE_WINDOW_DAYS} "
            f"days old ({days_elapsed} days elapsed) exceed the maximum manager override window. "
            "Offer to connect them with a customer care specialist if this is a manufacturer warranty claim."
        )
        error_msg = (
            f"Manager override denied for order '{order.order_id}': purchase is {days_elapsed} "
            f"days old (exceeds {MAX_OVERRIDE_WINDOW_DAYS}-day override limit). Recovery: {recovery}"
        )
        return RequestManagerOverrideApprovalOutput(
            status="denied",
            approved=False,
            override_approval_code=None,
            order_id=order.order_id,
            item_sku=item.sku,
            manager_notes=(
                f"Denied: {days_elapsed} days since purchase exceeds {MAX_OVERRIDE_WINDOW_DAYS}-day maximum limit."
            ),
            message=error_msg,
            error_code="OVERRIDE_DENIED_EXPIRED_WINDOW",
            recovery_instructions=recovery,
            error_detail=ErrorDetail(
                error_code="OVERRIDE_DENIED_EXPIRED_WINDOW",
                error_message=error_msg,
                recovery_instructions=recovery,
                failed_field="order_id",
            ),
        )

    # Check 3: Late returns (>30 days) or missing receipts requesting cash refund (unless Platinum or defective)
    if (
        (days_elapsed > item.return_window_days or not order.has_receipt_on_file)
        and validated.requested_resolution == "refund"
        and order.loyalty_tier != "platinum"
        and not claims_defect
    ):
        policy_issue = (
            f"outside the {item.return_window_days}-day window ({days_elapsed} days)"
            if days_elapsed > item.return_window_days
            else "missing a receipt on file"
        )
        recovery = (
            "Ask the customer if they will accept 'store_credit' or an 'exchange' instead of "
            "an original-payment refund, then call request_manager_override_approval again with "
            "requested_resolution='store_credit' or 'exchange'."
        )
        error_msg = (
            f"Manager override for a cash/card 'refund' denied because order '{order.order_id}' is "
            f"{policy_issue}. However, an override CAN be approved for 'store_credit' or 'exchange'. "
            f"Recovery: {recovery}"
        )
        return RequestManagerOverrideApprovalOutput(
            status="denied",
            approved=False,
            override_approval_code=None,
            order_id=order.order_id,
            item_sku=item.sku,
            manager_notes=(
                f"Denied cash refund for order {policy_issue}; customer is eligible for store_credit or exchange override."
            ),
            message=error_msg,
            error_code="OVERRIDE_DENIED_RESOLUTION_RESTRICTED",
            recovery_instructions=recovery,
            error_detail=ErrorDetail(
                error_code="OVERRIDE_DENIED_RESOLUTION_RESTRICTED",
                error_message=error_msg,
                recovery_instructions=recovery,
                failed_field="requested_resolution",
            ),
        )

    # Approve the manager override and register the approval token
    clean_ord = re.sub(r"[^A-Z0-9]", "", order.order_id.upper())
    sku_suffix = re.sub(r"[^A-Z0-9]", "", item.sku.upper())[-4:]
    approval_code = f"MGR-OVR-{clean_ord}-{sku_suffix}"

    MANAGER_OVERRIDE_APPROVALS[approval_code] = {
        "order_id": order.order_id,
        "item_sku": item.sku,
        "approved_resolution": validated.requested_resolution,
        "max_approved_amount": item.unit_price,
        "override_reason": validated.override_reason,
        "customer_justification": validated.customer_justification,
    }

    notes = (
        f"Approved {validated.requested_resolution} override for {order.customer_name} "
        f"(loyalty tier: {order.loyalty_tier}) on item '{item.sku}' (${item.unit_price:.2f}). "
        f"Reason: {validated.override_reason}."
    )
    return RequestManagerOverrideApprovalOutput(
        status="approved",
        approved=True,
        override_approval_code=approval_code,
        order_id=order.order_id,
        item_sku=item.sku,
        approved_resolution=validated.requested_resolution,
        max_approved_amount=item.unit_price,
        manager_notes=notes,
        message=(
            f"Manager override approved! Authorization code: '{approval_code}'. "
            f"You may now call process_refund_and_generate_label with "
            f"override_approval_code='{approval_code}' and resolution_type='{validated.requested_resolution}'."
        ),
    )


@log_tool_intent_and_outcome(
    predicted_purpose="Process customer refund/exchange/store credit and generate prepaid return shipping label or QR pass"
)
def process_refund_and_generate_label(
    order_id: str = "",
    item_sku: str = "",
    resolution_type: str = "refund",
    return_channel: str = "mail_carrier",
    override_approval_code: str | None = None,
    exchange_replacement_variant: str | None = None,
    has_receipt: bool = True,
    item_condition: str = "unopened",
    request: ProcessRefundAndGenerateLabelInput | None = None,
) -> ProcessRefundAndGenerateLabelOutput:
    """Processes a customer refund, exchange, or store credit and generates a prepaid return label.

    Validates that the item meets all return eligibility policies or carries a
    valid `override_approval_code` issued by `request_manager_override_approval`.
    Upon verification, issues an RMA number, calculates the refund/credit or
    reserves the exchange variant, and generates a prepaid shipping label or
    in-store QR drop-off pass.

    Args:
        order_id: Unique order identifier for the return (e.g., 'ORD-1001').
        item_sku: SKU of the item being returned or exchanged (e.g., 'SKU-TSHIRT-BLK-M').
        resolution_type: Desired financial/inventory resolution: 'refund',
            'exchange', or 'store_credit'. Defaults to 'refund'.
        return_channel: Preferred omnichannel return method: 'mail_carrier',
            'in_store_dropoff', or 'home_pickup'. Defaults to 'mail_carrier'.
        override_approval_code: Optional manager override authorization code
            (e.g., 'MGR-OVR-ORD1004-COATM') required when the item exceeds $100,
            is past the 30-day window, is missing a receipt, or is a defective
            Final Sale item.
        exchange_replacement_variant: Optional replacement size or color when
            `resolution_type` is 'exchange' (e.g., 'L', 'Silver Frost').
        has_receipt: Whether the customer has a valid receipt. Defaults to True.
        item_condition: Physical condition of the item ('unopened', 'like_new',
            'opened', 'worn', 'damaged', 'defective'). Defaults to 'unopened'.
        request: Optional pre-constructed `ProcessRefundAndGenerateLabelInput`
            Pydantic model.

    Returns:
        ProcessRefundAndGenerateLabelOutput: A strict Pydantic v2 model containing:
            - status: 'success' when processed, or 'error' if blocked/invalid.
            - processed: Boolean indicating whether the return was finalized.
            - rma_number: Generated Return Merchandise Authorization ID (e.g., 'RMA-ORD1001-TSHI').
            - order_id: Order ID processed.
            - item_sku: Item SKU processed.
            - item_name: Display name of the returned item.
            - resolution_type: Resolution applied ('refund', 'exchange', 'store_credit').
            - refund_amount: Dollar amount refunded or credited in USD.
            - refund_destination: Destination account or replacement shipment summary.
            - exchange_replacement_details: Reserved replacement item details for exchanges.
            - shipping_label: Structured `ReturnShippingLabel` with tracking number,
              carrier, download URL, and printerless QR code.
            - message: Confirmation summary or error with `Recovery:` instructions.
            - error_code: Error code when `status` is 'error'.
            - recovery_instructions: Actionable recovery instructions for the LLM.
            - error_detail: Structured `ErrorDetail` object when `status` is 'error'.
    """
    try:
        if isinstance(order_id, ProcessRefundAndGenerateLabelInput):
            validated = order_id
        elif isinstance(request, ProcessRefundAndGenerateLabelInput):
            validated = request
        elif isinstance(request, dict):
            validated = ProcessRefundAndGenerateLabelInput.model_validate(request)
        else:
            validated = ProcessRefundAndGenerateLabelInput(
                order_id=order_id,
                item_sku=item_sku,
                resolution_type=resolution_type,  # type: ignore[arg-type]
                return_channel=return_channel,  # type: ignore[arg-type]
                override_approval_code=override_approval_code,
                exchange_replacement_variant=exchange_replacement_variant,
                has_receipt=has_receipt,
                item_condition=item_condition,  # type: ignore[arg-type]
            )
    except ValidationError as exc:
        raw_order_id = order_id if isinstance(order_id, str) else ""
        raw_sku = item_sku if isinstance(item_sku, str) else ""
        raw_res = resolution_type if isinstance(resolution_type, str) else "refund"
        recovery = (
            "Ensure 'order_id' and 'item_sku' are non-empty, 'resolution_type' is one of "
            "['refund', 'exchange', 'store_credit'], and 'return_channel' is one of "
            "['mail_carrier', 'in_store_dropoff', 'home_pickup'], then call "
            "process_refund_and_generate_label again."
        )
        error_msg = (
            f"Validation error in process_refund_and_generate_label: {exc.errors()[0]['msg']} "
            f"Recovery: {recovery}"
        )
        return ProcessRefundAndGenerateLabelOutput(
            status="error",
            processed=False,
            order_id=raw_order_id,
            item_sku=raw_sku,
            resolution_type=raw_res,
            message=error_msg,
            error_code="VALIDATION_ERROR",
            recovery_instructions=recovery,
            error_detail=ErrorDetail(
                error_code="VALIDATION_ERROR",
                error_message=error_msg,
                recovery_instructions=recovery,
                failed_field="resolution_type | return_channel | order_id | item_sku",
            ),
        )

    normalized_order_id = validated.order_id.upper()
    order = MOCK_ORDERS_DB.get(normalized_order_id)
    if order is None:
        recovery = (
            "Prompt customer to provide their phone number or loyalty email to "
            "re-query the database via lookup_order_details before processing a return."
        )
        error_msg = f"Order '{validated.order_id}' not found. Recovery: {recovery}"
        return ProcessRefundAndGenerateLabelOutput(
            status="error",
            processed=False,
            order_id=validated.order_id,
            item_sku=validated.item_sku,
            resolution_type=validated.resolution_type,
            message=error_msg,
            error_code="ORDER_NOT_FOUND",
            recovery_instructions=recovery,
            error_detail=ErrorDetail(
                error_code="ORDER_NOT_FOUND",
                error_message=error_msg,
                recovery_instructions=recovery,
                failed_field="order_id",
            ),
        )

    item = _find_item_in_order(order, validated.item_sku)
    if item is None:
        available_skus = ", ".join(i.sku for i in order.items)
        recovery = (
            f"Confirm the correct SKU from order '{order.order_id}' (available SKUs: "
            f"{available_skus}) with the customer and retry process_refund_and_generate_label."
        )
        error_msg = (
            f"Item SKU '{validated.item_sku}' not found in order '{order.order_id}'. "
            f"Recovery: {recovery}"
        )
        return ProcessRefundAndGenerateLabelOutput(
            status="error",
            processed=False,
            order_id=order.order_id,
            item_sku=validated.item_sku,
            resolution_type=validated.resolution_type,
            message=error_msg,
            error_code="ITEM_NOT_FOUND_IN_ORDER",
            recovery_instructions=recovery,
            error_detail=ErrorDetail(
                error_code="ITEM_NOT_FOUND_IN_ORDER",
                error_message=error_msg,
                recovery_instructions=recovery,
                failed_field="item_sku",
            ),
        )

    # Evaluate eligibility to determine if an override approval code is required
    days_elapsed = _calculate_days_since_purchase(order.purchase_date)
    is_high_value = item.unit_price > HIGH_VALUE_THRESHOLD_USD
    receipt_verified = validated.has_receipt and order.has_receipt_on_file
    needs_override = (
        is_high_value
        or days_elapsed > item.return_window_days
        or not receipt_verified
        or item.is_final_sale
        or validated.item_condition in ("worn", "damaged")
    )

    if needs_override:
        if not validated.override_approval_code:
            # Build specific error message and recovery instructions for the policy block
            if item.is_final_sale and validated.item_condition != "defective":
                recovery = (
                    f"Inform the customer that '{item.name}' (SKU: '{item.sku}') is a Final Sale "
                    "item and cannot be returned unless defective. If defective, first call "
                    "request_manager_override_approval with override_reason='defective_final_sale'."
                )
                error_code = "FINAL_SALE_ITEM_INELIGIBLE"
                reason_desc = "is marked Final Sale"
            elif days_elapsed > item.return_window_days:
                recovery = (
                    f"Order '{order.order_id}' is {days_elapsed} days old (outside the "
                    f"{item.return_window_days}-day window). Call request_manager_override_approval "
                    f"with order_id='{order.order_id}', item_sku='{item.sku}', "
                    "override_reason='expired_return_window', and requested_resolution='store_credit' "
                    "or 'exchange', then pass the returned override_approval_code to "
                    "process_refund_and_generate_label."
                )
                error_code = "OUTSIDE_RETURN_WINDOW"
                reason_desc = f"is outside the {item.return_window_days}-day return window ({days_elapsed} days elapsed)"
            elif not receipt_verified:
                recovery = (
                    f"Order '{order.order_id}' is missing a verified receipt. Call "
                    f"request_manager_override_approval with order_id='{order.order_id}', "
                    f"item_sku='{item.sku}', override_reason='missing_receipt', and "
                    "requested_resolution='store_credit' or 'exchange', then pass the "
                    "override_approval_code to process_refund_and_generate_label."
                )
                error_code = "MISSING_RECEIPT"
                reason_desc = "is missing a verified receipt"
            elif is_high_value:
                recovery = (
                    f"Item '{item.sku}' (${item.unit_price:.2f}) exceeds the "
                    f"${HIGH_VALUE_THRESHOLD_USD:.2f} high-value threshold. Call "
                    f"request_manager_override_approval with order_id='{order.order_id}', "
                    f"item_sku='{item.sku}', override_reason='high_value_item', and the "
                    "customer's justification, then pass the resulting override_approval_code "
                    "to process_refund_and_generate_label."
                )
                error_code = "HIGH_VALUE_OVERRIDE_REQUIRED"
                reason_desc = f"exceeds the ${HIGH_VALUE_THRESHOLD_USD:.2f} high-value threshold (${item.unit_price:.2f})"
            else:
                recovery = (
                    f"Item '{item.sku}' is in '{validated.item_condition}' condition. "
                    "Verify if the item arrived defective or obtain manager override approval "
                    "via request_manager_override_approval before processing."
                )
                error_code = "MANAGER_OVERRIDE_REQUIRED"
                reason_desc = f"is in '{validated.item_condition}' condition"

            error_msg = (
                f"Cannot process return for item '{item.sku}' in order '{order.order_id}' "
                f"because it {reason_desc} and no manager override_approval_code was provided. "
                f"Recovery: {recovery}"
            )
            return ProcessRefundAndGenerateLabelOutput(
                status="error",
                processed=False,
                order_id=order.order_id,
                item_sku=item.sku,
                item_name=item.name,
                resolution_type=validated.resolution_type,
                message=error_msg,
                error_code=error_code,
                recovery_instructions=recovery,
                error_detail=ErrorDetail(
                    error_code=error_code,
                    error_message=error_msg,
                    recovery_instructions=recovery,
                    failed_field="override_approval_code",
                ),
            )

        # Validate the provided override_approval_code
        code_upper = validated.override_approval_code.upper()
        approval_record = MANAGER_OVERRIDE_APPROVALS.get(code_upper)
        if approval_record is None or not code_upper.startswith("MGR-OVR-"):
            recovery = (
                f"Call request_manager_override_approval for order_id='{order.order_id}' "
                f"and item_sku='{item.sku}' to obtain a valid authorization code, then retry "
                "process_refund_and_generate_label with that exact override_approval_code."
            )
            error_msg = (
                f"Invalid or unrecognized override_approval_code "
                f"'{validated.override_approval_code}'. Recovery: {recovery}"
            )
            return ProcessRefundAndGenerateLabelOutput(
                status="error",
                processed=False,
                order_id=order.order_id,
                item_sku=item.sku,
                item_name=item.name,
                resolution_type=validated.resolution_type,
                message=error_msg,
                error_code="INVALID_MANAGER_OVERRIDE_CODE",
                recovery_instructions=recovery,
                error_detail=ErrorDetail(
                    error_code="INVALID_MANAGER_OVERRIDE_CODE",
                    error_message=error_msg,
                    recovery_instructions=recovery,
                    failed_field="override_approval_code",
                ),
            )

        if (
            approval_record["order_id"].upper() != order.order_id.upper()
            or approval_record["item_sku"].upper() != item.sku.upper()
        ):
            recovery = (
                f"The override code '{code_upper}' was issued for order "
                f"'{approval_record['order_id']}' / SKU '{approval_record['item_sku']}'. "
                f"Call request_manager_override_approval specifically for order_id='{order.order_id}' "
                f"and item_sku='{item.sku}'."
            )
            error_msg = (
                f"Override approval code '{code_upper}' does not match order '{order.order_id}' "
                f"and SKU '{item.sku}'. Recovery: {recovery}"
            )
            return ProcessRefundAndGenerateLabelOutput(
                status="error",
                processed=False,
                order_id=order.order_id,
                item_sku=item.sku,
                item_name=item.name,
                resolution_type=validated.resolution_type,
                message=error_msg,
                error_code="MISMATCHED_MANAGER_OVERRIDE_CODE",
                recovery_instructions=recovery,
                error_detail=ErrorDetail(
                    error_code="MISMATCHED_MANAGER_OVERRIDE_CODE",
                    error_message=error_msg,
                    recovery_instructions=recovery,
                    failed_field="override_approval_code",
                ),
            )

    # Check exchange variant availability if resolution_type == 'exchange'
    exchange_details: str | None = None
    if validated.resolution_type == "exchange":
        if validated.exchange_replacement_variant:
            variant_clean = validated.exchange_replacement_variant.strip()
            if item.available_exchange_sizes and variant_clean.upper() not in {
                v.upper() for v in item.available_exchange_sizes
            }:
                avail_str = ", ".join(item.available_exchange_sizes)
                recovery = (
                    f"Inform the customer that variant '{variant_clean}' is out of stock "
                    f"for '{item.name}'. Ask them to choose from the available variants "
                    f"([{avail_str}]) or switch resolution_type to 'refund' or 'store_credit'."
                )
                error_msg = (
                    f"Requested exchange variant '{variant_clean}' for SKU '{item.sku}' is "
                    f"unavailable. Available variants: {avail_str}. Recovery: {recovery}"
                )
                return ProcessRefundAndGenerateLabelOutput(
                    status="error",
                    processed=False,
                    order_id=order.order_id,
                    item_sku=item.sku,
                    item_name=item.name,
                    resolution_type="exchange",
                    message=error_msg,
                    error_code="EXCHANGE_VARIANT_UNAVAILABLE",
                    recovery_instructions=recovery,
                    error_detail=ErrorDetail(
                        error_code="EXCHANGE_VARIANT_UNAVAILABLE",
                        error_message=error_msg,
                        recovery_instructions=recovery,
                        failed_field="exchange_replacement_variant",
                    ),
                )
            exchange_details = (
                f"Reserved replacement '{item.name}' in variant '{variant_clean}' "
                "(ships immediately upon carrier scan of return label)."
            )
        else:
            default_variant = (
                item.available_exchange_sizes[0]
                if item.available_exchange_sizes
                else "Same Size/Variant"
            )
            exchange_details = (
                f"Reserved replacement '{item.name}' (variant: {default_variant}); "
                "ships upon return label scan."
            )

    # Build RMA and shipping label / omnichannel pass
    clean_ord = re.sub(r"[^A-Z0-9]", "", order.order_id.upper())
    sku_tag = re.sub(r"[^A-Z0-9]", "", item.sku.upper())[-4:]
    rma_number = f"RMA-{clean_ord}-{sku_tag}"
    expires_date = (datetime.date.today() + datetime.timedelta(days=14)).isoformat()

    if validated.return_channel == "in_store_dropoff":
        carrier = "Omnichannel In-Store Express Desk"
        tracking_number = f"INSTORE-{clean_ord}-{sku_tag}"
        dropoff_instructions = (
            "Bring the item (no box or tape needed) to any retail store Customer Concierge "
            f"desk and present QR pass '{tracking_number}' before {expires_date}."
        )
    elif validated.return_channel == "home_pickup":
        carrier = "FedEx Same-Day Courier Pickup"
        tracking_number = f"1Z-PICKUP-{clean_ord}-{sku_tag}"
        dropoff_instructions = (
            "Pack and seal the item securely. A courier pickup has been scheduled for the "
            f"next business day using tracking number '{tracking_number}'."
        )
    else:
        carrier = "UPS Ground Return Services"
        tracking_number = f"1Z999AA101{clean_ord}{sku_tag}"
        dropoff_instructions = (
            "Place the item in its original packaging, affix the prepaid UPS Ground label "
            "or show the printerless QR code at any UPS Store drop-off counter by "
            f"{expires_date}."
        )

    shipping_label = ReturnShippingLabel(
        label_id=f"LBL-{clean_ord}-{sku_tag}",
        carrier=carrier,
        tracking_number=tracking_number,
        return_channel=validated.return_channel,
        label_download_url=f"https://returns.example-retail.com/labels/{rma_number}.pdf",
        qr_code_data=f"OMNI-RET://{rma_number}/{tracking_number}",
        dropoff_instructions=dropoff_instructions,
        expires_at=expires_date,
    )

    if validated.resolution_type == "refund":
        refund_amount = item.unit_price
        refund_destination = f"Original payment method ({order.payment_method})"
    elif validated.resolution_type == "store_credit":
        refund_amount = item.unit_price
        refund_destination = (
            f"Instant Digital Store Credit Gift Card sent to {order.customer_email}"
        )
    else:
        refund_amount = 0.0
        refund_destination = (
            f"Even exchange replacement order for {order.customer_email}"
        )

    output = ProcessRefundAndGenerateLabelOutput(
        status="success",
        processed=True,
        rma_number=rma_number,
        order_id=order.order_id,
        item_sku=item.sku,
        item_name=item.name,
        resolution_type=validated.resolution_type,
        refund_amount=refund_amount,
        refund_destination=refund_destination,
        exchange_replacement_details=exchange_details,
        shipping_label=shipping_label,
        message=(
            f"Return {rma_number} successfully processed for '{item.name}' (SKU: '{item.sku}'). "
            f"Resolution: {validated.resolution_type} (${refund_amount:.2f} -> {refund_destination}). "
            f"Prepaid {carrier} label generated (Tracking: {tracking_number}, Download: "
            f"{shipping_label.label_download_url})."
        ),
    )
    PROCESSED_RETURNS_LOG[rma_number] = output
    return output
