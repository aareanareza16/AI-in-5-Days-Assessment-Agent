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

"""Pydantic v2 data models and explicit tool input/output schemas.

This module defines strict Pydantic v2 BaseModel classes for all domain entities,
tool input arguments, structured tool outputs, and guided error responses used by
the Omnichannel Retail Return & Exchange Concierge agent.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

LoyaltyTier = Literal["standard", "silver", "gold", "platinum"]
PurchaseChannel = Literal["online", "in_store", "mobile_app"]
ItemCondition = Literal[
    "unopened",
    "like_new",
    "opened",
    "worn",
    "damaged",
    "defective",
]
ResolutionType = Literal["refund", "exchange", "store_credit"]
ReturnChannel = Literal["mail_carrier", "in_store_dropoff", "home_pickup"]


class StrictBaseModel(BaseModel):
    """Base Pydantic v2 model enforcing strict types and forbidding extra fields.

    Also provides read-only mapping helper methods (`__getitem__`, `get`,
    `__contains__`, `to_dict`) so callers and ADK runtime utilities can interact
    with tool outputs as either typed Pydantic models or dictionaries.
    """

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        validate_assignment=True,
    )

    def __getitem__(self, item: str) -> Any:
        """Allows dictionary-style key access on structured tool outputs."""
        try:
            return getattr(self, item)
        except AttributeError as exc:
            raise KeyError(item) from exc

    def get(self, key: str, default: Any = None) -> Any:
        """Returns the attribute value for `key` if present, else `default`."""
        return getattr(self, key, default)

    def __contains__(self, key: object) -> bool:
        """Returns True if `key` is a defined field on this model."""
        return isinstance(key, str) and key in type(self).model_fields

    def to_dict(self) -> dict[str, Any]:
        """Serializes the Pydantic model into a standard Python dictionary."""
        return self.model_dump()


class ErrorDetail(StrictBaseModel):
    """Structured error payload with actionable LLM recovery instructions."""

    error_code: str = Field(
        description="Machine-readable error classification code (e.g., ORDER_NOT_FOUND, OUTSIDE_RETURN_WINDOW)."
    )
    error_message: str = Field(
        description="Human-readable explanation of the validation or business logic failure."
    )
    recovery_instructions: str = Field(
        description="Actionable step-by-step instructions guiding the LLM on how to recover from this error."
    )
    failed_field: str | None = Field(
        default=None,
        description="Optional name of the input field or policy attribute that triggered the error.",
    )


class OrderItem(StrictBaseModel):
    """Represents a single line item within a retail customer order."""

    item_id: str = Field(
        min_length=1,
        description="Unique line-item identifier within the order (e.g., ITEM-1001-1).",
    )
    sku: str = Field(
        min_length=1,
        description="Stock Keeping Unit code for the product (e.g., SKU-TSHIRT-BLK-M).",
    )
    name: str = Field(
        min_length=1,
        description="Display name of the retail item.",
    )
    category: str = Field(
        min_length=1,
        description="Merchandise category (e.g., Apparel, Accessories, Outerwear, Footwear, Electronics).",
    )
    quantity: int = Field(
        ge=1,
        description="Number of units purchased for this SKU.",
    )
    unit_price: float = Field(
        ge=0.0,
        description="Unit purchase price in USD paid by the customer.",
    )
    is_final_sale: bool = Field(
        default=False,
        description="Whether the item was sold as Final Sale / non-returnable clearance.",
    )
    return_window_days: int = Field(
        default=30,
        ge=0,
        description="Standard return eligibility window in days from purchase_date.",
    )
    available_exchange_sizes: list[str] = Field(
        default_factory=list,
        description="List of currently in-stock sizes or variants available for direct exchange.",
    )


class OrderRecord(StrictBaseModel):
    """Represents a complete omnichannel retail order record."""

    order_id: str = Field(
        min_length=1,
        description="Unique order identifier (e.g., ORD-1001).",
    )
    customer_id: str = Field(
        min_length=1,
        description="Unique customer loyalty account ID (e.g., CUST-501).",
    )
    customer_name: str = Field(
        min_length=1,
        description="Full name of the customer who placed the order.",
    )
    customer_email: str = Field(
        min_length=3,
        description="Customer's registered loyalty email address.",
    )
    customer_phone: str = Field(
        min_length=7,
        description="Customer's contact phone number.",
    )
    loyalty_tier: LoyaltyTier = Field(
        default="standard",
        description="Customer's loyalty program tier (standard, silver, gold, or platinum).",
    )
    purchase_date: str = Field(
        min_length=10,
        description="ISO-8601 purchase date string in YYYY-MM-DD format.",
    )
    purchase_channel: PurchaseChannel = Field(
        default="online",
        description="Channel where the purchase originated (online, in_store, or mobile_app).",
    )
    has_receipt_on_file: bool = Field(
        default=True,
        description="Whether a verified digital or POS receipt is attached to the order record.",
    )
    payment_method: str = Field(
        min_length=1,
        description="Original payment method description (e.g., Visa ending in 4242, Cash).",
    )
    order_total: float = Field(
        ge=0.0,
        description="Total dollar amount of the order in USD.",
    )
    items: list[OrderItem] = Field(
        min_length=1,
        description="List of line items purchased in the order.",
    )


# ============================================================================
# Tool 1 Schemas: lookup_order_details
# ============================================================================


class LookupOrderDetailsInput(StrictBaseModel):
    """Strict input schema for the `lookup_order_details` tool."""

    order_id: str | None = Field(
        default=None,
        description="Primary order identifier to look up (e.g., 'ORD-1001').",
    )
    customer_email: str | None = Field(
        default=None,
        description="Customer loyalty email address used for fallback order lookup.",
    )
    customer_phone: str | None = Field(
        default=None,
        description="Customer phone number used for fallback order lookup (e.g., '555-0101').",
    )

    @field_validator("order_id", "customer_email", "customer_phone")
    @classmethod
    def normalize_optional_strings(cls, value: str | None) -> str | None:
        """Strips whitespace and converts blank strings to None."""
        if value is None:
            return None
        stripped = value.strip()
        return stripped if stripped else None

    @model_validator(mode="after")
    def ensure_at_least_one_identifier(self) -> LookupOrderDetailsInput:
        """Ensures at least one lookup identifier is provided."""
        if not any([self.order_id, self.customer_email, self.customer_phone]):
            raise ValueError(
                "At least one lookup identifier ('order_id', 'customer_email', or 'customer_phone') must be provided."
            )
        if self.customer_email and "@" not in self.customer_email:
            raise ValueError(
                f"Invalid customer_email '{self.customer_email}': must contain an '@' symbol."
            )
        return self


class LookupOrderDetailsOutput(StrictBaseModel):
    """Strict output schema for the `lookup_order_details` tool."""

    status: Literal["success", "error"] = Field(
        description="Overall execution status of the order lookup ('success' or 'error')."
    )
    found: bool = Field(
        description="True if at least one matching order was found in the database."
    )
    message: str = Field(
        description="Summary message describing the lookup outcome or guided error recovery."
    )
    order: OrderRecord | None = Field(
        default=None,
        description="Primary matched order record when found.",
    )
    matching_orders: list[OrderRecord] = Field(
        default_factory=list,
        description="All orders matched when querying by customer email or phone number.",
    )
    error_code: str | None = Field(
        default=None,
        description="Machine-readable error code when status is 'error'.",
    )
    recovery_instructions: str | None = Field(
        default=None,
        description="Actionable recovery instructions for the LLM when an error occurs.",
    )
    error_detail: ErrorDetail | None = Field(
        default=None,
        description="Structured error object containing code, message, and recovery guidance.",
    )


# ============================================================================
# Tool 2 Schemas: verify_item_return_eligibility
# ============================================================================


class VerifyItemReturnEligibilityInput(StrictBaseModel):
    """Strict input schema for the `verify_item_return_eligibility` tool."""

    order_id: str = Field(
        min_length=1,
        description="Unique order identifier containing the item to verify (e.g., 'ORD-1001').",
    )
    item_sku: str = Field(
        min_length=1,
        description="SKU of the specific item to check for return or exchange eligibility.",
    )
    item_condition: ItemCondition = Field(
        default="unopened",
        description="Physical condition of the item ('unopened', 'like_new', 'opened', 'worn', 'damaged', or 'defective').",
    )
    return_reason: str = Field(
        default="changed_mind",
        min_length=1,
        description="Customer's stated reason for returning or exchanging the item.",
    )
    has_receipt: bool = Field(
        default=True,
        description="Whether the customer has provided a receipt or proof of purchase.",
    )
    reference_date: str | None = Field(
        default=None,
        description="Optional ISO-8601 date (YYYY-MM-DD) used to evaluate the return window; defaults to today.",
    )

    @field_validator("order_id", "item_sku", "return_reason")
    @classmethod
    def strip_required_strings(cls, value: str) -> str:
        """Strips surrounding whitespace and rejects blank strings."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("Field cannot be empty or whitespace only.")
        return stripped


class VerifyItemReturnEligibilityOutput(StrictBaseModel):
    """Strict output schema for the `verify_item_return_eligibility` tool."""

    status: Literal["eligible", "ineligible", "requires_override", "error"] = Field(
        description="Eligibility determination status ('eligible', 'ineligible', 'requires_override', or 'error')."
    )
    is_eligible: bool = Field(
        description="True if the item can be returned directly or with manager override approval."
    )
    requires_manager_override: bool = Field(
        description="True if a manager override approval code is required before processing."
    )
    order_id: str = Field(description="Order identifier that was evaluated.")
    item_sku: str = Field(description="Item SKU that was evaluated.")
    item_name: str | None = Field(
        default=None,
        description="Display name of the evaluated item.",
    )
    unit_price: float | None = Field(
        default=None,
        description="Unit price of the evaluated item in USD.",
    )
    days_since_purchase: int | None = Field(
        default=None,
        description="Number of calendar days elapsed since the order purchase_date.",
    )
    return_window_days: int = Field(
        default=30,
        description="Applicable return window in days for the item.",
    )
    is_final_sale: bool = Field(
        default=False,
        description="Whether the item is flagged as Final Sale.",
    )
    is_high_value_item: bool = Field(
        default=False,
        description="True if the item's unit_price exceeds the $100.00 high-value threshold.",
    )
    eligible_resolutions: list[str] = Field(
        default_factory=list,
        description="List of resolution types permitted ('refund', 'exchange', 'store_credit').",
    )
    policy_findings: list[str] = Field(
        default_factory=list,
        description="Detailed list of policy checks evaluated for this item.",
    )
    message: str = Field(
        description="Summary of the eligibility result or guided error recovery instructions."
    )
    error_code: str | None = Field(
        default=None,
        description="Machine-readable policy or validation error code when not standard-eligible.",
    )
    recovery_instructions: str | None = Field(
        default=None,
        description="Actionable recovery instructions for the LLM when an error or policy block occurs.",
    )
    error_detail: ErrorDetail | None = Field(
        default=None,
        description="Structured error object with code, message, and recovery instructions.",
    )


# ============================================================================
# Tool 3 Schemas: request_manager_override_approval
# ============================================================================


class RequestManagerOverrideApprovalInput(StrictBaseModel):
    """Strict input schema for the `request_manager_override_approval` tool."""

    order_id: str = Field(
        min_length=1,
        description="Order identifier requiring a policy exception or high-value approval.",
    )
    item_sku: str = Field(
        min_length=1,
        description="SKU of the item requiring manager override approval.",
    )
    override_reason: str = Field(
        min_length=3,
        description="Policy exception category (e.g., 'high_value_item', 'expired_return_window', 'missing_receipt', 'defective_final_sale').",
    )
    customer_justification: str = Field(
        min_length=5,
        description="Detailed explanation of the customer's situation justifying the override request.",
    )
    requested_resolution: ResolutionType = Field(
        default="refund",
        description="Desired resolution type ('refund', 'exchange', or 'store_credit').",
    )

    @field_validator(
        "order_id", "item_sku", "override_reason", "customer_justification"
    )
    @classmethod
    def strip_and_validate_non_empty(cls, value: str) -> str:
        """Strips whitespace and ensures required string fields are non-empty."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("Field cannot be empty or whitespace only.")
        return stripped


class RequestManagerOverrideApprovalOutput(StrictBaseModel):
    """Strict output schema for the `request_manager_override_approval` tool."""

    status: Literal["approved", "denied", "error"] = Field(
        description="Outcome of the manager override request ('approved', 'denied', or 'error')."
    )
    approved: bool = Field(description="True if the manager override was granted.")
    override_approval_code: str | None = Field(
        default=None,
        description="Authorization code (e.g., 'MGR-OVR-ORD1004-8821') to pass to process_refund_and_generate_label.",
    )
    order_id: str = Field(description="Order ID associated with the override request.")
    item_sku: str = Field(description="Item SKU associated with the override request.")
    approved_resolution: str | None = Field(
        default=None,
        description="Resolution type authorized by the manager ('refund', 'exchange', or 'store_credit').",
    )
    max_approved_amount: float | None = Field(
        default=None,
        description="Maximum dollar value authorized under this override in USD.",
    )
    manager_notes: str = Field(
        description="Audit notes explaining the approval or denial decision."
    )
    message: str = Field(
        description="Summary message with approval details or recovery instructions."
    )
    error_code: str | None = Field(
        default=None,
        description="Machine-readable error code when status is 'denied' or 'error'.",
    )
    recovery_instructions: str | None = Field(
        default=None,
        description="Actionable recovery instructions for the LLM if denied or errored.",
    )
    error_detail: ErrorDetail | None = Field(
        default=None,
        description="Structured error payload with actionable recovery guidance.",
    )


# ============================================================================
# Tool 4 Schemas: process_refund_and_generate_label
# ============================================================================


class ReturnShippingLabel(StrictBaseModel):
    """Prepaid return shipping label or omnichannel drop-off pass details."""

    label_id: str = Field(
        min_length=1,
        description="Unique identifier for the generated shipping label.",
    )
    carrier: str = Field(
        min_length=1,
        description="Logistics carrier handling the return (e.g., 'UPS Ground', 'FedEx Express', 'In-Store Express Desk').",
    )
    tracking_number: str = Field(
        min_length=1,
        description="Carrier tracking number or in-store drop-off pass code.",
    )
    return_channel: ReturnChannel = Field(
        description="Selected omnichannel return method ('mail_carrier', 'in_store_dropoff', or 'home_pickup')."
    )
    label_download_url: str = Field(
        min_length=1,
        description="URL where the customer can download/print the prepaid return label PDF.",
    )
    qr_code_data: str = Field(
        min_length=1,
        description="Printerless QR code payload for carrier counter or in-store scan.",
    )
    dropoff_instructions: str = Field(
        min_length=1,
        description="Customer-facing instructions for packaging and dropping off the return.",
    )
    expires_at: str = Field(
        min_length=10,
        description="ISO-8601 expiration date by which the item must be shipped or dropped off.",
    )


class ProcessRefundAndGenerateLabelInput(StrictBaseModel):
    """Strict input schema for the `process_refund_and_generate_label` tool."""

    order_id: str = Field(
        min_length=1,
        description="Unique order identifier for the return transaction (e.g., 'ORD-1001').",
    )
    item_sku: str = Field(
        min_length=1,
        description="SKU of the item being returned or exchanged.",
    )
    resolution_type: ResolutionType = Field(
        default="refund",
        description="Type of financial/inventory resolution ('refund', 'exchange', or 'store_credit').",
    )
    return_channel: ReturnChannel = Field(
        default="mail_carrier",
        description="Preferred omnichannel return method ('mail_carrier', 'in_store_dropoff', or 'home_pickup').",
    )
    override_approval_code: str | None = Field(
        default=None,
        description="Manager override approval code (required for high-value items >$100 or policy exceptions).",
    )
    exchange_replacement_variant: str | None = Field(
        default=None,
        description="Requested replacement size/color variant when resolution_type is 'exchange'.",
    )
    has_receipt: bool = Field(
        default=True,
        description="Whether a valid receipt or proof of purchase was verified.",
    )
    item_condition: ItemCondition = Field(
        default="unopened",
        description="Physical condition of the returned item ('unopened', 'like_new', 'opened', 'worn', 'damaged', or 'defective').",
    )

    @field_validator("order_id", "item_sku")
    @classmethod
    def strip_required_ids(cls, value: str) -> str:
        """Strips whitespace from required identifiers."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("Identifier cannot be empty or whitespace only.")
        return stripped

    @field_validator("override_approval_code", "exchange_replacement_variant")
    @classmethod
    def normalize_optional_fields(cls, value: str | None) -> str | None:
        """Normalizes optional string parameters."""
        if value is None:
            return None
        stripped = value.strip()
        return stripped if stripped else None


class ProcessRefundAndGenerateLabelOutput(StrictBaseModel):
    """Strict output schema for the `process_refund_and_generate_label` tool."""

    status: Literal["success", "error"] = Field(
        description="Execution status of the refund/exchange and label generation ('success' or 'error')."
    )
    processed: bool = Field(
        description="True if the return/exchange was finalized and a label was generated."
    )
    rma_number: str | None = Field(
        default=None,
        description="Generated Return Merchandise Authorization (RMA) number.",
    )
    order_id: str = Field(description="Order identifier for the transaction.")
    item_sku: str = Field(description="Returned item SKU.")
    item_name: str | None = Field(
        default=None,
        description="Display name of the returned item.",
    )
    resolution_type: str = Field(
        description="Applied resolution type ('refund', 'exchange', or 'store_credit')."
    )
    refund_amount: float | None = Field(
        default=None,
        description="Dollar amount refunded or credited in USD (0.0 for even exchanges).",
    )
    refund_destination: str | None = Field(
        default=None,
        description="Destination of funds (e.g., original payment method, digital gift card, replacement shipment).",
    )
    exchange_replacement_details: str | None = Field(
        default=None,
        description="Details of the reserved replacement item when resolution_type is 'exchange'.",
    )
    shipping_label: ReturnShippingLabel | None = Field(
        default=None,
        description="Generated prepaid return shipping label or in-store QR pass.",
    )
    message: str = Field(
        description="Confirmation message or guided error recovery instructions."
    )
    error_code: str | None = Field(
        default=None,
        description="Machine-readable error code when status is 'error'.",
    )
    recovery_instructions: str | None = Field(
        default=None,
        description="Actionable recovery instructions for the LLM when processing fails.",
    )
    error_detail: ErrorDetail | None = Field(
        default=None,
        description="Structured error detail object when status is 'error'.",
    )
