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

"""Secure configuration and secret management for the Retail Returns Concierge.

Implements Requirement 3 of Category 5: Infrastructure & CI/CD:
- Ensures no API keys or credentials are hardcoded anywhere in source code.
- Reads configuration and secrets from environment variables (`os.getenv`) and
  optionally resolves secrets from Google Cloud Secret Manager at runtime.
"""

from __future__ import annotations

import functools
import os
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()


def get_secret(
    secret_name: str,
    project_id: str | None = None,
    version: str = "latest",
    default: str | None = None,
) -> str | None:
    """Retrieves a secret from environment variables or Google Cloud Secret Manager.

    Resolution order:
    1. Direct environment variable `os.getenv(secret_name)`.
    2. Google Cloud Secret Manager (`projects/{project_id}/secrets/{secret_name}/versions/{version}`)
       when `project_id` (or `GOOGLE_CLOUD_PROJECT`) is set and the Secret Manager
       client library is available.
    3. Provided `default` value.

    Args:
        secret_name: Environment variable name or GCP Secret Manager secret ID.
        project_id: Optional GCP project ID override.
        version: Secret version to access (default: 'latest').
        default: Fallback value if the secret is not found.

    Returns:
        Resolved secret string or `default` if unavailable.
    """
    env_val = os.getenv(secret_name)
    if env_val is not None and env_val.strip():
        return env_val.strip()

    resolved_project = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
    if (
        resolved_project
        and os.getenv("ENABLE_GCP_SECRET_MANAGER", "false").lower() == "true"
    ):
        try:
            from google.cloud import secretmanager  # type: ignore[import-untyped]

            client = secretmanager.SecretManagerServiceClient()
            resource_name = (
                f"projects/{resolved_project}/secrets/{secret_name}/versions/{version}"
            )
            response = client.access_secret_version(request={"name": resource_name})
            return response.payload.data.decode("utf-8").strip()
        except Exception:
            return default

    return default


class AgentSettings(BaseModel):
    """Validated runtime configuration loaded strictly from environment / Secret Manager."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    google_cloud_project: str = Field(
        default_factory=lambda: os.getenv("GOOGLE_CLOUD_PROJECT", ""),
        description="Google Cloud Project ID hosting Vertex AI / Cloud Run / Secret Manager.",
    )
    google_cloud_location: str = Field(
        default_factory=lambda: os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
        description="Google Cloud region or 'global' endpoint for Gemini models.",
    )
    google_genai_use_vertexai: bool = Field(
        default_factory=lambda: (
            os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "true").lower()
            in ("1", "true", "yes")
        ),
        description="Whether to route GenAI SDK calls through Vertex AI.",
    )
    gemini_api_key: str | None = Field(
        default_factory=lambda: (
            get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY")
        ),
        description="Optional Gemini API key loaded from env or Secret Manager (never hardcoded).",
    )
    fast_triage_model: str = Field(
        default_factory=lambda: os.getenv("FAST_TRIAGE_MODEL", "gemini-2.0-flash"),
        description="Fast low-latency Gemini model used for TriageCoordinator and FulfillmentAgent.",
    )
    deep_policy_model: str = Field(
        default_factory=lambda: os.getenv("DEEP_POLICY_MODEL", "gemini-2.0-pro"),
        description="Reasoning Gemini model used for ReturnPolicyEvaluator.",
    )
    hitl_refund_threshold_usd: float = Field(
        default_factory=lambda: float(os.getenv("HITL_REFUND_THRESHOLD_USD", "75.0")),
        description="Dollar threshold above which human supervisor HITL approval is required.",
    )
    hitl_return_window_days: int = Field(
        default_factory=lambda: int(os.getenv("HITL_RETURN_WINDOW_DAYS", "30")),
        description="Standard return window in calendar days before policy exception HITL triggers.",
    )
    log_level: str = Field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"),
        description="Structured JSON logging level.",
    )

    def safe_dump(self) -> dict[str, Any]:
        """Returns a dictionary of settings with any secret fields masked."""
        data = self.model_dump()
        if data.get("gemini_api_key"):
            data["gemini_api_key"] = "***REDACTED_SECRET***"
        return data


@functools.lru_cache(maxsize=1)
def get_settings() -> AgentSettings:
    """Returns the cached singleton `AgentSettings` instance."""
    return AgentSettings()


settings = get_settings()
