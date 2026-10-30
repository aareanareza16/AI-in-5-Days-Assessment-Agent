# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.30"
    }
  }
}

# ------------------------------------------------------------------------------
# Variables
# ------------------------------------------------------------------------------

variable "project_id" {
  description = "Google Cloud Project ID where the agent will be deployed."
  type        = string
}

variable "region" {
  description = "Google Cloud region for Cloud Run and Artifact Registry."
  type        = string
  default     = "us-central1"
}

variable "service_name" {
  description = "Name of the Cloud Run service for the Retail Returns Concierge."
  type        = string
  default     = "retail-returns-concierge"
}

variable "image_tag" {
  description = "Container image tag to deploy to Cloud Run."
  type        = string
  default     = "latest"
}

variable "hitl_webhook_secret_value" {
  description = "Sensitive secret value for the HITL supervisor webhook token."
  type        = string
  sensitive   = true
  default     = ""
}

# ------------------------------------------------------------------------------
# Provider Configuration
# ------------------------------------------------------------------------------

provider "google" {
  project = var.project_id
  region  = var.region
}

# ------------------------------------------------------------------------------
# Required Google Cloud APIs
# ------------------------------------------------------------------------------

locals {
  required_services = toset([
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "secretmanager.googleapis.com",
    "logging.googleapis.com",
    "cloudtrace.googleapis.com",
    "aiplatform.googleapis.com",
  ])
}

resource "google_project_service" "enabled_apis" {
  for_each           = local.required_services
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

# ------------------------------------------------------------------------------
# Artifact Registry Repository
# ------------------------------------------------------------------------------

resource "google_artifact_registry_repository" "agent_repo" {
  location      = var.region
  repository_id = "${var.service_name}-repo"
  description   = "Docker container repository for the Omnichannel Retail Return & Exchange Concierge"
  format        = "DOCKER"

  depends_on = [google_project_service.enabled_apis]
}

# ------------------------------------------------------------------------------
# Secret Manager Secrets
# ------------------------------------------------------------------------------

resource "google_secret_manager_secret" "gemini_api_key" {
  secret_id = "${var.service_name}-gemini-api-key"

  replication {
    auto {}
  }

  labels = {
    app       = var.service_name
    managed   = "terraform"
    component = "credentials"
  }

  depends_on = [google_project_service.enabled_apis]
}

resource "google_secret_manager_secret" "hitl_webhook_secret" {
  secret_id = "${var.service_name}-hitl-webhook-secret"

  replication {
    auto {}
  }

  labels = {
    app       = var.service_name
    managed   = "terraform"
    component = "hitl-auth"
  }

  depends_on = [google_project_service.enabled_apis]
}

resource "google_secret_manager_secret_version" "hitl_webhook_secret_v1" {
  count       = var.hitl_webhook_secret_value != "" ? 1 : 0
  secret      = google_secret_manager_secret.hitl_webhook_secret.id
  secret_data = var.hitl_webhook_secret_value
}

# ------------------------------------------------------------------------------
# Dedicated Service Account & Least-Privilege IAM Bindings
# ------------------------------------------------------------------------------

resource "google_service_account" "agent_runner" {
  account_id   = "${var.service_name}-sa"
  display_name = "Retail Returns Concierge Agent Runtime Service Account"
  description  = "Least-privilege identity for Cloud Run execution, Vertex AI, Secret Manager, and Cloud Logging/Trace."
}

resource "google_project_iam_member" "vertex_ai_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.agent_runner.email}"
}

resource "google_project_iam_member" "log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.agent_runner.email}"
}

resource "google_project_iam_member" "trace_agent" {
  project = var.project_id
  role    = "roles/cloudtrace.agent"
  member  = "serviceAccount:${google_service_account.agent_runner.email}"
}

resource "google_secret_manager_secret_iam_member" "gemini_api_key_access" {
  secret_id = google_secret_manager_secret.gemini_api_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.agent_runner.email}"
}

resource "google_secret_manager_secret_iam_member" "hitl_webhook_secret_access" {
  secret_id = google_secret_manager_secret.hitl_webhook_secret.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.agent_runner.email}"
}

# ------------------------------------------------------------------------------
# Cloud Logging Bucket & Log-Based Metric for HITL & Guardrail Events
# ------------------------------------------------------------------------------

resource "google_logging_project_bucket_config" "agent_audit_logs" {
  project        = var.project_id
  location       = "global"
  retention_days = 90
  bucket_id      = "${var.service_name}-audit-logs"
  description    = "Dedicated Cloud Logging bucket for retail returns agent intent, outcome, and HITL audit trails."

  depends_on = [google_project_service.enabled_apis]
}

resource "google_logging_project_sink" "agent_audit_sink" {
  name        = "${var.service_name}-audit-sink"
  destination = "logging.googleapis.com/projects/${var.project_id}/locations/global/buckets/${google_logging_project_bucket_config.agent_audit_logs.bucket_id}"
  filter      = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\""

  unique_writer_identity = true
}

resource "google_logging_metric" "hitl_pause_metric" {
  name        = "${var.service_name}/hitl_pause_count"
  description = "Counts high-stakes return requests paused for Human-in-the-Loop supervisor approval."
  filter      = "resource.type=\"cloud_run_revision\" AND jsonPayload.event_type=\"HITL_CHECKPOINT_TRIGGERED\""

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"
  }

  depends_on = [google_project_service.enabled_apis]
}

# ------------------------------------------------------------------------------
# Cloud Run v2 Service
# ------------------------------------------------------------------------------

resource "google_cloud_run_v2_service" "retail_returns_agent" {
  name     = var.service_name
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.agent_runner.email

    scaling {
      min_instance_count = 1
      max_instance_count = 10
    }

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.agent_repo.repository_id}/${var.service_name}:${var.image_tag}"

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
      }

      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }

      env {
        name  = "GOOGLE_CLOUD_LOCATION"
        value = "global"
      }

      env {
        name  = "GOOGLE_GENAI_USE_VERTEXAI"
        value = "True"
      }

      env {
        name  = "APP_ENV"
        value = "production"
      }

      env {
        name  = "LOG_LEVEL"
        value = "INFO"
      }

      env {
        name = "HITL_WEBHOOK_SECRET"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.hitl_webhook_secret.secret_id
            version = "latest"
          }
        }
      }
    }
  }

  depends_on = [
    google_project_service.enabled_apis,
    google_artifact_registry_repository.agent_repo,
    google_secret_manager_secret_iam_member.hitl_webhook_secret_access,
  ]
}

# ------------------------------------------------------------------------------
# Outputs
# ------------------------------------------------------------------------------

output "cloud_run_service_uri" {
  description = "Public HTTPS endpoint of the deployed Cloud Run service."
  value       = google_cloud_run_v2_service.retail_returns_agent.uri
}

output "artifact_registry_repo" {
  description = "Artifact Registry repository URL for pushing agent container images."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.agent_repo.repository_id}"
}

output "service_account_email" {
  description = "Service account identity running the Retail Returns Concierge."
  value       = google_service_account.agent_runner.email
}
