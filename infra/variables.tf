variable "project_id" {
  description = "GCP project ID where the infra is provisioned."
  type        = string
}

variable "region" {
  description = "GCP region for Cloud Run, Artifact Registry, and the dashboard."
  type        = string
  default     = "us-central1"
}

variable "service_name" {
  description = "Name of the Cloud Run service."
  type        = string
  default     = "latam-delay-api"
}

variable "ar_repo_id" {
  description = "Artifact Registry repository ID for Docker images."
  type        = string
  default     = "latam-images"
}

variable "deployer_sa_id" {
  description = "Service account ID used by GitHub Actions to deploy (Part IV)."
  type        = string
  default     = "latam-deployer"
}
