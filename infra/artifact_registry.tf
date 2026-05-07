resource "google_artifact_registry_repository" "api" {
  location      = var.region
  repository_id = var.ar_repo_id
  description   = "Docker images for the LATAM delay prediction API"
  format        = "DOCKER"
}
