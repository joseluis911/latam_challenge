# Service account used by the GitHub Actions CD pipeline (Part IV) to push
# images to Artifact Registry and update the Cloud Run service.
resource "google_service_account" "deployer" {
  account_id   = var.deployer_sa_id
  display_name = "GitHub Actions deployer"
  description  = "Used by CI/CD to push images to AR and deploy to Cloud Run"
}

# Push Docker images to Artifact Registry.
resource "google_project_iam_member" "deployer_ar_writer" {
  project = var.project_id
  role    = "roles/artifactregistry.writer"
  member  = "serviceAccount:${google_service_account.deployer.email}"
}

# Create / update Cloud Run revisions.
resource "google_project_iam_member" "deployer_run_admin" {
  project = var.project_id
  role    = "roles/run.admin"
  member  = "serviceAccount:${google_service_account.deployer.email}"
}

# Allow the deployer SA to act as the Cloud Run runtime service account
# (required when updating a Cloud Run service that runs as another SA).
resource "google_project_iam_member" "deployer_sa_user" {
  project = var.project_id
  role    = "roles/iam.serviceAccountUser"
  member  = "serviceAccount:${google_service_account.deployer.email}"
}
